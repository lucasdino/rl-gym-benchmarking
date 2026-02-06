import os, time, random
import imageio, wandb

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from algorithms.get_algorithm import get_algorithm
from configs.config import config_to_dict
from dataclass.primitives import BatchedTransition
from trainer.helper import ResultLogger, RunResults, record_vec_grid_video, make_vec_envs



class Trainer():
    def __init__(self, cfg, seed: int | None = None, seed_idx: int = 0):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed_idx = seed_idx
        self.current_seed = seed if seed is not None else cfg.algo.seed
        print(f"Using device: {self.device}")
        self._set_global_seeds(self.current_seed)

        start_time = time.time()
        self.train_envs = make_vec_envs(cfg.env, vectorization_mode="async", eval=False)
        self.eval_envs = make_vec_envs(cfg.env, vectorization_mode="sync", eval=True)
        self.algo = get_algorithm(cfg.algo.name)(cfg, self.train_envs.single_observation_space, self.train_envs.single_action_space, self.device)
        print(f"Successfully initialized env + algo in {(time.time() - start_time):.2f}s")

        self.wandb_run = None
        if "wandb" in self.cfg.train.logging_method:
            run_name = self._get_run_name()
            self.wandb_run = wandb.init(
                project=self.cfg.train.wandb_project,
                group=self.cfg.train.wandb_group,
                name=f"{run_name}_seed{seed_idx}",
                config=config_to_dict(self.cfg),
            )

        self.train_logger = ResultLogger(
            "Train",
            self.cfg.train.logging_method,
            max_steps=self.cfg.train.total_env_steps,
            wandb_run=self.wandb_run,
            dict_concat_stacked=True,
            seed_idx=seed_idx,
        )
        self.eval_logger = ResultLogger(
            "Eval",
            self.cfg.train.logging_method,
            max_steps=self.cfg.train.total_env_steps,
            wandb_run=self.wandb_run,
            dict_concat_stacked=True,
            seed_idx=seed_idx,
        )


    def train(self):
        print(f"Starting training ({self.cfg.train.total_env_steps} steps) [Seed {self.seed_idx}: {self.current_seed}]:\n")
        env_seed = self.current_seed
        self.num_env_steps, last_evaluation_step, num_meta_steps = 0, 0, 0
        last_log_step, log_interval = 0, self.cfg.train.log_interval
        action_repeat = max(1, int(self.cfg.algo.use_action_for_steps_train))

        cur_obs, _ = self.train_envs.reset(seed=env_seed)
        _, _ = self.eval_envs.reset(seed=env_seed + 10_000)
        while self.num_env_steps <= self.cfg.train.total_env_steps:
            cur_obs_tensor = self._to_tensor_obs(cur_obs)
            batched_actions = self.algo.act(cur_obs_tensor, eval_mode=False)
            env_actions = self._actions_to_env(batched_actions.action)

            for _ in range(action_repeat):   # This allows us to use the same action for 'n' frames
                next_obs, rewards, terminations, truncations, infos = self.train_envs.step(env_actions)
                transition = BatchedTransition(
                    obs = cur_obs_tensor,
                    act = batched_actions,
                    reward = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(1),
                    next_obs = self._to_tensor_obs(next_obs),
                    terminated = torch.as_tensor(terminations, device=self.device, dtype=torch.bool).unsqueeze(1),
                    truncated = torch.as_tensor(truncations, device=self.device, dtype=torch.bool).unsqueeze(1),
                    info = None
                )
                self.num_env_steps += rewards.shape[0]
                self.algo.step_info["rollout_steps"] = self.num_env_steps

                observe_results = self.algo.observe(transition)
                update_results = self.algo.update() if self.algo.ready_to_update() and num_meta_steps % self.cfg.algo.update_every_steps == 0 else None
                
                # Logging
                self.train_logger.update(observe_results)
                self.train_logger.update(update_results)
                self.train_logger.flush(step=self.num_env_steps)  # Check aggregation thresholds for CSV writes
                if (self.num_env_steps - last_log_step) >= log_interval:
                    self.train_logger.log(step=self.num_env_steps, console_log=self.cfg.train.console_log_train)
                    self.train_logger.zero()
                    last_log_step = self.num_env_steps

                if (self.num_env_steps - last_evaluation_step) >= self.cfg.train.eval_interval:
                    mean_eval_return = self.eval()
                    last_evaluation_step = self.num_env_steps
                    # Test for early exit
                    if (self.cfg.train.threshold_exit_training_on_eval_score is not None and mean_eval_return >= self.cfg.train.threshold_exit_training_on_eval_score):
                        print(f"\nTraining terminated early: ({mean_eval_return:.1f} >= {self.cfg.train.threshold_exit_training_on_eval_score:.1f})")
                        self._final_flush()
                        return

                cur_obs = next_obs
                cur_obs_tensor = self._to_tensor_obs(cur_obs)
                num_meta_steps += 1

                if self.num_env_steps >= self.cfg.train.total_env_steps:
                    break

            if self.num_env_steps >= self.cfg.train.total_env_steps:
                break
        
        self._final_flush()
            
    def _final_flush(self):
        """Flush all pending train logger data at end of training."""
        final_step = self.num_env_steps
        if self.cfg.train.total_env_steps is not None:
            final_step = min(final_step, self.cfg.train.total_env_steps)
        self.train_logger.flush(step=final_step, force=True)
        self.train_logger.log(step=final_step, console_log=False)

    def eval(self, save_video: bool = False) -> float:
        mean_reward, std_reward, all_rewards = self.eval_env(save_video=save_video)
        env_steps = self.num_env_steps if hasattr(self, "num_env_steps") else 0
        eval_runs = [
            RunResults("Avg. Episodic Reward", mean_reward, "mean"),
            RunResults("Episodic Rewards", all_rewards, "concat", category="histogram"),
            RunResults("Eval Reward", (env_steps, mean_reward, std_reward), "accumulating_writes", category="other", smoothing=False, show_ci=True, publish_to_wandb=True),
        ]
        self.eval_logger.update(eval_runs)
        self.eval_logger.log(step=env_steps)
        self.eval_logger.zero()
        return mean_reward

    def eval_env(self, save_video: bool = False) -> tuple[float, float, np.ndarray]:
        """Returns (mean_reward, std_reward, all_episode_rewards)."""
        eval_envs = self.cfg.train.eval_envs
        num_envs = self.cfg.env.num_envs
        num_batches = eval_envs // num_envs
        action_repeat = max(1, int(self.cfg.algo.use_action_for_steps_eval))
        all_episode_returns: list[np.ndarray] = []

        # Optional: record ONE tiled grid video (one episode per env tile)
        if save_video:
            grid_frames = record_vec_grid_video(
                env_cfg=self.cfg.env,
                algo=self.algo,
                to_tensor_obs=self._to_tensor_obs,
                actions_to_env=self._actions_to_env,
                grid_hw=(4, 4),
                seed=self.cfg.algo.seed + 10_000,
                pad=2,
                vectorization_mode="sync",
            )
            self._save_video(grid_frames)

        # Complete your env rollouts / get scores
        for batch_idx in range(num_batches):
            cur_obs, _ = self.eval_envs.reset()
            done = np.zeros(num_envs, dtype=bool)
            episode_returns = np.zeros(num_envs, dtype=np.float32)
            while not done.all():
                cur_obs_tensor = self._to_tensor_obs(cur_obs)
                batched_actions = self.algo.act(cur_obs_tensor, eval_mode=True)
                env_actions = self._actions_to_env(batched_actions.action)

                for _ in range(action_repeat):
                    next_obs, rewards, terminations, truncations, _ = self.eval_envs.step(env_actions)

                    active = ~done
                    episode_returns[active] += rewards[active]
                    done |= (terminations | truncations)
                    cur_obs = next_obs

                    if done.all():
                        break
            all_episode_returns.append(episode_returns)

        all_episode_rewards = np.concatenate(all_episode_returns, axis=0)
        mean_reward = float(np.mean(all_episode_rewards))
        std_reward = float(np.std(all_episode_rewards))

        return mean_reward, std_reward, all_episode_rewards

    def save_algo(self):
        algo_path = self._get_filepath(type="algo")
        self.algo.save(algo_path)
        print(f"Algo saved to: {algo_path}")

    def close(self):
        """Clean up resources to free memory between seed runs."""
        self.train_envs.close()
        self.eval_envs.close()
        del self.algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_video(self, frames: list[np.ndarray]):
        """Save recorded frames as a video file."""
        video_path = self._get_filepath(type="video")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to: {video_path}")

    # =======================
    # Other helpers
    # =======================
    def send_networks_to_device(self):
        for model in self.algo.networks.values():
            model.to(self.device)
        self.algo.device = self.device
        for attr in ("atom_values",):
            if hasattr(self.algo, attr):
                setattr(self.algo, attr, getattr(self.algo, attr).to(self.device))

    def _get_run_name(self) -> str:
        if self.cfg.train.run_name:
            return self.cfg.train.run_name
        return f"{self.cfg.env.name}_{self.cfg.algo.name}"

    def _get_filepath(self, type):
        run_name = self._get_run_name()
        run_name = run_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if type == "algo":
            algo_dir = self.cfg.train.algo_save_dir
            os.makedirs(algo_dir, exist_ok=True)
            save_path = os.path.join(algo_dir, f"{run_name}_{timestamp}.pkl")
        elif type == "video":
            video_dir = self.cfg.train.video_save_dir
            os.makedirs(video_dir, exist_ok=True)
            save_path = os.path.join(video_dir, f"{run_name}_{timestamp}.mp4")
        else:
            raise ValueError
        return save_path

    def _set_global_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

    def _actions_to_env(self, actions: torch.Tensor) -> np.ndarray:
        actions_np = actions.detach().cpu().numpy()
        if isinstance(self.train_envs.single_action_space, gym.spaces.Discrete) and actions_np.ndim == 2 and actions_np.shape[1] == 1:
            actions_np = actions_np.squeeze(1)
        return actions_np