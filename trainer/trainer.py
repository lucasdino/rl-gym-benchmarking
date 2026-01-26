import os, time, random
import imageio, wandb

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from algorithms.get_algorithm import get_algorithm
from dataclass.primitives import BatchedTransition
from trainer.helper import ResultLogger, RunResults, record_vec_grid_video



class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self._set_global_seeds(cfg.algo.seed)

        # Initialize env / algo
        start_time = time.time()
        self.train_envs = gym.make_vec(cfg.env.name, num_envs = cfg.env.num_envs, vectorization_mode="async", max_episode_steps=cfg.env.max_episode_steps)
        self.eval_envs = gym.make_vec(cfg.env.name, num_envs = cfg.env.num_envs, vectorization_mode="sync", max_episode_steps=cfg.env.max_episode_steps)
        self.algo = get_algorithm(cfg.algo.name)(cfg, self.train_envs.single_observation_space, self.train_envs.single_action_space, self.device)
        print(f"Successfully initialized env + algo in {(time.time() - start_time):.2f}s")

        self.wandb_run = None
        # if "wandb" in self.cfg.train.logging_method:
        #     self.wandb_run = wandb.init(
        #         project=self.cfg.train.wandb_project,
        #         group=self.cfg.train.wandb_group,
        #         config={
        #             "env": self.cfg.env,
        #             "algo": self.cfg.algo,
        #             "train": self.cfg.train,
        #             "sampler": self.cfg.sampler,
        #         },
        #     )

        self.train_logger = ResultLogger("Train", self.cfg.train.logging_method, max_steps=self.cfg.train.total_env_steps, wandb_run=self.wandb_run)
        self.eval_logger = ResultLogger("Eval", self.cfg.train.logging_method, max_steps=self.cfg.train.total_env_steps, wandb_run=self.wandb_run)


    def train(self):
        print(f"Starting training ({self.cfg.train.total_env_steps} steps).")
        num_env_steps, last_evaluation_step, num_meta_steps = 0, 0, 0
        last_log_step, log_interval = 0, self.cfg.train.log_interval
        env_seed = self.cfg.algo.seed

        cur_obs, _ = self.train_envs.reset(seed=env_seed)
        _, _ = self.eval_envs.reset(seed=env_seed + 10_000)
        while num_env_steps <= self.cfg.train.total_env_steps:
            cur_obs_tensor = self._to_tensor_obs(cur_obs)
            batched_actions = self.algo.act(cur_obs_tensor, eval_mode=False)
            env_actions = self._actions_to_env(batched_actions.action)
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
            num_env_steps += rewards.shape[0]
            self.algo.step_info["rollout_steps"] = num_env_steps
            observe_results = self.algo.observe(transition)

            update_results = None
            if self.algo.ready_to_update() and num_meta_steps % self.cfg.algo.update_every_steps == 0:
                update_results = self.algo.update()
            
            # Logging
            self.train_logger.update(observe_results)
            self.train_logger.update(update_results)
            if (num_env_steps - last_log_step) >= log_interval:
                self.train_logger.log(step=num_env_steps, console_log=self.cfg.train.console_log_train)
                self.train_logger.zero()
                last_log_step = num_env_steps

            if (num_env_steps - last_evaluation_step) >= self.cfg.train.eval_interval:
                self.eval(num_env_steps)
                last_evaluation_step = num_env_steps

            cur_obs = next_obs
            num_meta_steps += 1

        # Run final eval (with video saving if enabled)
        self.eval(num_env_steps, save_video=self.cfg.train.save_video_last_eval)
        # self.algo.save() # Will implement in the future

    def eval(self, num_env_steps: int, save_video: bool = False):
        eval_runs = self.eval_env(save_video=save_video)
        self.eval_logger.update(eval_runs)
        self.eval_logger.log(step=num_env_steps)
        self.eval_logger.zero()

    def eval_env(self, save_video: bool = False) -> list[RunResults]:
        eval_envs = self.cfg.train.extra["eval_envs"]
        num_envs = self.cfg.env.num_envs
        num_batches = eval_envs // num_envs

        all_episode_returns: list[np.ndarray] = []
        selected_action_value_sum = 0.0
        selected_action_value_count = 0

        # Optional: record ONE tiled grid video (one episode per env tile)
        grid_frames = []
        if save_video:
            side = int(round(num_envs ** 0.5))
            assert side * side == num_envs, f"num_envs={num_envs} must be a perfect square for a square grid"
            grid_frames = record_vec_grid_video(
                env_id=self.cfg.env.name,
                num_envs=num_envs,
                max_episode_steps=self.cfg.env.max_episode_steps,
                algo=self.algo,
                to_tensor_obs=self._to_tensor_obs,
                actions_to_env=self._actions_to_env,
                grid_hw=(side, side),
                seed=self.cfg.algo.seed + 10_000,
                pad=2,
                vectorization_mode="sync",
            )
        for batch_idx in range(num_batches):
            cur_obs, _ = self.eval_envs.reset()
            done = np.zeros(num_envs, dtype=bool)
            episode_returns = np.zeros(num_envs, dtype=np.float32)
            while not done.all():
                cur_obs_tensor = self._to_tensor_obs(cur_obs)
                batched_actions = self.algo.act(cur_obs_tensor, eval_mode=True)
                env_actions = self._actions_to_env(batched_actions.action)
                next_obs, rewards, terminations, truncations, _ = self.eval_envs.step(env_actions)

                active = ~done
                episode_returns[active] += rewards[active]

                if batched_actions.info is not None and "action_values" in batched_actions.info:
                    action_values = batched_actions.info["action_values"]
                    actions = batched_actions.action
                    selected = action_values.gather(1, actions.long())
                    active_mask = torch.as_tensor(active, device=selected.device)
                    selected_action_value_sum += selected[active_mask].sum().item()
                    selected_action_value_count += int(active_mask.sum().item())

                done |= (terminations | truncations)
                cur_obs = next_obs

            all_episode_returns.append(episode_returns)

        # Save tiled grid video
        if save_video and grid_frames:
            self._save_video(grid_frames)

        all_episode_returns = np.concatenate(all_episode_returns, axis=0)
        mean_return = float(np.mean(all_episode_returns))
        mean_selected_action_value = None
        if selected_action_value_count > 0:
            mean_selected_action_value = selected_action_value_sum / selected_action_value_count

        return [
            RunResults("Eval Mean Return", mean_return, "mean"),
            RunResults("Eval Mean Selected Action Value", mean_selected_action_value if mean_selected_action_value is not None else 0.0, "mean"),
        ]

    def _save_video(self, frames: list[np.ndarray]):
        """Save recorded frames as a video file."""        
        run_name = self.cfg.train.wandb_run if self.cfg.train.wandb_run else f"{self.cfg.env.name}_{self.cfg.algo.name}"
        run_name = run_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        video_dir = self.cfg.train.video_save_dir
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{run_name}.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to: {video_path}")


    # =======================
    # Other helpers
    # =======================
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