import time
import random

import torch
import numpy as np
import gymnasium as gym

from algorithms.get_algorithm import get_algorithm
from dataclass.primitives import BatchedTransition



class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self._set_global_seeds(cfg.algo.seed)

        # Initialize env / algo
        start_time = time.time()
        self.train_envs = gym.make_vec(
            cfg.env.name, 
            num_envs = cfg.env.num_envs, 
            vectorization_mode="async", 
        )
        self.eval_envs = gym.make_vec(
            cfg.env.name, 
            num_envs = cfg.env.num_envs, 
            vectorization_mode="async", 
        )
        self.algo = get_algorithm(cfg.algo.name)(cfg, self.train_envs.single_observation_space, self.train_envs.single_action_space, self.device)
        print(f"Successfully initialized env + algo in {(time.time() - start_time)/1e9:.2f}s")

        # Start WandB --> hold on this let's just print via console for now


    def train(self):
        train_start_time = time.time()
        print(f"Starting training ({self.cfg.train.total_env_steps} steps).")
        num_env_steps = 0

        cur_obs, _ = self.train_envs.reset(self.cfg.env.seed)
        _, _ = self.eval_envs.reset(self.cfg.env.seed + 10_000)
        while num_env_steps <= self.cfg.train.total_env_steps:
            actions = self.algo.act(cur_obs)
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
            transition = BatchedTransition(
                obs = cur_obs,
                act = actions,
                reward = rewards,
                next_obs = next_obs,
                terminated = terminations,
                truncated = truncations,
            )
            num_env_steps += rewards.shape[0]
            self.algo.observe(transition)
            
            if self.algo.ready_to_update():
                results = self.algo.update()
                # log / print our results (for now)

            # check and see if we exceed our cfg.train.eval_interval --> then run eval

    def eval(self):
        # implement an eval loop -- should reset our envs after they're all terminated
        # Start by resetting the env -- then 
        pass


    # =======================
    # Other helpers
    # =======================
    def _set_global_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)