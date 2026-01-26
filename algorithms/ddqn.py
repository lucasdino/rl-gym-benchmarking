import torch
import torch.optim as optim
import torch.nn.functional as F 

from typing import Any

from networks.build_network import build_network
from networks.helper import ActionSampler, build_cosine_warmup_schedulers
from algorithms.base import BaseAlgorithm
from dataclass.replay_buffer import ReplayBuffer
from dataclass.primitives import BatchedActionOutput, BatchedTransition
from configs.config import TrainConfig
from trainer.helper import RunResults


class DDQN(BaseAlgorithm):
    def __init__(self, cfg: TrainConfig, obs_space, act_space, device):
        """
        Implementation of https://arxiv.org/pdf/1509.06461.
        """
        super().__init__(
            cfg = cfg, 
            obs_space = obs_space, 
            act_space = act_space,
            device = device
        )
        self.replay_buffer = ReplayBuffer(cfg.algo.extra["buffer_size"])
        self.sampler = ActionSampler(self.cfg.sampler)
        self._instantiate_networks()
        self._instantiate_optimizer()
        self._instantiate_lr_schedulers()
        self.step_info = {
            "update_num": 0,
            "rollout_steps": 0 
        }

    # =========================================
    # Instantiation Helpers
    # =========================================
    def _instantiate_networks(self):
        """ Helper function to set-up all our networks """
        self.networks = {}
        for network_cfg in self.cfg.networks.networks.values():
            self.networks[network_cfg.name] = build_network(
                cfg = network_cfg, 
                obs_space = self.obs_space, 
                act_space = self.act_space
            )
        # Sending to GPU (if using GPU)
        for model in self.networks.values():
            model.to(self.device)
        
        assert all([req_net in self.networks for req_net in ("q_1", "q_2")])

    def _instantiate_optimizer(self):
        self.optimizers = {
            name: optim.Adam(model.parameters(), lr=self.cfg.algo.lr_start) for name, model in self.networks.items()
        }

    def _instantiate_lr_schedulers(self):
        self.lr_schedulers = build_cosine_warmup_schedulers(
            self.optimizers,
            total_env_steps=self.cfg.train.total_env_steps,
            warmup_env_steps=self.cfg.algo.lr_warmup_env_steps,
            start_lr=self.cfg.algo.lr_start,
            end_lr=self.cfg.algo.lr_end,
            warmup_start_lr=0.0,
        )


    # =========================================
    # API Functions
    # =========================================
    def act(self, obs: torch.Tensor, *, eval_mode: bool) -> BatchedActionOutput:
        """ Given an observation return an action. Optionally be allowed to set to 'eval' mode. 
        
        Inputs:
        - obs (torch.Tensor): Must be in the form of 'B x C' where 'C' can be any shape so long as that is what is expected by your network.
        - eval_mode (bool): Whether to compute in eval mode or not

        Returns an ActionOutput instance with shapes of 'B x C' for each element
        """
        obs = obs.to(self.device)
        self.networks['q_1'].eval()
        with torch.no_grad():
            action_values = self.networks['q_1'](obs)
        
        if eval_mode:
            actions = action_values.argmax(dim=1, keepdim=True)
        else:
            placeholder_action = torch.zeros((action_values.shape[0], 1), device=action_values.device, dtype=torch.long)
            sampled = self.sampler.sample(BatchedActionOutput(placeholder_action, {"action_values": action_values}))
            actions = sampled.action
        
        return BatchedActionOutput(actions, {"action_values": action_values})
        
    def observe(self, transition: BatchedTransition) -> list[RunResults]:
        """ Given a transition, store information in our replay buffer.

        Inputs:
        - transition: For each element in transition, it should be of shape 'B x C' where 'C' can be any arbtirary shape (so long as that's what we're working with in our networks)
        """
        self.replay_buffer.add(transition)
        
        avg_reward = torch.mean(transition.reward).item()
        avg_action_value = torch.mean(transition.act.info['action_values']).item()
        avg_max_action_value = torch.mean(transition.act.info['action_values'].max(dim=1).values).item()
        avg_selected_action_value = torch.mean(transition.act.info['action_values'].gather(1, transition.act.action.long())).item()

        observed_results = [
            RunResults("Avg. Reward", avg_reward, "mean"),
            RunResults("Avg. Action Value", avg_action_value, "mean"),
            RunResults("Avg. Max Action Value", avg_max_action_value, "mean"),
            RunResults("Avg. Selected Action Value", avg_selected_action_value, "mean"),
            RunResults("Avg. Epsilon", self.sampler.get_epsilon(), "mean")
        ]

        return observed_results

    def update(self) -> list[RunResults]:
        self.networks['q_1'].train()
        if self.step_info["update_num"] % self.cfg.algo.extra["overwrite_target_net_grad_updates"] == 0:
            self._copy_q1_to_q2()

        self.step_info["update_num"] += 1
        obs, actions, rewards, next_obs, terminated, truncated, act_info, info = self.replay_buffer.sample(self.cfg.algo.batch_size, self.device)
        _ = [optimizer.zero_grad() for optimizer in self.optimizers.values()]

        actions = actions.long()
        rewards = rewards.float()

        # Action selection a* = argmax_a Q_1(s', a)
        with torch.no_grad():
            next_action_values = self.networks['q_1'](next_obs)             # B x C
            greedy_actions = next_action_values.argmax(dim=1, keepdim=True)    # B x 1

        # Action evaluation w/ Q_2 (target net)
        with torch.no_grad():
            next_action_values_target = self.networks['q_2'](next_obs)
            action_values_target = next_action_values_target.gather(1, greedy_actions)   # B x 1

        # Compute bootstrapped values
        done = (terminated | truncated).float()
        target_values = rewards + self.cfg.algo.gamma * (1.0 - done) * action_values_target   # B x 1

        # Optimize (use MSE)
        q_values = self.networks['q_1'](obs)
        q_taken = q_values.gather(1, actions)
        loss = F.mse_loss(q_taken, target_values)
        loss.backward()
        _ = [optimizer.step() for optimizer in self.optimizers.values()]
        current_env_steps = int(self.step_info["rollout_steps"])
        _ = [scheduler.step(current_env_steps) for scheduler in self.lr_schedulers.values()]
        current_lr = self.lr_schedulers["q_1"].get_last_lr()

        update_results = [
            RunResults("Avg. Loss", loss.mean().item(), "mean"),
            RunResults("Learning Rate", current_lr, "mean"),
        ]

        return update_results

    def ready_to_update(self) -> bool:
        min_steps = max(self.cfg.algo.extra.get("warmup_buffer_size", 0), self.cfg.algo.batch_size)
        return len(self.replay_buffer) >= min_steps

    def save(self, path: str) -> None:
        # this should save all relevant parts to a pckl file.
        # so should basically make it so we save this entire class and all things associated down
        pass

    @classmethod
    def load(self, path: str, override_cfg: Any = None) -> "DDQN":
        # Should be able to load in a pckl file.
        # override cfg should simply override our cfg
        pass


    # =================================
    # Other helper methods
    # =================================
    def _copy_q1_to_q2(self):
        # Copy weights from q1 to q2
        self.networks["q_2"].load_state_dict(self.networks["q_1"].state_dict())
