import torch
import torch.optim as optim

from typing import Any

from networks.build_network import build_network
from algorithms.base import BaseAlgorithm
from dataclass.replay_buffer import ReplayBuffer
from dataclass.primitives import BatchedActionOutput, BatchedTransition
from configs.config import TrainConfig


class SoftActorCritic(BaseAlgorithm):
    def __init__(self, cfg: TrainConfig, obs_space, act_space, device):
        """
        Implementation of https://arxiv.org/pdf/1801.01290.
        """
        super().__init__(
            cfg = cfg, 
            obs_space = obs_space, 
            act_space = act_space,
            device = device
        )
        self.replay_buffer = ReplayBuffer(cfg.algo.extra["buffer_size"])
        self._instantiate_networks()
        self._instantiate_optimizer()


    # =========================================
    # Instantiation Helpers
    # =========================================
    def _instantiate_networks(self):
        """ Helper function to set-up all our networks """
        self.networks = {}
        for network_cfg in self.cfg.networks:
            self.networks[network_cfg.name] = build_network(
                cfg = network_cfg, 
                obs_space = self.obs_space, 
                act_space = self.act_space
            )
        # Sending to GPU (if using GPU)
        for model in self.networks.values():
            model.to(self.device)
        
        assert all([req_net in self.networks for req_net in ("q_1", "q_2", "policy", "value_network", "target_value_network")])

    def _instantiate_optimizer(self):
        self.optimizers = {
            name: optim.Adam(model.parameters(), lr=self.cfg.algo.lr) for name, model in self.networks.items()
        }


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
        obs.to(self.device)
        self.networks['policy'].eval()
        with torch.no_grad():
            raw_logits = self.networks['policy'](obs)

            # Need to compute probs for each action (do softmax)
            # Also for sampling this should be based on what it says in paper, probably ep greedy? Will need this as another param I can set in my config
        
        # Move to CPU and send back

        
    def observe(self, transition: BatchedTransition) -> None:
        """ Given a transition, store information in our replay buffer.

        Inputs:
        - transition: For each element in transition, it should be of shape 'B x C' where 'C' can be any arbtirary shape (so long as that's what we're working with in our networks)
        """
        self.replay_buffer.add(transition)

    def update(self) -> dict[str, Any]:
        obs, actions, rewards, next_obs, terminated, truncated, act_info, info = self.replay_buffer.sample(self.cfg.algo.batch_size, self.device)

        # Now update our various networks using SAC formulation
        
        pass

    def ready_to_update(self) -> bool:
        return max(self.cfg.algo.extra.get("warmup_buffer_size", 0), self.cfg.algo.batch_size) >= len(self.replay_buffer)

    def save(self, path: str) -> None:
        # this should save all relevant parts to a pckl file.
        # so should basically make it so we save this entire class and all things associated down
        pass

    @classmethod
    def load(self, path: str, override_cfg: Any = None) -> "SoftActorCritic":
        # Should be able to load in a pckl file.
        # override cfg should simply override our cfg
        pass