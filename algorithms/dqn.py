import torch
import torch.optim as optim
import torch.nn.functional as F 

from typing import Any

from algorithms.base import BaseAlgorithm
from algorithms.helper import ActionSampler, build_cosine_warmup_schedulers
from networks.build_network import build_network
from dataclass import BUFFER_MAPPING
from dataclass.primitives import BatchedActionOutput, BatchedTransition
from configs.config import TrainConfig
from trainer.helper import RunResults


class DQN(BaseAlgorithm):
    def __init__(self, cfg: TrainConfig, obs_space, act_space, device):
        """ Implementation of https://arxiv.org/pdf/1312.5602. """
        super().__init__(cfg=cfg, obs_space=obs_space, act_space=act_space, device=device)
        self._instantiate_buffer()
        self._instantiate_networks()
        self._instantiate_optimizer()
        self._instantiate_lr_schedulers()
        self.sampler = ActionSampler(self.cfg.sampler)
        self.step_info = {
            "update_num": 0,
            "rollout_steps": 0 
        }

    # =========================================
    # Instantiation Helpers
    # =========================================
    def _instantiate_buffer(self):
        """ Set up replay buffer / PER. """
        buffer_type = self.cfg.algo.extra["buffer_type"]
        buffer_cls = BUFFER_MAPPING[buffer_type]
        self.replay_buffer = buffer_cls(self.cfg.algo.extra["buffer_size"], self.cfg)

    def _instantiate_networks(self):
        """ Helper function to set-up all our networks. """
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
        self.networks["q_2"].eval()   # always set to eval for this

    def _instantiate_optimizer(self):
        self.optimizers = {
            name: optim.Adam(model.parameters(), lr=self.cfg.algo.lr_start) for name, model in self.networks.items() if name == "q_1"
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
        - eval_mode (bool): Whether to compute in eval mode or not. If using NoisyLinear (noisy nets) this makes them deterministic

        Returns an ActionOutput instance with shapes of 'B x C' for each element
        """
        obs = obs.to(self.device)
        
        # Non-eval mode samples (if using eps-greedy, for ex.), retains stochasticity if using noisy nets, or keeps dropout active if using dropout
        if eval_mode:
            self.networks['q_1'].eval()
            with torch.no_grad():
                action_values = self.networks['q_1'](obs)
            actions = action_values.argmax(dim=1, keepdim=True)
        else:
            self.networks['q_1'].train()
            with torch.no_grad():
                action_values = self.networks['q_1'](obs)
            placeholder_action = torch.zeros((action_values.shape[0], 1), device=action_values.device, dtype=torch.long)
            sampled = self.sampler.sample(BatchedActionOutput(placeholder_action, {"action_values": action_values}))
            actions = sampled.action
        
        return BatchedActionOutput(actions, {"action_values": action_values})
        
    def observe(self, transition: BatchedTransition) -> list[RunResults]:
        """ Given a transition, store information in our buffer.

        Inputs:
        - transition: For each element in transition, it should be of shape 'B x C' where 'C' can be any arbtirary shape (so long as that's what we're working with in our networks)
        """
        completed_episodes = self.replay_buffer.add(transition)
        if not completed_episodes:
            return []
        
        observed_results = []
        for completed in completed_episodes:
            avg_reward = completed.reward.item()
            action_values = completed.act.info['action_values']
            val_mean, val_std = action_values.mean().item(), action_values.std().item()

            observed_results.extend([
                RunResults("Avg. Episodic Reward", avg_reward, "batched_mean"),
                RunResults("Value Estimates", (val_mean, val_std), "batched_mean", category="val_est", value_format="mean_std", write_to_file=True, smoothing=False, aggregation_steps=50),
            ])
        
        if self.cfg.sampler.name == "epsilon_greedy": 
            observed_results.append(RunResults("Avg. Epsilon", self.sampler.get_epsilon(), "mean"))

        return observed_results

    def update(self) -> list[RunResults]:
        # Will do this at first step then every 'overwrite_target_net_grad_updates' update calls
        if self.step_info["update_num"] % self.cfg.algo.extra["overwrite_target_net_grad_updates"] == 0:
            self._copy_q1_to_q2()

        self.step_info["update_num"] += 1
        gamma = self.cfg.algo.gamma
        obs, actions, n_rewards, next_obs, terminated, truncated, act_info, info, weights, actual_n = self.replay_buffer.sample(
            self.cfg.algo.batch_size, self.device
        )
        self.optimizers['q_1'].zero_grad()   # only doing updates on q_1

        actions = actions.long()
        n_rewards = n_rewards.float()

        # Bootstrap using your target net: max Q_2(s', a). DDQN improves upon this by choosing your action greedily based on Q_1 then getting action_values_target based on the valuation of that in Q_2
        with torch.no_grad():
            action_values_target = self.networks["q_2"](next_obs).max(dim=1, keepdim=True).values       # B x 1

        # Compute n-step bootstrapped values: R_n + gamma^n * Q(s_{t+n}, a*)
        done = terminated.float()
        gamma_n = (gamma ** actual_n.float())                                                           # B x 1
        target_values = n_rewards + gamma_n * (1.0 - done) * action_values_target                       # B x 1

        # Optimize
        q_values = self.networks['q_1'](obs)
        q_taken = q_values.gather(1, actions)
        td_errors = (target_values - q_taken).detach()
        per_sample = F.smooth_l1_loss(q_taken, target_values, reduction="none").squeeze(-1)
        loss = (weights * per_sample).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks["q_1"].parameters(), max_norm=10.0)
        self.optimizers['q_1'].step()
        
        # Various clean-up + Logging
        current_env_steps = int(self.step_info["rollout_steps"])
        current_lr = self.lr_schedulers["q_1"].get_last_lr()
        self.lr_schedulers['q_1'].step(current_env_steps)       
        residual = td_errors.abs().flatten().tolist()
        self.replay_buffer.update(td_errors, step=current_env_steps)            # Necessary to pass info back to buffer if using PER (i.e., update priority)

        update_results = [
            RunResults("Loss", loss.mean().item(), "batched_mean", category="loss", write_to_file=True, smoothing=False, aggregation_steps=50),
            RunResults("Avg. Learning Rate", current_lr, "mean"),
            RunResults("Residual", residual, "concat", category="residual"),
        ]

        return update_results

    def ready_to_update(self) -> bool:
        min_steps = max(self.cfg.algo.extra.get("warmup_buffer_size", 0), self.cfg.algo.batch_size)
        return len(self.replay_buffer) >= min_steps


    # =================================
    # Defined at the base level
    # =================================
    def save(self, path: str) -> None:
        return super().save(path)

    @classmethod
    def load(cls, path: str, override_cfg: Any = None) -> "DQN":
        algo = super().load(path, override_cfg)
        assert isinstance(algo, DQN), f"Loaded algo type {type(algo)} does not match expected {DQN}"
        return algo

    # =================================
    # Other helper methods
    # =================================
    def _copy_q1_to_q2(self):
        # Copy weights from q1 to q2
        self.networks["q_2"].load_state_dict(self.networks["q_1"].state_dict())
        self.networks["q_2"].eval() # should always be in eval mode

    @staticmethod
    def _get_grad_magnitudes(model: torch.nn.Module, lr: float, eps: float = 1e-12) -> dict[str, float]:
        out = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().norm()
            w = p.detach().norm()
            out[name] = (lr * g / (w + eps)).item()
        return out