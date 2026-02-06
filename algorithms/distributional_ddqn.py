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


class Distributional_QN(BaseAlgorithm):
    def __init__(self, cfg: TrainConfig, obs_space, act_space, device):
        """ 
        Implementation of https://arxiv.org/pdf/1707.06887. This is 'Distributional RL' and is used in Rainbow (https://arxiv.org/pdf/1710.02298).

        Core idea is instead of predicting a single value per action, you learn a distribution over possible returns for each action (in predetermined 'value buckets'). Then you bootstrap learning the distribution (not the Q-value) -- minimizing KL-Divergence between your prediction at time 't' and your n-step future prediction.
        """
        super().__init__(cfg=cfg, obs_space=obs_space, act_space=act_space, device=device)
        self._instantiate_buffer()
        self._instantiate_networks()
        self._instantiate_optimizer()
        self._instantiate_lr_schedulers()
        self._instantiate_atom_buffer()
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

    def _instantiate_atom_buffer(self):
        """ Precompute the atom-value vector based on V_min + i * (V_max - V_min) / (N-1) """
        self.v_min = self.cfg.algo.extra["dist_rl_vmin"]
        self.v_max = self.cfg.algo.extra["dist_rl_vmax"]
        self.num_atoms = self.cfg.algo.extra["dist_num_atoms"]
        self.atom_values = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)   # 1-D tensor
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)


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
                action_atom_values = self.networks['q_1'](obs)                  # B x A x N
                action_atom_probs = torch.softmax(action_atom_values, dim=2)    # B x A x N
                action_values = action_atom_probs @ self.atom_values            # B x A
            actions = action_values.argmax(dim=1, keepdim=True)
        else:
            self.networks['q_1'].train()
            with torch.no_grad():
                action_atom_values = self.networks['q_1'](obs)
                action_atom_probs = torch.softmax(action_atom_values, dim=2)
                action_values = action_atom_probs @ self.atom_values
            placeholder_action = torch.zeros((action_values.shape[0], 1), device=action_values.device, dtype=torch.long)
            sampled = self.sampler.sample(BatchedActionOutput(placeholder_action, {"action_values": action_values}))
            actions = sampled.action
        
        return BatchedActionOutput(actions, {"action_values": action_values})
        
    def observe(self, transition: BatchedTransition) -> list[RunResults]:
        """ Given a transition, store information in our buffer.

        Inputs:
        - transition: For each element in transition, it should be of shape 'B x C' where 'C' can be any arbtirary shape (so long as that's what we're working with in our networks)
        """
        # Remains the same since your buffer / buffer computations don't change from DQN / DDQN
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

        self.optimizers['q_1'].zero_grad()   # only doing updates on q_1
        
        # Sample from buffer
        self.step_info["update_num"] += 1
        gamma = self.cfg.algo.gamma
        obs, actions, n_rewards, next_obs, terminated, truncated, act_info, info, weights, actual_n = self.replay_buffer.sample(
            self.cfg.algo.batch_size, self.device
        )
        actions = actions.long()
        n_rewards = n_rewards.float()
        done = terminated.float()
        gamma_n = (gamma ** actual_n.float())

        # Action selection a* = argmax_a Q_1(s', a)
        with torch.no_grad():
            self.networks["q_1"].eval()
            action_atom_values = self.networks['q_1'](next_obs)             # B x A x N
            action_atom_probs = torch.softmax(action_atom_values, dim=2)    # B x A x N
            next_q_online  = action_atom_probs @ self.atom_values           # B x A
            greedy_actions = next_q_online.argmax(dim=1, keepdim=True)
            self.networks["q_1"].train()
        
        # Action evaluation w/ Q_2 (target net) -- this is the 'Double' part of DDQN
        with torch.no_grad():
            B, A, N = action_atom_values.shape
            next_logits_target = self.networks['q_2'](next_obs)                      # B x A x N
            next_probs_target = torch.softmax(next_logits_target, dim=2)             # B x A x N
            greedy_actions_idx = greedy_actions.view(B, 1, 1).expand(B, 1, N)        # B x 1 x N
            next_dist = next_probs_target.gather(1, greedy_actions_idx).squeeze(1)   # B x N

            # Compute distriutional update
            z = self.atom_values.view(1, N)                     # 1 x N
            tz = n_rewards + (1.0 - done) * gamma_n * z         # B x N
            tz = tz.clamp(self.v_min, self.v_max)               # B x N in range of [v_min, v_max]
            b = (tz - self.v_min) / self.delta_z                # In range [0, N-1] (but not discrete)
            
            # Distribute updated prob mass using linear interp update
            # Added in case where b is an integer; if so need to just put all prob mass there
            l = b.floor().long()
            u = b.ceil().long()
            l = l.clamp(0, N - 1)
            u = u.clamp(0, N - 1)
            m = torch.zeros((B, N), device=self.device, dtype=next_dist.dtype)
            wl = (u.float() - b)
            wu = (b - l.float())
            m.scatter_add_(1, l, next_dist * wl)
            m.scatter_add_(1, u, next_dist * wu)
            eq = (u == l).float()
            m.scatter_add_(1, l, next_dist * eq)
            target_dist = m

        # Optimize using Cross-Entropy
        logits = self.networks["q_1"](obs)                  # B x A x N
        log_probs = torch.log_softmax(logits, dim=2)        # B x A x N
        a_idx = actions.view(B, 1, 1).expand(B, 1, N)       # B x 1 x N
        logp_taken = log_probs.gather(1, a_idx).squeeze(1)  # B x N
        per_sample = -(target_dist * logp_taken).sum(dim=1) # B, 
        w = weights.view(-1)                                # B,
        loss = (w * per_sample).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks["q_1"].parameters(), max_norm=10.0)
        current_lr = self.lr_schedulers["q_1"].get_last_lr()
        self.optimizers["q_1"].step()
        current_env_steps = int(self.step_info["rollout_steps"])
        self.lr_schedulers["q_1"].step(current_env_steps)

        # Need to update PER priority (will just use mean dist. for td residuals)
        # with torch.no_grad():
        #     pred_dist = torch.exp(logp_taken)                               # B x N
        #     pred_q = (pred_dist * self.atom_values.view(1, N)).sum(1)       # B,
        #     tgt_q = (target_dist * self.atom_values.view(1, N)).sum(1)      # B,
        #     td_errors = (tgt_q - pred_q).unsqueeze(1)
        td_errors = per_sample.detach().unsqueeze(1)   # This is just using our cross-entropy loss for 'residual' errors

        self.replay_buffer.update(td_errors, step=current_env_steps)
        residual = td_errors.abs().flatten().tolist()

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
    def load(cls, path: str, override_cfg: Any = None) -> "Distributional_QN":
        algo = super().load(path, override_cfg)
        assert isinstance(algo, Distributional_QN), f"Loaded algo type {type(algo)} does not match expected {Distributional_QN}"
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
