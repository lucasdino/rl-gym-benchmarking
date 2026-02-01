import torch

from configs.config import TrainConfig
from dataclass.base import BaseBuffer
from dataclass.primitives import BatchedActionOutput, BatchedTransition



class PriorityExperienceReplay(BaseBuffer):
    def __init__(self, buffer_length: int, cfg: TrainConfig, epsilon: float = 1e-6, alpha: float | None = None):
        """ Implementation of this paper: https://arxiv.org/pdf/1511.05952. Heavily vibe-coded but I audited the math. """
        self.buffer_length = buffer_length
        self.cfg = cfg
        self._cur_idx = 0
        self._full = False
        self._initialized = False
        self._epsilon = float(epsilon)
        cfg_extra = cfg.algo.extra or {}

        # These alpha / beta_start / beta_end use params + annealing schedule from the dueling nets paper (https://arxiv.org/pdf/1511.06581)
        self._alpha = cfg_extra.get("per_alpha", alpha if alpha is not None else 0.7)
        self._beta_start = cfg_extra.get("per_beta_start", 0.5)
        self._beta_end = cfg_extra.get("per_beta_end", 1.0)
        self._beta = cfg_extra.get("per_beta", self._beta_start)
        self._total_steps = cfg.train.total_env_steps
        self._priorities = torch.zeros(buffer_length)
        self._max_priority = 1.0
        self._last_indices = None
        self._rollout_cache: dict[int, list[dict]] = {}
        self._rollout_ids: set[int] = set()
        self._idx_to_rollout_id: torch.Tensor | None = None
        self._valid_count = 0
        self._gamma = cfg.algo.gamma
        self._n_step = cfg.algo.n_step
        self._last_update_step = 0


    # ===================================
    # Helpers
    # ===================================
    def _allocate_buffer(self, template: torch.Tensor) -> torch.Tensor:
        return torch.empty((self.buffer_length, *template.shape[1:]), dtype=template.dtype, device=template.device)

    def _initialize_from_transition(self, transition: BatchedTransition) -> None:
        self._obs = self._allocate_buffer(transition.obs)
        self._act_action = self._allocate_buffer(transition.act.action)
        self._reward = self._allocate_buffer(transition.reward)
        self._next_obs = self._allocate_buffer(transition.next_obs)
        self._terminated = self._allocate_buffer(transition.terminated)
        self._truncated = self._allocate_buffer(transition.truncated)
        self._act_info = {k: self._allocate_buffer(v) for k, v in transition.act.info.items()} if transition.act.info else None
        self._info = {k: self._allocate_buffer(v) for k, v in transition.info.items()} if transition.info else None
        self._priorities = self._priorities.to(transition.obs.device)
        self._idx_to_rollout_id = torch.full((self.buffer_length,), -1, dtype=torch.long, device=transition.obs.device)
        self._initialized = True

    def _new_rollout_id(self) -> int:
        while True:
            rid = torch.randint(0, 2**31, (1,)).item()
            if rid not in self._rollout_ids and rid != -1:
                self._rollout_ids.add(rid)
                return rid

    def _flush_rollout(self, env_idx: int) -> BatchedTransition:
        cache = self._rollout_cache.pop(env_idx)
        n = len(cache)
        rollout_id = self._new_rollout_id()
        device = self._obs.device
        indices = (torch.arange(n, device=device) + self._cur_idx) % self.buffer_length

        overwritten_rids = self._idx_to_rollout_id[indices]
        unique_overwritten = overwritten_rids.unique()
        for rid in unique_overwritten.tolist():
            if rid != -1:
                self._rollout_ids.discard(rid)
        self._idx_to_rollout_id[indices] = rollout_id

        was_invalid = overwritten_rids == -1
        self._valid_count += was_invalid.sum().item()

        self._priorities[indices] = self._max_priority ** self._alpha

        obs = torch.stack([c["obs"] for c in cache])
        act_action = torch.stack([c["act_action"] for c in cache])
        reward = torch.stack([c["reward"] for c in cache])
        next_obs = torch.stack([c["next_obs"] for c in cache])
        terminated = torch.stack([c["terminated"] for c in cache])
        truncated = torch.stack([c["truncated"] for c in cache])

        self._obs[indices] = obs
        self._act_action[indices] = act_action
        self._reward[indices] = reward
        self._next_obs[indices] = next_obs
        self._terminated[indices] = terminated
        self._truncated[indices] = truncated

        act_info = None
        if self._act_info:
            act_info = {}
            for k in self._act_info:
                stacked = torch.stack([c["act_info"][k] for c in cache])
                self._act_info[k][indices] = stacked
                act_info[k] = stacked.mean(dim=0, keepdim=True)

        info = None
        if self._info:
            info = {}
            for k in self._info:
                stacked = torch.stack([c["info"][k] for c in cache])
                self._info[k][indices] = stacked
                info[k] = stacked[-1:]

        self._cur_idx = (self._cur_idx + n) % self.buffer_length
        if not self._full and self._cur_idx < n:
            self._full = True

        return BatchedTransition(
            obs[0:1], BatchedActionOutput(act_action[0:1], act_info),
            reward.sum(dim=0, keepdim=True), next_obs[-1:], terminated[-1:], truncated[-1:], info
        )


    # ===================================
    # External functionality
    # ===================================
    def add(self, transition: BatchedTransition) -> list[BatchedTransition]:
        """Returns list of completed episode transitions (may be empty)."""
        if not self._initialized:
            self._initialize_from_transition(transition)

        batch_size = transition.obs.shape[0]
        done = (transition.terminated | transition.truncated).squeeze(-1)
        completed = []

        for i in range(batch_size):
            if i not in self._rollout_cache:
                self._rollout_cache[i] = []
            self._rollout_cache[i].append({
                "obs": transition.obs[i].clone(), "act_action": transition.act.action[i].clone(),
                "reward": transition.reward[i].clone(), "next_obs": transition.next_obs[i].clone(),
                "terminated": transition.terminated[i].clone(), "truncated": transition.truncated[i].clone(),
                "act_info": {k: v[i].clone() for k, v in transition.act.info.items()} if transition.act.info else None,
                "info": {k: v[i].clone() for k, v in transition.info.items()} if transition.info else None,
            })
            if done[i]:
                completed.append(self._flush_rollout(i))

        return completed

    def sample(self, num_samples: int, device):
        """ Samples n-step transitions with priority weighting. """
        if self._valid_count == 0:
            raise ValueError("Cannot sample from an empty buffer")

        max_idx = self.buffer_length if self._full else self._cur_idx
        probs = self._priorities[:max_idx]
        probs = probs / probs.sum()

        indices = torch.multinomial(probs, num_samples, replacement=True)
        self._last_indices = indices

        weights = (self._valid_count * probs[indices]) ** (-self._beta)
        weights = weights / weights.max()

        obs, actions, n_rewards, next_obs, terminated, truncated, actual_n = self._build_n_step_vectorized(indices, self._n_step, self._gamma)
        act_info = {k: v[indices].to(device) for k, v in self._act_info.items()} if self._act_info else None
        info = {k: v[indices].to(device) for k, v in self._info.items()} if self._info else None

        return (obs.to(device), actions.to(device), n_rewards.to(device), next_obs.to(device),
                terminated.to(device), truncated.to(device), act_info, info, weights.to(device), actual_n.to(device))

    def _build_n_step_vectorized(self, starts: torch.Tensor, n_step: int, gamma: float):
        """ Vectorized n-step return computation. """
        B = starts.shape[0]
        device = starts.device

        if n_step == 1:
            return (self._obs[starts], self._act_action[starts], self._reward[starts],
                    self._next_obs[starts], self._terminated[starts], self._truncated[starts],
                    torch.ones(B, 1, dtype=torch.long, device=device))

        # [B, n_step] offset indices
        offsets = torch.arange(n_step, device=device).unsqueeze(0)  # [1, n_step]
        all_indices = (starts.unsqueeze(1) + offsets) % self.buffer_length  # [B, n_step]

        start_rids = self._idx_to_rollout_id[starts]  # [B]
        step_rids = self._idx_to_rollout_id[all_indices]  # [B, n_step]
        same_rollout = step_rids == start_rids.unsqueeze(1)  # [B, n_step]

        step_rewards = self._reward[all_indices].squeeze(-1)  # [B, n_step]
        step_terminated = self._terminated[all_indices].squeeze(-1)  # [B, n_step]
        step_truncated = self._truncated[all_indices].squeeze(-1)  # [B, n_step]
        step_done = step_terminated | step_truncated  # [B, n_step]

        # Mask: valid if same rollout AND no prior done
        prior_done = torch.zeros(B, n_step, dtype=torch.bool, device=device)
        prior_done[:, 1:] = step_done[:, :-1].cumsum(dim=1) > 0
        valid_mask = same_rollout & ~prior_done  # [B, n_step]

        # Discounts [1, gamma, gamma^2, ...]
        discounts = (gamma ** torch.arange(n_step, device=device, dtype=step_rewards.dtype)).unsqueeze(0)
        n_rewards = (step_rewards * discounts * valid_mask).sum(dim=1, keepdim=True)  # [B, 1]

        # actual_n = number of valid steps
        actual_n = valid_mask.sum(dim=1, keepdim=True)  # [B, 1]

        # final_step_idx = last valid step index per sample
        step_indices = torch.arange(n_step, device=device).unsqueeze(0).expand(B, -1)
        final_step_idx = (step_indices * valid_mask).max(dim=1).values  # [B]
        final_indices = all_indices[torch.arange(B, device=device), final_step_idx]  # [B]

        return (self._obs[starts], self._act_action[starts], n_rewards,
                self._next_obs[final_indices], self._terminated[final_indices], self._truncated[final_indices], actual_n)

    def update(self, td_errors, step: int | None = None) -> None:
        if self._last_indices is None:
            raise ValueError("update called before sample")
        if step is not None:
            self._last_update_step = int(step)
            frac = min(max(self._last_update_step / max(self._total_steps, 1), 0.0), 1.0)
            self._beta = self._beta_start + frac * (self._beta_end - self._beta_start)
        td_tensor = torch.as_tensor(td_errors).detach().flatten()
        new_priorities = (td_tensor.abs().to(self._priorities.device) + self._epsilon) ** self._alpha
        self._priorities[self._last_indices] = new_priorities
        self._max_priority = max(self._max_priority, (td_tensor.abs().max().item() + self._epsilon))

    def __len__(self) -> int:
        return self._valid_count
