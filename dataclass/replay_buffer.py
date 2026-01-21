import torch

from dataclass.primitives import BatchedTransition



class ReplayBuffer():
    def __init__(self, buffer_length: int):
        self.buffer_length = buffer_length
        self._cur_idx = 0
        self._full = False
        self._initialized = False
        # NOTE: We instantiate our obs, act_action, etc. buffers upon seeing our first transition (to get shapes correct).


    # ===================================
    # Helpers to instantiate buffers upon first call to 'add'
    # ===================================
    def _allocate_buffer(self, template: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (self.buffer_length, *template.shape[1:]),
            dtype=template.dtype,
            device=template.device,
        )

    def _initialize_from_transition(self, transition: BatchedTransition) -> None:
        self._obs = self._allocate_buffer(transition.obs)
        self._act_action = self._allocate_buffer(transition.act.action)
        self._reward = self._allocate_buffer(transition.reward)
        self._next_obs = self._allocate_buffer(transition.next_obs)
        self._terminated = self._allocate_buffer(transition.terminated)
        self._truncated = self._allocate_buffer(transition.truncated)
        self._act_info = None if transition.act.info is None else {
            key: self._allocate_buffer(val) for key, val in transition.act.info.items()
        }
        self._info = None if transition.info is None else {
            key: self._allocate_buffer(val) for key, val in transition.info.items()
        }
        self._initialized = True


    # ===================================
    # External functionality
    # ===================================
    def add(self, transition: BatchedTransition) -> None:
        """ Add to our replay buffer. Upon first call to this function we initialize our buffers (since we then know shapes of obs, act, etc.) """
        if not self._initialized:
            self._initialize_from_transition(transition)

        # Add our new batched transitions to our buffer (using ring-buffer logic.)
        batch_size = transition.obs.shape[0]
        indices = (torch.arange(batch_size) + self._cur_idx) % self.buffer_length

        self._obs[indices] = transition.obs.clone()
        self._act_action[indices] = transition.act.action.clone()
        self._reward[indices] = transition.reward.clone()
        self._next_obs[indices] = transition.next_obs.clone()
        self._terminated[indices] = transition.terminated.clone()
        self._truncated[indices] = transition.truncated.clone()
        if self._act_info is not None:
            for key, buf in self._act_info.items():
                buf[indices] = transition.act.info[key].clone()
        if self._info is not None:
            for key, buf in self._info.items():
                buf[indices] = transition.info[key].clone()

        start_idx = self._cur_idx
        self._cur_idx = (start_idx + batch_size) % self.buffer_length
        if not self._full and (start_idx + batch_size) >= self.buffer_length:
            self._full = True

    def sample(self, num_samples: int, device):
        """ Samples desired num_samples and moves to device prior to return. """
        max_idx = self.buffer_length if self._full else self._cur_idx
        indices = torch.randint(0, max_idx, (num_samples,))

        act_info = None if self._act_info is None else {
            key: buf[indices].to(device) for key, buf in self._act_info.items()
        }
        info = None if self._info is None else {
            key: buf[indices].to(device) for key, buf in self._info.items()
        }

        return (
            self._obs[indices].to(device),
            self._act_action[indices].to(device),
            self._reward[indices].to(device),
            self._next_obs[indices].to(device),
            self._terminated[indices].to(device),
            self._truncated[indices].to(device),
            act_info,
            info,
        )

    def __len__(self) -> int:
        return self.buffer_length if self._full else self._cur_idx