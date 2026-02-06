from typing import Any
from abc import ABC, abstractmethod
import torch

from dataclass.primitives import BatchedActionOutput, BatchedTransition



class BaseAlgorithm(ABC):
    def __init__(self, cfg: Any, obs_space: Any, act_space: Any, device):
        """
        Base superclass for all our implemented algorithms.

        All algorithms will expose these functions but will have additional helpers / etc.. 
        """
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device

    @abstractmethod
    def act(self, obs: Any, *, eval_mode: bool) -> BatchedActionOutput:
        """ Run inference on your policy. """
        ...

    @abstractmethod
    def observe(self, transition: BatchedTransition) -> list:
        """ Log transitions to your buffer / rollout buffer. Returns list of RunResults. """
        ...

    @abstractmethod
    def update(self) -> dict[str, Any]:
        """ Update your model parameters across various models and return key stats used for logging. """
        ...

    @abstractmethod
    def ready_to_update(self) -> bool:
        """ Simple call to implement in training loop to see if ready to update (e.g., checks if buffer has enough samples & you've exceeded the number of warmup steps) """
        ...

    # Non-abstract methods defined at this base-level
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("replay_buffer", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.replay_buffer = None

    def save(self, path: str) -> None:
        """ Saves your class (e.g., parameters, cfg, etc.). """
        torch.save(self, path)

    @classmethod
    def load(cls, path: str, override_cfg: Any = None) -> "BaseAlgorithm":
        """ Loads in an instance of this class that was saved down. """
        algo = torch.load(path, map_location="cpu", weights_only=False)
        if cls is not BaseAlgorithm:
            assert isinstance(algo, cls), f"Loaded algo type {type(algo)} does not match expected {cls}"
        if override_cfg is not None:
            algo.cfg = override_cfg
        algo._instantiate_buffer()
        return algo