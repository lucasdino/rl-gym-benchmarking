from typing import Any
from abc import ABC, abstractmethod

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
    def observe(self, transition: BatchedTransition) -> None:
        """ Log transitions to your buffer / rollout buffer. """
        ...

    @abstractmethod
    def update(self) -> dict[str, Any]:
        """ Update your model parameters across various models and return key stats used for logging. """
        ...

    @abstractmethod
    def ready_to_update(self) -> bool:
        """ Simple call to implement in training loop to see if ready to update (e.g., checks if buffer has enough samples / you've exceeded the number of warmup steps) """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """ Saves your class (e.g., parameters, cfg, etc.). """
        ...

    @classmethod
    @abstractmethod
    def load(self, path: str, override_cfg: Any = None) -> "BaseAlgorithm":
        """ Loads in an instance of this class that was saved down. """
        ...