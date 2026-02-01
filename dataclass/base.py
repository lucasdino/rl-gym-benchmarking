from typing import Any
from abc import ABC, abstractmethod

from dataclass.primitives import BatchedTransition



class BaseBuffer(ABC):
    """
    Base superclass for all replay buffers.
    """

    @abstractmethod
    def add(self, transition: BatchedTransition) -> BatchedTransition | None:
        """ Add a transition to the buffer. Returns completed rollouts if any. """
        ...

    @abstractmethod
    def sample(self, num_samples: int, device) -> Any:
        """ Sample a batch from the buffer. If n_step > 1, returns n-step transitions. """
        ...

    @abstractmethod
    def update(self, td_errors, step: int | None = None) -> None:
        """ Update priorities (no-op for non-prioritized buffers). """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """ Return current number of transitions stored. """
        ...