import torch

from typing import Any, Optional
from dataclasses import dataclass



@dataclass(frozen=True)
class BatchedActionOutput:
    """ All values should be torch.Tensors of shape 'B x C' where 'C' can be any arbitrary shape that we expect to handle in our neural networks. """
    action:     torch.Tensor
    info:       dict[str, torch.Tensor]   # could be entropy, etc.

@dataclass(frozen=True)
class BatchedTransition:
    """ All values should be torch.Tensors of shape 'B x C' where 'C' can be any arbitrary shape that we expect to handle in our neural networks. """
    obs:        torch.Tensor
    act:        BatchedActionOutput
    reward:     torch.Tensor
    next_obs:   torch.Tensor
    terminated: torch.Tensor
    truncated:  torch.Tensor
    info:       Optional[dict[str, torch.Tensor]] = None