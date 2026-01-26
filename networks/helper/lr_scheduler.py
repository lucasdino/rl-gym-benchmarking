import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class CosineWarmupConfig:
    total_env_steps: int
    warmup_env_steps: int
    start_lr: float
    end_lr: float
    warmup_start_lr: float = 0.0


class CosineWarmupLRScheduler:
    """Cosine decay with linear warmup using env-step timebase.

    Call `step(env_steps)` on each optimizer update, where `env_steps` is the
    cumulative number of environment steps collected so far (vectorized OK).
    """

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: CosineWarmupConfig):
        self.optimizer = optimizer
        self.cfg = cfg
        self.last_lr = cfg.start_lr
        # Initialize optimizer lr to warmup start for consistency
        for group in self.optimizer.param_groups:
            group["lr"] = cfg.warmup_start_lr if cfg.warmup_env_steps > 0 else cfg.start_lr

    def _compute_lr(self, env_steps: int) -> float:
        total = max(1, int(self.cfg.total_env_steps))
        warmup = max(0, int(self.cfg.warmup_env_steps))
        start_lr = float(self.cfg.start_lr)
        end_lr = float(self.cfg.end_lr)
        warmup_start = float(self.cfg.warmup_start_lr)

        if warmup > 0 and env_steps <= warmup:
            progress = min(1.0, env_steps / max(1, warmup))
            return warmup_start + (start_lr - warmup_start) * progress

        decay_steps = max(1, total - warmup)
        progress = min(1.0, max(0.0, (env_steps - warmup) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_lr + (start_lr - end_lr) * cosine

    def step(self, env_steps: int) -> float:
        lr = self._compute_lr(env_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def get_last_lr(self) -> float:
        return self.last_lr


def build_cosine_warmup_schedulers(
    optimizers: dict[str, torch.optim.Optimizer],
    total_env_steps: int,
    warmup_env_steps: int,
    start_lr: float,
    end_lr: float,
    warmup_start_lr: float = 0.0,
) -> dict[str, CosineWarmupLRScheduler]:
    cfg = CosineWarmupConfig(
        total_env_steps=total_env_steps,
        warmup_env_steps=warmup_env_steps,
        start_lr=start_lr,
        end_lr=end_lr,
        warmup_start_lr=warmup_start_lr,
    )
    return {name: CosineWarmupLRScheduler(opt, cfg) for name, opt in optimizers.items()}
