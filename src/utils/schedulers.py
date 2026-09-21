from __future__ import annotations

import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
    base_lr: float,
) -> LambdaLR:
    """Linear warmup for ``warmup_epochs``, then cosine decay to ``min_lr``.

    The returned scheduler steps once per epoch (``scheduler.step()`` after each epoch).
    """
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    warmup_epochs = max(0, int(warmup_epochs))
    min_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0
    min_ratio = max(0.0, min(1.0, min_ratio))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_cast_lambda(lr_lambda))


def _cast_lambda(fn: Callable[[int], float]) -> Callable[[int], float]:
    return fn
