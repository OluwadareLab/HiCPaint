from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn

from .distributed import unwrap_model


class ModelEma:
    """Exponential moving average of model weights, used for eval and checkpoints.

    Diffusion samples from averaged weights are markedly more stable than from the
    raw optimizer trajectory. Buffers are copied, not averaged.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        self.module = copy.deepcopy(unwrap_model(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = unwrap_model(model).state_dict()
        for key, dst in self.module.state_dict().items():
            value = src[key].detach()
            if dst.dtype.is_floating_point:
                dst.mul_(self.decay).add_(value.to(dst.dtype), alpha=1.0 - self.decay)
            else:
                dst.copy_(value)

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.module.load_state_dict(state)

    def to(self, device: torch.device) -> "ModelEma":
        self.module.to(device)
        return self
