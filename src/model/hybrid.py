from __future__ import annotations

import torch
import torch.nn as nn

from .norm import LayerNorm2d


class ConvStem(nn.Module):
    """Local CNN encoder over ``[x_t, mask, masked]`` before DiT tokens."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        mid = max(out_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """Map mid-res features to noise / x0 channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = max(in_channels // 2, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HoleRefine(nn.Module):
    """Zero-init residual refine applied only inside the hole."""

    def __init__(self, channels: int = 1, hidden: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        delta = self.conv(torch.cat([x, mask], dim=1))
        return x + mask * delta
