from __future__ import annotations

import torch

Rope = tuple[torch.Tensor, torch.Tensor]


def build_2d_rope(
    grid: int,
    head_dim: int,
    theta: float = 10000.0,
    scale: float = 1.0,
    offset: float = 0.0,
) -> Rope:
    """2D rotary tables ``(cos, sin)``, each ``(grid*grid, head_dim)``, row-major tokens.

    Rotate-half pairs ``(i, i + head_dim/2)``: the first quarter of pairs rotates by
    the row position, the second quarter by the column position.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
    quarter = head_dim // 4
    inv_freq = theta ** (-torch.arange(quarter, dtype=torch.float64) / quarter)
    pos = torch.arange(grid, dtype=torch.float64) * scale + offset
    rows, cols = torch.meshgrid(pos, pos, indexing="ij")
    ang = torch.cat(
        [rows.reshape(-1, 1) * inv_freq, cols.reshape(-1, 1) * inv_freq], dim=-1
    )
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().float(), ang.sin().float()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``(B, H, N, D)`` by tables broadcastable to it; math in fp32."""
    xf = x.float()
    return (xf * cos + _rotate_half(xf) * sin).to(x.dtype)
