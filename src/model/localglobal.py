from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import modulate
from .mlp import Mlp


def _grid_size(num_tokens: int) -> int:
    g = int(num_tokens**0.5)
    if g * g != num_tokens:
        raise ValueError(f"token count {num_tokens} is not a square grid")
    return g


def window_partition(x: torch.Tensor, window: int) -> tuple[torch.Tensor, int, int]:
    """``(B, g*g, C)`` → ``(B * nw, window*window, C)`` over a row-major token grid."""
    b, n, c = x.shape
    g = _grid_size(n)
    if g % window != 0:
        raise ValueError(f"grid {g} not divisible by window {window}")
    nw_side = g // window
    x = x.view(b, nw_side, window, nw_side, window, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b * nw_side * nw_side, window * window, c), g, nw_side


def window_merge(x: torch.Tensor, b: int, g: int, window: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`."""
    nw_side = g // window
    c = x.shape[-1]
    x = x.view(b, nw_side, nw_side, window, window, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, g * g, c)


def window_attn_bias(patch_bias: torch.Tensor, window: int) -> torch.Tensor:
    """``(B, 1, g, g)`` key bias → ``(B * nw, 1, 1, window*window)`` for windows."""
    b, _, g, _ = patch_bias.shape
    nw_side = g // window
    x = patch_bias.view(b, 1, nw_side, window, nw_side, window)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(b * nw_side * nw_side, 1, 1, window * window)


def pooled_attn_bias(patch_bias: torch.Tensor, pool: int) -> torch.Tensor:
    """``(B, 1, g, g)`` key bias → ``(B, 1, 1, (g/pool)^2)`` for pooled keys."""
    pooled = F.avg_pool2d(patch_bias, kernel_size=pool, stride=pool)
    return pooled.flatten(2).unsqueeze(2)


class WindowSelfAttention(nn.Module):
    """Self-attention restricted to non-overlapping ``window x window`` token tiles."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = int(window)
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        xw, g, _ = window_partition(x, self.window)
        bw, n, c = xw.shape
        qkv = (
            self.qkv(xw)
            .reshape(bw, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        if attn_bias is not None:
            attn_bias = attn_bias.to(dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(bw, n, c)
        out = window_merge(out, b, g, self.window)
        return self.proj_drop(self.proj(out))


class PooledGlobalAttention(nn.Module):
    """All tokens attend to an average-pooled summary of the whole token grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        pool: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pool = int(pool)
        self.attn_drop = attn_drop

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        g = _grid_size(n)
        if g % self.pool != 0:
            raise ValueError(f"grid {g} not divisible by pool {self.pool}")
        ctx = x.transpose(1, 2).reshape(b, c, g, g)
        ctx = F.avg_pool2d(ctx, kernel_size=self.pool, stride=self.pool)
        ctx = ctx.flatten(2).transpose(1, 2)  # (B, (g/pool)^2, C)

        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = (
            self.kv(ctx)
            .reshape(b, ctx.shape[1], 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        if attn_bias is not None:
            attn_bias = attn_bias.to(dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class LocalGlobalBlock(nn.Module):
    """adaLN-Zero block: windowed local attention, pooled global attention, MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window: int = 4,
        pool: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm_local = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.local_attn = WindowSelfAttention(
            hidden_size,
            num_heads=num_heads,
            window=window,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm_global = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.global_attn = PooledGlobalAttention(
            hidden_size,
            num_heads=num_heads,
            pool=pool,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU,
            drop=proj_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        local_bias: torch.Tensor | None = None,
        global_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            shift_loc,
            scale_loc,
            gate_loc,
            shift_glb,
            scale_glb,
            gate_glb,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        x = x + gate_loc.unsqueeze(1) * self.local_attn(
            modulate(self.norm_local(x), shift_loc, scale_loc),
            attn_bias=local_bias,
        )
        x = x + gate_glb.unsqueeze(1) * self.global_attn(
            modulate(self.norm_global(x), shift_glb, scale_glb),
            attn_bias=global_bias,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        )
        return x
