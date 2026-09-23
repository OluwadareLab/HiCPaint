from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DiTBlock, FinalLayer
from .embeddings import LabelEmbedder, PatchEmbed, TimestepEmbedder
from .ffc import FFCBlock
from .hybrid import ConvDecoder, ConvStem, HoleRefine


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sin-cos positional embedding of shape ``(grid_size**2, embed_dim)``."""
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return torch.from_numpy(pos_embed).float()


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class DiffusionTransformer(nn.Module):
    """Blind-inpaint DiT: CNN stem → split cond tokens → mask-aware DiT → FFC → hole refine.

    Input is still stacked ``[x_t, mask, masked]`` (3 channels) for train/infer compatibility.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 24,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.2,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffc_blocks: int = 4,
        stem_channels: int = 64,
        mid_channels: int = 64,
        mask_attn_bias: float = 4.0,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        if in_channels != 3:
            raise ValueError("blind DiT expects in_channels=3 ([x_t, mask, masked])")

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.gt_channels = 1
        self.out_channels = self.gt_channels * (2 if learn_sigma else 1)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.img_size = img_size
        self.ffc_blocks = int(ffc_blocks)
        self.stem_channels = int(stem_channels)
        self.mid_channels = int(mid_channels)
        self.mask_attn_bias = float(mask_attn_bias)

        # CNN stem over stacked input, then patchify.
        self.stem = ConvStem(in_channels=in_channels, out_channels=self.stem_channels)
        self.feat_embedder = PatchEmbed(
            img_size, patch_size, self.stem_channels, hidden_size
        )
        # Explicit mask / known-context token streams (added to stem tokens).
        self.mask_embedder = PatchEmbed(img_size, patch_size, 1, hidden_size)
        self.masked_embedder = PatchEmbed(img_size, patch_size, 1, hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.feat_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(depth)
            ]
        )
        # Unpatchify to mid-channel maps; FFC runs here (not on 1-ch eps).
        self.final_layer = FinalLayer(hidden_size, patch_size, self.mid_channels)
        if self.ffc_blocks > 0:
            self.ffc = nn.Sequential(
                *[FFCBlock(self.mid_channels, ratio_global=0.5) for _ in range(self.ffc_blocks)]
            )
        else:
            self.ffc = None
        self.decoder = ConvDecoder(self.mid_channels, self.out_channels)
        self.hole_refine = HoleRefine(self.out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.feat_embedder.num_patches**0.5),
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        for embedder in (self.feat_embedder, self.mask_embedder, self.masked_embedder):
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        """``(N, T, patch**2 * C)`` → ``(N, C, H, W)``."""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], channels, h * p, w * p))

    def _mask_attn_bias(self, mask: torch.Tensor) -> torch.Tensor:
        """Soft bias: down-weight attention keys inside the hole (MAT-style)."""
        # mask hole=1 → patch hole fraction in [0, 1]
        patch_hole = F.avg_pool2d(mask, kernel_size=self.patch_size, stride=self.patch_size)
        patch_hole = patch_hole.flatten(2)  # (B, 1, N)
        # (B, 1, 1, N) broadcasts over heads and queries
        return -self.mask_attn_bias * patch_hole.unsqueeze(2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N, 3, H, W) stacked ``x_t``, mask, masked
        t: (N,) diffusion timesteps
        y: (N,) class labels
        returns: (N, out_channels, H, W)
        """
        mask = x[:, 1:2]
        masked = x[:, 2:3]

        feat = self.stem(x)
        tokens = (
            self.feat_embedder(feat)
            + self.mask_embedder(mask)
            + self.masked_embedder(masked)
            + self.pos_embed
        )
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb
        attn_bias = self._mask_attn_bias(mask)

        for block in self.blocks:
            tokens = block(tokens, c, attn_bias=attn_bias)

        h = self.final_layer(tokens, c)
        h = self.unpatchify(h, self.mid_channels)
        if self.ffc is not None:
            h = self.ffc(h)
        out = self.decoder(h)
        out = self.hole_refine(out, mask)
        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Classifier-free guidance forward pass."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.gt_channels], model_out[:, self.gt_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
