from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DiTBlock, DiTCrossBlock, FinalLayer
from .embeddings import LabelEmbedder, PatchEmbed, TimestepEmbedder
from .ffc import FFCEncoder
from .hybrid import ConvDecoder, HoleRefine, ResConvBlock
from .localglobal import LocalGlobalBlock, pooled_attn_bias, window_attn_bias


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
    """Blind-inpaint DiT with an FFC front-end and a local/global attention stage.

    Stage 1: ``FFCEncoder`` on both streams (local conv + Fourier global structure).
    Stage 2: main DiT trunk with interleaved cross-attention from the structure DiT.
    Stage 3: ``LocalGlobalBlock`` stack (windowed local + pooled global attention).
    Stage 4: residual conv head → hole refine.

    Main input stays stacked ``[x_t, mask, masked]`` (3 channels).
    Structure input is masked ``[edge, line, dot]`` (Canny / Hough / Shi-Tomasi).
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
        ffc_ratio_global: float = 0.5,
        ffc_downsample: int = 1,
        stem_channels: int = 64,
        mid_channels: int = 64,
        mask_attn_bias: float = 4.0,
        structure_depth: int = 4,
        structure_channels: int = 3,
        lg_blocks: int = 2,
        lg_window: int = 4,
        lg_pool: int = 4,
        conv_head_blocks: int = 2,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        if in_channels != 3:
            raise ValueError("blind DiT expects in_channels=3 ([x_t, mask, masked])")
        if structure_channels != 3:
            raise ValueError(
                "structure expects 3 channels ([edge-canny, line-hough, dot-shi-tomasi])"
            )
        if structure_depth < 1:
            raise ValueError(f"structure_depth must be >= 1, got {structure_depth}")
        if ffc_blocks < 0:
            raise ValueError(f"ffc_blocks must be >= 0, got {ffc_blocks}")
        if ffc_blocks > 0 and not (0.0 < float(ffc_ratio_global) < 1.0):
            raise ValueError(
                f"ffc_ratio_global must be in (0, 1), got {ffc_ratio_global}"
            )
        if ffc_downsample < 1 or patch_size % ffc_downsample != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by "
                f"ffc_downsample ({ffc_downsample})"
            )
        if lg_blocks < 0:
            raise ValueError(f"lg_blocks must be >= 0, got {lg_blocks}")
        if conv_head_blocks < 0:
            raise ValueError(f"conv_head_blocks must be >= 0, got {conv_head_blocks}")

        grid_size = img_size // patch_size
        if lg_blocks > 0:
            if grid_size % lg_window != 0:
                raise ValueError(
                    f"token grid ({grid_size}) must be divisible by lg_window ({lg_window})"
                )
            if grid_size % lg_pool != 0:
                raise ValueError(
                    f"token grid ({grid_size}) must be divisible by lg_pool ({lg_pool})"
                )

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.structure_channels = structure_channels
        self.gt_channels = 1
        self.out_channels = self.gt_channels * (2 if learn_sigma else 1)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.structure_depth = int(structure_depth)
        self.img_size = img_size
        self.ffc_blocks = int(ffc_blocks)
        self.ffc_ratio_global = float(ffc_ratio_global)
        self.ffc_downsample = int(ffc_downsample)
        self.stem_channels = int(stem_channels)
        self.mid_channels = int(mid_channels)
        self.mask_attn_bias = float(mask_attn_bias)
        self.lg_blocks = int(lg_blocks)
        self.lg_window = int(lg_window)
        self.lg_pool = int(lg_pool)
        self.conv_head_blocks = int(conv_head_blocks)
        self.grid_size = grid_size

        # FFC runs on the (optionally downsampled) feature map, so the token patch
        # size shrinks with it and the token count stays ``grid_size**2``.
        feat_size = img_size // self.ffc_downsample
        self.feat_size = feat_size
        self.token_patch_size = patch_size // self.ffc_downsample

        # Stage 1: local (conv) + global (Fourier) structure before patchify.
        self.ffc_encoder = FFCEncoder(
            in_channels=in_channels,
            out_channels=self.stem_channels,
            blocks=self.ffc_blocks,
            ratio_global=self.ffc_ratio_global,
            downsample=self.ffc_downsample,
        )
        self.feat_embedder = PatchEmbed(
            feat_size, self.token_patch_size, self.stem_channels, hidden_size
        )
        # Explicit mask / known-context token streams (added to stem tokens).
        self.mask_embedder = PatchEmbed(img_size, patch_size, 1, hidden_size)
        self.masked_embedder = PatchEmbed(img_size, patch_size, 1, hidden_size)

        # Structure stream: masked [edge, line, dot].
        self.structure_ffc_encoder = FFCEncoder(
            in_channels=structure_channels,
            out_channels=self.stem_channels,
            blocks=self.ffc_blocks,
            ratio_global=self.ffc_ratio_global,
            downsample=self.ffc_downsample,
        )
        self.structure_feat_embedder = PatchEmbed(
            feat_size, self.token_patch_size, self.stem_channels, hidden_size
        )
        self.edge_embedder = PatchEmbed(img_size, patch_size, 1, hidden_size)
        self.line_embedder = PatchEmbed(img_size, patch_size, 1, hidden_size)
        self.dot_embedder = PatchEmbed(img_size, patch_size, 1, hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.feat_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.structure_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

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
        self.structure_blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(self.structure_depth)
            ]
        )
        # Main ← structure cross-attn, interleaved evenly across main depth.
        self.cross_blocks = nn.ModuleList(
            [
                DiTCrossBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(self.structure_depth)
            ]
        )
        self._cross_interval = max(1, depth // self.structure_depth)

        # Stage 3: windowed local + pooled global attention over the same tokens.
        self.lg_stage = nn.ModuleList(
            [
                LocalGlobalBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    window=self.lg_window,
                    pool=self.lg_pool,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(self.lg_blocks)
            ]
        )

        # Stage 4: unpatchify to mid-channel maps, then a pure conv output path.
        self.final_layer = FinalLayer(
            hidden_size, self.token_patch_size, self.mid_channels
        )
        self.conv_head = nn.Sequential(
            *[ResConvBlock(self.mid_channels) for _ in range(self.conv_head_blocks)]
        )
        self.decoder = ConvDecoder(
            self.mid_channels, self.out_channels, upsample=self.ffc_downsample
        )
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
        self.structure_pos_embed.data.copy_(pos_embed.unsqueeze(0))

        for embedder in (
            self.feat_embedder,
            self.mask_embedder,
            self.masked_embedder,
            self.structure_feat_embedder,
            self.edge_embedder,
            self.line_embedder,
            self.dot_embedder,
        ):
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in (
            list(self.blocks)
            + list(self.structure_blocks)
            + list(self.cross_blocks)
            + list(self.lg_stage)
        ):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        """``(N, T, patch**2 * C)`` → ``(N, C, H, W)`` at feature resolution."""
        p = self.token_patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], channels, h * p, w * p))

    def _patch_key_bias(self, mask: torch.Tensor) -> torch.Tensor:
        """Per-token key bias grid ``(B, 1, g, g)``: down-weight hole keys (MAT-style)."""
        patch_hole = F.avg_pool2d(
            mask, kernel_size=self.patch_size, stride=self.patch_size
        )
        return -self.mask_attn_bias * patch_hole

    def _mask_attn_bias(self, mask: torch.Tensor) -> torch.Tensor:
        """Full-attention key bias ``(B, 1, 1, N)``, broadcast over heads and queries."""
        return self._patch_key_bias(mask).flatten(2).unsqueeze(2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        structure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (N, 3, H, W) stacked ``x_t``, mask, masked
        structure: (N, 3, H, W) stacked masked edge / line / dot
        t: (N,) diffusion timesteps
        y: (N,) class labels
        returns: (N, out_channels, H, W)
        """
        mask = x[:, 1:2]
        masked = x[:, 2:3]
        if structure is None:
            structure = torch.zeros(
                x.shape[0],
                self.structure_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )

        feat = self.ffc_encoder(x)
        tokens = (
            self.feat_embedder(feat)
            + self.mask_embedder(mask)
            + self.masked_embedder(masked)
            + self.pos_embed
        )

        edge, line, dot = structure[:, 0:1], structure[:, 1:2], structure[:, 2:3]
        struct_feat = self.structure_ffc_encoder(structure)
        struct_tokens = (
            self.structure_feat_embedder(struct_feat)
            + self.edge_embedder(edge)
            + self.line_embedder(line)
            + self.dot_embedder(dot)
            + self.structure_pos_embed
        )

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb
        patch_bias = self._patch_key_bias(mask)
        attn_bias = patch_bias.flatten(2).unsqueeze(2)

        for block in self.structure_blocks:
            struct_tokens = block(struct_tokens, c, attn_bias=attn_bias)

        cross_i = 0
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, c, attn_bias=attn_bias)
            if (
                cross_i < len(self.cross_blocks)
                and (i + 1) % self._cross_interval == 0
            ):
                tokens = self.cross_blocks[cross_i](
                    tokens, struct_tokens, c, attn_bias=attn_bias
                )
                cross_i += 1

        # If depth not divisible by structure_depth, apply remaining cross blocks.
        while cross_i < len(self.cross_blocks):
            tokens = self.cross_blocks[cross_i](
                tokens, struct_tokens, c, attn_bias=attn_bias
            )
            cross_i += 1

        if self.lg_blocks > 0:
            local_bias = window_attn_bias(patch_bias, self.lg_window)
            global_bias = pooled_attn_bias(patch_bias, self.lg_pool)
            for block in self.lg_stage:
                tokens = block(
                    tokens, c, local_bias=local_bias, global_bias=global_bias
                )

        h = self.final_layer(tokens, c)
        h = self.unpatchify(h, self.mid_channels)
        h = self.conv_head(h)
        out = self.decoder(h)
        out = self.hole_refine(out, mask)
        return out

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        structure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classifier-free guidance forward pass."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        struct = None
        if structure is not None:
            half_s = structure[: len(structure) // 2]
            struct = torch.cat([half_s, half_s], dim=0)
        model_out = self.forward(combined, t, y, structure=struct)
        eps, rest = model_out[:, : self.gt_channels], model_out[:, self.gt_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
