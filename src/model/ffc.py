from __future__ import annotations

import torch
import torch.nn as nn

from .hybrid import ConvStem
from .norm import LayerNorm2d


class FourierUnit(nn.Module):
    """FFT → 1x1 conv on stacked real/imag → IFFT."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(channels * 2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        dtype = x.dtype
        # cuFFT has no half/bfloat16 kernels, so the whole unit runs in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            ffted = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
            ffted = torch.stack([ffted.real, ffted.imag], dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view(b, c * 2, h, ffted.shape[-1])
            ffted = self.act(self.norm(self.conv(ffted)))
            ffted = ffted.view(b, c, 2, h, -1).permute(0, 1, 3, 4, 2).contiguous()
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])
            out = torch.fft.irfftn(ffted, s=(h, w), dim=(-2, -1), norm="ortho")
        return out.to(dtype)


class FFC(nn.Module):
    """Fast Fourier Convolution: local spatial features + global Fourier features."""

    def __init__(self, channels: int, ratio_global: float = 0.5):
        super().__init__()
        assert 0.0 < ratio_global < 1.0
        self.channels_g = max(1, int(channels * ratio_global))
        self.channels_l = channels - self.channels_g
        self.conv_l = nn.Conv2d(self.channels_l, self.channels_l, kernel_size=3, padding=1)
        self.conv_g = FourierUnit(self.channels_g)
        self.norm_l = LayerNorm2d(self.channels_l)
        self.norm_g = LayerNorm2d(self.channels_g)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = torch.split(x, [self.channels_l, self.channels_g], dim=1)
        y_l = self.act(self.norm_l(self.conv_l(x_l)))  # local spatial
        y_g = self.act(self.norm_g(self.conv_g(x_g)))  # global frequency
        return torch.cat([y_l, y_g], dim=1)


class FFCBlock(nn.Module):
    """Residual FFC: refine DiT mid-res maps with local + global features."""

    def __init__(self, channels: int, ratio_global: float = 0.5):
        super().__init__()
        self.ffc = FFC(channels, ratio_global=ratio_global)
        self.norm = LayerNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.ffc(x))


class FFCEncoder(nn.Module):
    """ConvStem then residual FFC blocks: local dots/lines/edges + global context.

    Runs before patchify so the DiT trunk receives tokens that already carry
    both spatial (conv branch) and whole-map frequency (Fourier branch) structure.
    ``downsample`` > 1 shrinks the map before the FFT blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int = 4,
        ratio_global: float = 0.5,
        downsample: int = 1,
    ):
        super().__init__()
        if blocks < 0:
            raise ValueError(f"blocks must be >= 0, got {blocks}")
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")
        if blocks > 0 and not (0.0 < float(ratio_global) < 1.0):
            raise ValueError(f"ratio_global must be in (0, 1), got {ratio_global}")

        self.downsample = int(downsample)
        self.stem = ConvStem(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=self.downsample,
        )
        self.blocks = nn.Sequential(
            *[
                FFCBlock(out_channels, ratio_global=ratio_global)
                for _ in range(int(blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))
