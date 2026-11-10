from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN for Hi-C maps, fed ``[map, mask, masked]``.

    Conditioning on the mask and known pixels makes D judge hole/context agreement
    rather than global map statistics. Spectral norm plus ``InstanceNorm`` keeps it
    stable at small batch sizes, where ``BatchNorm`` stats differ between the
    separate real and fake forward passes.
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [
            spectral_norm(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=False),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                spectral_norm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                ),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=False),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            spectral_norm(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            ),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            ),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
