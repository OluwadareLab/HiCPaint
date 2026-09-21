from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    """Linear beta DDPM schedule with blind-inpaint and reverse-sample helpers."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)

    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t.long())
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: ``x_t = sqrt(abar) x0 + sqrt(1-abar) eps``."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omb = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omb * noise, noise

    def blind_q_sample(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        masked: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blind forward: known pixels stay as ``masked``; only the hole is noised.

        Model main channel never sees clean ``gt`` — only ``x_t``, ``mask``, ``masked``.
        ``x0``/``gt`` is used solely to build the noisy hole (training target).
        """
        x_t_full, noise = self.q_sample(x0, t, noise)
        x_t = (1.0 - mask) * masked + mask * x_t_full
        return x_t, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate ``x0`` from noisy ``x_t`` and predicted noise ``eps``."""
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_omb = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_omb * eps) / sqrt_ab.clamp_min(1e-8)

    def p_sample(
        self,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One DDPM reverse step from ``x_t`` given noise prediction ``eps``."""
        beta_t = self._extract(self.betas, t, x_t.shape)
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        abar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - abar_t).clamp_min(1e-8)) * eps
        )
        noise = torch.randn_like(x_t)
        nonzero = (t > 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean + nonzero * torch.sqrt(var.clamp_min(1e-20)) * noise

    @torch.no_grad()
    def inpaint(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        masked: torch.Tensor,
        y: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """True blind inpainting: noise only in the hole; never uses ``gt``.

        Starts from ``(1-mask)*masked + mask*N(0,1)``, denoises, and reinjects
        known pixels after every reverse step.
        """
        b = masked.shape[0]
        device = masked.device
        if y is None:
            y = torch.zeros(b, dtype=torch.long, device=device)

        x = (1.0 - mask) * masked + mask * torch.randn_like(masked)

        if num_steps is None or num_steps >= self.num_timesteps:
            times = list(range(self.num_timesteps - 1, -1, -1))
        else:
            times = (
                torch.linspace(0, self.num_timesteps - 1, int(num_steps))
                .round()
                .long()
                .unique(sorted=True)
                .tolist()
            )
            times = list(reversed(times))

        for t_int in times:
            t = torch.full((b,), int(t_int), device=device, dtype=torch.long)
            model_in = torch.cat([x, mask, masked], dim=1)
            out = model(model_in, t, y)
            eps = out[:, :1]
            x = self.p_sample(x, eps, t)
            x = (1.0 - mask) * masked + mask * x

        return x.clamp(0.0, 1.0)
