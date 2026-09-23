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

    def predict_eps_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate ``eps`` from noisy ``x_t`` and predicted clean ``x0``."""
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_omb = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_ab * x0) / sqrt_omb.clamp_min(1e-8)

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

    def ddim_step(
        self,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM jump from ``t`` -> ``t_prev`` (``t_prev`` may be -1 for final ``x0``).

        Required when ``num_steps < num_timesteps``: plain ``p_sample`` only advances
        one schedule index, so skipped schedules left the hole near pure noise.
        """
        x0 = self.predict_x0_from_eps(x_t, t, eps)
        # Final step: return predicted clean image.
        if bool((t_prev < 0).all().item()):
            return x0

        abar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        # Clamp prev index to >=0 for buffer gather; masked out when t_prev < 0.
        t_prev_clamped = t_prev.clamp_min(0)
        abar_prev = self._extract(self.alphas_cumprod, t_prev_clamped, x_t.shape)
        # When t_prev < 0, abar_prev := 1 (fully clean).
        prev_ok = (t_prev >= 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
        abar_prev = prev_ok * abar_prev + (1.0 - prev_ok) * torch.ones_like(abar_prev)

        sigma = (
            float(eta)
            * torch.sqrt((1.0 - abar_prev) / (1.0 - abar_t).clamp_min(1e-8))
            * torch.sqrt((1.0 - abar_t / abar_prev.clamp_min(1e-8)).clamp_min(0.0))
        )
        dir_xt = torch.sqrt((1.0 - abar_prev - sigma**2).clamp_min(0.0)) * eps
        noise = torch.randn_like(x_t) if float(eta) > 0.0 else torch.zeros_like(x_t)
        return torch.sqrt(abar_prev) * x0 + dir_xt + sigma * noise

    @torch.no_grad()
    def inpaint_onestep(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        masked: torch.Tensor,
        y: torch.Tensor | None = None,
        t_value: int = -1,
        prediction: str = "x0",
    ) -> torch.Tensor:
        """Single-pass blind inpaint: noise in hole → predict clean ``x0`` → compose.

        ``t_value < 0`` uses ``num_timesteps - 1`` (near-pure noise in the hole).
        ``prediction`` is ``\"x0\"`` (model outputs clean map) or ``\"eps\"``.
        """
        b = masked.shape[0]
        device = masked.device
        if y is None:
            y = torch.zeros(b, dtype=torch.long, device=device)
        tv = int(t_value)
        if tv < 0:
            tv = self.num_timesteps - 1
        tv = max(0, min(tv, self.num_timesteps - 1))

        x = (1.0 - mask) * masked + mask * torch.randn_like(masked)
        t = torch.full((b,), tv, device=device, dtype=torch.long)
        out = model(torch.cat([x, mask, masked], dim=1), t, y)
        if prediction == "x0":
            x0 = out[:, :1]
        elif prediction == "eps":
            x0 = self.predict_x0_from_eps(x, t, out[:, :1])
        else:
            raise ValueError(f"prediction must be 'x0' or 'eps', got {prediction!r}")
        return ((1.0 - mask) * masked + mask * x0).clamp(0.0, 1.0)

    @torch.no_grad()
    def inpaint(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        masked: torch.Tensor,
        y: torch.Tensor | None = None,
        num_steps: int | None = None,
        eta: float = 0.0,
        prediction: str = "eps",
    ) -> torch.Tensor:
        """Multi-step blind inpainting (DDPM/DDIM). Prefer ``inpaint_onestep`` for x0.

        Starts from ``(1-mask)*masked + mask*N(0,1)``, denoises, and reinjects
        known pixels after every reverse step.
        """
        b = masked.shape[0]
        device = masked.device
        if y is None:
            y = torch.zeros(b, dtype=torch.long, device=device)

        x = (1.0 - mask) * masked + mask * torch.randn_like(masked)

        use_full = num_steps is None or num_steps >= self.num_timesteps
        if use_full:
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

        for i, t_int in enumerate(times):
            t = torch.full((b,), int(t_int), device=device, dtype=torch.long)
            model_in = torch.cat([x, mask, masked], dim=1)
            out = model(model_in, t, y)
            if prediction == "x0":
                x0 = out[:, :1]
                eps = self.predict_eps_from_x0(x, t, x0)
            else:
                eps = out[:, :1]
            if use_full:
                x = self.p_sample(x, eps, t)
            else:
                t_prev_int = times[i + 1] if i + 1 < len(times) else -1
                t_prev = torch.full(
                    (b,), int(t_prev_int), device=device, dtype=torch.long
                )
                x = self.ddim_step(x, eps, t, t_prev, eta=eta)
            x = (1.0 - mask) * masked + mask * x

        return x.clamp(0.0, 1.0)
