from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between ``pred`` and ``target``."""
    return F.mse_loss(pred, target)


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE averaged only over hole pixels where ``mask == 1``."""
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def _ssim_1ch(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable SSIM for a single ``(1,1,H,W)`` pair; returns scalar in ``[0,1]``."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (num / den.clamp_min(1e-8)).mean()


def masked_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    data_range: float = 1.0,
) -> torch.Tensor:
    """``1 - SSIM`` on each sample's hole bbox; optional per-sample weights.

    Known pixels match under ``_compose``, so full-image SSIM is misleading —
    this crops to the hole so the structure signal tracks masked SSIM.
    """
    b = pred.shape[0]
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for i in range(b):
        w_i = pred.new_ones(()) if sample_weight is None else sample_weight[i]
        if float(w_i.detach().item()) <= 0.0:
            continue
        m = mask[i, 0] > 0.5
        if not bool(m.any().item()):
            continue
        ys = m.any(dim=1).nonzero(as_tuple=False).squeeze(1)
        xs = m.any(dim=0).nonzero(as_tuple=False).squeeze(1)
        y0, y1 = int(ys[0].item()), int(ys[-1].item()) + 1
        x0, x1 = int(xs[0].item()), int(xs[-1].item()) + 1
        if (y1 - y0) < 3 or (x1 - x0) < 3:
            hole = mask[i : i + 1]
            losses.append(masked_mse_loss(pred[i : i + 1], target[i : i + 1], hole))
        else:
            p = pred[i : i + 1, :, y0:y1, x0:x1]
            t = target[i : i + 1, :, y0:y1, x0:x1]
            losses.append(1.0 - _ssim_1ch(p, t, data_range=data_range))
        weights.append(w_i.to(dtype=pred.dtype))
    if not losses:
        return pred.new_zeros(())
    stacked = torch.stack(losses)
    w = torch.stack(weights)
    return (stacked * w).sum() / w.sum().clamp_min(1.0)


def _broadcast_sample_weight(
    sample_weight: torch.Tensor,
    like: torch.Tensor,
) -> torch.Tensor:
    """Reshape ``(B,)`` weights to broadcast over ``like``'s non-batch dims."""
    return sample_weight.reshape(-1, *([1] * (like.ndim - 1))).to(dtype=like.dtype)


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Element MSE; optional per-sample weights (DDP-safe when some weights are 0)."""
    if sample_weight is None:
        return F.mse_loss(pred, target)
    w = _broadcast_sample_weight(sample_weight, pred)
    err = (pred - target) ** 2
    return (err * w).sum() / w.expand_as(err).sum().clamp_min(1.0)


class AdversarialLoss(nn.Module):
    """LSGAN adversarial losses for generator and discriminator.

    Soft labels (e.g. ``real_label=0.9``, ``fake_label=0.1``) reduce D saturation
    when fakes are still weak.
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.real_label = float(real_label)
        self.fake_label = float(fake_label)

    def g_loss(
        self,
        fake_logits: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generator loss: MSE of fake logits toward 1."""
        return _weighted_mse(fake_logits, torch.ones_like(fake_logits), sample_weight)

    def d_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discriminator loss: real→``real_label``, fake→``fake_label``."""
        real_tgt = torch.full_like(real_logits, self.real_label)
        fake_tgt = torch.full_like(fake_logits, self.fake_label)
        real_loss = _weighted_mse(real_logits, real_tgt, sample_weight)
        fake_loss = _weighted_mse(fake_logits, fake_tgt, sample_weight)
        return 0.5 * (real_loss + fake_loss)

    def forward(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """If ``real_logits`` is given, return D loss; otherwise G loss."""
        if real_logits is None:
            return self.g_loss(fake_logits, sample_weight)
        return self.d_loss(real_logits, fake_logits, sample_weight)


def adversarial_g_loss(
    fake_logits: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generator adversarial loss (LSGAN): MSE of fake logits toward 1."""
    return AdversarialLoss().g_loss(fake_logits, sample_weight)


def adversarial_d_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    real_label: float = 1.0,
    fake_label: float = 0.0,
) -> torch.Tensor:
    """Discriminator adversarial loss (LSGAN) with optional soft labels."""
    return AdversarialLoss(real_label=real_label, fake_label=fake_label).d_loss(
        real_logits, fake_logits, sample_weight
    )
