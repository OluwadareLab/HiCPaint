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


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 averaged only over hole pixels where ``mask == 1``.

    Optional ``sample_weight`` ``(B,)`` gates per sample (e.g. low-``t`` only).
    """
    err = (pred - target).abs() * mask
    if sample_weight is None:
        return err.sum() / mask.sum().clamp_min(1.0)
    w = sample_weight.to(dtype=err.dtype)
    if float(w.sum().detach().item()) <= 0.0:
        return pred.new_zeros(())
    w_b = w.reshape(-1, *([1] * (err.ndim - 1)))
    per_sample_hole = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1.0)
    denom = (per_sample_hole * w).sum().clamp_min(1.0)
    return (err * w_b).sum() / denom


def masked_charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sqrt((p - t)^2 + eps^2)`` over hole pixels: L1-like tails, L2-like near zero.

    Less blur-prone than MSE while staying smooth at the origin.
    """
    err = (torch.sqrt((pred - target) ** 2 + eps * eps) - eps) * mask
    if sample_weight is None:
        return err.sum() / mask.sum().clamp_min(1.0)
    w = sample_weight.to(dtype=err.dtype)
    if float(w.sum().detach().item()) <= 0.0:
        return pred.new_zeros(())
    w_b = w.reshape(-1, *([1] * (err.ndim - 1)))
    per_sample_hole = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1.0)
    denom = (per_sample_hole * w).sum().clamp_min(1.0)
    return (err * w_b).sum() / denom


def _hole_bbox(mask_2d: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Tight bounding box ``(y0, y1, x0, x1)`` of ``mask_2d > 0.5``, or ``None``."""
    m = mask_2d > 0.5
    if not bool(m.any().item()):
        return None
    ys = m.any(dim=1).nonzero(as_tuple=False).squeeze(1)
    xs = m.any(dim=0).nonzero(as_tuple=False).squeeze(1)
    return (
        int(ys[0].item()),
        int(ys[-1].item()) + 1,
        int(xs[0].item()),
        int(xs[-1].item()) + 1,
    )


def _reduce_per_sample(
    losses: list[torch.Tensor],
    weights: list[torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    if not losses:
        return like.new_zeros(())
    w = torch.stack(weights)
    return (torch.stack(losses) * w).sum() / w.sum().clamp_min(1.0)


def _crop_hole_rect(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    index: int,
    min_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Hole bbox crop for sample ``index``; ``None`` if either side < ``min_size``."""
    box = _hole_bbox(mask[index, 0])
    if box is None:
        return None
    y0, y1, x0, x1 = box
    if min(y1 - y0, x1 - x0) < min_size:
        return None
    return pred[index, 0, y0:y1, x0:x1], target[index, 0, y0:y1, x0:x1]


def _crop_hole_on_diagonal(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    index: int,
    min_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Square crop of the hole bbox that lies on the map's main diagonal.

    Diagonal-anchored scores (insulation) are undefined for off-diagonal holes,
    which return ``None``.
    """
    box = _hole_bbox(mask[index, 0])
    if box is None:
        return None
    y0, y1, x0, x1 = box
    a, b = max(y0, x0), min(y1, x1)
    if b - a < min_size:
        return None
    return pred[index, 0, a:b, a:b], target[index, 0, a:b, a:b]


def _strata_offsets(h: int, w: int) -> range:
    """Every diagonal offset of an ``(h, w)`` crop; each one is a single genomic distance.

    Both signs are kept: an off-diagonal or asymmetric hole has no mirror strata.
    """
    return range(-(h - 1), w)


def stratum_corr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    max_stratum: int | None = None,
) -> torch.Tensor:
    """``1 - mean_k Pearson(diag_k)`` over the hole crop (differentiable HiCRep/SCC).

    Hi-C intensity decays with genomic distance, so plain pixel losses are dominated
    by the near-diagonal band; correlating each distance stratum separately keeps the
    far-from-diagonal structure in the objective.
    """
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for i in range(pred.shape[0]):
        w_i = pred.new_ones(()) if sample_weight is None else sample_weight[i]
        if float(w_i.detach().item()) <= 0.0:
            continue
        crop = _crop_hole_rect(pred, target, mask, i, min_size=4)
        if crop is None:
            continue
        p, t = crop
        corrs: list[torch.Tensor] = []
        for k in _strata_offsets(*p.shape):
            if max_stratum is not None and abs(k) >= int(max_stratum):
                continue
            pk = torch.diagonal(p, offset=k)
            tk = torch.diagonal(t, offset=k)
            if pk.numel() < 3:
                continue
            pk = pk - pk.mean()
            tk = tk - tk.mean()
            t_norm = tk.norm()
            if float(t_norm.detach().item()) < 1e-6:
                continue  # constant GT stratum: correlation undefined
            corrs.append((pk * tk).sum() / (pk.norm() * t_norm).clamp_min(1e-8))
        if not corrs:
            continue
        losses.append(1.0 - torch.stack(corrs).mean())
        weights.append(w_i.to(dtype=pred.dtype))
    return _reduce_per_sample(losses, weights, pred)


def distance_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Per-stratum L1 on the hole crop, each diagonal scaled by ``1 / (std_k + eps)``.

    Rebalances the contact-decay profile so low-variance distant strata still matter.
    """
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for i in range(pred.shape[0]):
        w_i = pred.new_ones(()) if sample_weight is None else sample_weight[i]
        if float(w_i.detach().item()) <= 0.0:
            continue
        crop = _crop_hole_rect(pred, target, mask, i, min_size=2)
        if crop is None:
            continue
        p, t = crop
        terms: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        for k in _strata_offsets(*p.shape):
            pk = torch.diagonal(p, offset=k)
            tk = torch.diagonal(t, offset=k)
            if pk.numel() < 2:
                continue  # single-element stratum has no spread to normalize by
            scale = 1.0 / (tk.detach().std(unbiased=False) + eps)
            terms.append((pk - tk).abs().mean() * scale)
            scales.append(scale)
        if not terms:
            continue
        total_scale = torch.stack(scales).sum().clamp_min(1e-8)
        losses.append(torch.stack(terms).sum() / total_scale)
        weights.append(w_i.to(dtype=pred.dtype))
    return _reduce_per_sample(losses, weights, pred)


def _insulation_profile(m: torch.Tensor, window: int) -> torch.Tensor:
    """Diamond insulation score along the diagonal of a square map ``(n, n)``."""
    n = m.shape[-1]
    kernel = m.new_ones(1, 1, window, window) / float(window * window)
    box = F.conv2d(m.view(1, 1, n, n), kernel)
    # box[j, j + window] averages rows [j, j+window) against cols [j+window, j+2*window)
    return torch.diagonal(box[0, 0], offset=window)


def insulation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window: int = 8,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 between insulation-score profiles of the hole crop (TAD boundary fidelity)."""
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for i in range(pred.shape[0]):
        w_i = pred.new_ones(()) if sample_weight is None else sample_weight[i]
        if float(w_i.detach().item()) <= 0.0:
            continue
        crop = _crop_hole_on_diagonal(pred, target, mask, i, min_size=2 * window + 1)
        if crop is None:
            continue
        p, t = crop
        losses.append(
            (_insulation_profile(p, window) - _insulation_profile(t, window))
            .abs()
            .mean()
        )
        weights.append(w_i.to(dtype=pred.dtype))
    return _reduce_per_sample(losses, weights, pred)


_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
).view(1, 1, 3, 3)


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 on Sobel gradients inside the hole: sharpens loops/dots and the hole seam."""
    kx = _SOBEL_X.to(device=pred.device, dtype=pred.dtype)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(pred, kx, padding=1) - F.conv2d(target, kx, padding=1)
    gy = F.conv2d(pred, ky, padding=1) - F.conv2d(target, ky, padding=1)
    err = (gx.abs() + gy.abs()) * mask
    if sample_weight is None:
        return err.sum() / mask.sum().clamp_min(1.0)
    w = sample_weight.to(dtype=err.dtype)
    if float(w.sum().detach().item()) <= 0.0:
        return pred.new_zeros(())
    w_b = w.reshape(-1, *([1] * (err.ndim - 1)))
    per_sample_hole = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1.0)
    denom = (per_sample_hole * w).sum().clamp_min(1.0)
    return (err * w_b).sum() / denom


def spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 on ``rfft2`` magnitude of the hole crop; pairs with the FFC front-end."""
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for i in range(pred.shape[0]):
        w_i = pred.new_ones(()) if sample_weight is None else sample_weight[i]
        if float(w_i.detach().item()) <= 0.0:
            continue
        crop = _crop_hole_rect(pred, target, mask, i, min_size=4)
        if crop is None:
            continue
        p, t = crop
        with torch.autocast(device_type=p.device.type, enabled=False):
            fp = torch.fft.rfft2(p.float(), norm="ortho").abs()
            ft = torch.fft.rfft2(t.float(), norm="ortho").abs()
            losses.append((fp - ft).abs().mean().to(pred.dtype))
        weights.append(w_i.to(dtype=pred.dtype))
    return _reduce_per_sample(losses, weights, pred)


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


def feature_matching_loss(
    fake_feats: list[torch.Tensor],
    real_feats: list[torch.Tensor],
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-layer L1 between D features on fake vs real; optional per-sample weights."""
    total = fake_feats[0].new_zeros(())
    for f, r in zip(fake_feats, real_feats):
        err = (f.float() - r.detach().float()).abs().flatten(1).mean(dim=1)
        if sample_weight is None:
            total = total + err.mean()
        else:
            w = sample_weight.to(err.dtype)
            total = total + (err * w).sum() / w.sum().clamp_min(1.0)
    return total / max(1, len(fake_feats))


def r1_penalty(
    real_logits: torch.Tensor,
    real_input: torch.Tensor,
) -> torch.Tensor:
    """R1 gradient penalty on D's real logits w.r.t. its input.

    Averaged per input element rather than summed, so one ``gamma`` works across
    image sizes. ``real_input`` must have ``requires_grad=True`` before the D forward.
    """
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_input,
        create_graph=True,
    )[0]
    return grad.reshape(grad.shape[0], -1).pow(2).mean(dim=1).mean()


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
