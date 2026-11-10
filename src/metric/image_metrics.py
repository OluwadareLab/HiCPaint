from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


# Typical metric scales so equal weights ≈ equal vote after normalization.
_PSNR_SCALE = 40.0
_SSIM_SCALE = 1.0
_FID_SCALE = 350.0


def masked_selection_score(
    metrics: Dict[str, float],
    w_psnr: float = 1.0,
    w_ssim: float = 1.0,
    w_fid: float = 1.0,
) -> float:
    """Higher is better on masked hole (scale-normalized):

    ``w_psnr*(PSNR/40) + w_ssim*(SSIM/1) - w_fid*(FID/350)``

    Default weights=1 → equal contribution at typical ranges. FID may be NaN
    early (too few samples); then score uses PSNR+SSIM only.
    """
    psnr = float(metrics.get("psnr_masked", float("nan")))
    ssim = float(metrics.get("ssim_masked", float("nan")))
    fid = float(metrics.get("fid_masked", float("nan")))
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        return float("-inf")
    score = w_psnr * (psnr / _PSNR_SCALE) + w_ssim * (ssim / _SSIM_SCALE)
    if math.isfinite(fid):
        score -= w_fid * (fid / _FID_SCALE)
    return score


def _eligible(
    history: List[Dict[str, float]], min_epoch: int = 0
) -> Tuple[List[Dict[str, float]], Tuple[str, ...]]:
    """Entries competing for best, plus the metric keys in play.

    Once any epoch has a finite FID, epochs without one drop out so all
    candidates are compared on the same three metrics.
    """
    rows = [
        h
        for h in history
        if int(h["epoch"]) >= min_epoch
        and math.isfinite(float(h["psnr"]))
        and math.isfinite(float(h["ssim"]))
    ]
    with_fid = [h for h in rows if math.isfinite(float(h.get("fid", float("nan"))))]
    if with_fid:
        return with_fid, ("psnr", "ssim", "fid")
    return rows, ("psnr", "ssim")


def _oriented(h: Dict[str, float], keys: Tuple[str, ...]) -> Tuple[float, ...]:
    """Metric vector where higher is better (FID negated)."""
    return tuple(-float(h[k]) if k == "fid" else float(h[k]) for k in keys)


def pareto_front(
    history: List[Dict[str, float]], min_epoch: int = 0
) -> List[Dict[str, float]]:
    """Non-dominated entries over masked PSNR (max), SSIM (max), FID (min)."""
    rows, keys = _eligible(history, min_epoch)
    vecs = [_oriented(h, keys) for h in rows]
    front = []
    for i, vi in enumerate(vecs):
        dominated = any(
            all(a >= b for a, b in zip(vj, vi)) and any(a > b for a, b in zip(vj, vi))
            for j, vj in enumerate(vecs)
            if j != i
        )
        if not dominated:
            front.append(rows[i])
    return front


def balanced_pick(
    history: List[Dict[str, float]], min_epoch: int = 0
) -> Tuple[Optional[int], Dict[str, float], List[int]]:
    """Pareto epoch with the highest worst-case normalized metric (maximin).

    Each metric is min-max normalized to [0,1] over all eligible epochs (FID
    flipped). Ties on the minimum are broken by the mean. Returns
    ``(best_epoch or None, normalized metrics of best, pareto epochs)``.
    """
    rows, keys = _eligible(history, min_epoch)
    front = pareto_front(history, min_epoch)
    if not front:
        return None, {}, []
    vecs = [_oriented(h, keys) for h in rows]
    lo = [min(v[i] for v in vecs) for i in range(len(keys))]
    hi = [max(v[i] for v in vecs) for i in range(len(keys))]

    def norm(h: Dict[str, float]) -> Tuple[float, ...]:
        v = _oriented(h, keys)
        return tuple(
            (v[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 1.0
            for i in range(len(keys))
        )

    best = max(front, key=lambda h: (min(norm(h)), sum(norm(h)) / len(keys)))
    best_norm = dict(zip(keys, norm(best)))
    return int(best["epoch"]), best_norm, sorted(int(h["epoch"]) for h in front)


def _to_rgb01(x: torch.Tensor) -> torch.Tensor:
    """(N,1,H,W) or (N,3,H,W) in [0,1] -> (N,3,H,W) float."""
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def _crop_hole(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop each sample to the axis-aligned bbox of the hole (``mask > 0.5``)."""
    pred_crops: List[torch.Tensor] = []
    gt_crops: List[torch.Tensor] = []
    for i in range(pred.shape[0]):
        m = mask[i, 0] > 0.5
        if not bool(m.any().item()):
            continue
        ys = m.any(dim=1).nonzero(as_tuple=False).squeeze(1)
        xs = m.any(dim=0).nonzero(as_tuple=False).squeeze(1)
        y0, y1 = int(ys[0].item()), int(ys[-1].item()) + 1
        x0, x1 = int(xs[0].item()), int(xs[-1].item()) + 1
        pred_crops.append(pred[i : i + 1, :, y0:y1, x0:x1])
        gt_crops.append(gt[i : i + 1, :, y0:y1, x0:x1])
    if not pred_crops:
        # Fallback: empty hole — return 1x1 zeros so SSIM still gets a tensor.
        z = pred.new_zeros((pred.shape[0], pred.shape[1], 1, 1))
        return z, z.clone()
    return torch.cat(pred_crops, dim=0), torch.cat(gt_crops, dim=0)


class ValImageMetrics(nn.Module):
    """Accumulates PSNR/SSIM/FID (full + masked-hole) over a val/test/infer pass."""

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        # Rank-0-only val must not all-reduce on compute (DDP is initialized).
        _sync = {"sync_on_compute": False}
        self.psnr = PeakSignalNoiseRatio(data_range=data_range, **_sync)
        self.psnr_masked = PeakSignalNoiseRatio(data_range=data_range, **_sync)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range, **_sync)
        self.ssim_masked = StructuralSimilarityIndexMeasure(
            data_range=data_range, **_sync
        )
        self.fid = FrechetInceptionDistance(
            feature=2048, normalize=True, **_sync
        )
        self.fid_masked = FrechetInceptionDistance(
            feature=2048, normalize=True, **_sync
        )

    def reset(self) -> None:
        self.psnr.reset()
        self.psnr_masked.reset()
        self.ssim.reset()
        self.ssim_masked.reset()
        self.fid.reset()
        self.fid_masked.reset()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> None:
        pred = pred.clamp(0.0, 1.0)
        gt = gt.clamp(0.0, 1.0)
        self.psnr.update(pred, gt)
        self.ssim.update(pred, gt)
        pred_hole, gt_hole = _crop_hole(pred, gt, mask)
        self.psnr_masked.update(pred_hole, gt_hole)
        self.ssim_masked.update(pred_hole, gt_hole)
        real = _to_rgb01(gt)
        fake = _to_rgb01(pred)
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        real_h = _to_rgb01(gt_hole)
        fake_h = _to_rgb01(pred_hole)
        self.fid_masked.update(real_h, real=True)
        self.fid_masked.update(fake_h, real=False)

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        out = {
            "psnr": float(self.psnr.compute().item()),
            "psnr_masked": float(self.psnr_masked.compute().item()),
            "ssim": float(self.ssim.compute().item()),
            "ssim_masked": float(self.ssim_masked.compute().item()),
        }
        try:
            out["fid"] = float(self.fid.compute().item())
        except Exception:
            out["fid"] = float("nan")
        try:
            out["fid_masked"] = float(self.fid_masked.compute().item())
        except Exception:
            out["fid_masked"] = float("nan")
        return out
