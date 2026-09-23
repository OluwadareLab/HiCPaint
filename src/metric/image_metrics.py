from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


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
