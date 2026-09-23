import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_EPSILON = 1e-8


def normalize_hic(array: np.ndarray) -> np.ndarray:
    """log2(clip) then per-map min-max to [0, 1]. Returns float32 HxW."""
    values = np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.squeeze(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D Hi-C map, got shape {array.shape}")

    values = np.log2(np.clip(values, _EPSILON, None))
    low = float(values.min())
    high = float(values.max())
    if high > low:
        values = (values - low) / (high - low)
    else:
        values = np.zeros_like(values)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def make_diagonal_mask(
    height: int, width: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Random square hole constrained to the main diagonal. 1 = hole."""
    if size > min(height, width):
        raise ValueError(f"Mask size {size} exceeds image {height}x{width}")
    span_y = height - size
    span_x = width - size
    fraction = float(rng.random())
    top = int(round(span_y * fraction))
    left = int(round(span_x * fraction))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[top: top + size, left: left + size] = 1.0
    return mask


def make_canny_edge(gray01: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Canny edges from [0,1] gray map -> float32 HxW in {0,1}."""
    u8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    try:
        from skimage.color import rgb2gray
        from skimage.feature import canny

        rgb = np.stack([gray01, gray01, gray01], axis=-1)
        edge = canny(rgb2gray(rgb), sigma=sigma).astype(np.float32)
    except Exception:
        blured = cv2.GaussianBlur(u8, ksize=(7, 7), sigmaX=sigma, sigmaY=sigma)
        edge = cv2.Canny(
            blured, threshold1=int(25.5), threshold2=int(51.0)
        ).astype(np.float32) / 255.0
    return edge


def make_diagonal_line(
    height: int, width: int, band: float = 8.0
) -> np.ndarray:
    """Soft main-diagonal prior as 'line' substitute for Hi-C."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.abs(yy - xx)
    line = np.exp(-(dist ** 2) / (2.0 * band * band)).astype(np.float32)
    return line


def _to_1c(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_size: int = 64,
        edge_sigma: float = 3.0,
        line_band: float = 8.0,
        seed: int = 42,
        deterministic_masks: bool = False,
        image_size: Optional[int] = None,
    ):
        self.image_paths = image_paths
        self.mask_size = mask_size
        self.edge_sigma = edge_sigma
        self.line_band = line_band
        self.seed = seed
        self.deterministic_masks = deterministic_masks
        self.image_size = image_size
        self._epoch_rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.image_paths)

    def set_epoch(self, epoch: int):
        self._epoch_rng = np.random.default_rng(self.seed + epoch * 10007)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        gray = normalize_hic(np.load(path))
        if self.image_size is not None and gray.shape != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"{path}: expected {self.image_size}x{self.image_size}, got {gray.shape}"
            )

        if self.deterministic_masks:
            rng = np.random.default_rng(self.seed + idx)
        else:
            rng = self._epoch_rng

        h, w = gray.shape
        mask = make_diagonal_mask(h, w, self.mask_size, rng)
        keep = 1.0 - mask
        masked = gray * keep
        edge = make_canny_edge(gray, sigma=self.edge_sigma) * keep
        line = make_diagonal_line(h, w, band=self.line_band) * keep

        return {
            "gt": _to_1c(gray),
            "mask": _to_1c(mask),
            "masked": _to_1c(masked),
            "edge": _to_1c(edge),
            "line": _to_1c(line),
        }


class CustomDataset:
    def __init__(self, record_file: str, img_dir: str):
        self.record_file = record_file
        self.img_dir = img_dir

    def _prep_paths(self) -> List[str]:
        with open(self.record_file, "r") as fid:
            lines = [line.strip() for line in fid if line.strip()]

        paths = []
        for entry in lines:
            if not entry.endswith(".npy"):
                continue
            if os.path.isabs(entry):
                path = entry
            else:
                path = os.path.join(self.img_dir, entry)
            paths.append(path)
        return paths

    @staticmethod
    def _subset_paths(
        paths: List[str],
        seed: int,
        subset_fraction: float = 1.0,
    ) -> List[str]:
        """Keep a seeded random subset of ``paths`` by fraction."""
        n = len(paths)
        if n == 0:
            return paths
        frac = float(subset_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"subset_fraction must be in (0, 1], got {frac}")
        n_keep = n if frac >= 1.0 else max(1, int(round(n * frac)))
        if n_keep >= n:
            return list(paths)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n_keep, replace=False)
        idx.sort()
        return [paths[i] for i in idx]

    def get_dataset(
        self,
        mask_size: int = 64,
        edge_sigma: float = 3.0,
        line_band: float = 8.0,
        seed: int = 42,
        deterministic_masks: bool = False,
        image_size: Optional[int] = None,
        augment: bool = False,
        subset_fraction: float = 1.0,
    ) -> ImageDataset:
        del augment  # reserved; flips not applied to structured Hi-C yet
        paths = self._subset_paths(
            self._prep_paths(),
            seed=seed,
            subset_fraction=subset_fraction,
        )
        return ImageDataset(
            image_paths=paths,
            mask_size=mask_size,
            edge_sigma=edge_sigma,
            line_band=line_band,
            seed=seed,
            deterministic_masks=deterministic_masks,
            image_size=image_size,
        )
