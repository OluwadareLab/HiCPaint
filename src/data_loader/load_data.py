import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_EPSILON = 1e-8


def _as_2d(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.squeeze(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D Hi-C map, got shape {array.shape}")
    return values


def normalize_hic(array: np.ndarray, clip_pct: float = 99.5) -> np.ndarray:
    """log1p then per-map percentile scale to [0, 1]. Returns float32 HxW.

    Zero counts stay at 0 (not a far-negative log value), so dropouts don't
    dominate the dynamic range.
    """
    values = np.log1p(np.clip(_as_2d(array), 0.0, None))
    high = float(np.percentile(values, clip_pct))
    if high > _EPSILON:
        values = values / high
    else:
        values = np.zeros_like(values)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def zero_fraction(array: np.ndarray) -> float:
    return float(np.mean(_as_2d(array) <= 0.0))


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


MASK_PLACEMENTS = ("diagonal", "random", "mixed")


def make_random_mask(
    height: int, width: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Random square hole anywhere (independent row/col offsets). 1 = hole."""
    if size > min(height, width):
        raise ValueError(f"Mask size {size} exceeds image {height}x{width}")
    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[top: top + size, left: left + size] = 1.0
    return mask


def make_mask(
    height: int,
    width: int,
    size: int,
    rng: np.random.Generator,
    placement: str = "mixed",
) -> np.ndarray:
    """Hole mask by ``placement``: diagonal | random | mixed (50/50)."""
    if placement not in MASK_PLACEMENTS:
        raise ValueError(f"mask placement must be one of {MASK_PLACEMENTS}, got {placement!r}")
    if placement == "mixed":
        placement = "diagonal" if rng.random() < 0.5 else "random"
    if placement == "diagonal":
        return make_diagonal_mask(height, width, size, rng)
    return make_random_mask(height, width, size, rng)


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


def make_hough_line(
    gray01: np.ndarray,
    edge: Optional[np.ndarray] = None,
    threshold: int = 40,
    min_line_length: int = 20,
    max_line_gap: int = 8,
) -> np.ndarray:
    """Probabilistic Hough lines from Canny edges -> float32 HxW in {0,1}."""
    if edge is None:
        edge = make_canny_edge(gray01)
    u8 = np.clip(edge * 255.0, 0, 255).astype(np.uint8)
    lines = cv2.HoughLinesP(
        u8,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )
    out = np.zeros(gray01.shape[:2], dtype=np.float32)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, 1)
    return out


def make_shi_tomasi_dots(
    gray01: np.ndarray,
    max_corners: int = 256,
    quality_level: float = 0.01,
    min_distance: float = 8.0,
) -> np.ndarray:
    """Shi-Tomasi corners as binary dots -> float32 HxW in {0,1}."""
    u8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        u8,
        maxCorners=int(max_corners),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance),
        useHarrisDetector=False,
    )
    out = np.zeros(gray01.shape[:2], dtype=np.float32)
    if corners is not None:
        for x, y in corners[:, 0]:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < out.shape[0] and 0 <= xi < out.shape[1]:
                out[yi, xi] = 1.0
    return out


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
        mask_placement: str = "mixed",
    ):
        if mask_placement not in MASK_PLACEMENTS:
            raise ValueError(
                f"mask_placement must be one of {MASK_PLACEMENTS}, got {mask_placement!r}"
            )
        self.image_paths = image_paths
        self.mask_size = mask_size
        self.mask_placement = mask_placement
        self.edge_sigma = edge_sigma
        self.line_band = line_band  # kept for API compat; unused (Hough replaces it)
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
        mask = make_mask(h, w, self.mask_size, rng, self.mask_placement)
        keep = 1.0 - mask
        masked = gray * keep
        edge_full = make_canny_edge(gray, sigma=self.edge_sigma)
        edge = edge_full * keep
        line = make_hough_line(gray, edge=edge_full) * keep
        dot = make_shi_tomasi_dots(gray) * keep

        return {
            "gt": _to_1c(gray),
            "mask": _to_1c(mask),
            "masked": _to_1c(masked),
            "edge": _to_1c(edge),
            "line": _to_1c(line),
            "dot": _to_1c(dot),
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

    def _filter_sparse(self, paths: List[str], max_zero_frac: float) -> List[str]:
        """Drop maps with more than ``max_zero_frac`` zero pixels (cached per record file)."""
        if max_zero_frac >= 1.0 or not paths:
            return paths
        cache = f"{self.record_file}.zf{max_zero_frac:g}.keep"
        if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(
            self.record_file
        ):
            with open(cache, "r") as fid:
                keep = {line.strip() for line in fid if line.strip()}
            return [p for p in paths if p in keep]
        kept = [p for p in paths if zero_fraction(np.load(p)) <= max_zero_frac]
        tmp = f"{cache}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as fid:
                fid.write("\n".join(kept) + "\n")
            os.replace(tmp, cache)
        except OSError:
            pass
        return kept

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
        mask_placement: str = "mixed",
        max_zero_frac: float = 0.5,
    ) -> ImageDataset:
        del augment  # reserved; flips not applied to structured Hi-C yet
        paths = self._subset_paths(
            self._filter_sparse(self._prep_paths(), float(max_zero_frac)),
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
            mask_placement=mask_placement,
        )
