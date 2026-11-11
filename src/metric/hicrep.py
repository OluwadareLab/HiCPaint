from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
from scipy.stats import rankdata


def fast_mean_filter(mat: np.ndarray, h: int) -> np.ndarray:
    if h == 0:
        return np.asarray(mat, dtype=np.float64)
    return ndimage.uniform_filter(
        np.asarray(mat, dtype=np.float64), size=2 * h + 1, mode="nearest"
    )


def vstran(d: np.ndarray) -> np.ndarray:
    n_rows = d.shape[0]
    out = np.empty((n_rows, 2), dtype=np.float64)
    for c in range(2):
        perm = np.random.permutation(n_rows)
        ranks = np.empty(n_rows, dtype=np.float64)
        ranks[perm] = rankdata(d[perm, c], method="ordinal")
        out[:, c] = ranks / n_rows
    return out


def compute_hicrep_scc(
    mat1: np.ndarray,
    mat2: np.ndarray,
    resol: int = 25000,
    h: int = 1,
    lbr: int = 0,
    ubr: int = 25000 * 253,
) -> float:
    smt_r1 = fast_mean_filter(mat1, h)
    smt_r2 = fast_mean_filter(mat2, h)

    lb = int(np.floor(lbr / resol))
    ub = min(int(np.floor(ubr / resol)), smt_r1.shape[0] - 1)

    corr_list = []
    wei_list = []
    for dist in range(lb, ub + 1):
        ffd1 = np.diagonal(smt_r1, offset=-dist)
        ffd2 = np.diagonal(smt_r2, offset=-dist)
        keep = ~((ffd1 == 0) & (ffd2 == 0))
        ffd1 = ffd1[keep]
        ffd2 = ffd2[keep]
        n = len(ffd1)
        if n == 0 or len(np.unique(ffd1)) == 1 or len(np.unique(ffd2)) == 1:
            continue
        nd = vstran(np.column_stack((ffd1, ffd2)))
        corr_list.append(np.corrcoef(ffd1, ffd2)[0, 1])
        wei_list.append(
            np.sqrt(np.var(nd[:, 0], ddof=1) * np.var(nd[:, 1], ddof=1)) * n
        )

    corr = np.asarray(corr_list)
    wei = np.asarray(wei_list)
    valid = ~np.isnan(corr) & ~np.isnan(wei)
    corr = corr[valid]
    wei = wei[valid] + 1e-10
    if len(corr) == 0:
        return float("nan")
    return float(np.dot(corr, wei) / np.sum(wei))


def get_hicrep_scc(
    mat1: np.ndarray, mat2: np.ndarray, resol: int = 25000, h: int = 1
) -> float:
    """HiCRep SCC over all diagonals of a square matrix."""
    size = min(mat1.shape[-2], mat1.shape[-1])
    try:
        return compute_hicrep_scc(mat1, mat2, resol, h, 0, resol * (size - 1))
    except Exception as e:
        print(f"Error computing HiCRep SCC: {e}")
        return float("nan")
