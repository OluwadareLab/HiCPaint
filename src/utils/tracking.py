from __future__ import annotations

import csv
import os
from typing import List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def append_csv_row(path: str, row: Mapping[str, object], fieldnames: Sequence[str]) -> None:
    """Append one CSV row; write header if the file does not exist yet."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_train_val_loss(
    epochs: Sequence[int],
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    out_path: str,
) -> None:
    """Draw and save a train vs val loss curve."""
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(epochs), list(train_losses), label="train_loss", linewidth=2)
    ax.plot(list(epochs), list(val_losses), label="val_loss", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def select_fixed_indices(n_dataset: int, n_samples: int, seed: int) -> List[int]:
    """Random but fixed sample indices for visualization."""
    n = min(int(n_samples), int(n_dataset))
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_dataset, size=n, replace=False).tolist())


@torch.no_grad()
def visualize_gt_masked_pred(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    schedule: nn.Module,
    device: torch.device,
    out_path: str,
    epoch: int,
    prediction: str = "x0",
    infer_t: int = -1,
) -> None:
    """Save fixed samples: gt | masked | predicted (one-step blind inpaint)."""
    if not indices:
        return
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    model.eval()
    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        gt = sample["gt"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        masked = sample["masked"].unsqueeze(0).to(device)
        y = torch.zeros(1, dtype=torch.long, device=device)
        pred = schedule.inpaint_onestep(
            model, mask, masked, y=y, t_value=infer_t, prediction=prediction
        )

        panels = [
            (gt[0, 0].detach().cpu().numpy(), "gt"),
            (masked[0, 0].detach().cpu().numpy(), "masked"),
            (pred[0, 0].detach().cpu().numpy(), "predicted"),
        ]
        for col, (arr, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(arr, cmap="Reds", vmin=0.0, vmax=1.0, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"idx={idx}", fontsize=9)

    fig.suptitle(f"Best epoch {epoch}  (one-step x0, blind)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
