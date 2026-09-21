from __future__ import annotations
from src.data_loader.load_data import CustomDataset

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


KEYS = ("gt", "mask", "masked", "edge", "line")
CMAPS = {
    "gt": "Reds",
    "mask": "gray",
    "masked": "Reds",
    "edge": "gray",
    "line": "gray",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--record-file",
        type=Path,
        default=ROOT / "dataset" / "dataset_dict_5000_256.test",
    )
    p.add_argument(
        "--img-dir",
        type=Path,
        default=Path(
            "/home/hc0783.unt.ad.unt.edu/workspace/"
            "hicinterpolate/datasets/timeseries/new_triplets"
        ),
    )
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "load_data_preview.png",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ds = CustomDataset(
        record_file=str(args.record_file),
        img_dir=str(args.img_dir),
    ).get_dataset(
        mask_size=args.mask_size,
        image_size=args.image_size,
        seed=args.seed,
        deterministic_masks=True,
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.n, replace=False)

    fig, axes = plt.subplots(
        args.n, len(KEYS), figsize=(3 * len(KEYS), 3 * args.n), squeeze=False
    )
    for row, idx in enumerate(indices):
        sample = ds[int(idx)]
        path = Path(ds.image_paths[int(idx)])
        for col, key in enumerate(KEYS):
            ax = axes[row, col]
            arr = sample[key].squeeze(0).numpy()
            ax.imshow(arr, cmap=CMAPS[key], vmin=0.0, vmax=1.0, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(key)
            if col == 0:
                ax.set_ylabel(f"idx={idx}\n{path.name}", fontsize=8)

    fig.suptitle(
        f"ImageDataset preview  seed={args.seed}  mask={args.mask_size}x{args.mask_size}",
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"indices={indices.tolist()}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
