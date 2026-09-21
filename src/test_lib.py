from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.configs import DEFAULT_CONFIG_PATH, config_to_test_defaults, load_config
from src.data_loader.load_data import CustomDataset
from src.model import DiffusionTransformer
from src.utils import DiffusionSchedule, write_log


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre.parse_known_args(argv)
    d = config_to_test_defaults(load_config(pre_args.config))

    p = argparse.ArgumentParser(
        description="Blind inpainting inference (noise only in the hole; no gt input)."
    )
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--checkpoint", type=str, default=d["checkpoint"])
    p.add_argument("--record-file", type=str, default=d["record_file"])
    p.add_argument("--img-dir", type=str, default=d["img_dir"])
    p.add_argument("--output-dir", type=str, default=d["output_dir"])
    p.add_argument("--device", type=str, default=d["device"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--num-samples", type=int, default=d["num_samples"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--mask-size", type=int, default=d["mask_size"])
    p.add_argument("--image-size", type=int, default=d["image_size"])
    p.add_argument("--patch-size", type=int, default=d["patch_size"])
    p.add_argument("--hidden-size", type=int, default=d["hidden_size"])
    p.add_argument("--depth", type=int, default=d["depth"])
    p.add_argument("--num-heads", type=int, default=d["num_heads"])
    p.add_argument("--ffc-blocks", type=int, default=d["ffc_blocks"])
    p.add_argument("--num-timesteps", type=int, default=d["num_timesteps"])
    p.add_argument(
        "--sample-steps",
        type=int,
        default=d["sample_steps"],
        help="Number of reverse steps (subsample schedule). Use 0 for full T.",
    )
    p.add_argument("--learn-sigma", action="store_true", default=d["learn_sigma"])
    p.add_argument("--no-learn-sigma", action="store_false", dest="learn_sigma")
    p.add_argument("--save-npy", action="store_true")
    return p.parse_args(argv)


def load_model(args: argparse.Namespace, device: torch.device) -> DiffusionTransformer:
    model = DiffusionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        class_dropout_prob=0.0,
        num_classes=1,
        learn_sigma=args.learn_sigma,
        ffc_blocks=args.ffc_blocks,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "inference.log")
    write_log(f"device={device} checkpoint={args.checkpoint}", log_path)

    ds = CustomDataset(
        record_file=args.record_file,
        img_dir=args.img_dir,
    ).get_dataset(
        mask_size=args.mask_size,
        image_size=args.image_size,
        seed=args.seed,
        deterministic_masks=True,
    )
    n = min(int(args.num_samples), len(ds))
    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
    write_log(f"indices={indices}", log_path)

    model = load_model(args, device)
    schedule = DiffusionSchedule(num_timesteps=args.num_timesteps).to(device)
    sample_steps = None if args.sample_steps <= 0 else int(args.sample_steps)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for row, idx in enumerate(tqdm(indices, desc="infer")):
        sample = ds[int(idx)]
        # Inference inputs: mask + masked only (no gt to the model)
        mask = sample["mask"].unsqueeze(0).to(device)
        masked = sample["masked"].unsqueeze(0).to(device)
        gt = sample["gt"].unsqueeze(0)  # visualization / optional metrics only

        pred = schedule.inpaint(
            model,
            mask=mask,
            masked=masked,
            num_steps=sample_steps,
        )

        if args.save_npy:
            np.save(
                os.path.join(args.output_dir, f"pred_{idx:05d}.npy"),
                pred[0, 0].detach().cpu().numpy(),
            )

        panels = [
            (gt[0, 0].numpy(), "gt"),
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

    out_fig = os.path.join(args.output_dir, "inference_grid.png")
    fig.suptitle("Blind inpainting inference (no gt model input)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_log(f"saved {out_fig}", log_path)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()
