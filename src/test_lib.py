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
from src.metric import ValImageMetrics
from src.model import DiffusionTransformer
from src.utils import DiffusionSchedule, append_csv_row, write_log


METRIC_FIELDS = [
    "psnr",
    "psnr_masked",
    "ssim",
    "ssim_masked",
    "fid",
    "fid_masked",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre.parse_known_args(argv)
    d = config_to_test_defaults(load_config(pre_args.config))

    p = argparse.ArgumentParser(
        description="Blind inpainting inference (settings from config by default)."
    )
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--checkpoint", type=str, default=d["checkpoint"])
    p.add_argument("--dataset-dict", type=str, default=d["dataset_dict"])
    p.add_argument(
        "--record-file",
        type=str,
        default=None,
        help="Override record list; default = {dataset_dict}.{split}",
    )
    p.add_argument("--img-dir", type=str, default=d["img_dir"])
    p.add_argument("--output-dir", type=str, default=d["output_dir"])
    p.add_argument("--device", type=str, default=d["device"])
    p.add_argument("--split", type=str, default=d["split"], choices=("train", "val", "test"))
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument(
        "--num-samples",
        type=int,
        default=d["num_samples"],
        help="Maps to inpaint+viz; 0 = entire split (after subset_fraction).",
    )
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--mask-size", type=int, default=d["mask_size"])
    p.add_argument("--subset-fraction", type=float, default=d["subset_fraction"])
    p.add_argument("--image-size", type=int, default=d["image_size"])
    p.add_argument("--patch-size", type=int, default=d["patch_size"])
    p.add_argument("--hidden-size", type=int, default=d["hidden_size"])
    p.add_argument("--depth", type=int, default=d["depth"])
    p.add_argument("--num-heads", type=int, default=d["num_heads"])
    p.add_argument("--ffc-blocks", type=int, default=d["ffc_blocks"])
    p.add_argument("--stem-channels", type=int, default=d["stem_channels"])
    p.add_argument("--mid-channels", type=int, default=d["mid_channels"])
    p.add_argument("--mask-attn-bias", type=float, default=d["mask_attn_bias"])
    p.add_argument("--num-timesteps", type=int, default=d["num_timesteps"])
    p.add_argument(
        "--prediction",
        type=str,
        default=d["prediction"],
        choices=("x0", "eps"),
    )
    p.add_argument(
        "--infer-t",
        type=int,
        default=d["infer_t"],
        help="One-step infer timestep; <0 means T-1.",
    )
    p.add_argument("--learn-sigma", action="store_true", default=d["learn_sigma"])
    p.add_argument("--no-learn-sigma", action="store_false", dest="learn_sigma")
    p.add_argument(
        "--save-npy",
        action=argparse.BooleanOptionalAction,
        default=d["save_npy"],
        help="Save predicted matrices as .npy (default from config).",
    )
    p.add_argument("--viz-name", type=str, default=d["viz_name"])
    args = p.parse_args(argv)
    if not args.record_file:
        args.record_file = f"{args.dataset_dict}.{args.split}"
    return args


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
        stem_channels=args.stem_channels,
        mid_channels=args.mid_channels,
        mask_attn_bias=args.mask_attn_bias,
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
    matrix_dir = os.path.join(args.output_dir, "matrices")
    if args.save_npy:
        os.makedirs(matrix_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "inference.log")
    metrics_csv = os.path.join(args.output_dir, "inference_metrics.csv")
    write_log(
        f"device={device} checkpoint={args.checkpoint} split={args.split} "
        f"prediction={args.prediction} infer_t={args.infer_t} "
        f"subset_fraction={args.subset_fraction} "
        f"save_npy={int(bool(args.save_npy))}",
        log_path,
    )

    ds = CustomDataset(
        record_file=args.record_file,
        img_dir=args.img_dir,
    ).get_dataset(
        mask_size=args.mask_size,
        image_size=args.image_size,
        seed=args.seed,
        deterministic_masks=True,
        subset_fraction=float(args.subset_fraction),
    )
    n_all = len(ds)
    if int(args.num_samples) <= 0:
        indices = list(range(n_all))
        n = n_all
    else:
        n = min(int(args.num_samples), n_all)
        if n <= 0:
            raise RuntimeError(f"No samples in {args.record_file} after subset_fraction")
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(n_all, size=n, replace=False).tolist())
    if n <= 0:
        raise RuntimeError(f"No samples in {args.record_file} after subset_fraction")
    # Full-split grids are huge; cap figure rows (matrices still saved for all).
    viz_cap = 16
    viz_indices = indices[:viz_cap]
    write_log(
        f"dataset_size={n_all} num_samples={n} viz_rows={len(viz_indices)} "
        f"indices={indices[:20]}{'...' if n > 20 else ''}",
        log_path,
    )

    model = load_model(args, device)
    schedule = DiffusionSchedule(num_timesteps=args.num_timesteps).to(device)
    prediction = str(args.prediction).strip().lower()
    infer_t = int(args.infer_t)
    image_metrics = ValImageMetrics().to(device)
    image_metrics.reset()

    fig, axes = plt.subplots(
        len(viz_indices), 3, figsize=(9, 3 * len(viz_indices)), squeeze=False
    )
    viz_row = 0
    for idx in tqdm(indices, desc=f"infer[{args.split}]"):
        sample = ds[int(idx)]
        mask = sample["mask"].unsqueeze(0).to(device)
        masked = sample["masked"].unsqueeze(0).to(device)
        gt = sample["gt"].unsqueeze(0).to(device)

        pred = schedule.inpaint_onestep(
            model,
            mask=mask,
            masked=masked,
            t_value=infer_t,
            prediction=prediction,
        )
        image_metrics.update(pred, gt, mask)
        pred_np = pred[0, 0].detach().cpu().numpy()

        if args.save_npy:
            np.save(os.path.join(matrix_dir, f"pred_{idx:05d}.npy"), pred_np)
            np.save(
                os.path.join(matrix_dir, f"gt_{idx:05d}.npy"),
                gt[0, 0].detach().cpu().numpy(),
            )
            np.save(
                os.path.join(matrix_dir, f"masked_{idx:05d}.npy"),
                masked[0, 0].detach().cpu().numpy(),
            )

        if viz_row < len(viz_indices) and int(idx) == int(viz_indices[viz_row]):
            panels = [
                (gt[0, 0].detach().cpu().numpy(), "gt"),
                (masked[0, 0].detach().cpu().numpy(), "masked"),
                (pred_np, "predicted"),
            ]
            for col, (arr, title) in enumerate(panels):
                ax = axes[viz_row, col]
                ax.imshow(arr, cmap="Reds", vmin=0.0, vmax=1.0, origin="upper")
                ax.set_xticks([])
                ax.set_yticks([])
                if viz_row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(f"idx={idx}", fontsize=9)
            viz_row += 1

    metrics = image_metrics.compute()
    append_csv_row(
        metrics_csv,
        {k: f"{float(metrics.get(k, float('nan'))):.4f}" for k in METRIC_FIELDS},
        METRIC_FIELDS,
    )
    write_log(
        f"metrics psnr={metrics.get('psnr', float('nan')):.4f} "
        f"psnr_masked={metrics.get('psnr_masked', float('nan')):.4f} "
        f"ssim={metrics.get('ssim', float('nan')):.4f} "
        f"ssim_masked={metrics.get('ssim_masked', float('nan')):.4f} "
        f"fid={metrics.get('fid', float('nan')):.4f} "
        f"fid_masked={metrics.get('fid_masked', float('nan')):.4f} "
        f"-> {metrics_csv}",
        log_path,
    )

    out_fig = os.path.join(args.output_dir, args.viz_name)
    fig.suptitle(
        f"Blind inpaint [{args.split}]  prediction={args.prediction} "
        f"infer_t={args.infer_t}  showing {len(viz_indices)}/{n}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_log(f"saved viz -> {out_fig}", log_path)
    if args.save_npy:
        write_log(f"saved matrices -> {matrix_dir}", log_path)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()
