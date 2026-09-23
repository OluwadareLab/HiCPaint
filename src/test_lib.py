from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.configs import DEFAULT_CONFIG_PATH, config_to_test_defaults, load_config
from src.data_loader.load_data import CustomDataset
from src.metric import ValImageMetrics
from src.model import DiffusionTransformer
from src.utils import (
    DiffusionSchedule,
    append_csv_row,
    select_fixed_indices,
    write_log,
)


METRIC_FIELDS = [
    "dataset",
    "chromosome",
    "n",
    "psnr",
    "psnr_masked",
    "ssim",
    "ssim_masked",
    "fid",
    "fid_masked",
]

_CHR_RE = re.compile(r"(?:^|/)(chr(?:\d+|X|Y|M|MT))(?:/|$)", re.IGNORECASE)
# Drop duration suffix: dmso_control_30m -> dmso_control, dtag_v1_60m -> dtag_v1
_DURATION_RE = re.compile(r"_\d+m$", re.IGNORECASE)


def _chromosome_from_path(path: str) -> str:
    m = _CHR_RE.search(str(path).replace("\\", "/"))
    return m.group(1) if m else "unknown"


def _dataset_from_path(path: str) -> str:
    """Experiment folder above ``chr*``, without duration suffix (``_30m`` / ``_60m``)."""
    parts = str(path).replace("\\", "/").split("/")
    raw = "unknown"
    for i, part in enumerate(parts):
        if re.fullmatch(r"chr(?:\d+|X|Y|M|MT)", part, flags=re.IGNORECASE):
            raw = parts[i - 1] if i > 0 else "unknown"
            break
    return _DURATION_RE.sub("", raw)


def _labels_from_path(path: str) -> Tuple[str, str]:
    return _dataset_from_path(path), _chromosome_from_path(path)


def _chr_sort_key(name: str) -> Tuple[int, int, str]:
    m = re.fullmatch(r"chr(\d+|X|Y|M|MT)", name, flags=re.IGNORECASE)
    if not m:
        return (1, 0, name)
    tok = m.group(1).upper()
    if tok.isdigit():
        return (0, int(tok), name)
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return (0, order.get(tok, 99), name)


def _group_sort_key(key: Tuple[str, str]) -> Tuple[str, Tuple[int, int, str]]:
    dataset, chrom = key
    return (dataset, _chr_sort_key(chrom))


def group_indices_by_dataset_chromosome(
    paths: Sequence[str],
) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, path in enumerate(paths):
        groups[_labels_from_path(path)].append(i)
    return {
        key: groups[key]
        for key in sorted(groups.keys(), key=_group_sort_key)
    }


def group_indices_by_chromosome(paths: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, path in enumerate(paths):
        groups[_chromosome_from_path(path)].append(i)
    return {
        chrom: groups[chrom]
        for chrom in sorted(groups.keys(), key=_chr_sort_key)
    }


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
        help="Maps to inpaint; 0 = entire split (after subset_fraction).",
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
        help="Save predicted matrices as .npy (default from config; off for now).",
    )
    p.add_argument(
        "--viz-per-chromosome",
        type=int,
        default=d["viz_per_chromosome"],
        help="Random-but-fixed viz rows per dataset×chromosome group.",
    )
    p.add_argument("--viz-name", type=str, default=d["viz_name"])
    args = p.parse_args(argv)
    if not args.record_file:
        args.record_file = f"{args.dataset_dict}.{args.split}"
    return args


def load_model(args: argparse.Namespace, device: torch.device) -> DiffusionTransformer:
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # LabelEmbedder is num_classes + 1 when trained with class_dropout_prob > 0.
    num_classes = 1
    y_key = "y_embedder.embedding_table.weight"
    y_rows = int(state[y_key].shape[0]) if y_key in state else num_classes
    class_dropout_prob = 0.1 if y_rows > num_classes else 0.0
    model = DiffusionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        class_dropout_prob=class_dropout_prob,
        num_classes=num_classes,
        learn_sigma=args.learn_sigma,
        ffc_blocks=args.ffc_blocks,
        stem_channels=args.stem_channels,
        mid_channels=args.mid_channels,
        mask_attn_bias=args.mask_attn_bias,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _metrics_row(
    dataset: str,
    chromosome: str,
    n: int,
    metrics: Dict[str, float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": dataset,
        "chromosome": chromosome,
        "n": int(n),
    }
    for k in METRIC_FIELDS:
        if k in ("dataset", "chromosome", "n"):
            continue
        row[k] = f"{float(metrics.get(k, float('nan'))):.4f}"
    return row


def _save_sample_viz(
    dataset: str,
    chrom: str,
    idx: int,
    gt: np.ndarray,
    masked: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    split: str,
    prediction: str,
    infer_t: int,
) -> None:
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), squeeze=False)
    panels = [(gt, "gt"), (masked, "masked"), (pred, "predicted")]
    for col, (arr, title) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(arr, cmap="Reds", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel(f"idx={idx}", fontsize=9)
    fig.suptitle(
        f"Blind inpaint [{split}/{dataset}/{chrom}]  "
        f"prediction={prediction} infer_t={infer_t}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    matrix_dir = os.path.join(args.output_dir, "matrices")
    if args.save_npy:
        os.makedirs(matrix_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "inference.log")
    metrics_csv = os.path.join(args.output_dir, "inference_metrics.csv")
    write_log(
        f"device={device} checkpoint={args.checkpoint} split={args.split} "
        f"prediction={args.prediction} infer_t={args.infer_t} "
        f"subset_fraction={args.subset_fraction} "
        f"viz_per_chromosome={args.viz_per_chromosome} "
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
    if n_all <= 0:
        raise RuntimeError(f"No samples in {args.record_file} after subset_fraction")

    by_group = group_indices_by_dataset_chromosome(ds.image_paths)
    write_log(
        "groups "
        + " ".join(f"{d}/{c}={len(idxs)}" for (d, c), idxs in by_group.items()),
        log_path,
    )

    if int(args.num_samples) <= 0:
        indices = list(range(n_all))
        n = n_all
    else:
        n = min(int(args.num_samples), n_all)
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(n_all, size=n, replace=False).tolist())
    index_set = set(int(i) for i in indices)

    viz_per_chr = max(0, int(args.viz_per_chromosome))
    viz_by_group: Dict[Tuple[str, str], List[int]] = {}
    for key, group_idxs in by_group.items():
        eligible = [i for i in group_idxs if i in index_set]
        if not eligible or viz_per_chr <= 0:
            viz_by_group[key] = []
            continue
        dataset, chrom = key
        # Seed offset per dataset×chr so picks stay fixed but independent.
        group_seed = int(args.seed) + sum(ord(ch) for ch in f"{dataset}/{chrom}")
        local = select_fixed_indices(len(eligible), viz_per_chr, group_seed)
        viz_by_group[key] = [eligible[j] for j in local]
    viz_index_set = {i for idxs in viz_by_group.values() for i in idxs}
    write_log(
        f"dataset_size={n_all} num_samples={n} "
        f"viz="
        + " ".join(f"{d}/{c}:{viz_by_group[(d, c)]}" for d, c in by_group),
        log_path,
    )

    model = load_model(args, device)
    schedule = DiffusionSchedule(num_timesteps=args.num_timesteps).to(device)
    prediction = str(args.prediction).strip().lower()
    infer_t = int(args.infer_t)

    overall_metrics = ValImageMetrics().to(device)
    overall_metrics.reset()
    group_metrics = {key: ValImageMetrics().to(device) for key in by_group}
    for meter in group_metrics.values():
        meter.reset()
    group_counts = {key: 0 for key in by_group}
    dataset_metrics: Dict[str, ValImageMetrics] = {}
    dataset_counts: Dict[str, int] = defaultdict(int)
    for dataset, _chrom in by_group:
        if dataset not in dataset_metrics:
            dataset_metrics[dataset] = ValImageMetrics().to(device)
            dataset_metrics[dataset].reset()
    viz_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for idx in tqdm(indices, desc=f"infer[{args.split}]"):
        idx = int(idx)
        sample = ds[idx]
        mask = sample["mask"].unsqueeze(0).to(device)
        masked = sample["masked"].unsqueeze(0).to(device)
        gt = sample["gt"].unsqueeze(0).to(device)
        dataset, chrom = _labels_from_path(ds.image_paths[idx])
        key = (dataset, chrom)

        pred = schedule.inpaint_onestep(
            model,
            mask=mask,
            masked=masked,
            t_value=infer_t,
            prediction=prediction,
        )
        overall_metrics.update(pred, gt, mask)
        if key in group_metrics:
            group_metrics[key].update(pred, gt, mask)
            group_counts[key] += 1
        if dataset in dataset_metrics:
            dataset_metrics[dataset].update(pred, gt, mask)
            dataset_counts[dataset] += 1

        gt_np = gt[0, 0].detach().cpu().numpy()
        masked_np = masked[0, 0].detach().cpu().numpy()
        pred_np = pred[0, 0].detach().cpu().numpy()

        if args.save_npy:
            np.save(os.path.join(matrix_dir, f"pred_{idx:05d}.npy"), pred_np)
            np.save(os.path.join(matrix_dir, f"gt_{idx:05d}.npy"), gt_np)
            np.save(os.path.join(matrix_dir, f"masked_{idx:05d}.npy"), masked_np)

        if idx in viz_index_set:
            viz_cache[idx] = (gt_np, masked_np, pred_np)

    if os.path.isfile(metrics_csv):
        os.remove(metrics_csv)

    def _log_metrics(label: str, count: int, metrics: Dict[str, float]) -> None:
        write_log(
            f"{label} n={count} "
            f"psnr={metrics.get('psnr', float('nan')):.4f} "
            f"psnr_masked={metrics.get('psnr_masked', float('nan')):.4f} "
            f"ssim={metrics.get('ssim', float('nan')):.4f} "
            f"ssim_masked={metrics.get('ssim_masked', float('nan')):.4f} "
            f"fid={metrics.get('fid', float('nan')):.4f} "
            f"fid_masked={metrics.get('fid_masked', float('nan')):.4f}",
            log_path,
        )

    for key in by_group:
        dataset, chrom = key
        metrics = group_metrics[key].compute()
        append_csv_row(
            metrics_csv,
            _metrics_row(dataset, chrom, group_counts[key], metrics),
            METRIC_FIELDS,
        )
        _log_metrics(f"{dataset}/{chrom}", group_counts[key], metrics)

    for dataset in sorted(dataset_metrics.keys()):
        metrics = dataset_metrics[dataset].compute()
        append_csv_row(
            metrics_csv,
            _metrics_row(dataset, "all", dataset_counts[dataset], metrics),
            METRIC_FIELDS,
        )
        _log_metrics(f"{dataset}/all", dataset_counts[dataset], metrics)

    all_metrics = overall_metrics.compute()
    append_csv_row(
        metrics_csv,
        _metrics_row("all", "all", n, all_metrics),
        METRIC_FIELDS,
    )
    _log_metrics("all/all", n, all_metrics)
    write_log(f"metrics -> {metrics_csv}", log_path)

    viz_stem = os.path.splitext(args.viz_name)[0] or "inference_grid"
    for key, vidxs in viz_by_group.items():
        dataset, chrom = key
        for idx in vidxs:
            if idx not in viz_cache:
                continue
            gt_np, masked_np, pred_np = viz_cache[idx]
            out_fig = os.path.join(
                viz_dir, f"{viz_stem}_{dataset}_{chrom}_idx{idx:05d}.png"
            )
            _save_sample_viz(
                dataset=dataset,
                chrom=chrom,
                idx=idx,
                gt=gt_np,
                masked=masked_np,
                pred=pred_np,
                out_path=out_fig,
                split=args.split,
                prediction=args.prediction,
                infer_t=args.infer_t,
            )
            write_log(f"saved viz -> {out_fig}", log_path)

    if args.save_npy:
        write_log(f"saved matrices -> {matrix_dir}", log_path)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()
