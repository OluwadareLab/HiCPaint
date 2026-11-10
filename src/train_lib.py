from __future__ import annotations

import argparse
import contextlib
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.configs import DEFAULT_CONFIG_PATH, config_to_train_defaults, load_config
from src.data_loader.load_data import MASK_PLACEMENTS, CustomDataset, ImageDataset
from src.loss import (
    AdversarialLoss,
    distance_weighted_loss,
    gradient_loss,
    insulation_loss,
    masked_charbonnier_loss,
    masked_l1_loss,
    masked_mse_loss,
    masked_ssim_loss,
    r1_penalty,
    spectral_loss,
    stratum_corr_loss,
)
from src.metric import ValImageMetrics, balanced_pick, masked_selection_score
from src.model import DiffusionTransformer, PatchDiscriminator
from src.utils import (
    DiffusionSchedule,
    ModelEma,
    append_csv_row,
    barrier,
    build_warmup_cosine_scheduler,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    plot_train_val_loss,
    select_fixed_indices,
    unwrap_model,
    visualize_gt_masked_pred,
    write_log,
)


def build_dataloaders(
    record_prefix: str,
    img_dir: str,
    batch_size: int,
    image_size: int = 256,
    mask_size: int = 64,
    mask_placement: str = "mixed",
    num_workers: int = 4,
    seed: int = 42,
    subset_fraction: float = 1.0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[Dict[str, DataLoader], Dict[str, Optional[DistributedSampler]]]:
    """Build train / val / test loaders; optional DistributedSampler on train."""
    loaders: Dict[str, DataLoader] = {}
    samplers: Dict[str, Optional[DistributedSampler]] = {}
    for split, shuffle, deterministic in (
        ("train", True, False),
        ("val", False, True),
        ("test", False, True),
    ):
        record_file = f"{record_prefix}.{split}"
        if not os.path.isfile(record_file):
            raise FileNotFoundError(f"Missing record file: {record_file}")
        # Different seed offset per split so subsets don't share the same indices.
        split_seed = int(seed) + {"train": 0, "val": 1, "test": 2}[split]
        dataset: ImageDataset = CustomDataset(
            record_file=record_file,
            img_dir=img_dir,
        ).get_dataset(
            mask_size=mask_size,
            mask_placement=mask_placement,
            seed=split_seed,
            deterministic_masks=deterministic,
            image_size=image_size,
            subset_fraction=subset_fraction,
        )
        sampler: Optional[DistributedSampler] = None
        if distributed and split == "train":
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        samplers[split] = sampler
    return loaders, samplers


def _prepare_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gt = batch["gt"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    masked = batch["masked"].to(device, non_blocking=True)
    structure = torch.cat(
        [
            batch["edge"].to(device, non_blocking=True),
            batch["line"].to(device, non_blocking=True),
            batch["dot"].to(device, non_blocking=True),
        ],
        dim=1,
    )
    return gt, mask, masked, structure


def _model_input(x_t: torch.Tensor, mask: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Blind DiT input: noisy hole map + mask + known pixels (no clean gt)."""
    return torch.cat([x_t, mask, masked], dim=1)


def _pred_channels(model_out: torch.Tensor, gt_channels: int = 1) -> torch.Tensor:
    return model_out[:, :gt_channels]


def _compose(mask: torch.Tensor, masked: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return ((1.0 - mask) * masked + mask * x0).clamp(0.0, 1.0)


def _disc_input(
    image: torch.Tensor,
    mask: torch.Tensor,
    masked: torch.Tensor,
) -> torch.Tensor:
    """Conditional PatchGAN input: judge the map against its mask and known context."""
    return torch.cat([image, mask, masked], dim=1)


def _amp_dtype(use_amp: bool, device: torch.device) -> Optional[torch.dtype]:
    """bf16 when the device supports it; ``None`` disables autocast (fp32)."""
    if not use_amp or device.type != "cuda":
        return None
    if not torch.cuda.is_bf16_supported():
        return None
    return torch.bfloat16


def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Decay matmul/conv weights only; norms, biases and gains stay undecayed."""
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@dataclass
class LossConfig:
    """Loss weights and ``t``-gating fractions for one training step."""

    charbonnier_weight: float = 1.0
    charbonnier_eps: float = 1.0e-3
    x0_l1_weight: float = 0.0
    x0_l1_t_max_frac: float = 0.3
    ssim_weight: float = 0.1
    ssim_t_max_frac: float = 1.0
    strat_weight: float = 0.5
    insulation_weight: float = 0.25
    insulation_window: int = 8
    grad_weight: float = 0.05
    dist_weight: float = 0.0
    spectral_weight: float = 0.0
    struct_t_max_frac: float = 1.0
    adv_weight: float = 0.05
    adv_t_max_frac: float = 1.0
    min_snr_gamma: float = 0.0
    high_t_bias: float = 0.0
    zero_hole_prob: float = 0.0
    d_r1_gamma: float = 10.0
    d_r1_every: int = 16


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    image_metrics: Optional[ValImageMetrics] = None,
    prediction: str = "x0",
    infer_t: int = -1,
) -> Dict[str, float]:
    """Val/test: train-matched hole loss + one-step image metrics."""
    model.eval()
    raw = unwrap_model(model)
    if image_metrics is not None:
        image_metrics.reset()

    total = 0.0
    total_infer = 0.0
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        gt, mask, masked, structure = _prepare_batch(batch, device)
        b = gt.shape[0]
        # Deterministic low-discrepancy t (golden-ratio stride) for epoch-comparable val_loss.
        n_t = schedule.num_timesteps
        stride = max(1, round(0.6180339887 * n_t))
        while math.gcd(stride, n_t) != 1:
            stride += 1
        t = ((torch.arange(b, device=device) + count) * stride) % n_t
        x_t, noise = schedule.blind_q_sample(gt, mask, masked, t)
        y = torch.zeros(b, dtype=torch.long, device=device)
        out = raw(_model_input(x_t, mask, masked), t, y, structure=structure)
        if prediction == "x0":
            x0_pred = _pred_channels(out)
            loss = masked_mse_loss(x0_pred, gt, mask)
        else:
            eps = _pred_channels(out)
            loss = masked_mse_loss(eps, noise, mask)
            x0_pred = schedule.predict_x0_from_eps(x_t, t, eps)
        total += loss.item() * b
        count += b
        if image_metrics is not None:
            pred = schedule.inpaint_onestep(
                raw,
                mask,
                masked,
                y=y,
                t_value=infer_t,
                prediction=prediction,
                structure=structure,
            )
            image_metrics.update(pred, gt, mask)
            total_infer += masked_mse_loss(pred, gt, mask).item() * b

    metrics: Dict[str, float] = {"val_loss": total / max(1, count)}
    if image_metrics is not None:
        metrics["val_loss_infer"] = total_infer / max(1, count)
    if image_metrics is not None:
        metrics.update(image_metrics.compute())
    return metrics


def train_one_epoch(
    model: nn.Module,
    disc: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    adv_loss: AdversarialLoss,
    device: torch.device,
    epoch: int,
    grad_clip: float,
    loss_cfg: LossConfig,
    use_adv: bool = False,
    use_ssim: bool = False,
    use_struct: bool = False,
    sampler: Optional[DistributedSampler] = None,
    prediction: str = "x0",
    ema: Optional[ModelEma] = None,
    amp_dtype: Optional[torch.dtype] = None,
    global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    """One training epoch with blind inpainting (noise only in the hole)."""
    model.train()
    disc.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)

    cfg = loss_cfg
    use_adv = bool(use_adv) and cfg.adv_weight > 0
    use_ssim = bool(use_ssim) and cfg.ssim_weight > 0
    use_struct = bool(use_struct)
    use_x0_l1 = cfg.x0_l1_weight > 0
    t_max_adv = int(cfg.adv_t_max_frac * schedule.num_timesteps)
    t_max_ssim = int(cfg.ssim_t_max_frac * schedule.num_timesteps)
    t_max_struct = int(cfg.struct_t_max_frac * schedule.num_timesteps)
    t_max_x0_l1 = int(cfg.x0_l1_t_max_frac * schedule.num_timesteps)

    sums: Dict[str, float] = {
        k: 0.0
        for k in (
            "mse",
            "base",
            "x0_l1",
            "ssim",
            "strat",
            "insul",
            "grad",
            "dist",
            "spec",
            "adv_g",
            "adv_d",
            "total",
        )
    }
    n_batches = 0

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch in pbar:
        gt, mask, masked, structure = _prepare_batch(batch, device)
        b = gt.shape[0]
        t = schedule.sample_timesteps(b, device, high_t_bias=cfg.high_t_bias)
        x_t, noise = schedule.blind_q_sample(gt, mask, masked, t)
        if prediction == "x0":
            x_t, t = schedule.apply_zero_hole(x_t, t, mask, masked, cfg.zero_hole_prob)
        y = torch.zeros(b, dtype=torch.long, device=device)
        snr_w = schedule.min_snr_weights(t, cfg.min_snr_gamma, prediction)
        adv_w = (t <= t_max_adv).float() if use_adv else None
        ssim_w = (t <= t_max_ssim).float() if use_ssim else None
        struct_w = (t <= t_max_struct).float() if use_struct else None
        x0_l1_w = (t <= t_max_x0_l1).float() if use_x0_l1 else None

        optimizer_g.zero_grad(set_to_none=True)
        with _autocast(device, amp_dtype):
            out = model(_model_input(x_t, mask, masked), t, y, structure=structure)
        # Losses stay fp32: SSIM/stratum/FFT terms are numerically touchy in bf16.
        out = out.float()
        if prediction == "x0":
            x0_pred = _pred_channels(out)
            base_pred, base_target = x0_pred, gt
        else:
            eps = _pred_channels(out)
            base_pred, base_target = eps, noise
            x0_pred = schedule.predict_x0_from_eps(x_t, t, eps)
        # Reported unweighted so the *_mse_threshold unlock gates stay comparable.
        loss_mse = masked_mse_loss(base_pred, base_target, mask)
        loss_base = masked_charbonnier_loss(
            base_pred,
            base_target,
            mask,
            eps=cfg.charbonnier_eps,
            sample_weight=snr_w,
        )
        pred = _compose(mask, masked, x0_pred)

        zero = pred.new_zeros(())
        loss_x0_l1 = (
            masked_l1_loss(pred, gt, mask, sample_weight=x0_l1_w) if use_x0_l1 else zero
        )
        loss_ssim = (
            masked_ssim_loss(pred, gt, mask, sample_weight=ssim_w) if use_ssim else zero
        )

        loss_strat = zero
        loss_insul = zero
        loss_grad = zero
        loss_dist = zero
        loss_spec = zero
        if use_struct:
            if cfg.strat_weight > 0:
                loss_strat = stratum_corr_loss(pred, gt, mask, sample_weight=struct_w)
            if cfg.insulation_weight > 0:
                loss_insul = insulation_loss(
                    pred,
                    gt,
                    mask,
                    window=cfg.insulation_window,
                    sample_weight=struct_w,
                )
            if cfg.grad_weight > 0:
                loss_grad = gradient_loss(pred, gt, mask, sample_weight=struct_w)
            if cfg.dist_weight > 0:
                loss_dist = distance_weighted_loss(
                    pred, gt, mask, sample_weight=struct_w
                )
            if cfg.spectral_weight > 0:
                loss_spec = spectral_loss(pred, gt, mask, sample_weight=struct_w)

        if use_adv:
            fake_logits = unwrap_model(disc)(_disc_input(pred, mask, masked))
            loss_adv_g = adv_loss.g_loss(fake_logits, sample_weight=adv_w)
        else:
            loss_adv_g = zero

        loss_g = (
            cfg.charbonnier_weight * loss_base
            + cfg.x0_l1_weight * loss_x0_l1
            + cfg.ssim_weight * loss_ssim
            + cfg.strat_weight * loss_strat
            + cfg.insulation_weight * loss_insul
            + cfg.grad_weight * loss_grad
            + cfg.dist_weight * loss_dist
            + cfg.spectral_weight * loss_spec
            + cfg.adv_weight * loss_adv_g
        )
        loss_g.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer_g.step()
        if ema is not None:
            ema.update(model)

        if use_adv:
            optimizer_d.zero_grad(set_to_none=True)
            raw_d = unwrap_model(disc)
            real_in = _disc_input(gt, mask, masked)
            do_r1 = cfg.d_r1_gamma > 0 and (global_step % max(1, cfg.d_r1_every) == 0)
            if do_r1:
                real_in = real_in.detach().requires_grad_(True)
            real_logits = raw_d(real_in)
            fake_logits = raw_d(_disc_input(pred.detach(), mask, masked))
            loss_d = adv_loss.d_loss(real_logits, fake_logits, sample_weight=adv_w)
            if do_r1:
                loss_d = loss_d + 0.5 * cfg.d_r1_gamma * r1_penalty(
                    real_logits, real_in
                )
            loss_d.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
            optimizer_d.step()
        else:
            loss_d = zero

        global_step += 1
        batch_losses = {
            "mse": loss_mse,
            "base": loss_base,
            "x0_l1": loss_x0_l1,
            "ssim": loss_ssim,
            "strat": loss_strat,
            "insul": loss_insul,
            "grad": loss_grad,
            "dist": loss_dist,
            "spec": loss_spec,
            "adv_g": loss_adv_g,
            "adv_d": loss_d,
            "total": loss_g,
        }
        for key, value in batch_losses.items():
            sums[key] += float(value.item())
        n_batches += 1
        pbar.set_postfix(
            mse=f"{float(loss_mse.item()):.4f}",
            base=f"{float(loss_base.item()):.4f}",
            ssim=f"{float(loss_ssim.item()):.4f}",
            strat=f"{float(loss_strat.item()):.4f}",
            adv_g=f"{float(loss_adv_g.item()):.4f}",
        )

    denom = max(1, n_batches)
    metrics = {f"train_{k}": v / denom for k, v in sums.items()}
    metrics["train_loss"] = metrics.pop("train_total")
    return metrics, global_step


def save_checkpoint(
    path: str,
    model: nn.Module,
    disc: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_score: float,
    metrics: Optional[Dict[str, float]] = None,
    adv_unlocked: bool = False,
    ssim_unlocked: bool = False,
    struct_unlocked: bool = False,
    ema: Optional[ModelEma] = None,
    global_step: int = 0,
    sel_history: Optional[List[Dict[str, float]]] = None,
    best_epoch: Optional[int] = None,
) -> None:
    """Save full training checkpoint (unwraps DDP)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "best_val": best_score,  # alias for older resume paths
            "adv_unlocked": bool(adv_unlocked),
            "ssim_unlocked": bool(ssim_unlocked),
            "struct_unlocked": bool(struct_unlocked),
            "global_step": int(global_step),
            "sel_history": list(sel_history or []),
            "best_epoch": best_epoch,
            "ema": ema.state_dict() if ema is not None else None,
            "model": unwrap_model(model).state_dict(),
            "disc": unwrap_model(disc).state_dict(),
            "optimizer": optimizer_g.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics or {},
        },
        path,
    )


def save_best_weights(
    path: str,
    model: nn.Module,
    epoch: int,
    score: float,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save best DiT weights (criteria: masked PSNR/SSIM/FID selection score).

    ``model`` is the EMA copy when EMA is enabled, so inference loads EMA weights
    from the usual ``model`` key without any extra handling.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    m = metrics or {}
    torch.save(
        {
            "epoch": epoch,
            "best_score": score,
            "val_loss": float(m.get("val_loss", float("nan"))),
            "psnr_masked": float(m.get("psnr_masked", float("nan"))),
            "ssim_masked": float(m.get("ssim_masked", float("nan"))),
            "fid_masked": float(m.get("fid_masked", float("nan"))),
            "model": unwrap_model(model).state_dict(),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    distributed = bool(args.distributed)
    if distributed:
        enabled, rank, local_rank, world_size = init_distributed()
        distributed = enabled
    else:
        rank, local_rank, world_size = 0, 0, 1

    main = is_main_process(rank)
    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(
            args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
        )

    if main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "viz"), exist_ok=True)

    viz_dir = os.path.join(args.output_dir, "viz")
    log_path = args.log_path or os.path.join(args.output_dir, "train.log")
    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    pareto_dir = os.path.join(args.checkpoint_dir, "pareto")
    best_selection = str(args.best_selection).strip().lower()
    if best_selection not in ("balanced", "weighted"):
        raise ValueError(
            f"best_selection must be 'balanced' or 'weighted', got {best_selection!r}"
        )
    val_csv = (
        args.val_metrics_csv
        if isinstance(args.val_metrics_csv, str) and args.val_metrics_csv
        else os.path.join(args.output_dir, "val_metrics.csv")
    )
    loss_csv = (
        args.loss_csv
        if isinstance(args.loss_csv, str) and args.loss_csv
        else os.path.join(args.output_dir, "train_val_loss.csv")
    )
    loss_plot = (
        args.loss_plot
        if isinstance(args.loss_plot, str) and args.loss_plot
        else os.path.join(args.output_dir, "train_val_loss_plot.png")
    )

    val_fields = [
        "epoch",
        "val_loss",
        "psnr",
        "psnr_masked",
        "ssim",
        "ssim_masked",
        "fid",
        "fid_masked",
        "val_loss_infer",
    ]
    loss_fields = ["epoch", "train_mse", "val_mse"]

    if main:
        write_log(
            f"device={device} distributed={distributed} world_size={world_size} blind_inpaint=True",
            log_path,
        )

    loaders, samplers = build_dataloaders(
        record_prefix=args.record_prefix,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mask_size=args.mask_size,
        mask_placement=args.mask_placement,
        num_workers=args.num_workers,
        seed=args.seed,
        subset_fraction=float(args.subset_fraction),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if main:
        write_log(
            f"splits train={len(loaders['train'].dataset)} "
            f"val={len(loaders['val'].dataset)} "
            f"test={len(loaders['test'].dataset)}",
            log_path,
        )

    viz_indices = select_fixed_indices(
        n_dataset=len(loaders["val"].dataset),
        n_samples=args.num_visualization_samples,
        seed=args.seed,
    )
    if main:
        write_log(f"fixed viz indices={viz_indices}", log_path)

    drop = float(args.dropout)
    model = DiffusionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        # num_classes=1 means there is nothing to guide; no CFG label dropout.
        class_dropout_prob=0.0,
        num_classes=1,
        learn_sigma=args.learn_sigma,
        attn_drop=0.0,
        proj_drop=drop,
        ffc_blocks=args.ffc_blocks,
        ffc_ratio_global=args.ffc_ratio_global,
        ffc_downsample=args.ffc_downsample,
        stem_channels=args.stem_channels,
        mid_channels=args.mid_channels,
        mask_attn_bias=args.mask_attn_bias,
        structure_depth=args.structure_depth,
        lg_blocks=args.lg_blocks,
        lg_window=args.lg_window,
        lg_pool=args.lg_pool,
        conv_head_blocks=args.conv_head_blocks,
    ).to(device)
    # Conditional PatchGAN: [map, mask, masked].
    disc = PatchDiscriminator(in_channels=3).to(device)
    adv_loss = AdversarialLoss(
        real_label=float(args.adv_real_label),
        fake_label=float(args.adv_fake_label),
    )
    schedule = DiffusionSchedule(num_timesteps=args.num_timesteps).to(device)
    image_metrics = ValImageMetrics().to(device) if main else None
    prediction = str(args.prediction).strip().lower()
    if prediction not in ("x0", "eps"):
        raise ValueError(f"prediction must be 'x0' or 'eps', got {prediction!r}")
    infer_t = int(args.infer_t)

    loss_cfg = LossConfig(
        charbonnier_weight=float(args.charbonnier_weight),
        charbonnier_eps=float(args.charbonnier_eps),
        x0_l1_weight=float(args.x0_l1_weight),
        x0_l1_t_max_frac=float(args.x0_l1_t_max_frac),
        ssim_weight=float(args.ssim_weight),
        ssim_t_max_frac=float(args.ssim_t_max_frac),
        strat_weight=float(args.strat_weight),
        insulation_weight=float(args.insulation_weight),
        insulation_window=int(args.insulation_window),
        grad_weight=float(args.grad_weight),
        dist_weight=float(args.dist_weight),
        spectral_weight=float(args.spectral_weight),
        struct_t_max_frac=float(args.struct_t_max_frac),
        adv_weight=float(args.adv_weight),
        adv_t_max_frac=float(args.adv_t_max_frac),
        min_snr_gamma=float(args.min_snr_gamma),
        high_t_bias=float(args.high_t_bias),
        zero_hole_prob=float(args.zero_hole_prob),
        d_r1_gamma=float(args.d_r1_gamma),
        d_r1_every=int(args.d_r1_every),
    )
    amp_dtype = _amp_dtype(bool(args.use_amp), device)

    n_params = sum(p.numel() for p in model.parameters())
    if main:
        write_log(
            f"params={n_params / 1e6:.2f}M img_size={args.image_size} patch_size={args.patch_size} "
            f"depth={args.depth} structure_depth={args.structure_depth} "
            f"ffc_blocks={args.ffc_blocks} ffc_ratio_global={args.ffc_ratio_global} "
            f"ffc_downsample={args.ffc_downsample} "
            f"lg_blocks={args.lg_blocks} lg_window={args.lg_window} lg_pool={args.lg_pool} "
            f"stem_ch={args.stem_channels} mid_ch={args.mid_channels} "
            f"conv_head_blocks={args.conv_head_blocks} mask_placement={args.mask_placement} "
            f"mask_attn_bias={args.mask_attn_bias} learn_sigma={int(args.learn_sigma)} "
            f"prediction={prediction} infer_t={infer_t} "
            f"amp={amp_dtype if amp_dtype is not None else 'fp32'} "
            f"ema_decay={args.ema_decay} min_snr_gamma={args.min_snr_gamma} "
            f"high_t_bias={args.high_t_bias} zero_hole_prob={args.zero_hole_prob} "
            f"charbonnier_weight={args.charbonnier_weight} "
            f"strat_weight={args.strat_weight} "
            f"insulation_weight={args.insulation_weight} "
            f"grad_weight={args.grad_weight} dist_weight={args.dist_weight} "
            f"spectral_weight={args.spectral_weight} "
            f"struct_t_max_frac={args.struct_t_max_frac} "
            f"adv_weight={args.adv_weight} d_r1_gamma={args.d_r1_gamma} "
            f"d_r1_every={args.d_r1_every} "
            f"adv_mse_threshold={args.adv_mse_threshold} ssim_weight={args.ssim_weight} "
            f"ssim_mse_threshold={args.ssim_mse_threshold} "
            f"x0_l1_weight={args.x0_l1_weight} x0_l1_t_max_frac={args.x0_l1_t_max_frac} "
            f"best_selection={best_selection} best_min_epoch={args.best_min_epoch} "
            f"weighted_score=w_psnr*{args.best_psnr_weight}*(psnr/40)"
            f"+w_ssim*{args.best_ssim_weight}*(ssim/1)"
            f"-w_fid*{args.best_fid_weight}*(fid/350) "
            f"dropout={drop} lr={args.lr} weight_decay={args.weight_decay}",
            log_path,
        )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        disc = DDP(
            disc,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    ema = (
        ModelEma(model, decay=float(args.ema_decay)).to(device)
        if float(args.ema_decay) > 0.0
        else None
    )
    # EMA weights are what inference loads, so validate and select on them.
    eval_model: nn.Module = ema.module if ema is not None else model

    optimizer_g = torch.optim.AdamW(
        param_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    optimizer_d = torch.optim.AdamW(
        param_groups(disc, args.weight_decay),
        lr=float(args.d_lr),
        betas=(0.9, 0.95),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer_g,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr,
        base_lr=args.lr,
    )

    start_epoch = 1
    best_score = float("-inf")
    epochs_no_improve = 0
    ssim_unlocked = False
    adv_unlocked = False
    struct_unlocked = False
    global_step = 0
    sel_history: List[Dict[str, float]] = []
    best_epoch: Optional[int] = None
    hist_epochs: list[int] = []
    hist_train: list[float] = []
    hist_val: list[float] = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model"])
        if "disc" in ckpt:
            unwrap_model(disc).load_state_dict(ckpt["disc"])
        opt_g = ckpt.get("optimizer_g", ckpt.get("optimizer"))
        if opt_g is not None:
            optimizer_g.load_state_dict(opt_g)
        if ckpt.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "best_score" in ckpt:
            best_score = float(ckpt["best_score"])
        elif "best_val" in ckpt:
            # Older ckpts stored min val_loss; force re-compete on masked score.
            best_score = float("-inf")
        ssim_unlocked = bool(ckpt.get("ssim_unlocked", False))
        adv_unlocked = bool(ckpt.get("adv_unlocked", False))
        struct_unlocked = bool(ckpt.get("struct_unlocked", ssim_unlocked))
        global_step = int(ckpt.get("global_step", 0))
        sel_history = list(ckpt.get("sel_history") or [])
        best_epoch = ckpt.get("best_epoch")
        if ema is not None and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        if main:
            write_log(
                f"resumed from {args.resume} epoch={start_epoch - 1} "
                f"best_score={best_score:.6f} best_epoch={best_epoch} "
                f"sel_history={len(sel_history)} ssim_unlocked={ssim_unlocked} "
                f"adv_unlocked={adv_unlocked} struct_unlocked={struct_unlocked} "
                f"global_step={global_step}",
                log_path,
            )

    if main:
        if not ssim_unlocked:
            write_log(
                f"ssim + hi-c structure losses locked "
                f"(unlock when train_mse <= {args.ssim_mse_threshold})",
                log_path,
            )
        if not adv_unlocked:
            write_log(
                f"adv locked (unlock when train_mse <= {args.adv_mse_threshold})",
                log_path,
            )

    for epoch in range(start_epoch, args.epochs + 1):
        if main:
            write_log(f"{epoch}/{args.epochs}", log_path)
        metrics, global_step = train_one_epoch(
            model=model,
            disc=disc,
            loader=loaders["train"],
            schedule=schedule,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            adv_loss=adv_loss,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            loss_cfg=loss_cfg,
            use_adv=adv_unlocked,
            use_ssim=ssim_unlocked,
            use_struct=struct_unlocked,
            sampler=samplers["train"],
            prediction=prediction,
            ema=ema,
            amp_dtype=amp_dtype,
            global_step=global_step,
        )

        mean_train_mse = float(metrics["train_mse"])
        if not ssim_unlocked and mean_train_mse <= float(args.ssim_mse_threshold):
            ssim_unlocked = True
            struct_unlocked = True
            if main:
                write_log(
                    f"ssim + hi-c structure losses unlocked "
                    f"(train_mse={mean_train_mse:.6f} <= {args.ssim_mse_threshold})",
                    log_path,
                )
        if not adv_unlocked and mean_train_mse <= float(args.adv_mse_threshold):
            adv_unlocked = True
            if main:
                write_log(
                    f"adv unlocked (train_mse={mean_train_mse:.6f} <= "
                    f"{args.adv_mse_threshold})",
                    log_path,
                )

        if distributed:
            flags = torch.tensor(
                [
                    1 if ssim_unlocked else 0,
                    1 if adv_unlocked else 0,
                    1 if struct_unlocked else 0,
                ],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(flags, src=0)
            ssim_unlocked = bool(flags[0].item())
            adv_unlocked = bool(flags[1].item())
            struct_unlocked = bool(flags[2].item())

        barrier()
        if main:
            val_metrics = evaluate(
                eval_model,
                loaders["val"],
                schedule,
                device,
                image_metrics=image_metrics,
                prediction=prediction,
                infer_t=infer_t,
            )
            val_loss = float(val_metrics["val_loss"])
            metrics.update(val_metrics)
            lr = optimizer_g.param_groups[0]["lr"]
            metrics["lr"] = lr
            sel_score = masked_selection_score(
                val_metrics,
                w_psnr=float(args.best_psnr_weight),
                w_ssim=float(args.best_ssim_weight),
                w_fid=float(args.best_fid_weight),
            )
            metrics["sel_score"] = sel_score

            def _fmt4(key: str, src: Dict[str, float]) -> str:
                if key not in src:
                    return ""
                return f"{float(src[key]):.4f}"

            append_csv_row(
                val_csv,
                {
                    "epoch": epoch,
                    "val_loss": f"{val_loss:.4f}",
                    "psnr": _fmt4("psnr", val_metrics),
                    "psnr_masked": _fmt4("psnr_masked", val_metrics),
                    "ssim": _fmt4("ssim", val_metrics),
                    "ssim_masked": _fmt4("ssim_masked", val_metrics),
                    "fid": _fmt4("fid", val_metrics),
                    "fid_masked": _fmt4("fid_masked", val_metrics),
                    "val_loss_infer": _fmt4("val_loss_infer", val_metrics),
                },
                val_fields,
            )
            append_csv_row(
                loss_csv,
                {
                    "epoch": epoch,
                    "train_mse": f"{float(metrics['train_mse']):.4f}",
                    "val_mse": f"{val_loss:.4f}",
                },
                loss_fields,
            )
            hist_epochs.append(epoch)
            hist_train.append(float(metrics["train_mse"]))
            hist_val.append(val_loss)
            plot_train_val_loss(hist_epochs, hist_train, hist_val, loss_plot)

            write_log(
                f"epoch={epoch} train_loss={metrics['train_loss']:.6f} "
                f"train_mse={metrics['train_mse']:.6f} "
                f"train_base={metrics.get('train_base', 0.0):.6f} "
                f"train_x0_l1={metrics.get('train_x0_l1', 0.0):.6f} "
                f"train_ssim={metrics.get('train_ssim', 0.0):.6f} "
                f"train_strat={metrics.get('train_strat', 0.0):.6f} "
                f"train_insul={metrics.get('train_insul', 0.0):.6f} "
                f"train_grad={metrics.get('train_grad', 0.0):.6f} "
                f"train_dist={metrics.get('train_dist', 0.0):.6f} "
                f"train_spec={metrics.get('train_spec', 0.0):.6f} "
                f"train_adv_g={metrics.get('train_adv_g', 0.0):.6f} "
                f"train_adv_d={metrics.get('train_adv_d', 0.0):.6f} "
                f"ssim_unlocked={int(ssim_unlocked)} "
                f"struct_unlocked={int(struct_unlocked)} "
                f"adv_unlocked={int(adv_unlocked)} "
                f"val_loss={val_loss:.6f} "
                f"val_loss_infer={val_metrics.get('val_loss_infer', float('nan')):.6f} "
                f"psnr={val_metrics.get('psnr', float('nan')):.4f} "
                f"psnr_masked={val_metrics.get('psnr_masked', float('nan')):.4f} "
                f"ssim={val_metrics.get('ssim', float('nan')):.4f} "
                f"ssim_masked={val_metrics.get('ssim_masked', float('nan')):.4f} "
                f"fid={val_metrics.get('fid', float('nan')):.4f} "
                f"fid_masked={val_metrics.get('fid_masked', float('nan')):.4f} "
                f"sel_score={sel_score:.4f} "
                f"lr={lr:.6e}",
                log_path,
            )

            viz_current = False
            if best_selection == "balanced":
                sel_history.append(
                    {
                        "epoch": int(epoch),
                        "psnr": float(val_metrics.get("psnr_masked", float("nan"))),
                        "ssim": float(val_metrics.get("ssim_masked", float("nan"))),
                        "fid": float(val_metrics.get("fid_masked", float("nan"))),
                    }
                )
                new_best, best_norm, front = balanced_pick(
                    sel_history, min_epoch=int(args.best_min_epoch)
                )
                os.makedirs(pareto_dir, exist_ok=True)

                def _pareto_file(e: int) -> str:
                    return os.path.join(pareto_dir, f"epoch_{e:04d}.pt")

                if epoch in front:
                    save_best_weights(
                        _pareto_file(epoch), eval_model, epoch, sel_score, val_metrics
                    )
                for fname in os.listdir(pareto_dir):
                    if not (fname.startswith("epoch_") and fname.endswith(".pt")):
                        continue
                    try:
                        e = int(fname[len("epoch_") : -len(".pt")])
                    except ValueError:
                        continue
                    if e not in front:
                        os.remove(os.path.join(pareto_dir, fname))

                improved = (
                    new_best is not None
                    and new_best != best_epoch
                    and os.path.isfile(_pareto_file(new_best))
                )
                if improved:
                    best_epoch = new_best
                    best_score = min(best_norm.values())
                    shutil.copyfile(_pareto_file(best_epoch), best_path)
                    viz_current = best_epoch == epoch
                elif best_norm:
                    best_score = min(best_norm.values())
                norm_str = ",".join(f"{k}={v:.3f}" for k, v in best_norm.items())
                write_log(
                    f"pareto={front} best_epoch={best_epoch} norm({norm_str})",
                    log_path,
                )
            else:
                improved = sel_score > best_score
                if improved:
                    best_score = sel_score
                    best_epoch = epoch
                    save_best_weights(best_path, eval_model, epoch, best_score, val_metrics)
                    viz_current = True

            if improved:
                epochs_no_improve = 0
                viz_msg = ""
                if viz_current:
                    viz_path = os.path.join(viz_dir, f"best_epoch_{epoch:04d}.png")
                    visualize_gt_masked_pred(
                        model=unwrap_model(eval_model),
                        dataset=loaders["val"].dataset,
                        indices=viz_indices,
                        schedule=schedule,
                        device=device,
                        out_path=viz_path,
                        epoch=epoch,
                        prediction=prediction,
                        infer_t=infer_t,
                    )
                    viz_msg = f"; viz -> {viz_path}"
                best_row = next(
                    (h for h in sel_history if int(h["epoch"]) == best_epoch), None
                )
                if best_row is None:
                    best_row = {
                        "psnr": val_metrics.get("psnr_masked", float("nan")),
                        "ssim": val_metrics.get("ssim_masked", float("nan")),
                        "fid": val_metrics.get("fid_masked", float("nan")),
                    }
                write_log(
                    f"new best epoch={best_epoch} score={best_score:.6f} "
                    f"(psnr_m={float(best_row['psnr']):.4f} "
                    f"ssim_m={float(best_row['ssim']):.4f} "
                    f"fid_m={float(best_row['fid']):.4f}) "
                    f"-> {best_path}{viz_msg}",
                    log_path,
                )
            else:
                epochs_no_improve += 1

            if epoch % args.save_every == 0 or improved or epoch == args.epochs:
                save_checkpoint(
                    ckpt_path,
                    model,
                    disc,
                    optimizer_g,
                    optimizer_d,
                    scheduler,
                    epoch,
                    best_score,
                    metrics,
                    adv_unlocked=adv_unlocked,
                    ssim_unlocked=ssim_unlocked,
                    struct_unlocked=struct_unlocked,
                    ema=ema,
                    global_step=global_step,
                    sel_history=sel_history,
                    best_epoch=best_epoch,
                )
                write_log(f"checkpoint saved -> {ckpt_path}", log_path)

            stop = args.patience > 0 and epochs_no_improve >= args.patience
            if stop:
                write_log(
                    f"early stop at epoch={epoch} (patience={args.patience})",
                    log_path,
                )
        else:
            stop = False
            improved = False

        scheduler.step()
        barrier()

        # broadcast early-stop decision
        if distributed:
            flag = torch.tensor(
                [1 if (main and stop) else 0], device=device, dtype=torch.int32
            )
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            break

    if main:
        test_metrics = evaluate(
            eval_model,
            loaders["test"],
            schedule,
            device,
            image_metrics=image_metrics,
            prediction=prediction,
            infer_t=infer_t,
        )
        write_log(
            f"test_loss={test_metrics['val_loss']:.6f} "
            f"test_psnr={test_metrics.get('psnr', float('nan')):.4f} "
            f"test_psnr_masked={test_metrics.get('psnr_masked', float('nan')):.4f} "
            f"test_ssim={test_metrics.get('ssim', float('nan')):.4f} "
            f"test_ssim_masked={test_metrics.get('ssim_masked', float('nan')):.4f} "
            f"test_fid={test_metrics.get('fid', float('nan')):.4f} "
            f"test_fid_masked={test_metrics.get('fid_masked', float('nan')):.4f} "
            f"best_score={best_score:.6f}",
            log_path,
        )
        append_csv_row(
            os.path.join(args.output_dir, "test_metrics.csv"),
            {
                "psnr": f"{test_metrics.get('psnr', float('nan')):.4f}",
                "psnr_masked": f"{test_metrics.get('psnr_masked', float('nan')):.4f}",
                "ssim": f"{test_metrics.get('ssim', float('nan')):.4f}",
                "ssim_masked": f"{test_metrics.get('ssim_masked', float('nan')):.4f}",
                "fid": f"{test_metrics.get('fid', float('nan')):.4f}",
                "fid_masked": f"{test_metrics.get('fid_masked', float('nan')):.4f}",
                "test_loss": f"{test_metrics['val_loss']:.4f}",
                "best_score": f"{best_score:.4f}",
            },
            [
                "psnr",
                "psnr_masked",
                "ssim",
                "ssim_masked",
                "fid",
                "fid_masked",
                "test_loss",
                "best_score",
            ],
        )

    cleanup_distributed()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre.parse_known_args(argv)
    d = config_to_train_defaults(load_config(pre_args.config))

    p = argparse.ArgumentParser(description="Train Diffusion Vision Transformer (Hi-C)")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--record-prefix", type=str, default=d["record_prefix"])
    p.add_argument("--img-dir", type=str, default=d["img_dir"])
    p.add_argument("--output-dir", type=str, default=d["output_dir"])
    p.add_argument("--checkpoint-dir", type=str, default=d["checkpoint_dir"])
    p.add_argument("--log-path", type=str, default=d["log_path"])
    p.add_argument("--device", type=str, default=d["device"])
    p.add_argument("--distributed", action="store_true", help="Enable DDP (use torchrun)")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--min-lr", type=float, default=d["min_lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--warmup-epochs", type=int, default=d["warmup_epochs"])
    p.add_argument("--grad-clip", type=float, default=d["grad_clip"])
    p.add_argument(
        "--patience",
        type=int,
        default=d["patience"],
        help="Early-stop patience; 0 disables early stopping.",
    )
    p.add_argument("--save-every", type=int, default=d["save_every"])
    p.add_argument(
        "--ema-decay",
        type=float,
        default=d["ema_decay"],
        help="EMA decay for eval/checkpoint weights; 0 disables EMA.",
    )
    p.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=d["use_amp"],
        help="bf16 autocast for the backbone (losses and FFTs stay fp32).",
    )
    p.add_argument(
        "--min-snr-gamma",
        type=float,
        default=d["min_snr_gamma"],
        help="Min-SNR-gamma loss weighting; 0 disables (uniform over t).",
    )
    p.add_argument(
        "--high-t-bias",
        type=float,
        default=d["high_t_bias"],
        help="Fraction of samples drawn from the top decile of t (matches infer_t).",
    )
    p.add_argument(
        "--zero-hole-prob",
        type=float,
        default=d["zero_hole_prob"],
        help="Fraction of samples trained at t=T-1 with a zeroed hole (one-step infer input).",
    )
    p.add_argument(
        "--charbonnier-weight",
        type=float,
        default=d["charbonnier_weight"],
        help="Weight for the masked Charbonnier base reconstruction loss.",
    )
    p.add_argument(
        "--charbonnier-eps",
        type=float,
        default=d["charbonnier_eps"],
        help="Charbonnier epsilon (smaller = closer to L1).",
    )
    p.add_argument(
        "--strat-weight",
        type=float,
        default=d["strat_weight"],
        help="Weight for per-stratum correlation loss (differentiable HiCRep/SCC).",
    )
    p.add_argument(
        "--insulation-weight",
        type=float,
        default=d["insulation_weight"],
        help="Weight for the insulation-score (TAD boundary) loss.",
    )
    p.add_argument(
        "--insulation-window",
        type=int,
        default=d["insulation_window"],
        help="Diamond window size for the insulation profile.",
    )
    p.add_argument(
        "--grad-weight",
        type=float,
        default=d["grad_weight"],
        help="Weight for the masked Sobel gradient loss.",
    )
    p.add_argument(
        "--dist-weight",
        type=float,
        default=d["dist_weight"],
        help="Weight for distance-stratified (1/std per diagonal) L1; 0 disables.",
    )
    p.add_argument(
        "--spectral-weight",
        type=float,
        default=d["spectral_weight"],
        help="Weight for rFFT-magnitude loss on the hole crop; 0 disables.",
    )
    p.add_argument(
        "--struct-t-max-frac",
        type=float,
        default=d["struct_t_max_frac"],
        help="Apply Hi-C structure losses only for samples with t < frac * T.",
    )
    p.add_argument(
        "--d-r1-gamma",
        type=float,
        default=d["d_r1_gamma"],
        help="R1 gradient-penalty strength on real discriminator inputs; 0 disables.",
    )
    p.add_argument(
        "--d-r1-every",
        type=int,
        default=d["d_r1_every"],
        help="Apply the R1 penalty every N discriminator steps (lazy regularization).",
    )
    p.add_argument(
        "--adv-weight",
        type=float,
        default=d["adv_weight"],
        help="Weight for LSGAN generator adversarial loss on low-t x0 predictions.",
    )
    p.add_argument(
        "--adv-mse-threshold",
        type=float,
        default=d["adv_mse_threshold"],
        help="Unlock adversarial training once mean train_mse <= this value.",
    )
    p.add_argument(
        "--adv-t-max-frac",
        type=float,
        default=d["adv_t_max_frac"],
        help="Apply adversarial loss only for samples with t < frac * num_timesteps.",
    )
    p.add_argument(
        "--d-lr",
        type=float,
        default=d["d_lr"],
        help="Discriminator AdamW learning rate.",
    )
    p.add_argument(
        "--adv-real-label",
        type=float,
        default=d["adv_real_label"],
        help="Soft real label for discriminator LSGAN loss.",
    )
    p.add_argument(
        "--adv-fake-label",
        type=float,
        default=d["adv_fake_label"],
        help="Soft fake label for discriminator LSGAN loss.",
    )
    p.add_argument(
        "--ssim-weight",
        type=float,
        default=d["ssim_weight"],
        help="Weight for masked-region SSIM loss on low-t x0 predictions.",
    )
    p.add_argument(
        "--ssim-mse-threshold",
        type=float,
        default=d["ssim_mse_threshold"],
        help="Unlock SSIM loss once mean train_mse <= this value (MSE-only before).",
    )
    p.add_argument(
        "--ssim-t-max-frac",
        type=float,
        default=d["ssim_t_max_frac"],
        help="Apply SSIM loss only for samples with t < frac * num_timesteps.",
    )
    p.add_argument(
        "--x0-l1-weight",
        type=float,
        default=d["x0_l1_weight"],
        help="Weight for masked x0 L1 loss on low-t predictions (0 disables).",
    )
    p.add_argument(
        "--x0-l1-t-max-frac",
        type=float,
        default=d["x0_l1_t_max_frac"],
        help="Apply x0 L1 only for samples with t < frac * num_timesteps.",
    )
    p.add_argument(
        "--best-selection",
        type=str,
        default=d["best_selection"],
        choices=("balanced", "weighted"),
        help="balanced: Pareto front + maximin over history-normalized masked "
        "PSNR/SSIM/FID; weighted: legacy fixed-scale weighted sum.",
    )
    p.add_argument(
        "--best-min-epoch",
        type=int,
        default=d["best_min_epoch"],
        help="Epochs before this are ignored for best-model selection.",
    )
    p.add_argument(
        "--best-psnr-weight",
        type=float,
        default=d["best_psnr_weight"],
        help="Weight on (psnr_masked/40) in best-model selection score.",
    )
    p.add_argument(
        "--best-ssim-weight",
        type=float,
        default=d["best_ssim_weight"],
        help="Weight on (ssim_masked/1) in best-model selection score.",
    )
    p.add_argument(
        "--best-fid-weight",
        type=float,
        default=d["best_fid_weight"],
        help="Weight on (fid_masked/350) in best-model selection score (subtracted).",
    )
    p.add_argument("--num-timesteps", type=int, default=d["num_timesteps"])
    p.add_argument(
        "--prediction",
        type=str,
        default=d["prediction"],
        choices=("x0", "eps"),
        help="Network target: direct x0 (one-step infer) or eps.",
    )
    p.add_argument(
        "--infer-t",
        type=int,
        default=d["infer_t"],
        help="Timestep for one-step infer/viz; <0 means T-1.",
    )
    p.add_argument("--image-size", type=int, default=d["image_size"])
    p.add_argument("--mask-size", type=int, default=d["mask_size"])
    p.add_argument(
        "--mask-placement",
        choices=MASK_PLACEMENTS,
        default=d["mask_placement"],
        help="Hole position: on the main diagonal, anywhere, or 50/50 per sample.",
    )
    p.add_argument(
        "--subset-fraction",
        type=float,
        default=d["subset_fraction"],
        help="Use this fraction of each split (0 < f <= 1).",
    )
    p.add_argument("--patch-size", type=int, default=d["patch_size"])
    p.add_argument("--hidden-size", type=int, default=d["hidden_size"])
    p.add_argument("--depth", type=int, default=d["depth"], help="Main DiTBlock count.")
    p.add_argument(
        "--structure-depth",
        type=int,
        default=d["structure_depth"],
        help="Structure DiTBlock + DiTCrossBlock count each.",
    )
    p.add_argument("--num-heads", type=int, default=d["num_heads"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument(
        "--ffc-blocks",
        type=int,
        default=d["ffc_blocks"],
        help="FFC blocks on mid features; 0 disables FFC.",
    )
    p.add_argument(
        "--ffc-ratio-global",
        type=float,
        default=d["ffc_ratio_global"],
        help="Fraction of stem channels through Fourier branch (0, 1).",
    )
    p.add_argument(
        "--ffc-downsample",
        type=int,
        default=d["ffc_downsample"],
        help="Run the FFC front-end at 1/N resolution (2 is 4x cheaper FFTs).",
    )
    p.add_argument(
        "--lg-blocks",
        type=int,
        default=d["lg_blocks"],
        help="Local+global attention blocks after the DiT trunk; 0 disables.",
    )
    p.add_argument("--lg-window", type=int, default=d["lg_window"])
    p.add_argument("--lg-pool", type=int, default=d["lg_pool"])
    p.add_argument("--stem-channels", type=int, default=d["stem_channels"])
    p.add_argument("--mid-channels", type=int, default=d["mid_channels"])
    p.add_argument(
        "--conv-head-blocks",
        type=int,
        default=d["conv_head_blocks"],
        help="Residual conv blocks in the output head.",
    )
    p.add_argument("--mask-attn-bias", type=float, default=d["mask_attn_bias"])
    p.add_argument("--learn-sigma", action="store_true", default=d["learn_sigma"])
    p.add_argument("--no-learn-sigma", action="store_false", dest="learn_sigma")
    p.add_argument("--num-workers", type=int, default=d["num_workers"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--resume", type=str, default="")
    p.add_argument(
        "--num-visualization-samples",
        type=int,
        default=d["num_visualization_samples"],
    )
    p.add_argument("--val-metrics-csv", type=str, default=d["val_metrics_csv"])
    p.add_argument("--loss-csv", type=str, default=None)
    p.add_argument("--loss-plot", type=str, default=d["loss_plot"])
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
