from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class DirConfig:
    data: str
    image: str
    model_state: str
    output: str


@dataclass
class FileConfig:
    dataset_dict: str
    snapshot: str
    model: str
    test_hic_map: str
    val_metrics: str
    num_visualization_samples: int
    psnr_val_plot: str
    ssim_val_plot: str
    ms_ssim_val_plot: str
    train_val_loss_plot: str
    lr_plot: str
    log: str


@dataclass
class DataConfig:
    patch: int
    resolution: int
    batch_size: int
    mask_size: int = 64
    num_workers: int = 4
    seed: int = 42
    # Use a fraction of each split (0 < f <= 1). Combined with max_samples if set.
    subset_fraction: float = 1.0
    # Optional hard cap on samples per split; null/omit = no cap.
    max_samples: Optional[int] = None


@dataclass
class TrainingConfig:
    epochs: int
    save_every: int
    lr: float
    min_lr: float
    weight_decay: float = 0.0
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    patience: int = 0
    adv_weight: float = 0.1
    adv_mse_threshold: float = 0.4
    adv_t_max_frac: float = 0.5
    d_lr: float = 1.0e-4
    adv_real_label: float = 0.9
    adv_fake_label: float = 0.1
    ssim_weight: float = 0.1
    ssim_mse_threshold: float = 0.5
    ssim_t_max_frac: float = 0.5
    val_sample_steps: int = 20


@dataclass
class FlowConfig:
    num_of_convs: List[int]
    out_channels: List[int]


@dataclass
class ModelConfig:
    name: str
    image_size: int = 256
    patch_size: int = 16
    hidden_size: int = 768
    depth: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    ffc_blocks: int = 4
    learn_sigma: bool = False
    num_timesteps: int = 1000


@dataclass
class InferenceConfig:
    batch_size: int = 4
    num_samples: int = 8
    sample_steps: int = 50


@dataclass
class LossWeightConfig:
    name: str
    boundaries: List[int]
    values: List[float]


@dataclass
class LossConfig:
    weight_parameters: List[LossWeightConfig]


@dataclass
class EvalMetricConfig:
    psnr: bool = True
    ssim: bool = True
    ssim_masked: bool = True
    fid: bool = True
    hicrep: bool = False


@dataclass
class EvaluationConfig:
    monitor: str = "loss"
    metrics: EvalMetricConfig = field(default_factory=EvalMetricConfig)


@dataclass
class Config:
    device: str
    dir: DirConfig
    file: FileConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _from_dict(cls: type, data: Any) -> Any:
    """Recursively build a dataclass (or list of dataclasses) from a nested dict."""
    if data is None:
        return None
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(data).__name__}")
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        nested = hints.get(f.name, f.type)
        origin = get_origin(nested)
        if origin is Union:
            args = [a for a in get_args(nested) if a is not type(None)]
            nested = args[0] if len(args) == 1 else nested
            origin = get_origin(nested)
        if is_dataclass(nested) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(nested, value)
        elif origin is list:
            item_type = get_args(nested)[0] if get_args(nested) else None
            if item_type is not None and is_dataclass(item_type) and isinstance(value, list):
                kwargs[f.name] = [_from_dict(item_type, v) for v in value]
            else:
                kwargs[f.name] = value
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: Union[str, Path, None] = None) -> Config:
    """Load and resolve ``config.yaml`` (OmegaConf ``${...}`` interpolation)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = OmegaConf.load(str(cfg_path))
    resolved = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Config root must be a mapping, got {type(resolved)}")
    return _from_dict(Config, resolved)


def config_to_train_defaults(cfg: Config) -> Dict[str, Any]:
    """Flat argparse defaults for ``train_lib.parse_args``."""
    return {
        "device": cfg.device,
        "record_prefix": cfg.dir.data,
        "img_dir": cfg.dir.image,
        "output_dir": cfg.dir.output,
        "checkpoint_dir": cfg.dir.model_state,
        "log_path": cfg.file.log,
        "val_metrics_csv": cfg.file.val_metrics,
        "loss_plot": cfg.file.train_val_loss_plot,
        "num_visualization_samples": cfg.file.num_visualization_samples,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.data.batch_size,
        "lr": cfg.training.lr,
        "min_lr": cfg.training.min_lr,
        "weight_decay": cfg.training.weight_decay,
        "warmup_epochs": cfg.training.warmup_epochs,
        "grad_clip": cfg.training.grad_clip,
        "patience": cfg.training.patience,
        "save_every": cfg.training.save_every,
        "adv_weight": cfg.training.adv_weight,
        "adv_mse_threshold": cfg.training.adv_mse_threshold,
        "adv_t_max_frac": cfg.training.adv_t_max_frac,
        "d_lr": cfg.training.d_lr,
        "adv_real_label": cfg.training.adv_real_label,
        "adv_fake_label": cfg.training.adv_fake_label,
        "ssim_weight": cfg.training.ssim_weight,
        "ssim_mse_threshold": cfg.training.ssim_mse_threshold,
        "ssim_t_max_frac": cfg.training.ssim_t_max_frac,
        "val_sample_steps": cfg.training.val_sample_steps,
        "num_timesteps": cfg.model.num_timesteps,
        "image_size": cfg.model.image_size,
        "mask_size": cfg.data.mask_size,
        "subset_fraction": cfg.data.subset_fraction,
        "max_samples": cfg.data.max_samples,
        "patch_size": cfg.model.patch_size,
        "hidden_size": cfg.model.hidden_size,
        "depth": cfg.model.depth,
        "num_heads": cfg.model.num_heads,
        "dropout": cfg.model.dropout,
        "ffc_blocks": cfg.model.ffc_blocks,
        "learn_sigma": cfg.model.learn_sigma,
        "num_workers": cfg.data.num_workers,
        "seed": cfg.data.seed,
    }


def config_to_test_defaults(cfg: Config) -> Dict[str, Any]:
    """Flat argparse defaults for ``test_lib.parse_args``."""
    return {
        "checkpoint": cfg.file.model,
        "record_file": f"{cfg.file.dataset_dict}.test",
        "img_dir": cfg.dir.image,
        "output_dir": cfg.file.test_hic_map,
        "device": cfg.device,
        "batch_size": cfg.inference.batch_size,
        "num_samples": cfg.inference.num_samples,
        "seed": cfg.data.seed,
        "mask_size": cfg.data.mask_size,
        "image_size": cfg.model.image_size,
        "patch_size": cfg.model.patch_size,
        "hidden_size": cfg.model.hidden_size,
        "depth": cfg.model.depth,
        "num_heads": cfg.model.num_heads,
        "ffc_blocks": cfg.model.ffc_blocks,
        "num_timesteps": cfg.model.num_timesteps,
        "sample_steps": cfg.inference.sample_steps,
        "learn_sigma": cfg.model.learn_sigma,
    }
