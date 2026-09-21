from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl") -> Tuple[bool, int, int, int]:
    """Initialize torch.distributed if launched via torchrun / env ranks.

    Returns ``(enabled, rank, local_rank, world_size)``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Destroy the process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    return int(rank) == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module if wrapped in DDP/DP."""
    return model.module if hasattr(model, "module") else model
