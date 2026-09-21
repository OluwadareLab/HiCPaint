from __future__ import annotations

import os
from datetime import datetime
from typing import Optional


def write_log(
    message: str,
    log_path: Optional[str] = None,
    also_print: bool = True,
) -> None:
    """Append a timestamped line to ``log_path`` and optionally print it."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    if also_print:
        print(line, flush=True)
    if log_path is None:
        return
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
