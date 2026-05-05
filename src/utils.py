"""General utility helpers for reproducible experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed all supported random generators for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_running_in_colab() -> bool:
    """Detect whether execution is happening in a Google Colab runtime."""
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False
