"""
Core utilities for TerraSpectra: path management and reproducible random seeding.
"""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Optional
import numpy as np


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Current file is at <root>/src/terraspectra/core/utils.py
    return Path(__file__).resolve().parent.parent.parent.parent


def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists; create it if not present."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """
    Configure global random seeds for deterministic reproducibility.
    Seeds Python's random, NumPy, and PyTorch (if available without errors).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Conditionally seed PyTorch if it can be safely imported
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # PyTorch not present or blocked by OS policy; seed NumPy and standard random only
        pass
