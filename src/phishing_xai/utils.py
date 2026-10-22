"""Utility functions: seeding, device management, logging, metadata, and I/O helpers."""

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds across random, numpy, and torch (CPU and CUDA) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return torch.device: cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "phishing_xai", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, filepath: Path | str, indent: int = 2) -> Path:
    """Save dictionary/list to JSON file."""
    p = Path(filepath)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    return p


def load_json(filepath: Path | str) -> Any:
    """Load JSON file and return contents."""
    p = Path(filepath)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_environment_metadata(seed: int = 42, extra_config: dict | None = None) -> dict:
    """Collect runtime environment metadata for tracking and reproducibility."""
    import transformers
    import sklearn

    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "seed": seed,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    if extra_config:
        metadata["config"] = extra_config
    return metadata
