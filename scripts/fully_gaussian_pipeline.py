"""Shared helpers for fully Gaussian experiments."""

from __future__ import annotations

import random

import numpy as np


DATASETS = [
    "asset",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
