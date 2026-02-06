"""Shared helpers for gaussian mixture main/fast/baseline pipelines."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from dataset_aliases import normalize_dataset_list, normalize_gaussian_mixture_dataset
except ModuleNotFoundError:  # pragma: no cover - direct script import fallback
    import sys
    from pathlib import Path

    _SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
    if str(_SRC_ROOT) not in sys.path:
        sys.path.append(str(_SRC_ROOT))
    from dataset_aliases import normalize_dataset_list, normalize_gaussian_mixture_dataset


VALIDATION_FRACTION = 0.15
CARE_SVD_GAMMA_GRID = [0.1, 0.2, 0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10]
# Backward compatibility alias.
CARESL_GAMMA_GRID = CARE_SVD_GAMMA_GRID
PREFERENCE_DATASETS = {"pku_better"}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        torch.cuda.manual_seed_all(seed)


def select_dataset_configs(
    dataset_configs: Sequence[dict],
    *,
    datasets: Iterable[str] | None = None,
    preference_only: bool = False,
    preserve_config_order: bool = True,
) -> list[dict]:
    if datasets:
        requested = normalize_dataset_list(datasets, normalize_gaussian_mixture_dataset)
        name_to_cfg = {cfg["name"]: cfg for cfg in dataset_configs}
        missing = [name for name in requested if name not in name_to_cfg]
        if missing:
            raise ValueError(f"Unknown dataset(s): {', '.join(sorted(missing))}")
        if preserve_config_order:
            return [cfg for cfg in dataset_configs if cfg["name"] in requested]
        return [name_to_cfg[name] for name in requested]
    if preference_only:
        return [cfg for cfg in dataset_configs if cfg["name"] in PREFERENCE_DATASETS]
    return list(dataset_configs)


def split_indices(
    labels: np.ndarray,
    *,
    seed: int,
    val_fraction: float = VALIDATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    unique_labels = np.unique(labels)
    if unique_labels.size > 1:
        idx_rest, idx_val = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=labels,
        )
    else:
        val_size = max(1, int(round(val_fraction * len(labels))))
        perm = np.random.default_rng(seed).permutation(indices)
        idx_val = perm[:val_size]
        idx_rest = perm[val_size:]
        if idx_rest.size == 0:
            idx_rest = idx_val
    return idx_val, idx_rest


def tune_care_svd_gamma(
    judge_df,
    labels: np.ndarray,
    *,
    idx_val: np.ndarray,
    threshold: float,
    gamma_grid: Sequence[float] = CARE_SVD_GAMMA_GRID,
) -> tuple[float, float, np.ndarray]:
    from pgm_tools import caresl_aggregate, sanitize_correlation

    best_gamma = float(gamma_grid[0])
    best_val_acc = float("nan")
    best_scores = None
    best_acc = -1.0

    corr_matrix = sanitize_correlation(judge_df.corr())
    for gamma in gamma_grid:
        try:
            scores = caresl_aggregate(
                judge_df,
                gamma=float(gamma),
                verbose=False,
                corr_matrix=corr_matrix,
            )
            preds = np.asarray(scores >= threshold, dtype=int)
            val_acc = float(accuracy_score(labels[idx_val], preds[idx_val]))
        except Exception:
            continue
        if val_acc > best_acc:
            best_acc = val_acc
            best_gamma = float(gamma)
            best_val_acc = val_acc
            best_scores = scores

    if best_scores is None:
        best_scores = caresl_aggregate(
            judge_df,
            gamma=best_gamma,
            verbose=False,
            corr_matrix=corr_matrix,
        )
    return best_gamma, best_val_acc, np.asarray(best_scores, dtype=float)


# Backward compatibility alias.
tune_caresl_gamma = tune_care_svd_gamma


def prepare_binary_matrix(judge_df, threshold: float) -> np.ndarray:
    raw = np.asarray(judge_df.to_numpy(dtype=float))
    finite_mask = np.isfinite(raw)
    bin_arr = np.zeros_like(raw, dtype=float)
    bin_arr[finite_mask] = (raw[finite_mask] >= threshold).astype(float)
    bin_arr[~finite_mask] = np.nan
    return bin_arr
