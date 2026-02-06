#!/usr/bin/env python3
"""Shared configs and dataset-loading helpers for Gaussian-mixture experiments."""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
DATA_ROOT = REPO_ROOT / "data"
JUDGE_OUTPUT_ROOT = REPO_ROOT / "judge_outputs" / "gaussian_mixture"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "outputs" / "gaussian_mixture"

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


FAST_TENSOR_OPTS = {
    "max_iters": 6,
    "early_stop_patience": 50,
    "improvement_tol": 1e-3,
}

SCORE_COLUMN_CANDIDATES = [
    "parsed_output",
    "score_original_order",
    "score_ab",
    "pred_label_num",
    "pred_label_binary",
]

PREF_LABEL_MAP = {
    "a": 0.0,
    "model_a": 0.0,
    "left": 0.0,
    "b": 1.0,
    "model_b": 1.0,
    "right": 1.0,
}

PREF_NULL_VALUES = {"tie", "none", "nan", ""}

DATASET_CONFIGS = [
    {
        "name": "civilcomments",
        "label_source": "csv",
        "label_path": DATA_ROOT / "binary" / "civilcomments.csv",
        "label_column": "label",
        "judge_subdir": "civilcomments",
        "min_rating": 0.0,
        "max_rating": 9.0,
        "binary_threshold": 4.5,
        "score_columns": ["parsed_output"],
        "ranks": (4, 5, 6, 7),
    },
    {
        "name": "pku_better",
        "label_source": "judge",
        "label_column": "gold_label_binary",
        "pref_label_column": "pref_A_or_B",
        "judge_subdir": "pku_better",
        "min_rating": -3.0,
        "max_rating": 3.0,
        "binary_threshold": 0.0,
        "score_columns": ["score_original_order", "score_ab"],
        "ranks": (4, 5, 6, 7),
    },
]

LAM_L_GRID = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
LAM_S_GRID = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


def maybe_remap_score_range(series: pd.Series, dataset: str, judge: str) -> pd.Series:
    """Map legacy 0-6 preference scores to -3..3 when needed."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric.astype(float)

    min_val = float(valid.min())
    max_val = float(valid.max())
    unique = int(valid.nunique())

    if unique > 1 and min_val >= 0.0 and max_val <= 6.0 and (max_val - min_val) >= 4.5:
        scaled = (numeric - min_val) / (max_val - min_val)
        remapped = scaled * 6.0 - 3.0
        print(f"{dataset}/{judge}: remapped score range [{min_val:.2f}, {max_val:.2f}] -> [-3, 3]")
        return remapped.astype(float)

    return numeric.astype(float)


def _extract_score_from_text(value, min_rating: float, max_rating: float):
    from data_tools import extract_score_from_parsed_output

    return extract_score_from_parsed_output(
        value,
        min_rating=min_rating,
        max_rating=max_rating,
    )


def extract_pref_labels(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df.columns:
        return None
    normalized = df[column].astype(str).str.strip().str.lower()
    mapped = normalized.map(PREF_LABEL_MAP)
    mapped = mapped.where(~normalized.isin(PREF_NULL_VALUES), np.nan)
    series = pd.to_numeric(mapped, errors="coerce").astype(float)
    finite = series.dropna()
    if finite.empty or np.unique(finite).size <= 1:
        return None
    return pd.Series(series.to_numpy(dtype=float), index=df.index, dtype=float)


def _maybe_flip_all_positive_labels(
    labels: np.ndarray,
    judge_df: pd.DataFrame,
    cfg: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    # Kept as a dedicated hook for future datasets where label polarity is inverted.
    return labels, judge_df


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def write_individual_json(directory: Path, dataset: str, payload: dict | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(payload or {})
    data.setdefault("dataset", dataset)
    target_path = directory / f"gaussian_mixture_{dataset}.json"
    target_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default))


def _maybe_extract_pku_better_archive(judge_dir: Path) -> None:
    if judge_dir.exists() and any(judge_dir.glob("*.csv")):
        return
    archive_path = JUDGE_OUTPUT_ROOT / "allenai_preference_test_sets_pku_better.tar.gz"
    if not archive_path.exists():
        return

    target_parent = judge_dir.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        root = target_parent.resolve()
        for member in tar.getmembers():
            member_path = (target_parent / member.name).resolve()
            if root not in member_path.parents and member_path != root:
                raise ValueError(f"Unsafe archive member path: {member.name}")
        tar.extractall(path=target_parent)


def load_dataset(cfg: dict) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    judge_dir = JUDGE_OUTPUT_ROOT / cfg["judge_subdir"]
    if cfg.get("name") == "pku_better":
        _maybe_extract_pku_better_archive(judge_dir)
    csv_paths = sorted(judge_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No judge outputs found in {judge_dir}")

    judge_columns: dict[str, pd.Series] = {}
    label_series: pd.Series | None = None
    expected_len = None

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if expected_len is None:
            expected_len = len(df)
        elif len(df) != expected_len:
            raise ValueError(f"Row count mismatch for {csv_path.name}: {len(df)} vs {expected_len}")

        if label_series is None:
            if cfg["label_source"] == "csv":
                label_df = pd.read_csv(cfg["label_path"])
                label_series = pd.to_numeric(label_df[cfg["label_column"]], errors="coerce")
            else:
                if cfg["label_column"] in df.columns:
                    label_series = pd.to_numeric(df[cfg["label_column"]], errors="coerce")
                elif "gold_label_num" in df.columns:
                    label_series = (pd.to_numeric(df["gold_label_num"], errors="coerce") > 0).astype(float)
                else:
                    raise ValueError(
                        f"Could not infer labels for {cfg['name']} from {csv_path.name}: "
                        f"missing {cfg['label_column']}"
                    )

            transform = cfg.get("label_transform")
            if transform is not None:
                label_series = transform(label_series)

            # Standardize common {-1, +1} encoding to {0, 1}.
            finite = label_series.dropna().to_numpy(dtype=float)
            if finite.size:
                uniq = set(np.unique(finite))
                if uniq.issubset({-1.0, 0.0, 1.0}) and uniq.intersection({-1.0, 1.0}):
                    label_series = (label_series > 0).astype(float)

            # Fallback to textual A/B labels when numeric labels are degenerate.
            if label_series.nunique(dropna=True) <= 1 and cfg.get("pref_label_column"):
                pref_series = extract_pref_labels(df, cfg["pref_label_column"])
                if pref_series is not None and pref_series.nunique(dropna=True) > 1:
                    label_series = pref_series

        score_series = None
        for column in cfg.get("score_columns", SCORE_COLUMN_CANDIDATES):
            if column not in df.columns:
                continue
            series = pd.to_numeric(df[column], errors="coerce")
            if column == "parsed_output" and series.isna().any():
                series = df[column].apply(
                    lambda x: _extract_score_from_text(
                        x,
                        min_rating=cfg["min_rating"],
                        max_rating=cfg["max_rating"],
                    )
                )
                series = pd.to_numeric(series, errors="coerce")
            score_series = series
            break

        if score_series is None:
            continue

        judge_name = csv_path.stem.replace("_prefs", "")
        score_series = maybe_remap_score_range(score_series, cfg["name"], judge_name)
        judge_columns[judge_name] = score_series

    if label_series is None:
        raise ValueError(f"Could not infer labels for dataset {cfg['name']}")

    judge_df = pd.DataFrame(judge_columns)
    if judge_df.empty:
        raise ValueError(f"No usable judge scores found in {judge_dir}")

    min_rating = cfg["min_rating"]
    max_rating = cfg["max_rating"]
    within_bounds = judge_df.ge(min_rating) & judge_df.le(max_rating)
    mask = (~label_series.isna()) & within_bounds.all(axis=1) & judge_df.notna().all(axis=1)

    labels = label_series[mask].astype(int).reset_index(drop=True)
    judge_df = judge_df[mask].reset_index(drop=True)

    label_array = labels.to_numpy(dtype=int)
    label_array, judge_df = _maybe_flip_all_positive_labels(label_array, judge_df, cfg)

    return label_array.astype(int, copy=False), judge_df, list(judge_df.columns)
