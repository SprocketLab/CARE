"""Data loading helpers used by the cleaned CARE pipelines."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_aliases import normalize_fully_gaussian_dataset


def extract_score_from_parsed_output(x, min_rating: float = -3.0, max_rating: float = 3.0):
    """Extract the first numeric token and keep it only if it is in range."""
    match = re.search(r"[-+]?\d*\.?\d+", str(x))
    if not match:
        return np.nan
    value = pd.to_numeric(match.group(0), errors="coerce")
    if pd.isna(value):
        return np.nan
    if min_rating <= float(value) <= max_rating:
        return float(value)
    return np.nan


def _collect_asset_judge_outputs(
    judge_output_dir: Path,
    *,
    min_rating: float,
    max_rating: float,
    valid_ratio_threshold: float,
) -> pd.DataFrame:
    parsed_outputs: dict[str, pd.Series] = {}
    for csv_path in sorted(judge_output_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "parsed_output" not in df.columns:
            continue
        scores = df["parsed_output"].map(
            lambda x: extract_score_from_parsed_output(
                x,
                min_rating=min_rating,
                max_rating=max_rating,
            )
        )
        valid_ratio = float(scores.notna().mean()) if len(scores) else 0.0
        if valid_ratio >= valid_ratio_threshold:
            parsed_outputs[csv_path.stem] = scores
    return pd.DataFrame(parsed_outputs)


def load_judge_dataset_bundle(
    dataset_name,
    project_root=None,
    allow_trim: bool = True,
    valid_ratio_threshold: float = 0.7,
):
    """Load the judge matrix and human labels for a fully Gaussian dataset.

    This cleaned repository ships only one fully Gaussian dataset (``asset``).
    """
    dataset_key = normalize_fully_gaussian_dataset(dataset_name)
    if dataset_key != "asset":
        raise ValueError(
            f"Unsupported fully Gaussian dataset: {dataset_name}. "
            "Supported dataset in this repo: asset"
        )

    project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "score" / "asset.csv"
    judge_output_dir = project_root / "judge_outputs" / "fully_gaussian" / "asset"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing labels file: {data_path}")
    if not judge_output_dir.exists():
        raise FileNotFoundError(f"Missing judge output directory: {judge_output_dir}")

    labels_df = pd.read_csv(data_path)
    if "human_rating" not in labels_df.columns:
        raise ValueError("data/score/asset.csv must contain a human_rating column")

    human_eval = pd.to_numeric(labels_df["human_rating"], errors="coerce")
    min_rating = float(human_eval.min())
    max_rating = float(human_eval.max())

    judge_df = _collect_asset_judge_outputs(
        judge_output_dir,
        min_rating=min_rating,
        max_rating=max_rating,
        valid_ratio_threshold=valid_ratio_threshold,
    )

    if judge_df.empty:
        raise ValueError(f"No usable judge outputs found in {judge_output_dir}")

    judge_df = judge_df.reset_index(drop=True)
    human_eval = human_eval.reset_index(drop=True)

    if len(judge_df) != len(human_eval):
        if not allow_trim:
            raise ValueError(
                f"Size mismatch: judge outputs={len(judge_df)}, human labels={len(human_eval)}"
            )
        min_len = min(len(judge_df), len(human_eval))
        warnings.warn(
            f"asset: trimming to {min_len} rows (judges={len(judge_df)}, human={len(human_eval)})",
            RuntimeWarning,
            stacklevel=2,
        )
        judge_df = judge_df.iloc[:min_len].reset_index(drop=True)
        human_eval = human_eval.iloc[:min_len].reset_index(drop=True)

    judge_df = judge_df.apply(pd.to_numeric, errors="coerce").astype(float)
    judge_df = judge_df.replace([np.inf, -np.inf], np.nan)

    valid_mask = judge_df.notna().all(axis=1) & human_eval.notna()
    judge_df = judge_df.loc[valid_mask].reset_index(drop=True)
    human_eval = human_eval.loc[valid_mask].reset_index(drop=True)

    stds = judge_df.std(axis=0)
    non_constant_cols = stds[stds > 0].index.tolist()
    if not non_constant_cols:
        raise ValueError("All judge columns are constant after filtering")
    judge_df = judge_df[non_constant_cols]

    return judge_df, human_eval.astype(float)
