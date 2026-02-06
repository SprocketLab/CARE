#!/usr/bin/env python
"""Fully Gaussian CARE-SVD evaluation with validation-based gamma tuning."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    def display(*_args, **_kwargs):
        return None

csv.field_size_limit(10**9)


LLM_ROOT = Path(__file__).resolve().parents[1]
if not (LLM_ROOT / 'src').exists():
    raise RuntimeError(f'Could not locate repository root from script path: {LLM_ROOT}')

NOTEBOOK_DIR = LLM_ROOT / 'notebooks' if (LLM_ROOT / 'notebooks').exists() else LLM_ROOT
SRC_ROOT = LLM_ROOT / 'src'
SCRIPT_DIR = LLM_ROOT / 'scripts'

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from eval_tools import collect_metrics  # noqa: E402
from binary_baselines import BASELINE_METHODS, run_all_methods  # noqa: E402
from dataset_aliases import normalize_dataset_list, normalize_fully_gaussian_dataset  # noqa: E402

VAL_FRACTION = 0.1
DEFAULT_RANDOM_SEED = 2024
GAMMA_GRID = [0.1, 0.2, 0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10]
LEARN_STRUCTURE_SOLVER_KW = dict(max_iters=10000)

DATASETS = ["asset"]

ORDERED_METHODS = ['MV', 'AVG', 'WS', 'UWS', 'CARE-SVD']
METHOD_LABELS = {
    'mv': 'MV',
    'avg': 'AVG',
    'ws': 'WS',
    'uws': 'UWS',
    'care_svd': 'CARE-SVD',
}
DATASET_LABELS = {
    'asset': 'asset',
}


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fully Gaussian CARE-SVD evaluation with validation-based gamma search.')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED, help='Random seed for validation splits (default: 2024).')
    parser.add_argument('--output-dir', type=Path, help='Optional override for the results directory.')
    parser.add_argument('--datasets', nargs='+', help='Optional subset of dataset names.')
    parser.add_argument('--skip-main', action='store_true', help='Skip the core aggregation table and run only baselines.')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip binary baselines.')
    parser.add_argument(
        '--baseline-methods',
        nargs='+',
        choices=list(BASELINE_METHODS.keys()),
        help='Optional subset of baseline methods.',
    )
    parser.add_argument(
        '--baseline-output',
        type=Path,
        help='Path for baseline CSV (defaults to <output-dir>/fully_gaussian_baselines.csv).',
    )
    return parser.parse_args(argv)


def _is_binary_labels(series: pd.Series) -> bool:
    vals = pd.unique(series.dropna())
    if vals.size == 0:
        return False
    rounded = np.unique(np.rint(vals))
    return set(rounded).issubset({0.0, 1.0})


def _prepare_binary_matrix(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    rounded = np.rint(arr)
    clipped = np.clip(rounded, 0.0, 1.0)
    clipped[np.isnan(arr)] = np.nan
    return clipped


def tune_gamma_with_validation(train_df: pd.DataFrame, val_df: pd.DataFrame, val_eval: pd.Series, dataset_name: str, corr_matrix_train: pd.DataFrame):
    """Return best gamma and validation telemetry for the given dataset."""
    import pgm_tools

    validation_entries = []
    gamma_candidates = []
    for gamma in GAMMA_GRID:
        try:
            _, weights = pgm_tools.caresl_aggregate(
                train_df,
                gamma=gamma,
                verbose=False,
                corr_matrix=corr_matrix_train,
                return_weights=True,
                **LEARN_STRUCTURE_SOLVER_KW,
            )
            val_pred = pgm_tools.caresl_aggregate(val_df, weights=weights)
            val_mae, val_tau = collect_metrics(val_pred, val_eval)
        except Exception as exc:
            print(f'Skipping gamma {gamma} for {dataset_name}: {exc}')
            continue
        validation_entries.append({
            'dataset': dataset_name,
            'gamma': gamma,
            'val_mae': val_mae,
            'val_kendall_tau': val_tau,
        })
        if not np.isnan(val_mae):
            gamma_candidates.append((gamma, val_mae))

    best_gamma = GAMMA_GRID[0]
    if gamma_candidates:
        best_gamma = min(gamma_candidates, key=lambda item: (item[1], item[0]))[0]
    return best_gamma, validation_entries


def load_dataset_bundle(dataset_name: str):
    from data_tools import load_judge_dataset_bundle

    return load_judge_dataset_bundle(
        dataset_name,
        project_root=LLM_ROOT,
        allow_trim=True,
        valid_ratio_threshold=0.7,
    )


def run(
    seed: int,
    output_dir: Path | None = None,
    datasets: list[str] | None = None,
    *,
    run_main: bool = True,
    run_baselines: bool = True,
    baseline_methods=None,
    baseline_output: Path | None = None,
) -> Path | None:
    if not run_main and not run_baselines:
        print("Nothing to run: both main and baselines are disabled.")
        return None

    selected_datasets = DATASETS
    if datasets:
        normalized = normalize_dataset_list(datasets, normalize_fully_gaussian_dataset)
        missing = [d for d in normalized if d not in DATASETS]
        if missing:
            raise ValueError(f"Unknown dataset(s): {', '.join(missing)}")
        selected_datasets = normalized

    results = []
    validation_records = []
    best_gamma_records = []
    judge_performance_records = []
    baseline_records: list[dict[str, object]] = []

    for dataset_name in selected_datasets:
        import pgm_tools

        print('Processing dataset:', dataset_name)
        try:
            judge_df, human_eval = load_dataset_bundle(dataset_name)
        except (FileNotFoundError, ValueError) as exc:
            print(f'Skipping {dataset_name}: {exc}')
            continue

        judge_df = judge_df.reset_index(drop=True)
        human_eval = human_eval.reset_index(drop=True)

        if run_main:
            numeric_judges = judge_df.select_dtypes(include=[np.number])
            for judge_name in numeric_judges.columns:
                judge_mae, _ = collect_metrics(numeric_judges[judge_name], human_eval)
                judge_performance_records.append({
                    'dataset': dataset_name,
                    'judge': judge_name,
                    'mae': judge_mae,
                })

            best_gamma = GAMMA_GRID[0]
            try:
                train_df, val_df, _, val_eval = train_test_split(
                    judge_df,
                    human_eval,
                    test_size=VAL_FRACTION,
                    random_state=seed,
                    shuffle=True,
                )
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
                val_eval = val_eval.reset_index(drop=True)
                corr_matrix_train = pgm_tools.sanitize_correlation(train_df.corr())
                best_gamma, gamma_validation_entries = tune_gamma_with_validation(
                    train_df,
                    val_df,
                    val_eval,
                    dataset_name,
                    corr_matrix_train,
                )
                validation_records.extend(gamma_validation_entries)
            except Exception as exc:
                print(f'Validation split failed for {dataset_name}: {exc}')

            best_gamma_records.append({'dataset': dataset_name, 'best_gamma': best_gamma})

            predictions = {
                'mv': pgm_tools.majority_vote(judge_df),
                'avg': judge_df.mean(axis=1),
            }

            corr_matrix_full = pgm_tools.sanitize_correlation(judge_df.corr())

            try:
                encoded_df, inverse_mapping = pgm_tools.encode_for_label_models(judge_df)
                ws_indices = pgm_tools.run_label_model(encoded_df)
                predictions['ws'] = np.array([inverse_mapping.get(idx, np.nan) for idx in ws_indices], dtype=float)
            except Exception as exc:
                print(f'Skipping WS for {dataset_name} due to error: {exc}')
                predictions['ws'] = None

            try:
                predictions['uws'] = pgm_tools.uws_aggregate(judge_df)
            except Exception as exc:
                print(f'Skipping UWS for {dataset_name} due to error: {exc}')
                predictions['uws'] = None

            try:
                predictions['care_svd'], _ = pgm_tools.caresl_aggregate(
                    judge_df,
                    gamma=best_gamma,
                    verbose=False,
                    corr_matrix=corr_matrix_full,
                    return_weights=True,
                    **LEARN_STRUCTURE_SOLVER_KW,
                )
            except Exception as exc:
                print(f'Skipping CARE-SVD for {dataset_name}: {exc}')
                predictions['care_svd'] = None

            for name, pred in predictions.items():
                mae, kendall = collect_metrics(pred, human_eval)
                row = {
                    'dataset': dataset_name,
                    'pred': name,
                    'mae': mae,
                    'kendall_tau': kendall,
                }
                if name == 'care_svd':
                    row['gamma'] = best_gamma
                results.append(row)

        if run_baselines:
            if not _is_binary_labels(human_eval):
                reason = "non-binary labels; skipped"
                for method_name in baseline_methods or BASELINE_METHODS.keys():
                    baseline_records.append(
                        {
                            "dataset": dataset_name,
                            "method": method_name,
                            "status": "skipped",
                            "message": reason,
                            "accuracy": np.nan,
                            "f1": np.nan,
                            "coverage": 0.0,
                        }
                    )
                continue

            label_matrix = _prepare_binary_matrix(judge_df)
            gold = np.asarray(np.rint(human_eval).astype(int))
            for res in run_all_methods(label_matrix, gold, seed=seed, methods=baseline_methods):
                baseline_records.append(
                    {
                        "dataset": dataset_name,
                        "method": res.method,
                        "status": res.status,
                        "message": res.message,
                        "accuracy": res.accuracy,
                        "f1": res.f1,
                        "coverage": res.coverage,
                    }
                )

    results_dir = output_dir or (NOTEBOOK_DIR / 'results')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = None
    baseline_path = None

    if run_main:
        results_df = pd.DataFrame(results)
        best_gamma_df = pd.DataFrame(best_gamma_records)
        validation_df = pd.DataFrame(validation_records)
        judge_results_df = pd.DataFrame(judge_performance_records)

        try:
            display(best_gamma_df)
            if not validation_df.empty:
                display(validation_df)
            display(results_df)
            if not judge_results_df.empty:
                display(judge_results_df)
        except Exception:  # pragma: no cover - display only works in notebooks
            pass

        results_path = results_dir / 'fully_gaussian_main.csv'
        results_df.to_csv(results_path, index=False)
        best_gamma_path = results_dir / 'fully_gaussian_main_best_gamma.csv'
        best_gamma_df.to_csv(best_gamma_path, index=False)
        if not validation_df.empty:
            validation_path = results_dir / 'fully_gaussian_main_validation.csv'
            validation_df.to_csv(validation_path, index=False)
        if not judge_results_df.empty:
            judge_results_path = results_dir / 'fully_gaussian_main_judges.csv'
            judge_results_df.to_csv(judge_results_path, index=False)

        mae_table = (
            results_df
            .replace({'pred': METHOD_LABELS})
            .replace({'dataset': DATASET_LABELS})
            .pivot_table(index='pred', columns='dataset', values='mae')
            .rename_axis(index='method_label', columns='dataset_label')
        ).reindex(ORDERED_METHODS)

        try:
            display(mae_table.style.set_caption('MAE by method and dataset'))
        except Exception:  # pragma: no cover
            pass

    if run_baselines:
        baseline_df = pd.DataFrame(baseline_records)
        baseline_path = baseline_output or (results_dir / "fully_gaussian_baselines.csv")
        baseline_df.to_csv(baseline_path, index=False)
        print(f"Wrote fully_gaussian baselines to {baseline_path}")

    if results_path is not None:
        return results_path
    return baseline_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed = int(args.seed)
    set_global_seed(seed)
    run(
        seed=seed,
        output_dir=args.output_dir,
        datasets=args.datasets,
        run_main=not args.skip_main,
        run_baselines=not args.skip_baselines,
        baseline_methods=args.baseline_methods,
        baseline_output=args.baseline_output,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
