#!/usr/bin/env python3
"""Run Gaussian mixture aggregation using CARE-SVD and CARE-Tensor."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from binary_baselines import BASELINE_METHODS, evaluate_predictions, run_all_methods
import gaussian_mixture_fast_common as gm_common
import gaussian_mixture_pipeline as gm_pipeline


DEFAULT_CACHE_PATH = gm_common.CACHE_DIR / "care_tensor_results.json"
CARE_TENSOR_STATE_DIR = gm_common.CACHE_DIR / "care_tensor_state"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Gaussian mixture aggregation with CARE-SVD and CARE-Tensor.")
    parser.add_argument("--datasets", nargs="+", help="Optional subset of dataset names to run.")
    parser.add_argument(
        "--preference-only",
        action="store_true",
        help="Restrict to preference datasets (pku_better in this cleaned repo).",
    )
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached results when available.")
    parser.add_argument("--output", type=Path, help="Path for the metrics CSV (defaults to results/gaussian_mixture_results.csv).")
    parser.add_argument("--judge-summary", type=Path, help="Optional path to write the dataset-to-judge mapping CSV.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH, help="Location of the care_tensor cache JSON.")
    parser.add_argument("--individual-cache-dir", type=Path, help="Directory for per-dataset JSON cache outputs.")
    parser.add_argument("--solver-max-iters", type=int, help="Override the underlying solver's max_iters parameter.")
    parser.add_argument("--solver-eps", type=float, help="Override the underlying solver's eps tolerance (SCS).")
    parser.add_argument("--solver-verbose", action="store_true", help="Enable verbose output from the precision solver.")
    parser.add_argument("--state-dir", type=Path, help="Directory for cached CARE-Tensor state (defaults to outputs/gaussian_mixture/care_tensor_state).")
    parser.add_argument("--skip-main", action="store_true", help="Skip CARE-Tensor/CARE-SVD/MV/WS/UWS main metrics.")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline methods.")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        choices=list(BASELINE_METHODS.keys()),
        help="Optional subset of baseline methods.",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        help="Path for baseline CSV (defaults to results/gaussian_mixture_baselines.csv).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splits and numpy RNG.")
    return parser.parse_args(argv)


def _ged_from_scores(judge_binary: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        frac_one = np.nanmean(judge_binary, axis=1)
    return frac_one


def _normalize_metrics_method_names(metrics: dict | None) -> dict | None:
    if not isinstance(metrics, dict):
        return metrics
    if "caresl" in metrics and "care_svd" not in metrics:
        metrics["care_svd"] = metrics.pop("caresl")
    if "caret" in metrics and "care_tensor" not in metrics:
        metrics["care_tensor"] = metrics.pop("caret")
    if "caresl_gamma" in metrics and "care_svd_gamma" not in metrics:
        metrics["care_svd_gamma"] = metrics.pop("caresl_gamma")
    if "caresl_val_acc" in metrics and "care_svd_val_acc" not in metrics:
        metrics["care_svd_val_acc"] = metrics.pop("caresl_val_acc")
    return metrics


def _normalize_cache_entry_names(entry: dict | None) -> dict | None:
    if not isinstance(entry, dict):
        return entry
    if "metrics" in entry:
        entry["metrics"] = _normalize_metrics_method_names(entry.get("metrics"))
    hyperparams = entry.get("hyperparams")
    if isinstance(hyperparams, dict):
        if "caresl_gamma_grid" in hyperparams and "care_svd_gamma_grid" not in hyperparams:
            hyperparams["care_svd_gamma_grid"] = hyperparams.pop("caresl_gamma_grid")
    if "caret_meta" in entry and "care_tensor_meta" not in entry:
        entry["care_tensor_meta"] = entry.pop("caret_meta")
    return entry


def run(
    *,
    seed: int,
    datasets: list[str] | None = None,
    preference_only: bool = False,
    use_cache: bool = False,
    output: Path | None = None,
    judge_summary: Path | None = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    individual_cache_dir: Path | None = None,
    solver_max_iters: int | None = None,
    solver_eps: float | None = None,
    solver_verbose: bool = False,
    state_dir: Path | None = None,
    run_main: bool = True,
    run_baselines: bool = True,
    baseline_output: Path | None = None,
    baseline_methods=None,
) -> tuple[Path | None, Path | None]:
    if not run_main and not run_baselines:
        print("Nothing to run: both main and baselines are disabled.")
        return None, None

    force_rerun = not use_cache
    gm_pipeline.set_global_seed(seed)

    results_dir = gm_common.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = output or (results_dir / "gaussian_mixture_results.csv")
    judge_summary_path = judge_summary or (results_dir / "gaussian_mixture_judges.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    judge_summary_path.parent.mkdir(parents=True, exist_ok=True)

    if baseline_output is not None:
        baseline_output_path = baseline_output
    elif output is not None:
        out_name = output.name.replace("results", "baselines")
        baseline_output_path = output.with_name(out_name)
    else:
        baseline_output_path = results_dir / "gaussian_mixture_baselines.csv"
    baseline_output_path.parent.mkdir(parents=True, exist_ok=True)

    individual_dir = individual_cache_dir or results_dir
    individual_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = gm_pipeline.select_dataset_configs(
        gm_common.DATASET_CONFIGS,
        datasets=datasets,
        preference_only=preference_only,
    )

    if run_main:
        from pgm_tools import FastCaretAggregator, majority_vote, uws_aggregate, ws_aggregate

    solver_kwargs: dict[str, float | bool] = {}
    if solver_max_iters is not None:
        solver_kwargs["max_iters"] = int(solver_max_iters)
    if solver_eps is not None:
        solver_kwargs["eps"] = float(solver_eps)
    if solver_verbose:
        solver_kwargs["verbose"] = True

    state_dir = state_dir or CARE_TENSOR_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)

    cache_data = gm_common.load_cache(cache_path)
    cache_meta = cache_data.setdefault(
        "__meta__",
        {
            "variant": "gaussian_mixture_main",
            "created": datetime.utcnow().isoformat(timespec="seconds"),
        },
    )
    cache_meta.setdefault("variant", "gaussian_mixture_main")
    dataset_cache = cache_data.setdefault("datasets", {})

    results: list[dict] = []
    baseline_records: list[dict[str, object]] = []
    judge_registry: dict[str, list[str]] = {}
    skipped_datasets: dict[str, str] = {}

    for cfg in selected_configs:
        ds_name = cfg["name"]
        cached_entry = None if force_rerun else dataset_cache.get(ds_name)
        if cached_entry:
            cached_entry = _normalize_cache_entry_names(cached_entry)
            dataset_cache[ds_name] = cached_entry

        if cached_entry:
            status = cached_entry.get("status")
            cached_metrics = cached_entry.get("metrics")
            cached_baselines = cached_entry.get("baseline_records")

            can_use_main = run_main and status == "ok" and bool(cached_metrics)
            can_use_baselines = (not run_baselines) or bool(cached_baselines)

            if can_use_main and can_use_baselines:
                print(f"{ds_name}: using cached results")
                results.append(cached_metrics)
                if run_baselines and cached_baselines:
                    baseline_records.extend(cached_baselines)
                judge_registry[ds_name] = cached_entry.get("judges", [])
                gm_common.write_individual_json(individual_dir, ds_name, cached_entry)
                continue

            if (not run_main) and run_baselines and cached_baselines:
                print(f"{ds_name}: using cached baselines")
                baseline_records.extend(cached_baselines)
                judge_registry[ds_name] = cached_entry.get("judges", [])
                gm_common.write_individual_json(individual_dir, ds_name, cached_entry)
                continue

            if status == "error":
                reason = cached_entry.get("error", "previous failure")
                print(f"{ds_name}: skipping (cached error: {reason})")
                skipped_datasets[ds_name] = reason
                gm_common.write_individual_json(individual_dir, ds_name, cached_entry)
                continue

        try:
            y, judge_df, judge_names = gm_common.load_dataset(cfg)
        except Exception as exc:  # pragma: no cover - defensive logging
            reason = str(exc)
            print(f"{ds_name}: skipped during load -> {reason}")
            dataset_cache[ds_name] = {
                "status": "error",
                "error": reason,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            }
            gm_common.save_cache(cache_data, cache_path)
            gm_common.write_individual_json(individual_dir, ds_name, dataset_cache[ds_name])
            skipped_datasets[ds_name] = reason
            continue

        judge_registry[ds_name] = judge_names
        print(f"{ds_name}: {len(y)} examples, {len(judge_names)} judges")
        print("Judges:", ", ".join(judge_names))

        if run_main and judge_df.shape[1] < 3:
            reason = "CARE-Tensor requires at least 3 judges to form groups"
            print(f"  skipping {ds_name}: {reason}")
            dataset_cache[ds_name] = {
                "status": "error",
                "error": reason,
                "judges": judge_names,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            }
            gm_common.save_cache(cache_data, cache_path)
            gm_common.write_individual_json(individual_dir, ds_name, dataset_cache[ds_name])
            skipped_datasets[ds_name] = reason
            continue

        judge_binary_df = (judge_df >= cfg["binary_threshold"]).astype(int)
        judge_binary = judge_binary_df.to_numpy(dtype=float)

        metrics = None
        best_meta = None
        ds_baseline_records: list[dict[str, object]] = []

        if run_main:
            pos_rate = float(np.clip(np.mean(y), 0.0, 1.0))
            class_balance = float(np.clip((1.0 - pos_rate) * 100.0, 0.0, 100.0))

            idx_val, idx_test = gm_pipeline.split_indices(
                y,
                seed=seed,
                val_fraction=gm_pipeline.VALIDATION_FRACTION,
            )

            judge_scores_matrix = judge_df.to_numpy(dtype=float)
            aggregator = FastCaretAggregator(
                judge_scores_matrix,
                class_balance=class_balance,
                ranks=cfg.get("ranks", (4, 5, 6, 7)),
                tensor_opts=gm_common.FAST_TENSOR_OPTS,
                solver_kwargs=solver_kwargs,
                dataset_name=ds_name,
                state_cache_dir=state_dir,
            )

            best_acc = -1.0
            best_params: tuple[float | None, float | None] = (None, None)
            best_preds = None
            last_error = None

            for lam_L in gm_common.LAM_L_GRID:
                for lam_S in gm_common.LAM_S_GRID:
                    try:
                        preds, meta = aggregator.predict(lam_L=lam_L, lam_S=lam_S)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        last_error = str(exc)
                        continue

                    preds = np.asarray(preds, dtype=int)
                    val_acc = accuracy_score(y[idx_val], preds[idx_val])
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_params = (lam_L, lam_S)
                        best_preds = preds
                        best_meta = meta
                        print(
                            "  new best:",
                            f"lam_L={lam_L}",
                            f"lam_S={lam_S}",
                            f"val_acc={val_acc:.3f}",
                            f"precision_cache_hit={meta['precision_cache_hit']}",
                            f"tensor_cache_hit={meta['tensor_cache_hit']}",
                        )

            if best_preds is None:
                reason = last_error or "care_tensor_aggregate did not return any predictions"
                print(f"  skipping {ds_name}: {reason}")
                dataset_cache[ds_name] = {
                    "status": "error",
                    "error": reason,
                    "judges": judge_names,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                }
                gm_common.save_cache(cache_data, cache_path)
                gm_common.write_individual_json(individual_dir, ds_name, dataset_cache[ds_name])
                skipped_datasets[ds_name] = reason
                continue

            care_tensor_pred = np.asarray(best_preds, dtype=int)
            mv_pred = np.asarray(majority_vote(judge_binary_df), dtype=int)
            ws_pred = np.asarray(ws_aggregate(judge_binary_df, seed=seed), dtype=int)
            avg_scores = judge_df.mean(axis=1)
            avg_pred = np.asarray((avg_scores >= cfg["binary_threshold"]).astype(int))
            uws_scores = np.asarray(uws_aggregate(judge_df), dtype=float)
            uws_pred = np.asarray((uws_scores >= cfg["binary_threshold"]).astype(int))

            care_svd_gamma, care_svd_val_acc, care_svd_scores = gm_pipeline.tune_care_svd_gamma(
                judge_df,
                y,
                idx_val=idx_val,
                threshold=cfg["binary_threshold"],
                gamma_grid=gm_pipeline.CARE_SVD_GAMMA_GRID,
            )
            care_svd_pred = np.asarray((care_svd_scores >= cfg["binary_threshold"]).astype(int))

            def acc(pred):
                arr = np.asarray(pred, dtype=int)
                return float((arr[idx_test] == y[idx_test]).mean())

            metrics = {
                "dataset": ds_name,
                "n_examples": int(len(y)),
                "n_judges": int(judge_df.shape[1]),
                "mv": acc(mv_pred),
                "avg": acc(avg_pred),
                "ws": acc(ws_pred),
                "uws": acc(uws_pred),
                "care_svd": acc(care_svd_pred),
                "care_tensor": acc(care_tensor_pred),
                "care_svd_gamma": care_svd_gamma,
                "care_svd_val_acc": care_svd_val_acc,
                "lam_L": best_params[0],
                "lam_S": best_params[1],
                "val_acc": best_acc,
                "val_size": int(idx_val.size),
                "test_size": int(idx_test.size),
                "class_balance": class_balance,
            }
            results.append(metrics)

            cache_snapshot = aggregator.cache_info()
        else:
            cache_snapshot = {"tensor_cache_entries": 0, "state_cache_enabled": state_dir is not None}

        if run_baselines:
            mv_scores = np.nanmean(judge_binary, axis=1)
            mv_scores = np.where(np.isnan(mv_scores), 0.5, mv_scores)
            mv_pred = (mv_scores >= 0.5).astype(int)
            mv_acc = accuracy_score(y, mv_pred) if mv_scores.size else np.nan

            for res in run_all_methods(judge_binary, y, seed=seed, methods=baseline_methods):
                ds_baseline_records.append(
                    {
                        "dataset": ds_name,
                        "method": res.method,
                        "status": res.status,
                        "message": res.message,
                        "accuracy": res.accuracy,
                        "f1": res.f1,
                        "coverage": res.coverage,
                        "majority_vote_acc": mv_acc,
                    }
                )

            try:
                ged_probs = _ged_from_scores(judge_binary)
                mask = np.isfinite(ged_probs)
                if mask.any():
                    ged_acc, ged_f1, coverage = evaluate_predictions(ged_probs, y)
                    ds_baseline_records.append(
                        {
                            "dataset": ds_name,
                            "method": "ged_pref",
                            "status": "ok",
                            "message": "from_thresholded_scores",
                            "accuracy": ged_acc,
                            "f1": ged_f1,
                            "coverage": coverage,
                            "majority_vote_acc": mv_acc,
                        }
                    )
                else:
                    ds_baseline_records.append(
                        {
                            "dataset": ds_name,
                            "method": "ged_pref",
                            "status": "skipped",
                            "message": "no finite GED predictions",
                            "accuracy": np.nan,
                            "f1": np.nan,
                            "coverage": 0.0,
                            "majority_vote_acc": mv_acc,
                        }
                    )
            except Exception as exc:  # pragma: no cover - defensive
                ds_baseline_records.append(
                    {
                        "dataset": ds_name,
                        "method": "ged_pref",
                        "status": "error",
                        "message": str(exc),
                        "accuracy": np.nan,
                        "f1": np.nan,
                        "coverage": 0.0,
                        "majority_vote_acc": mv_acc,
                    }
                )

            baseline_records.extend(ds_baseline_records)

        entry = dataset_cache.get(ds_name, {})
        entry.update(
            {
                "status": "ok",
                "judges": judge_names,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
        if metrics is not None:
            entry["metrics"] = metrics
            entry["hyperparams"] = {
                "lam_L_grid": list(gm_common.LAM_L_GRID),
                "lam_S_grid": list(gm_common.LAM_S_GRID),
                "ranks": list(cfg.get("ranks", (4, 5, 6, 7))),
                "care_svd_gamma_grid": list(gm_pipeline.CARE_SVD_GAMMA_GRID),
            }
            entry["care_tensor_meta"] = {
                "best": best_meta,
                "tensor_cache_entries": cache_snapshot["tensor_cache_entries"],
                "state_cache_enabled": cache_snapshot["state_cache_enabled"],
            }
            print(
                "  best val acc:",
                f"{metrics['val_acc']:.3f}",
                "with lam_L",
                metrics["lam_L"],
                "lam_S",
                metrics["lam_S"],
            )
        if run_baselines:
            entry["baseline_records"] = ds_baseline_records

        dataset_cache[ds_name] = entry
        gm_common.save_cache(cache_data, cache_path)
        gm_common.write_individual_json(individual_dir, ds_name, entry)

    main_path = None
    baselines_path = None

    if run_main:
        if results:
            result_df = pd.DataFrame(results).sort_values("dataset")
            result_df.to_csv(output_path, index=False)
            print(f"Wrote metrics to {output_path}")
            main_path = output_path
        else:
            print("No results to write.")

        if judge_registry:
            judge_df = pd.DataFrame(
                {
                    "dataset": list(judge_registry.keys()),
                    "judges": [", ".join(v) for v in judge_registry.values()],
                }
            ).sort_values("dataset")
            judge_df.to_csv(judge_summary_path, index=False)
            print(f"Wrote judge registry to {judge_summary_path}")

    if run_baselines:
        baseline_df = pd.DataFrame(baseline_records)
        baseline_df.to_csv(baseline_output_path, index=False)
        baselines_path = baseline_output_path
        print(f"Wrote gaussian mixture baselines to {baseline_output_path}")

    if skipped_datasets:
        print("Skipped datasets:")
        for name, reason in skipped_datasets.items():
            print(f"  {name}: {reason}")

    return main_path, baselines_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(
        seed=int(args.seed),
        datasets=args.datasets,
        preference_only=args.preference_only,
        use_cache=args.use_cache,
        output=args.output,
        judge_summary=args.judge_summary,
        cache_path=args.cache_path,
        individual_cache_dir=args.individual_cache_dir,
        solver_max_iters=args.solver_max_iters,
        solver_eps=args.solver_eps,
        solver_verbose=args.solver_verbose,
        state_dir=args.state_dir,
        run_main=not args.skip_main,
        run_baselines=not args.skip_baselines,
        baseline_output=args.baseline_output,
        baseline_methods=args.baseline_methods,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
