#!/usr/bin/env python3
"""Dawid-Skene, GLAD, MACE, and GED baselines for the binary experiments."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


LLM_ROOT = Path(__file__).resolve().parents[1]
if not (LLM_ROOT / "src").exists():
    raise RuntimeError(f"Could not locate repository root from script path: {LLM_ROOT}")

SRC_ROOT = LLM_ROOT / "src"
NOTEBOOK_DIR = LLM_ROOT / "notebooks" if (LLM_ROOT / "notebooks").exists() else LLM_ROOT
JUDGE_OUTPUTS_BINARY_ROOT = LLM_ROOT / "judge_outputs" / "binary"
BINARY_DATA_ROOT = LLM_ROOT / "data" / "binary"

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

try:  # noqa: E402
    import binary_judge_experiments as bje
except ModuleNotFoundError:
    bje = None


VALIDATION_DATASET_MAP = OrderedDict(
    [
        ("civilcomments", "civilcomments"),
        ("pku_better", "pku_better"),
    ]
)

DS_MAX_ITER = 200
GLAD_MAX_ITER = 120
GLAD_LR = 0.02
GLAD_REG = 1e-2
MACE_MAX_ITER = 80
RESTARTS = 5
VALIDATION_FRACTION = 0.15

# Hyperparameter grids for validation tuning (expanded for DS/GLAD)
DS_ITER_GRID = [150, 300, 500]
GLAD_LR_GRID = [0.005, 0.01, 0.02, 0.05]
GLAD_ITER_GRID = [80, 160, 240]
MACE_ITER_GRID = [60, 80, 120]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _prepare_binary_matrix(df: pd.DataFrame) -> np.ndarray:
    """Round scores to the nearest label and clip to {0,1}; keep NaNs as missing."""
    arr = df.to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    rounded = np.rint(arr)
    clipped = np.clip(rounded, 0.0, 1.0)
    clipped[np.isnan(arr)] = np.nan
    return clipped


def dawid_skene(
    labels: np.ndarray,
    *,
    max_iter: int = DS_MAX_ITER,
    tol: float = 1e-6,
    smoothing: float = 1e-2,
    seed: int | None = None,
) -> np.ndarray:
    """Classic Dawid-Skene EM for binary labels (P(y=1))."""
    if seed is not None:
        np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=float)
    n_items, n_workers = labels.shape
    mask = np.isfinite(labels)
    if mask.sum() == 0:
        raise ValueError("All labels are missing; cannot run Dawid-Skene.")

    L = 2
    # Initialize confusion matrices close to identity
    pi = np.full((n_workers, L, L), smoothing, dtype=float)
    for j in range(n_workers):
        pi[j, 0, 0] = 0.8
        pi[j, 1, 1] = 0.8
    pi = pi / pi.sum(axis=2, keepdims=True)

    # Initial posteriors from majority vote (fallback to 0.5)
    mv = np.nanmean(labels, axis=1)
    mv = np.where(np.isnan(mv), 0.5, mv)
    posteriors = np.clip(mv, 1e-3, 1 - 1e-3)

    for _ in range(max_iter):
        prev = posteriors.copy()
        # E-step: compute P(y_i = k | labels)
        log_lik = np.zeros((n_items, L), dtype=float)
        for k in range(L):
            ll = np.zeros(n_items, dtype=float)
            for j in range(n_workers):
                worker_mask = mask[:, j]
                if not worker_mask.any():
                    continue
                obs = labels[worker_mask, j].astype(int)
                obs = np.clip(obs, 0, 1)
                ll_subset = np.log(pi[j, k, obs] + 1e-12)
                ll[worker_mask] += ll_subset
            log_lik[:, k] = ll
        log_lik[:, 0] += math.log(0.5)
        log_lik[:, 1] += math.log(0.5)
        # Normalize in log-space
        logit = log_lik[:, 1] - log_lik[:, 0]
        posteriors = _sigmoid(logit)

        # M-step: update confusion matrices
        weights = np.stack([1 - posteriors, posteriors], axis=1)  # (n_items, 2)
        for j in range(n_workers):
            worker_mask = mask[:, j]
            if not worker_mask.any():
                continue
            obs = labels[worker_mask, j].astype(int)
            obs = np.clip(obs, 0, 1)
            counts = np.zeros((L, L), dtype=float)
            for k in range(L):
                wk = weights[worker_mask, k]
                for l in range(L):
                    counts[k, l] = np.sum(wk * (obs == l))
            counts += smoothing
            counts = counts / counts.sum(axis=1, keepdims=True)
            pi[j] = counts

        delta = float(np.max(np.abs(prev - posteriors)))
        if delta < tol:
            break

    return np.clip(posteriors, 0.0, 1.0)


def glad(
    labels: np.ndarray,
    *,
    max_iter: int = GLAD_MAX_ITER,
    lr: float = GLAD_LR,
    reg: float = GLAD_REG,
    seed: int | None = None,
) -> np.ndarray:
    """Binary GLAD (Whitehill et al., 2009): logistic ability * difficulty.

    labels: (n_items, n_workers) with entries in {0,1} and NaN for missing.
    Returns: (n_items,) with P(y=1) for each item.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=float)
    n_items, n_workers = labels.shape
    mask = np.isfinite(labels)
    if mask.sum() == 0:
        raise ValueError("All labels are missing; cannot run GLAD.")

    # Ensure strict 0/1 labels
    labels_bin = np.zeros_like(labels, dtype=np.int8)
    labels_bin[mask] = (labels[mask] >= 0.5).astype(np.int8)

    # Parameters: abilities α_j (workers), difficulties β_i (items)
    abilities = rng.normal(loc=0.0, scale=0.5, size=n_workers)
    difficulties = np.abs(rng.normal(loc=1.0, scale=0.1, size=n_items))
    prior = 0.5
    posteriors = np.full(n_items, prior, dtype=float)  # P(z_i = 1)

    for _ in range(max_iter):
        # P(worker j is correct on item i) = σ(α_j * β_i)
        tau = _sigmoid(np.outer(difficulties, abilities))  # (n_items, n_workers)
        tau = np.clip(tau, 1e-6, 1.0 - 1e-6)

        # ---------- E-step: p(z_i | labels, α, β) ----------
        ll1 = np.full(n_items, np.log(prior + 1e-9), dtype=float)        # log p(z=1)
        ll0 = np.full(n_items, np.log(1.0 - prior + 1e-9), dtype=float)  # log p(z=0)

        for j in range(n_workers):
            m = mask[:, j]
            if not m.any():
                continue
            col = labels_bin[m, j]    # observed labels 0/1
            t   = tau[m, j]           # prob correct

            # If z=1, label is correct iff col==1
            ll1[m] += np.where(col == 1, np.log(t), np.log(1.0 - t))
            # If z=0, label is correct iff col==0
            ll0[m] += np.where(col == 0, np.log(t), np.log(1.0 - t))

        posteriors = _sigmoid(ll1 - ll0)  # P(z=1 | data)

        # ---------- M-step: gradient updates for α, β ----------
        grad_alpha = np.zeros_like(abilities)
        grad_beta = np.zeros_like(difficulties)

        for i in range(n_items):
            obs_mask = mask[i]
            if not obs_mask.any():
                continue
            y_row = labels_bin[i, obs_mask]    # labels for item i
            tau_row = tau[i, obs_mask]         # current p(correct)
            alpha_row = abilities[obs_mask]

            # Expected correctness under posterior:
            #  - if y_ij = 1, correctness when z_i=1
            #  - if y_ij = 0, correctness when z_i=0
            e_correct = np.where(y_row == 1, posteriors[i], 1.0 - posteriors[i])

            # Gradient term: E[correct] - model p(correct)
            residual = e_correct - tau_row

            grad_alpha[obs_mask] += residual * difficulties[i]
            grad_beta[i]         += np.sum(residual * alpha_row)

        # L2 regularization (pull α -> 0, β -> 1)
        grad_alpha -= reg * abilities
        grad_beta  -= reg * (difficulties - 1.0)

        abilities   += lr * grad_alpha
        difficulties += lr * grad_beta
        difficulties = np.clip(difficulties, 1e-3, 50.0)

        # Optional: update class prior to current mean
        prior = float(np.mean(posteriors))

    return np.clip(posteriors, 0.0, 1.0)

def mace(
    labels: np.ndarray,
    *,
    max_iter: int = MACE_MAX_ITER,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    delta: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Binary MACE EM (Hovy et al., 2013) with spamming indicator.

    labels: (n_items, n_workers) with entries in {0,1} and NaN for missing.
    alpha_prior, beta_prior: Beta prior on theta_j (spam probability).
    delta: Dirichlet smoothing for spam distribution xi_j.
    Returns: (n_items,) with P(true label = 1) for each item.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=float)
    n_items, n_workers = labels.shape
    mask = np.isfinite(labels)
    if mask.sum() == 0:
        raise ValueError("All labels are missing; cannot run MACE.")

    # Binary 0/1 labels
    labels_bin = np.zeros_like(labels, dtype=np.int8)
    labels_bin[mask] = (labels[mask] >= 0.5).astype(np.int8)

    L = 2  # number of classes {0,1}

    # theta_j: probability worker j is spamming (S_ij = 1)
    theta = np.full(n_workers, 0.2, dtype=float)  # start with mostly non-spammers
    # xi_j: spam label multinomial over {0,1}
    xi = np.full((n_workers, L), 1.0 / L, dtype=float)

    # Posterior over true labels T_i
    post_T = np.full((n_items, L), 1.0 / L, dtype=float)

    for _ in range(max_iter):
        # ========== E-step ==========
        # 1) Posterior over true labels T_i
        log_post_T = np.zeros((n_items, L), dtype=float)  # log p(T_i = k | A)

        for k in range(L):
            # For each possible true label k
            ll = np.zeros(n_items, dtype=float)
            for j in range(n_workers):
                m = mask[:, j]
                if not m.any():
                    continue
                obs = labels_bin[m, j]  # observed labels (0/1)

                # P(A_ij = obs | T_i = k) =
                #   (1 - theta_j) * 1_{obs == k} + theta_j * xi_j[obs]
                correct = (obs == k).astype(float)
                xi_j_obs = xi[j, obs]  # xi_j evaluated at each observed label
                term = (1.0 - theta[j]) * correct + theta[j] * xi_j_obs
                ll[m] += np.log(term + 1e-12)

            log_post_T[:, k] = ll

        # Normalize to get posterior over T_i
        log_post_T -= log_post_T.max(axis=1, keepdims=True)
        post_T = np.exp(log_post_T)
        post_T /= post_T.sum(axis=1, keepdims=True)

        # 2) Expected spamming indicators E[S_ij]
        exp_s = np.zeros((n_items, n_workers), dtype=float)
        for j in range(n_workers):
            m = mask[:, j]
            if not m.any():
                continue
            obs = labels_bin[m, j]

            for idx, i in enumerate(np.where(m)[0]):
                l = obs[idx]           # observed label for item i, worker j
                p_T_i = post_T[i]      # posterior over T_i (length 2)

                # P(S_ij = 1 | T_i = l, A_ij = l)
                denom_eq = (1.0 - theta[j]) + theta[j] * xi[j, l]
                p_s_eq = theta[j] * xi[j, l] / (denom_eq + 1e-12)

                # For k != l, S_ij must be 1 (cannot be correct non-spam)
                # E[S_ij] = (1 - P(T_i = l)) * 1 + P(T_i = l) * P(S_ij = 1 | T_i = l, A_ij = l)
                #         = 1 - P(T_i = l) * (1 - P(S_ij = 1 | T_i = l, A_ij = l))
                exp_s[i, j] = 1.0 - p_T_i[l] * (1.0 - p_s_eq)

        # ========== M-step ==========
        for j in range(n_workers):
            m = mask[:, j]
            n_obs = m.sum()
            if n_obs == 0:
                continue

            # Update theta_j (spam probability) using Beta(alpha_prior, beta_prior)
            sum_s = exp_s[m, j].sum()
            theta[j] = (sum_s + alpha_prior - 1.0) / (
                n_obs + alpha_prior + beta_prior - 2.0
            )
            theta[j] = np.clip(theta[j], 1e-4, 1.0 - 1e-4)

            # Update xi_j (spam distribution) from expected spam counts
            counts = np.zeros(L, dtype=float)
            obs = labels_bin[m, j]
            for ell in range(L):
                counts[ell] = np.sum(exp_s[m, j] * (obs == ell))
            counts += delta
            xi[j] = counts / counts.sum()

    # Return P(T_i = 1) as in your original code
    return np.clip(post_T[:, 1], 0.0, 1.0)


def ged(_: np.ndarray) -> np.ndarray:
    """PGED-style GED requires pairwise preference graphs; not applicable here."""
    raise NotImplementedError("GED requires preference graphs (pairwise comparisons), not per-item binary votes.")


BASELINE_METHODS: dict[str, Callable[..., np.ndarray]] = OrderedDict(
    [
        ("dawid_skene", dawid_skene),
        ("glad", glad),
        ("mace", mace),
        ("ged", ged),
    ]
)


@dataclass
class BaselineResult:
    dataset: str
    method: str
    accuracy: float
    f1: float
    coverage: float
    status: str
    message: str = ""


def evaluate_predictions(probs: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(probs)
    if mask.sum() == 0:
        return float("nan"), float("nan"), 0.0
    preds = (probs >= 0.5).astype(int)
    gold_masked = gold[mask]
    preds_masked = preds[mask]
    accuracy = accuracy_score(gold_masked, preds_masked)
    f1 = f1_score(gold_masked, preds_masked, zero_division=0)
    coverage = mask.mean()
    return float(accuracy), float(f1), float(coverage)


def run_all_methods(
    label_matrix: np.ndarray,
    gold_labels: np.ndarray,
    *,
    seed: int,
    methods: Iterable[str] | None = None,
) -> list[BaselineResult]:
    chosen_methods = list(methods) if methods else list(BASELINE_METHODS.keys())
    results: list[BaselineResult] = []
    # validation split for hyperparameter tuning
    indices = np.arange(len(gold_labels))
    stratify_labels = gold_labels if np.unique(gold_labels).size > 1 else None
    _, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
        stratify=stratify_labels,
    )

    for name in chosen_methods:
        func = BASELINE_METHODS.get(name)
        if func is None:
            continue
        best = None
        seeds = [seed + i for i in range(RESTARTS)] if name in {"dawid_skene", "glad", "mace"} else [seed]
        if name == "dawid_skene":
            hyper_grid = [{"max_iter": it} for it in DS_ITER_GRID]
        elif name == "glad":
            hyper_grid = [{"max_iter": it, "lr": lr, "reg": GLAD_REG} for it in GLAD_ITER_GRID for lr in GLAD_LR_GRID]
        elif name == "mace":
            hyper_grid = [{"max_iter": it} for it in MACE_ITER_GRID]
        else:
            hyper_grid = [{}]

        for hyper in hyper_grid:
            for s in seeds:
                try:
                    if name in {"dawid_skene", "glad", "mace"}:
                        probs = func(label_matrix, seed=s, **hyper)
                    else:
                        probs = func(label_matrix, **hyper) if hyper else func(label_matrix)
                    # validation accuracy on val_idx
                    acc_val, _, _, = evaluate_predictions(probs[val_idx], gold_labels[val_idx])
                    if best is None or acc_val > best["acc_val"]:
                        acc_full, f1_full, cov_full = evaluate_predictions(probs, gold_labels)
                        best = {
                            "acc": acc_full,
                            "f1": f1_full,
                            "cov": cov_full,
                            "acc_val": acc_val,
                            "seed": s,
                            "hyper": hyper,
                        }
                except NotImplementedError as exc:
                    best = {"acc": float("nan"), "f1": float("nan"), "cov": 0.0, "status": "skipped", "msg": str(exc)}
                    hyper_grid = []
                    break
                except Exception:
                    continue

        if best is None:
            acc = f1 = float("nan")
            cov = 0.0
            status = "error"
            msg = "all hyperparameter trials failed"
        else:
            acc = best["acc"]
            f1 = best["f1"]
            cov = best["cov"]
            status = "ok" if not pd.isna(acc) else "error"
            msg = f"seed={best.get('seed')},hyper={best.get('hyper')}"
            if "status" in best:
                status = best["status"]
                msg = best["msg"]
        results.append(
            BaselineResult(
                dataset="",
                method=name,
                accuracy=acc,
                f1=f1,
                coverage=cov,
                status=status,
                message=msg,
            )
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dawid-Skene, GLAD, MACE, and GED baselines for binary datasets."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional subset of dataset keys (e.g., civilcomments pku_better). Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write the CSV output (defaults to notebooks/results).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(BASELINE_METHODS.keys()),
        help="Limit to specific baselines.",
    )
    return parser.parse_args(argv)


def run(seed: int, output_dir: Path | None = None, datasets: list[str] | None = None, methods=None) -> Path:
    if bje is None:
        raise RuntimeError(
            "binary_judge_experiments could not be imported. "
            "Ensure src/ is available on PYTHONPATH before running binary_baselines.py."
        )

    random.seed(seed)
    np.random.seed(seed)

    selected = VALIDATION_DATASET_MAP
    if datasets:
        missing = [d for d in datasets if d not in VALIDATION_DATASET_MAP]
        if missing:
            raise ValueError(f"Unknown dataset(s): {', '.join(missing)}")
        selected = OrderedDict((d, VALIDATION_DATASET_MAP[d]) for d in datasets)

    results_dir = output_dir or (NOTEBOOK_DIR / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "binary_baselines.csv"

    records: list[dict[str, object]] = []
    for short_name, dataset_id in selected.items():
        try:
            judge_df, gold_labels = bje._load_binary_validation_data(
                dataset_id,
                data_root=BINARY_DATA_ROOT,
                judge_root=JUDGE_OUTPUTS_BINARY_ROOT,
            )
        except Exception as exc:  # pragma: no cover - defensive
            records.append(
                {
                    "dataset": short_name,
                    "method": "all",
                    "status": "error",
                    "message": f"load failed: {exc}",
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "coverage": 0.0,
                }
            )
            continue

        label_matrix = _prepare_binary_matrix(judge_df)
        gold = gold_labels.astype(int)
        print(f"{short_name}: {label_matrix.shape[0]} examples, {label_matrix.shape[1]} judges")

        for res in run_all_methods(label_matrix, gold, seed=seed, methods=methods):
            res.dataset = short_name
            records.append(
                {
                    "dataset": res.dataset,
                    "method": res.method,
                    "status": res.status,
                    "message": res.message,
                    "accuracy": res.accuracy,
                    "f1": res.f1,
                    "coverage": res.coverage,
                }
            )

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Wrote binary baselines to {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(seed=int(args.seed), output_dir=args.output_dir, datasets=args.datasets, methods=args.methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
