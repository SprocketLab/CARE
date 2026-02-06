import math
import re
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import scipy as sp


def compute_mae(pred, true):
    return np.mean(np.abs(np.array(pred) - np.array(true)))


def compute_corr(pred, true):
    return np.corrcoef(np.array(pred), np.array(true))[0, 1]


def compute_weighted_predictions(df_subset, weights):
    """Return normalized weighted scores for the provided dataframe."""
    weights = np.asarray(weights, dtype=float)
    denom = weights.sum()
    if np.isclose(denom, 0):
        return None
    return df_subset.to_numpy() @ weights / denom


def collect_metrics(pred, true):
    """Compute MAE and Kendall tau for predictions, handling invalid inputs."""
    if pred is None:
        return np.nan, np.nan
    arr = np.asarray(pred, dtype=float)
    if np.isnan(arr).any():
        return np.nan, np.nan
    truth = np.asarray(true, dtype=float)
    mae = compute_mae(arr, truth)
    kendall = sp.stats.kendalltau(arr, truth).correlation
    return mae, kendall


def map_yes_no_mixed(value) -> int:
    """Map "yes"/"no" style responses to 1/0 and return -1 when absent."""
    if value is None:
        return -1
    if isinstance(value, (int, np.integer)):
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0
        return -1
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return -1
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0
        return -1
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return -1
        if normalized in {"yes", "y", "true", "1"}:
            return 1
        if normalized in {"no", "n", "false", "0"}:
            return 0
    return -1


_STRIP_CHARS = "\"'`*"
_ANSWER_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"therefore,?\s*the\s*answer\s*is[:\s]*", re.IGNORECASE),
    re.compile(r"final\s*answer[:\s]*", re.IGNORECASE),
    re.compile(r"the\s*answer\s*is[:\s]*", re.IGNORECASE),
)


def _coerce_to_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ""
    return str(value).strip()


def _strip_answer(text: str) -> str:
    cleaned = text.strip().strip(_STRIP_CHARS)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _extract_final_answer(response) -> str:
    text = _coerce_to_string(response)
    if not text:
        return ""
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = text[match.end():].strip()
            if candidate:
                first_line = candidate.splitlines()[0]
                return _strip_answer(first_line)
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return _strip_answer(stripped)
    return ""


def _canonicalize_answer(value) -> str:
    text = _coerce_to_string(value)
    if not text:
        return ""
    text = text.replace("，", ",").replace("。", ".")
    text = _strip_answer(text)
    text = text.rstrip(".!?:;, ")
    return text.lower()


def evaluate_multisubject_rlvr(records: Iterable[dict]):
    """Evaluate judge outputs for the Multi-subject RLVR dataset."""
    all_results = []
    subset_totals = defaultdict(lambda: {"n": 0, "correct": 0})
    judge_correct = 0
    response_correct = 0
    total = 0

    for row in records:
        reference = _strip_answer(_coerce_to_string(row.get("reference")))
        predicted = _extract_final_answer(row.get("response"))
        is_response_correct = int(_canonicalize_answer(predicted) == _canonicalize_answer(reference))
        vote = map_yes_no_mixed(row.get("parsed_output"))
        is_judge_correct = int(vote == is_response_correct) if vote in (0, 1) else 0

        all_results.append(
            {
                "question": row.get("question"),
                "subset": row.get("subset"),
                "subject": row.get("subject"),
                "attack_id": row.get("attack_id"),
                "response_correct": is_response_correct,
                "judge_vote": vote,
                "correct": is_judge_correct,
                "answer_extracted": predicted,
                "reference": reference,
            }
        )

        subset_key = row.get("subset") or "unknown"
        subset_stats = subset_totals[subset_key]
        subset_stats["n"] += 1
        subset_stats["correct"] += is_judge_correct

        response_correct += is_response_correct
        judge_correct += is_judge_correct
        total += 1

    overall_acc = judge_correct / total if total else np.nan
    response_acc = response_correct / total if total else np.nan
    acc_dict = {
        "overall": overall_acc,
        "judge_accuracy": overall_acc,
        "response_accuracy": response_acc,
    }
    per_subset = {
        subset: {
            "n": stats["n"],
            "accuracy": stats["correct"] / stats["n"] if stats["n"] else np.nan,
        }
        for subset, stats in subset_totals.items()
    }
    return all_results, acc_dict, per_subset
