#!/usr/bin/env python3
"""Generate LLM judge outputs for the lightweight reproducibility datasets.

Supported datasets
------------------
- ``asset_ratings`` (alias: ``asset``)
- ``civilcomments_binary`` (alias: ``civilcomments``)
- ``allenai_preference_test_sets/pku_better_binary`` (aliases: ``pku_better_binary``, ``pku_better``)

These outputs feed the two experiment entrypoints documented in ``README.md``:
- ``scripts/fully_gaussian_main.py``
- ``scripts/gaussian_mixture_main.py``
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional runtime dependency
    LLM = None
    SamplingParams = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SRC_DIR = REPO_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from llm_tools import (  # noqa: E402
    ASSET_SIMPLIFICATION_JUDGE_PROMPT,
    CIVILCOMMENTS_BINARY_JUDGE_PROMPT,
    CIVILCOMMENTS_SCORE_JUDGE_PROMPT,
    JUDGEBENCH_PREF_NO_TIE_PROMPT,
    PREFERENCE_BINARY_JUDGE_PROMPT,
)


DEFAULT_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "microsoft/Phi-4-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

DATASET_ALIASES = {
    "asset": "asset_ratings",
    "asset_ratings": "asset_ratings",
    "civilcomments": "civilcomments_binary",
    "civilcomments_binary": "civilcomments_binary",
    "pku_better": "allenai_preference_test_sets/pku_better_binary",
    "pku_better_binary": "allenai_preference_test_sets/pku_better_binary",
    "allenai_preference_test_sets/pku_better_binary": "allenai_preference_test_sets/pku_better_binary",
}

SIGNED_PREF_MAP = {
    0: -3,
    1: -2,
    2: -1,
    4: 1,
    5: 2,
    6: 3,
}


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def parse_first_int_in_range(text: str, min_value: int, max_value: int) -> Optional[int]:
    if not text:
        return None
    rating_match = re.search(r"TOTAL\s*RATING\s*:\s*([+-]?\d+)", text, flags=re.IGNORECASE)
    candidates = []
    if rating_match:
        candidates.append(rating_match.group(1))
    candidates.extend(re.findall(r"[+-]?\d+", text))
    for token in candidates:
        try:
            value = int(token)
        except ValueError:
            continue
        if min_value <= value <= max_value:
            return value
    return None


def parse_binary_rating(text: str) -> Optional[int]:
    return parse_first_int_in_range(text, 0, 1)


def parse_civilcomments_score(text: str) -> Optional[int]:
    return parse_first_int_in_range(text, 0, 9)


def parse_asset_rating(text: str) -> Optional[int]:
    return parse_first_int_in_range(text, 0, 100)


def parse_pref_signed_score(text: str) -> Optional[int]:
    raw = parse_first_int_in_range(text, 0, 6)
    if raw is None:
        return None
    return SIGNED_PREF_MAP.get(raw)


def run_transformers_generation(model_name: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Greedy generation fallback when vLLM cannot load a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    outputs: list[str] = []
    batch_size = 4
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for prompt, text in zip(batch_prompts, decoded):
            outputs.append(text[len(prompt) :].strip() if text.startswith(prompt) else text.strip())

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return outputs


def run_generation(model_name: str, prompts: list[str], max_new_tokens: int) -> Optional[list[str]]:
    engine = None
    try:
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vLLM is not installed")

        # Qwen3 still has incomplete vLLM support in some local environments.
        if "Qwen3" in model_name:
            raise RuntimeError("forcing transformers fallback for Qwen3")

        engine = LLM(
            model=model_name,
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        )
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        )
        outputs = engine.generate(prompts, params)
        return [item.outputs[0].text.strip() for item in outputs]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] vLLM failed for {model_name}: {exc}; using transformers fallback", flush=True)
        try:
            return run_transformers_generation(model_name, prompts, max_new_tokens=max_new_tokens)
        except Exception as tf_exc:  # noqa: BLE001
            print(f"[error] generation failed for {model_name}: {tf_exc}", flush=True)
            return None
    finally:
        if engine is not None:
            del engine
        torch.cuda.empty_cache()
        gc.collect()


def fully_gaussian_output_dir(dataset_name: str) -> Path:
    out = REPO_ROOT / "judge_outputs" / "fully_gaussian" / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def mode_output_dir(mode: str, dataset_name: str) -> Path:
    root = "gaussian_mixture" if mode == "gaussian_mixture" else "binary"
    out = REPO_ROOT / "judge_outputs" / root / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_asset_ratings(models: list[str]) -> None:
    frame = pd.read_csv(REPO_ROOT / "data" / "score" / "asset.csv")
    prompts = [
        ASSET_SIMPLIFICATION_JUDGE_PROMPT.format(
            aspect=row.get("aspect", ""),
            original=str(row.get("original", "")),
            simplification=str(row.get("simplification", "")),
        )
        for _, row in frame.iterrows()
    ]

    output_dir = fully_gaussian_output_dir("asset")
    for model_name in models:
        raw = run_generation(model_name, prompts, max_new_tokens=4)
        if raw is None:
            continue
        parsed = [parse_asset_rating(text) for text in raw]
        out = frame.copy()
        out["raw_output"] = raw
        out["parsed_output"] = parsed
        out["model"] = model_slug(model_name)
        out.to_csv(output_dir / f"{model_slug(model_name)}.csv", index=False)


def run_civilcomments_binary(models: list[str], mode: str) -> None:
    frame = pd.read_csv(REPO_ROOT / "data" / "binary" / "civilcomments.csv")
    texts = frame["text"].astype(str).tolist()

    if mode == "gaussian_mixture":
        prompt_template = CIVILCOMMENTS_SCORE_JUDGE_PROMPT
        parser: Callable[[str], Optional[int]] = parse_civilcomments_score
        max_new_tokens = 8
    else:
        prompt_template = CIVILCOMMENTS_BINARY_JUDGE_PROMPT
        parser = parse_binary_rating
        max_new_tokens = 6

    prompts = [prompt_template.format(text=text) for text in texts]
    output_dir = mode_output_dir(mode, "civilcomments")

    for model_name in models:
        raw = run_generation(model_name, prompts, max_new_tokens=max_new_tokens)
        if raw is None:
            continue

        parsed = [parser(text) for text in raw]
        out = frame[["text", "label"]].copy()
        out["raw_output"] = raw
        out["parsed_output"] = parsed
        if mode == "gaussian_mixture":
            out["pred_label_binary"] = [None if value is None else int(value >= 5) for value in parsed]
        else:
            out["pred_label_binary"] = parsed
        out["scoring_mode"] = mode
        out["model"] = model_slug(model_name)
        out.to_csv(output_dir / f"{model_slug(model_name)}.csv", index=False)


def run_pku_better_binary(models: list[str], mode: str, swap_ab: bool, seed: int) -> None:
    rng = random.Random(seed)
    frame = pd.read_csv(REPO_ROOT / "data" / "preference" / "pku_better.csv")
    frame = frame.copy()
    frame["gold_label_binary"] = (pd.to_numeric(frame["gold_label_num"], errors="coerce") > 0).astype("Int64")

    question = frame["question"].astype(str).tolist()
    original_a = frame["response_A"].astype(str).tolist()
    original_b = frame["response_B"].astype(str).tolist()

    response_a: list[str] = []
    response_b: list[str] = []
    was_swapped: list[bool] = []
    for resp_a, resp_b in zip(original_a, original_b):
        do_swap = bool(swap_ab and rng.random() < 0.5)
        if do_swap:
            response_a.append(resp_b)
            response_b.append(resp_a)
        else:
            response_a.append(resp_a)
            response_b.append(resp_b)
        was_swapped.append(do_swap)

    if mode == "gaussian_mixture":
        prompt_template = JUDGEBENCH_PREF_NO_TIE_PROMPT
        parser = parse_pref_signed_score
        max_new_tokens = 8
    else:
        prompt_template = PREFERENCE_BINARY_JUDGE_PROMPT
        parser = parse_binary_rating
        max_new_tokens = 6

    prompts = [
        prompt_template.format(question=q, answer_a=a, answer_b=b)
        for q, a, b in zip(question, response_a, response_b)
    ]

    output_dir = mode_output_dir(mode, "pku_better")
    for model_name in models:
        raw = run_generation(model_name, prompts, max_new_tokens=max_new_tokens)
        if raw is None:
            continue

        score_ab: list[Optional[int]] = [parser(text) for text in raw]
        score_original: list[Optional[int]] = []
        pref_labels: list[str] = []
        pred_binary: list[Optional[int]] = []

        for idx, value in enumerate(score_ab):
            if value is None:
                score_original.append(None)
                pref_labels.append("")
                pred_binary.append(None)
                continue

            if mode == "gaussian_mixture":
                orig = -value if was_swapped[idx] else value
                score_original.append(orig)
                if orig < 0:
                    pref_labels.append("A")
                    pred_binary.append(0)
                elif orig > 0:
                    pref_labels.append("B")
                    pred_binary.append(1)
                else:
                    pref_labels.append("tie")
                    pred_binary.append(None)
            else:
                orig = 1 - value if was_swapped[idx] else value
                score_original.append(orig)
                if orig == 0:
                    pref_labels.append("A")
                    pred_binary.append(0)
                elif orig == 1:
                    pref_labels.append("B")
                    pred_binary.append(1)
                else:
                    pref_labels.append("")
                    pred_binary.append(None)

        out = frame.copy()
        out["response_a"] = response_a
        out["response_b"] = response_b
        out["was_swapped"] = was_swapped
        out["raw_output"] = raw
        out["score_ab"] = score_ab
        out["score_original_order"] = score_original
        out["pref_A_or_B"] = pref_labels
        out["pred_label_binary"] = pred_binary
        out["scoring_mode"] = mode
        out["model"] = model_slug(model_name)
        out.to_csv(output_dir / f"{model_slug(model_name)}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate judge outputs for the light CARE datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Dataset keys to generate. Supported: "
            "asset_ratings, civilcomments_binary, allenai_preference_test_sets/pku_better_binary "
            "(plus aliases asset, civilcomments, pku_better)."
        ),
    )
    parser.add_argument("--models", nargs="+", default=None, help="Override model list.")
    parser.add_argument("--mode", choices=["binary", "gaussian_mixture"], default="gaussian_mixture")
    parser.add_argument("--swap-ab", action="store_true", help="Randomly swap A/B in pairwise prompts.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for swap-ab decisions.")
    return parser.parse_args()


def normalize_requested_datasets(names: Iterable[str]) -> list[str]:
    requested = list(names)
    if "all" in requested:
        return [
            "asset_ratings",
            "civilcomments_binary",
            "allenai_preference_test_sets/pku_better_binary",
        ]

    normalized: list[str] = []
    seen: set[str] = set()
    for name in requested:
        key = DATASET_ALIASES.get(str(name).strip().lower())
        if key is None:
            raise ValueError(f"Unsupported dataset: {name}")
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized


def main() -> None:
    args = parse_args()
    models = args.models or DEFAULT_MODELS
    datasets = normalize_requested_datasets(args.datasets)

    for dataset in datasets:
        print(f"[run] dataset={dataset} mode={args.mode}")
        if dataset == "asset_ratings":
            run_asset_ratings(models)
        elif dataset == "civilcomments_binary":
            run_civilcomments_binary(models, mode=args.mode)
        elif dataset == "allenai_preference_test_sets/pku_better_binary":
            run_pku_better_binary(
                models,
                mode=args.mode,
                swap_ab=bool(args.swap_ab),
                seed=int(args.seed),
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    main()
