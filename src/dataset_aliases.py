"""Canonical dataset names and accepted aliases for this cleaned repository."""

from __future__ import annotations

from typing import Callable, Iterable


FULLY_GAUSSIAN_DATASETS = [
    "asset",
]

GAUSSIAN_MIXTURE_DATASETS = [
    "civilcomments",
    "pku_better",
]


FULLY_GAUSSIAN_ALIASES = {
    "asset": "asset",
    "asset_ratings": "asset",
}

GAUSSIAN_MIXTURE_ALIASES = {
    "civilcomments": "civilcomments",
    "civilcomments_binary": "civilcomments",
    "pku_better": "pku_better",
    "pku_better_binary": "pku_better",
    "allenai_preference_test_sets/pku_better": "pku_better",
    "allenai_preference_test_sets/pku_better_binary": "pku_better",
}


def normalize_fully_gaussian_dataset(name: str) -> str:
    key = str(name).strip().lower()
    return FULLY_GAUSSIAN_ALIASES.get(key, key)


def normalize_gaussian_mixture_dataset(name: str) -> str:
    key = str(name).strip().lower()
    return GAUSSIAN_MIXTURE_ALIASES.get(key, key)


def normalize_dataset_list(
    names: Iterable[str],
    normalizer: Callable[[str], str],
) -> list[str]:
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        canon = normalizer(name)
        if canon not in seen:
            seen.add(canon)
            ordered_unique.append(canon)
    return ordered_unique
