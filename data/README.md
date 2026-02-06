# Data Files (Cleaned)

This repository ships only the lightweight datasets used by the current pipeline.

## Included files

- `data/score/asset.csv`
  - Continuous human ratings for ASSET simplification examples.
  - Used by `scripts/fully_gaussian_main.py`.

- `data/binary/civilcomments.csv`
  - Binary toxicity labels (`label`) and raw comments (`text`).
  - Used by `scripts/gaussian_mixture_main.py` (classification setting).

- `data/preference/pku_better.csv`
  - Pairwise preference data (`question`, `response_A`, `response_B`, `gold_label_num`).
  - Used by `scripts/gaussian_mixture_main.py` (preference setting).

## Scope

These files are a small reproducibility subset aligned with the paper’s experiment types.
