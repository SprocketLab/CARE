# Judge Output Layout (Cleaned)

Judge outputs are stored by experiment setting:

- `judge_outputs/fully_gaussian/asset/`
- `judge_outputs/gaussian_mixture/civilcomments/`
- `judge_outputs/gaussian_mixture/pku_better/`

Each file is one judge model output CSV.

For `pku_better`, this repo keeps a compressed archive
`judge_outputs/gaussian_mixture/allenai_preference_test_sets_pku_better.tar.gz`.
`scripts/gaussian_mixture_main.py` auto-extracts it on first use.
