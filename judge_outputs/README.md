# Judge Output Layout

The `judge_outputs/` directory now groups model outputs by the downstream pipeline that consumes them:

- `fully_gaussian/` – continuous-score experiments used in `notebooks/score_caresl_main_exp.ipynb`.  Each dataset keeps its original name (for example `fully_gaussian/feedbackqa/`) and any archived copies (`*.tar.gz`) live alongside the CSVs.
- `binary/` – all pairwise/binary judgements formerly under `judge_outputs_binary/`, plus the datasets loaded by `notebooks/tensor_binary_with_validation.ipynb` (for example `binary/civilcomments/`, `binary/helpsteer3/`, `binary/judgebench/`, `binary/chatbot_arena_conversations/`, and the AllenAI preference splits such as `binary/anthropic_harmless/`).
- `gaussian_mixture/` – outputs used in `notebooks/real_exp_tensor_end2end_score2cls_with_validation_set.ipynb` together with everything that used to sit under `judge_outputs_gaussian_mixtures/`.  This includes the mixture runs for `civilcomments`, `yelp`, and `liar2`, plus the AllenAI preference splits (e.g. `gaussian_mixture/anthropic_harmless/`).
- `miscellaneous/` – legacy or ad-hoc runs that are not part of the main scoring pipelines (Anthropic HH-RLHF variants, injected-bias sweeps, masterkey experiments, etc.).

If a dataset has both raw CSVs and archived tarballs, the archives stay beside the data inside the appropriate category directory.  Future judge runs should be dropped into the category that matches the pipeline you plan to evaluate.
