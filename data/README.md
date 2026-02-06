# Data Asset Catalog

This catalog documents every CSV shipped in `data/`. For each asset we list the upstream dataset, the split that was copied, and any sampling or preprocessing performed before the file was materialised locally. Paths are relative to the repository root. Script and notebook names reference the helpers that regenerate the file today.

## Score datasets (continuous labels)

| Path | Upstream dataset & split | Sampling / preprocessing | Regeneration notes |
| --- | --- | --- | --- |
| `data/score/feedbackqa.csv` | FeedbackQA (`feedback_train.json`) | Textual rubric labels (Excellent/Acceptable/…) mapped to 1–4 and averaged across the two reviewers. | `src/data_tools.load_feedbackqa` (expects the original JSON payload).
| `data/score/helpsteer2.csv` | NVIDIA HelpSteer2 validation | Prompts/answers renamed to `question`/`answer`, helpfulness scores cast to `float`. No subsampling. | `src/data_tools.load_helpsteer2`.
| `data/score/helpsteer3_sampled.csv` | `nvidia/HelpSteer3` train | Sample ≤625 rows per domain (seed = 42), preserve both responses and the individual/overall preferences. | `scripts/helpsteer3_sampling.py`.
| `data/score/ultrafeedback_sampled.csv` | `argilla/ultrafeedback-binarized-preferences-cleaned` train | Group by `source`, sample ≤250 conversations per source (seed = 42), flatten chosen/rejected replies into separate rows with binary labels. | `notebooks/ultrafeedback_sampling.ipynb`.
| `data/score/review_5k_merged.csv` | `WestlakeNLP/Review-5K` train + test | Cache built by `load_review5k`; retains all metadata plus averaged review scores. | `src/data_tools.load_review5k` (`_build_review5k_cache`).
| `data/score/review_5k_minimal.csv.gz` | `WestlakeNLP/Review-5K` train + test | Compressed cache with the minimal columns needed by `load_review5k`; regenerated alongside the merged file. | `src/data_tools.load_review5k`.
| `data/score/asset.csv` | `facebook/asset` ratings split | Full set of 4 500 human simplification ratings. | Reload via `load_dataset('facebook/asset', 'ratings', split='full')`.
| `data/score/summarize_from_feedback.csv` | OpenAI Summarize-from-Feedback test | Entire public test split with model policy tags. | Download from the OpenAI release.
| `data/score/tripadvisor_reviews.csv` | `nhull/tripadvisor-split-dataset-v2` test | 5 000 sampled hotel reviews with 5-class sentiment labels (seed = 42). | Recreate via `load_dataset('nhull/tripadvisor-split-dataset-v2', split='test')` and sample 5 000 examples.
| `data/score/tripadvisor_reviews/samples.csv` | Same as above | Duplicate copy of the 5 000-example sample kept for exploratory work. | Same regeneration as the previous row.
| `data/score/yelp_with_scores.csv` | `Yelp/yelp_review_full` test | 5 000-sample with the original 0–4 star ratings preserved. | Recreate via `load_dataset('Yelp/yelp_review_full', split='test')` and sample 5 000 examples (seed = 42).

## Preference datasets (pairwise labels)

| Path | Upstream dataset & split | Sampling / preprocessing | Regeneration notes |
| --- | --- | --- | --- |
| `data/preference/chatbot_arena_sampled.csv` | `lmsys/chatbot_arena_conversations` train | Keep only the first two turns, drop examples lacking both answers, sample 5 000 comparisons (seed = 42), and emit numeric labels (`-1`, `0`, `1`) plus a binary view. | `_load_chatbot_arena_pairs` in `scripts/save_judge_outputs.py` (invoked with `sample_size=5000`).
| `data/preference/helpsteer3.csv` | `nvidia/HelpSteer3` train | Pairwise prompt with two responses and binary labels; this is the repository copy used by preference experiments (2317 rows after filtering). | Local cache; regenerate by re-exporting the same split from `nvidia/HelpSteer3` (no script checked in).
| `data/preference/judgebench.csv` | `ScalerLab/JudgeBench` `claude` + `gpt` splits | Concatenate both official splits and keep the provided `A>B`/`B>A` label plus a derived binary column. No subsampling. | `_load_judgebench_pairs` in `scripts/save_judge_outputs.py`.
| `data/preference/anthropic_harmless.csv` | AllenAI Preference Test Sets (`anthropic_harmless`) | Direct copy of the published evaluation split; no local sampling. | Download from the AllenAI Preference Test Sets release.
| `data/preference/anthropic_helpful.csv` | AllenAI Preference Test Sets (`anthropic_helpful`) | Direct copy of the published evaluation split; no local sampling. | Same as above.
| `data/preference/mtbench_gpt4.csv` | AllenAI Preference Test Sets (`mtbench_gpt4`) | Direct copy of the published evaluation split; no local sampling. | Same as above.
| `data/preference/mtbench_human.csv` | AllenAI Preference Test Sets (`mtbench_human`) | Direct copy of the published evaluation split; no local sampling. | Same as above.
| `data/preference/pku_better.csv` | AllenAI Preference Test Sets (`pku_better`) | Direct copy of the published evaluation split; no local sampling. | Same as above.
| `data/preference/pku_safer.csv` | AllenAI Preference Test Sets (`pku_safer`) | Direct copy of the published evaluation split; no local sampling. | Same as above.
| `data/preference/shp.csv` | AllenAI Preference Test Sets (`shp`) | Direct copy of the published evaluation split; no local sampling. | Same as above.
| `data/preference/summarize.csv` | AllenAI Preference Test Sets (`summarize`) | Direct copy of the published evaluation split; no local sampling. | Same as above.

## Binary classification datasets

| Path | Upstream dataset & split | Sampling / preprocessing | Regeneration notes |
| --- | --- | --- | --- |
| `data/binary/civilcomments.csv` | `civil_comments` test | Threshold toxicity ≥0.5 for the positive class, then sample 2 500 comments per class (seed = 42) and shuffle. | `src/data_tools.prepare_civilcomments`.
| `data/binary/yelp.csv` | `Yelp/yelp_review_full` (star ratings) | Local 5 000-example sample with balanced labels (2 500 per class) collapsed into a binary sentiment label. | Local cache; recreate by sampling the desired star buckets from `Yelp/yelp_review_full` (script not in repo).
| `data/binary/liar2.csv` | LIAR2 (PolitiFact statements) | Binary fact-check labels retained; no sampling metadata stored. | Local cache; regenerate from the LIAR2 release (script not in repo).

## Miscellaneous supporting datasets

| Path | Upstream dataset & split | Sampling / preprocessing | Regeneration notes |
| --- | --- | --- | --- |
| `data/miscellaneous/anthropic_hh_rlhf/hh_rlhf_sampled.csv` | `Anthropic/hh-rlhf` train | Random sample of 2 500 prompt/response pairs (seed = 42). | `scripts/hhrlhf_sampling.py`.
| `data/miscellaneous/hhrlhf.csv` | `Anthropic/hh-rlhf` train | Same 2 500-sample as above, trimmed to the two responses plus binary label for convenience. | Derived from the sampled file; regenerate by post-processing `hh_rlhf_sampled.csv`.
| `data/miscellaneous/tulu3/tulu3_8b_sampled.csv` | `allenai/llama-3.1-tulu-3-8b-preference-mixture` train | Sample ≤357 interactions per `source` (seed = 42) and retain the full JSON conversations alongside extracted chosen/rejected replies. | `scripts/tulu3_8b_sampling.py`.
| `data/miscellaneous/tulu3_8b.csv` | `allenai/llama-3.1-tulu-3-8b-preference-mixture` train | Simplified view of the sampled file with only `question`, `response_1/2`, and the binary label. | Derived from `tulu3/tulu3_8b_sampled.csv`.
| `data/miscellaneous/ultrafeedback_preference.csv` | Ultrafeedback preference data | 5 000-example, near-balanced sample of pairwise comparisons; stored as a local cache for preference baselines. | Local cache; regenerate from the Ultrafeedback release (script not in repo).

## Maintenance notes

- Files marked “local cache” do not yet have regeneration scripts checked in; if you rebuild them, please add the script path here for reproducibility.
- Judge output caches (`judge_outputs/…`) are tracked separately and are intentionally excluded from this inventory.
