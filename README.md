# CARE: Confounder-Aware Aggregation for Reliable LLM Evaluation

[Jitian Zhao*](https://jzhao326.github.io/), [Changho Shin*](https://ch-shin.github.io/), [Tzu-Heng Huang](https://zihengh1.github.io/), [Satya Sai Srinath Namburi GNVV](https://namburisrinath.github.io/index.html), [Frederic Sala](https://pages.cs.wisc.edu/~fredsala/)

Paper Link: [TBD](https://arxiv.org/abs/2412.03881)

<img width="1024" height="225" alt="image" src="https://github.com/user-attachments/assets/8b964d70-127f-4aad-a4cc-847512dfac60" />

## Install

```bash
pip install -r requirements.txt
```

## Run pipeline

### 1) Generate LLM judge outputs

```bash
python scripts/save_judge_outputs.py \
  --datasets asset_ratings civilcomments_binary allenai_preference_test_sets/pku_better_binary \
  --mode gaussian_mixture
```

Output path example: `judge_outputs/fully_gaussian/asset/Qwen3-8B.csv`

### 2) Run aggregations

Fully Gaussian (table 1 experiment):

```bash
python scripts/fully_gaussian_main.py --seed 2024
```

Gaussian mixture (table 2 experiment):

```bash
python scripts/gaussian_mixture_main.py --seed 42 --datasets civilcomments pku_better
```

## Citation

TBD
