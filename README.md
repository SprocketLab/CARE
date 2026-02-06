# CARE: Confounder-Aware Aggregation for Reliable LLM Evaluation

[Jitian Zhao*](https://jzhao326.github.io/), [Changho Shin*](https://ch-shin.github.io/), [Tzu-Heng Huang](https://zihengh1.github.io/), [Satya Sai Srinath Namburi GNVV](https://namburisrinath.github.io/index.html), [Frederic Sala](https://pages.cs.wisc.edu/~fredsala/)

Paper Link: [TBD](https://arxiv.org/abs/2412.03881)

## Abstract
LLM-as-a-judge ensembles are the standard paradigm for scalable evaluation, but their aggregation mechanisms suffer from a fundamental flaw: they implicitly assume that judges provide independent estimates of true quality. However, in practice, LLM judges exhibit correlated errors caused by shared latent confounders---such as verbosity, stylistic preferences, or training artifacts---causing standard aggregation rules like majority vote or averaging to provide little gain or even amplify systematic mistakes. To address this, we introduce CARE, a confounder-aware aggregation framework that explicitly models LLM judge scores as arising from both a latent true-quality signal and shared confounding factors. Rather than heuristically re-weighting judges, CARE separates quality from confounders without access to ground-truth labels. We provide theoretical guarantees for identifiability and finite-sample recovery under shared confounders, and we quantify the systematic bias incurred when aggregation models omit confounding latent factors. Across 12 public benchmarks spanning continuous scoring, binary classification, and pairwise preference settings, CARE improves aggregation accuracy, reducing error by up to 26.8%.

## Installation
TODO

## Running experiments
TODO

## Citation
TBD
