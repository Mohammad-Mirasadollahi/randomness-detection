# LRD-Hybrid — Research-Grade Randomness Detector

**LRD-Hybrid** (*Linguistic Randomness Detector — Hybrid*) extends the production
scorer with signals designed for paper-grade evaluation:

| Layer | Signal | Purpose |
|-------|--------|---------|
| Statistical | bigram freq, entropy, DEFLATE | character-random strings |
| Lexical | coverage, word count, segmentation | compounds vs word-salad |
| **Language model** | character n-gram perplexity | natural language plausibility |
| **PMI** | adjacent word co-occurrence | dictionary-DGA detection |
| Ensemble | HistGradientBoosting + calibration | non-linear fusion |

> **Note:** LRD-Hybrid is a **research module**. The production API and `install.sh`
> use the logistic-regression ensemble. Train and score Hybrid separately (below).

---

## Measured results (reproducible on this repo)

Hold-out set: **1,821 strings** (784 natural, 1,037 random) — same protocol as
`test_quality_benchmark.py` (979 calibration + per-method threshold tuning).

### Quality benchmark — synthetic hold-out

| Method | F1 | ROC-AUC | FPR | FNR |
|--------|-----|---------|-----|-----|
| **lrd_hybrid** | **1.000** | **1.000** | 0% | 0% |
| randomness_detection (product) | 1.000 | 1.000 | 0% | 0% |
| freqpy (external) | 0.987 | 0.999 | 1.3% | 1.6% |

Both models hit the ceiling here. Gains appear on robustness and dictionary-DGA.

### Robustness — hand-curated real-world strings

Same buckets as `test_robustness.py` (data **not** from the training generator):

| Bucket | Production | LRD-Hybrid |
|--------|------------|------------|
| `core_natural` FP rate | 8% | **5%** |
| `clear_random` FN rate | 0% | 0% |

### Dictionary-DGA — 4+ word salad (curated)

| Model | Recall (4+ segmented words) |
|-------|----------------------------|
| Production | ~75% |
| **LRD-Hybrid** | **100%** |

Examples that Hybrid catches: `theirtheandaloneinto`, `whenwherehowwhatwhy`,
`youonehimoutnow`.

**Diagnostic only (not pass/fail):** 3-word salad such as `boxcarmittenglow` —
structurally similar to short compounds; both models struggle.

### Ablation study (`research_benchmark.py`)

| Variant | F1 | ROC-AUC | FPR |
|---------|-----|---------|-----|
| lrd_hybrid_full | **1.000** | **1.000** | 0% |
| lrd_hybrid_no_lm | 1.000 | 1.000 | 0% |
| lrd_hybrid_no_pmi | 1.000 | 1.000 | 0% |
| lrd_hybrid_no_lexical | 1.000 | 1.000 | 0% |
| lrd_hybrid_statistical_only | 0.999 | 1.000 | 0.26% |
| production_lr | 0.999 | 1.000 | 0.13% |

On this synthetic split, LM/PMI ablations stay at F1 1.000 — improvement is
primarily visible on **robustness FP** and **dictionary-DGA**, not headline F1.

Results saved to `research_benchmark_results.json` and `quality_benchmark_results.json`.

---

## Quick start

```bash
# Train hybrid model (uses production freq table + new LM/PMI/ensemble)
PYTHONPATH=. .venv/bin/python research_train.py --verbose

# Integration tests (smoke + robustness + quality + DGA)
PYTHONPATH=. .venv/bin/python test_research_hybrid.py

# Ablation benchmark vs production
PYTHONPATH=. .venv/bin/python research_benchmark.py
PYTHONPATH=. .venv/bin/python research_benchmark.py --quick

# Score a string
PYTHONPATH=. .venv/bin/python research_score.py "theirtheandaloneinto" --json
```

Python API:

```python
from randomness_detection.research import HybridScorer

scorer = HybridScorer()
print(scorer.score("hello"))                    # natural
print(scorer.score("theirtheandaloneinto"))     # likely_random
```

---

## Artifacts (in cache dir)

| File | Content |
|------|---------|
| `hybrid_lm.pkl` | Character 5-gram LM |
| `hybrid_pmi.pkl` | Word bigram PMI model |
| `hybrid_ensemble.pkl` | Calibrated gradient-boosting classifier |
| `hybrid_metadata.json` | Training metrics + version |

Artifacts are written **atomically** (`.tmp` + rename) to avoid corruption when
training runs overlap.

---

## Ablation groups (for paper Table)

Defined in `randomness_detection/research/hybrid_features.py`:

| Group | Features |
|-------|----------|
| `statistical` | freq, entropy, compression |
| `structural` | length, digit ratio, vowel ratio, … |
| `lexical` | coverage, word count, longest word |
| `language_model` | cross-entropy, log-prob, perplexity |
| `pmi` | mean/min PMI, pair count |

Variants: `lrd_hybrid_full`, `_no_lm`, `_no_pmi`, `_no_lexical`, `_statistical_only`.

---

## Paper claim template (honest)

> We propose LRD-Hybrid, an interpretable multi-signal framework combining
> statistical randomness tests, dictionary segmentation, character-level language
> model perplexity, and word co-occurrence PMI, fused by a calibrated gradient-
> boosting ensemble. On a held-out synthetic benchmark both production and Hybrid
> reach F1 1.000; Hybrid reduces robustness false positives (8% → 5%) and improves
> dictionary-DGA recall on 4+ word salad (75% → 100%) on curated real-world strings.

---

## What you still need for Q1 submission

This code provides the **method + reproducible experiments**. A Q1 paper also needs:

1. Comparison with **5–10 published DGA/randomness methods** (not only internal baselines)
2. Evaluation on **multiple public datasets** (Tranco, andrewaeva/DGA, UTL_DGA22)
3. **Statistical significance** (bootstrap CI or paired tests)
4. **Error analysis** section with FP/FN examples
5. Related work survey (40+ references)

See [Tests & Metrics Explained](tests-and-metrics-explained.md) and
[Benchmark Methodology](benchmark-methodology.md).

---

## Module layout

```
randomness_detection/research/
  ngram_lm.py          # Character n-gram LM + perplexity
  pmi_model.py         # Word PMI from natural compounds
  segmentation.py      # Dictionary DP segmentation
  hybrid_features.py   # Full feature vector + ablation groups
  hybrid_trainer.py    # HistGradientBoosting + calibration
  hybrid_scorer.py     # Scoring API
  hybrid_bootstrap.py  # End-to-end training pipeline
  io_utils.py          # Atomic pickle writes
```
