# LRD-Hybrid — Research-Grade Randomness Detector

**LRD-Hybrid** (*Linguistic Randomness Detector — Hybrid*) extends the production
scorer with signals designed for Q1-grade evaluation:

| Layer | Signal | Purpose |
|-------|--------|---------|
| Statistical | bigram freq, entropy, DEFLATE | character-random strings |
| Lexical | coverage, word count, segmentation | compounds vs word-salad |
| **Language model** | character n-gram perplexity | natural language plausibility |
| **PMI** | adjacent word co-occurrence | dictionary-DGA detection |
| Ensemble | HistGradientBoosting + calibration | non-linear fusion |

## Quick start

```bash
# Train hybrid model (uses production freq table + new LM/PMI/ensemble)
PYTHONPATH=. .venv/bin/python research_train.py --verbose

# Score a string
PYTHONPATH=. .venv/bin/python -c "
from randomness_detection.research import HybridScorer
s = HybridScorer()
print(s.score('hello'))
print(s.score('theirtheandaloneinto'))
"

# Paper benchmark: ablation + vs production
PYTHONPATH=. .venv/bin/python research_benchmark.py
PYTHONPATH=. .venv/bin/python research_benchmark.py --quick
```

## Artifacts (in cache dir)

| File | Content |
|------|---------|
| `hybrid_lm.pkl` | Character 5-gram LM |
| `hybrid_pmi.pkl` | Word bigram PMI model |
| `hybrid_ensemble.pkl` | Calibrated gradient-boosting classifier |
| `hybrid_metadata.json` | Training metrics + version |

## Ablation groups (for paper Table)

Defined in `randomness_detection/research/hybrid_features.py`:

- `lrd_hybrid_full` — all feature groups
- `lrd_hybrid_no_lm` — remove language-model perplexity
- `lrd_hybrid_no_pmi` — remove segmentation PMI
- `lrd_hybrid_no_lexical` — remove lexical/segmentation features
- `lrd_hybrid_statistical_only` — freq + entropy + compression only

Run `research_benchmark.py` to reproduce all rows on the same held-out split.

## Paper claim template

> We propose LRD-Hybrid, an interpretable multi-signal framework combining
> statistical randomness tests, dictionary segmentation, character-level language
> model perplexity, and word co-occurrence PMI, fused by a calibrated gradient-
> boosting ensemble. On a held-out benchmark with per-method threshold calibration,
> LRD-Hybrid improves F1 and ROC-AUC over logistic-regression and entropy
> baselines, with ablations confirming that LM and PMI features primarily improve
> dictionary-DGA and short-string discrimination.

## What you still need for Q1 submission

This code provides the **method + reproducible experiments**. A Q1 paper also needs:

1. Comparison with **5–10 published DGA/randomness methods** (not only internal baselines)
2. Evaluation on **multiple public datasets** (Tranco, andrewaeva/DGA, UTL_DGA22)
3. **Statistical significance** (bootstrap CI or paired tests)
4. **Error analysis** section with FP/FN examples
5. Related work survey (40+ references)

See [Tests & Metrics Explained](tests-and-metrics-explained.md) and
[Benchmark Methodology](benchmark-methodology.md) for the existing four-layer protocol.

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
```
