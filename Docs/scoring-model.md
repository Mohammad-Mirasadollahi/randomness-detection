# Scoring Model

## Score Scale

| Range | Label | Meaning |
|-------|-------|---------|
| 1–30 | `natural` | Looks like normal language |
| 31–59 | `uncertain` | Ambiguous |
| 60–100 | `likely_random` | Looks random or machine-generated |
| 0 | `excluded` | Matched an exclusion rule (API only) |

**Higher score = more random.**

## Feature Signals

### Statistical

1. **Bigram frequency (`freq`)** — aligned with [MarkBaggett/freq](https://github.com/MarkBaggett/freq) style analysis
2. **Shannon entropy (`entropy`)** — normalized by unique character count
3. **DEFLATE compression ratio (`compression`)** — raw DEFLATE (`zlib` with `wbits=-15`)

### Lexical & structural

4. **Lexical coverage** — DP word-segmentation against the training dictionary
5. **Word count** — minimal number of dictionary words in the segmentation (compounds vs word-salad)
6. **Structure** — vowel ratio, consonant runs, digit/uppercase ratios, base64-like shape

### Language model & PMI

7. **Character LM perplexity** — 5-gram model trained on the corpus; high perplexity ⇒ less language-like
8. **Word PMI** — pointwise mutual information between adjacent segmented words; low PMI ⇒ concatenated salad

## Ensemble

**HistGradientBoostingClassifier** with sigmoid calibration:

```
StandardScaler → HistGradientBoosting → CalibratedClassifierCV (sigmoid)
```

Training (~50,000 natural + 50,000 synthetic random samples):

- GridSearchCV over `max_depth` and `learning_rate` (5-fold stratified CV, ROC-AUC)
- Character LM and PMI models fit on the same corpus before ensemble training

Typical metrics after bootstrap:

| Metric | Value |
|--------|-------|
| F1 | ~1.000 |
| ROC-AUC | ~1.000 |
| Brier score | ~0.009 |

## Response Breakdown

Each score includes per-component breakdown (0–100):

```json
{
  "breakdown": {
    "freq": 91,
    "entropy": 97,
    "compression": 100,
    "language_model": 85,
    "pmi": 72,
    "lexical": 15
  }
}
```

## Confidence

`confidence` is `high` when breakdown components spread by ≥25 points; otherwise `low`.  
Strings shorter than 4 characters always get `confidence: low`.

## Cache artifacts

| File | Content |
|------|---------|
| `english.freq` | Bigram frequency table + lexicon |
| `language_model.pkl` | Character 5-gram LM |
| `word_pmi.pkl` | Word bigram PMI |
| `ensemble.pkl` | Calibrated gradient-boosting classifier |
| `metadata.json` | Bootstrap version + training metrics |

## Python API

```python
from randomness_detection import Scorer

scorer = Scorer()  # auto-bootstrap on first use
result = scorer.score("hello")

print(result.score)       # 1-100
print(result.label)       # natural | uncertain | likely_random
print(result.breakdown)   # per-component scores
print(result.to_dict())   # JSON-serializable
```

## CLI Output

```
Score: 1
Label: natural
Confidence: high
Breakdown: freq=91, entropy=97, compression=100, language_model=12, pmi=8, lexical=5
```

## Ablation

Remove feature groups one at a time to measure contribution:

```bash
PYTHONPATH=. .venv/bin/python ablation_benchmark.py
```

Variants: `ensemble_full`, `ensemble_no_lm`, `ensemble_no_pmi`, `ensemble_no_lexical`, `ensemble_statistical_only`.
