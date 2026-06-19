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

### 1. Bigram Frequency (`freq`)

Aligned with [MarkBaggett/freq](https://github.com/MarkBaggett/freq) style analysis:

- Builds an English bigram frequency table from the training corpus
- Measures how "English-like" the string's character pairs are
- Low freq score → looks random; high → looks natural

### 2. Shannon Entropy (`entropy`)

```
H = -Σ p(c) * log2(p(c))
normalized = H / log2(unique_char_count)
```

High normalized entropy suggests random character distribution.

### 3. Compression Ratio (`compression`)

Uses raw DEFLATE (`zlib` with `wbits=-15`):

- Random data compresses poorly → high compression score
- Repetitive/natural text compresses well → low compression score

## Ensemble

A **logistic regression** pipeline:

```
StandardScaler → LogisticRegression (class_weight=balanced)
```

Training:

- **GridSearchCV** over `C ∈ {0.01, 0.1, 1.0, 10.0}` with 5-fold stratified CV
- **Platt calibration** (`CalibratedClassifierCV`, sigmoid method)
- ~50,000 natural + 50,000 synthetic random samples

Typical metrics after bootstrap:

| Metric | Value |
|--------|-------|
| Accuracy | ~99.2% |
| AUC | ~99.9% |
| Brier score | ~0.008 |

## Response Breakdown

Each score includes per-method breakdown (0–100):

```json
{
  "breakdown": {
    "freq": 91,
    "entropy": 97,
    "compression": 100
  }
}
```

## Confidence

`confidence` is `high` when the three breakdown values agree (low standard deviation), otherwise `low`.

Strings shorter than 4 characters always get `confidence: low`.

## Labels in API Responses

Beyond scoring labels, the API adds metadata fields:

| Field | When |
|-------|------|
| `excluded: true` | Matched exclusion rule |
| `cached: true` | Returned from score cache (no re-inference) |
| `skipped: true` | Either excluded or cache-hit |
| `skipped_reason` | `"excluded"` or `"score_cache_below_threshold"` |

## Python API

```python
from randomness_detection import Scorer

scorer = Scorer()  # auto-bootstrap on first use
result = scorer.score("hello")

print(result.score)       # 1-100
print(result.label)       # natural | uncertain | likely_random
print(result.breakdown)   # per-method scores
print(result.to_dict())   # JSON-serializable
```

## CLI Output

```
Score: 1
Label: natural
Confidence: high
Breakdown: freq=91, entropy=97, compression=100
```
