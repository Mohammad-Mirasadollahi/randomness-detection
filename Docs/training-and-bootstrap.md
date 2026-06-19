# Training & Bootstrap

The bootstrap pipeline automatically prepares everything needed for scoring: corpus download, frequency table, and ensemble model training.

## Triggering Bootstrap

Bootstrap runs automatically when:

1. `Scorer()` is created and cache is missing or outdated
2. API server starts (`lifespan` calls `Scorer(auto_bootstrap=True)`)
3. CLI is invoked with `--bootstrap`

Force re-training:

```bash
PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap
```

## Pipeline Steps

```
1. Download word lists
       ↓
2. Validate corpus safety
       ↓
3. Build freq table (parallel)
       ↓
4. Generate training data (parallel)
       ↓
5. Extract features (parallel)
       ↓
6. Train ensemble (sklearn GridSearchCV + calibration)
       ↓
7. Save models + metadata
```

### Step 1: Corpus Download

Sources (from [dwyl/english-words](https://github.com/dwyl/english-words)):

| File | URL |
|------|-----|
| `words_alpha.txt` | alphabetic words only |
| `words.txt` | full word list |

~466K unique words after merge.

### Step 2: Corpus Validation

`corpus_validator.py` checks for:

- Minimum word length
- Unsafe patterns (base64-like strings, etc.)
- Training eligibility

Bootstrap **fails** if validation does not pass.

### Step 3: Frequency Table

`english.freq` — bigram frequency table built from eligible words using parallel tally (`parallel.py`).

### Step 4: Training Data

| Class | Source | Count |
|-------|--------|-------|
| Natural (0) | Real English words + compounds | 50,000 |
| Random (1) | Cryptographic synthetic (`secrets`, `uuid`, etc.) | 50,000 |

No external malicious datasets are used.

### Step 5: Feature Extraction

Parallel feature extraction for all 100,000 samples using the freq table.

### Step 6: Model Training

```
StandardScaler → LogisticRegression
  ├── GridSearchCV (C: 0.01, 0.1, 1.0, 10.0)
  └── CalibratedClassifierCV (Platt sigmoid)
```

Uses 50% of CPU cores. Training pool is released before sklearn spawns joblib workers.

### Step 7: Save Artifacts

| File | Content |
|------|---------|
| `ensemble.pkl` | Pickled sklearn pipeline |
| `metadata.json` | Version, metrics, corpus stats |

## Metadata Example

```json
{
  "version": 5,
  "natural_sources": ["words_alpha.txt", "words.txt"],
  "random_source": "synthetic",
  "total_words_loaded": 466550,
  "training_words_available": 415000,
  "samples_per_class": 50000,
  "natural_real_word_ratio": 0.7,
  "cpu_workers": 24,
  "thread_workers": 24,
  "parallel_backend": "hybrid",
  "metrics": {
    "accuracy": 0.992,
    "auc": 0.999,
    "brier_score": 0.008,
    "best_C": 1.0,
    "calibration": "sigmoid"
  }
}
```

## Version Management

`BOOTSTRAP_VERSION` in `config.py` controls cache invalidation. When the version increases, bootstrap re-runs automatically on next startup.

Current version: **5**

## Cache Directory

```
~/.cache/randomness_detection/
├── words_alpha.txt
├── words.txt
├── english.freq
├── ensemble.pkl
├── metadata.json
└── exclude.db          # separate — managed by exclusion system
```

## Training Performance

On a 48-core machine:

| Metric | Value |
|--------|-------|
| Duration | ~40–50 seconds |
| CPU usage | ~50% peak |
| Worker processes | ~24–28 |
| Accuracy | ~99.2% |

## Parallel Training Details

| Task | Backend |
|------|---------|
| Word tally | Process pool (hybrid) |
| Synthetic generation | Process pool |
| Feature extraction | Process pool (chunked) |
| GridSearchCV | sklearn joblib |
| Calibration | sklearn joblib |

`shutdown_joblib()` is called after training to prevent resource leaks before inference pools start.

## Safety

- **Natural data only** from public English word lists
- **Random data** generated locally with `secrets` module
- **No malicious URL/domain datasets**
- Corpus validator blocks suspicious patterns

## Custom Cache Directory

```bash
export RANDOMNESS_CACHE_DIR=/data/randomness_detection
python -m randomness_detection --bootstrap --cache-dir /data/randomness_detection
```

Multiple API instances can share the same cache directory (read-only model files). The exclusion database (`exclude.db`) should be per-instance or use SQLite WAL with care for concurrent writes.
