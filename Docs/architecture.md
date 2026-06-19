# Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Request                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI (asyncio)          Auth middleware                     │
│  /score  /score/batch  /exclude  /health                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Fast Path (no inference CPU)                                   │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ ExcludeManager   │ →  │ Score Cache      │                   │
│  │ SQLite + Trie    │    │ SQLite lookup    │                   │
│  └──────────────────┘    └──────────────────┘                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ only if not excluded/cached
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  InferencePool (hybrid / process / thread)                       │
│  ProcessPoolExecutor → Scorer.score() per worker                │
│  ThreadPoolExecutor  → concurrent API dispatch                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Scorer                                                         │
│  features.py → freq + entropy + compression                      │
│  ensemble.pkl → logistic regression probability → score 1-100     │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Bootstrap & Training (`bootstrap.py`, `trainer.py`)

Runs once (or on `--bootstrap`) to:

1. Download English word lists from [dwyl/english-words](https://github.com/dwyl/english-words)
2. Validate corpus safety (`corpus_validator.py`)
3. Build bigram frequency table in parallel
4. Generate synthetic random training samples
5. Train calibrated logistic regression ensemble

Training uses **50% of CPU cores** by default (`CPU_FRACTION=0.5`).

### Feature Extraction (`features.py`, `freq_model.py`)

Per string:

| Feature | Method |
|---------|--------|
| Frequency | Bigram analysis vs English freq table |
| Entropy | Shannon entropy / log2(unique chars) |
| Compression | Raw DEFLATE ratio (`wbits=-15`) |

### Inference Pool (`inference_pool.py`)

Handles CPU-bound scoring for the API. Bypasses Python GIL via `ProcessPoolExecutor`.

Three backends — see [Parallel Processing](parallel-processing.md).

### Exclusion Layer (`exclude/`)

Pre-inference filter. See [Exclusion System](exclusion.md).

- **SQLite** for exact/domain rules and score cache (millions of rows)
- **In-memory suffix trie** for wildcard domain patterns
- **Prefix/glob matchers** for pattern rules

### API Layer (`api/`)

- `app.py` — FastAPI routes and lifespan
- `scoring.py` — batch pre-filter then inference
- `auth.py` — API key validation (constant-time)
- `models.py` — Pydantic schemas

## Request Flow (Score)

1. **Validate** request body (length, no null bytes)
2. **Authenticate** API key (unless disabled for dev)
3. **Exclude check** — SQLite + in-memory trie (~sub-ms per item)
4. **Score cache check** — return cached result if score ≤ threshold
5. **Inference** — only remaining texts go to process pool
6. **Store cache** — save newly scored results for future skip
7. **Return** enriched response with `excluded`, `cached`, `skipped` flags

## Cache Directory Layout

```
~/.cache/randomness_detection/
├── words_alpha.txt      # Corpus
├── words.txt
├── english.freq         # Bigram table
├── ensemble.pkl         # Trained model
├── metadata.json        # Version + metrics
└── exclude.db           # Exclusion rules + score cache (SQLite WAL)
```

## Process Model

| Layer | Technology | Purpose |
|-------|------------|---------|
| HTTP | asyncio + uvicorn | Concurrent connections |
| Pre-filter | sync SQLite (thread-locked) | Exclude/cache lookup |
| Scoring | ProcessPoolExecutor | CPU-bound inference |
| Dispatch | ThreadPoolExecutor | Hybrid batch coordination |
| Training | ProcessPool + sklearn joblib | Bootstrap pipeline |

## Design Principles

1. **Fail fast** — excluded/cached items never touch the inference pool
2. **Fork-safe** — uses `forkserver` context after joblib/sklearn to avoid deadlocks
3. **Modular** — scorer, exclude, API, and training are independent modules
4. **Real data only** — training uses English corpus + cryptographic synthetic random (no malicious datasets)
