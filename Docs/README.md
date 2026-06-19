# Randomness Detection Documentation

English documentation for the **randomness_detection** package — a modular system that detects how random a string looks on a **1–100 scale** (higher = more random).

**License:** [MIT](../LICENSE)

## Documentation Index

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Install, bootstrap, first score |
| [Architecture](architecture.md) | System design and request flow |
| [Scoring Model](scoring-model.md) | Features, ensemble, labels, breakdown |
| [CLI](cli.md) | Command-line usage |
| [API Reference](api-reference.md) | REST endpoints, request/response schemas |
| [Authentication](authentication.md) | API key setup and security |
| [Parallel Processing](parallel-processing.md) | Multiprocessing, threading, hybrid mode |
| [Exclusion System](exclusion.md) | Domain exclude, wildcards, score cache |
| [Configuration](configuration.md) | Environment variables and defaults |
| [Training & Bootstrap](training-and-bootstrap.md) | Corpus, training pipeline, cache layout |
| [Testing](testing.md) | Real integration tests (no mocks) |
| [Benchmark Methodology](benchmark-methodology.md) | How algorithm quality is measured across all benchmarks |
| [Benchmarks](#benchmarks) | Real throughput numbers (this page) |

## Benchmarks

### Detection quality (randomness_detection vs baselines)

Hold-out test set: **1,821 strings** (784 natural, 1,037 random) from real corpus words + cryptographic random tokens.  
Calibration: **979 strings** used only to tune each method's decision threshold (max F1).

**External tools** (installed/downloaded automatically by `test_quality_benchmark.py`):

| Tool | Source | Role |
|------|--------|------|
| freqpy | [Mark Baggett freq.py](https://github.com/MarkBaggett/freq) — git clone | Bigram frequency baseline |
| ent | [Fourmilab ent](https://fourmilab.ch/random/) — `apt install ent` | Shannon entropy via CLI |
| deflate_cli | `benchmark_tools/deflate_score.py` — subprocess | Raw DEFLATE compression ratio |

```bash
PYTHONPATH=. .venv/bin/python test_quality_benchmark.py
```

| Method | Source | Accuracy | F1 | ROC-AUC | FPR on natural | FNR on random | Threshold |
|--------|--------|----------|-----|---------|----------------|---------------|-----------|
| **randomness_detection** | product | **100.0%** | **1.000** | **1.000** | **0.1%** | **0.0%** | 23 |
| freqpy (Mark Baggett) | external | 98.5% | 0.986 | 0.999 | 1.3% | 1.7% | 95 |
| freq (internal bigram) | internal | 98.4% | 0.986 | 0.999 | 2.7% | 0.8% | 94 |
| avg3 (mean of 3 signals) | internal | 95.8% | 0.964 | 0.990 | 6.9% | 2.1% | 97 |
| ent (Fourmilab CLI) | external | 66.1% | 0.756 | 0.739 | 68.1% | 8.0% | 35 |
| deflate_cli / entropy / compression alone | — | ~57% | ~0.726 | ~0.50 | 100% | 0% | — |

Entropy, compression, and standalone DEFLATE cannot separate short natural English from random strings (scores overlap ~97–100). **ent** alone is also weak on short strings. Frequency-based methods work better but still miss more naturals than **randomness_detection**.

#### Why this is a root-cause result, not benchmark-fitting

The synthetic benchmark above shares its generator family with training, so a high score there is necessary but **not sufficient**. The real risk of adding a lexical-coverage feature plus richer training data is introducing *new* errors on real-world strings. Two failure modes are specifically guarded against:

1. **Surface-form shortcut.** An earlier iteration added "random" samples decorated with mixed case / separators / digits. The model learned `mixed-case + separator ⇒ random` and then flagged legitimate identifiers (`getUserById`), hyphenated phrases (`the-quick-brown-fox`), and digit-suffixed names (`annual_report_2024`) — even at 100% dictionary coverage. **Root fix:** the same surface decoration (casing, `-`/`_`/`.`, digit affixes) is now applied *identically to both classes* during training, so surface form carries **zero** label information. The model must judge the linguistic nature of the underlying tokens.
2. **Partial-coverage identifiers.** Natural training samples used to be built only from whole dictionary words (coverage ≈ 1.0), so real identifiers mixing words with abbreviations (`api`, `json`, `v3`) fell into an unseen middle region and were flagged random. **Root fix:** ~30% of natural samples now include a short non-dictionary affix (added only when the real-word content dominates, so coverage stays high and the affix does not collide with short pronounceable random tokens).
3. **Dictionary word-salad vs short compounds.** Lexical coverage alone cannot tell a real 2-word name (`wikipedia`, `hello-world`) from concatenated dictionary-DGA (`theirtheandaloneinto`) — both reach ~100% coverage. The distinguishing signal is **word count**: legitimate compounds use 1–3 words (length ≈ 7–10), while dictionary-DGA concatenates 4–7 words (length ≈ 23–27). **Root fix:** a `word_count` feature (minimal dictionary words in the DP segmentation) was added, and ~20% of random training samples are now 4–7-word salads, so the model learns "many concatenated words ⇒ machine-like" without relying on surface form. The feature is the **raw, uncapped count** — the pipeline's `StandardScaler` normalizes it, and a monotonic count (no hard ceiling) keeps the signal correct for *any* number of concatenated words, from 1-word names to 20-word salad, well beyond the training range.

This is validated by an **independent, hand-curated stress test** whose data is *not* produced by the training generator (`test_robustness.py`):

```bash
PYTHONPATH=. .venv/bin/python test_robustness.py
```

| Bucket | Metric | Result |
|--------|--------|--------|
| core_natural (words, identifiers, phrases, digit-names) | false-positive rate | **8%** |
| clear_random (UUIDs, SHAs, base64, API keys, DGA) | false-negative rate | **0%** |

Across the redesign the real-world false-positive rate fell **21% → 15% → 8%** while held-out FN stayed at **0%** — generalization improving from principled data changes, not test tuning.

**Honest limitations** (reported as diagnostics, not gamed away): short out-of-dictionary brand/jargon tokens (`nvidia`, `figma`, `nginx`) are *fundamentally* hard — they are not reliably separable from structure alone and would require a named-entity/brand list (memorization). The `word_count` signal now catches concatenated dictionary-word salad, but this is a genuine trade-off: human-chosen 4-word passphrases (`correcthorsebatterystaple`) and 4-word phrases (`the-quick-brown-fox`) are structurally identical to 4-word dictionary-DGA and therefore lean `random` — correct for a security/randomness detector, borderline as "natural." Under production thresholds (`natural ≤ 30`, `likely_random ≥ 60`) most of these land in the `uncertain` band rather than a hard misclassification.

### Real-world public-dataset validation (DGA detection)

The ultimate real-data test: the actual production job, on real public datasets that have nothing to do with the training pipeline (`test_real_world_data.py`).

```bash
PYTHONPATH=. .venv/bin/python test_real_world_data.py
```

- **Legit:** [Tranco top-1M](https://tranco-list.eu/) registrable labels.
- **DGA:** [andrewaeva/DGA](https://github.com/andrewaeva/DGA) — 800K+ real malware domains.

| Measure (real data) | Result |
|---------------------|--------|
| Random-character DGA recall (real malware) | **95.4%** |
| Legit vs random-char DGA, clean ROC-AUC | **0.966** |
| Dictionary-DGA recall (families 4 & 7) | **42–65%** (was ~1–7% before the word-count signal) |
| Top-list "legit" contamination, flagged by independent freqpy | ~16% |

The top-domain "legit" label is intrinsically noisy: traffic-ranked lists contain many machine-generated labels (CDN nodes, hashes, telemetry). The **independent** Mark Baggett freqpy tool flags ~16% of the legit sample as random, confirming the model is correct on most of its raw "false positives." After freqpy de-noising, legit vs random-char DGA separates at **ROC-AUC 0.966**. The `word_count` feature lifted dictionary-DGA recall from a near-zero structural ceiling (~1–7%) to **42–65%** by learning that 4+ concatenated words is machine-like; the residual miss (4-word salads overlapping real short compounds) is an inherent ambiguity, reported not hidden.

### Throughput (speed)

Run it yourself:

```bash
cd randomness_detection
PYTHONUNBUFFERED=1 \
RANDOMNESS_PARALLEL_BACKEND=hybrid \
RANDOMNESS_INFERENCE_WORKERS=24 \
PYTHONPATH=. .venv/bin/python test_benchmark.py
```

Results are printed to the terminal and saved to `benchmark_results.json`.

| Benchmark | Throughput | Notes |
|-----------|------------|-------|
| CLI batch scoring | **219 texts/s** | Single-process `Scorer`, 2,000 mixed corpus + random strings |
| Inference pool | **876 texts/s** | 10s sustained load, process pool + async dispatch |
| Exclude pre-filter | **81,646 checks/s** | 50,000 domain rules, 10,000 lookups (no ML inference) |
| API `POST /score` | **211 req/s** | 800 requests, concurrency 32; p50 **52 ms**, p95 **80 ms** |
| API `POST /score/batch` | **2,943 items/s** | 40 batches × 40 items; p50 **150 ms**, p95 **195 ms** |

Exclude and score-cache hits skip the inference pool entirely — useful when filtering millions of known domains or re-scoring cached low-score strings.

#### CPU scaling under load

`test_real_parallel.py` confirms that heavy concurrent traffic actually fans out across the configured cores. On a 48-core machine, sweeping `RANDOMNESS_INFERENCE_WORKERS` shows live worker processes and peak CPU rise with the setting (6 → 17% peak, 24 → 54%, 48 → 100%). Full method and table: [Parallel Processing → Verified CPU Scaling](parallel-processing.md#verified-cpu-scaling).

## Quick Overview

**One-command install:**

```bash
chmod +x install.sh && ./install.sh
```

The scorer combines several signals:

1. **Bigram frequency** (aligned with [freq.py](https://github.com/MarkBaggett/freq) style analysis)
2. **Shannon entropy** (normalized by unique character count)
3. **Raw DEFLATE compression ratio** (`wbits=-15`)
4. **Lexical coverage** — DP word-segmentation against a 414K-word dictionary (how much of the string decomposes into real words)
5. **Word count** — minimal number of dictionary words in the segmentation (separates short real compounds from concatenated dictionary-DGA / word-salad)
6. **Structure signals** — longest dictionary-word fragment, vowel ratio, longest consonant run

A **logistic regression ensemble** with Platt calibration merges these into a single **1–100 score**.

## Typical Workflow

```bash
# 1. Install
cd randomness_detection
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Bootstrap (download corpus + train models)
export RANDOMNESS_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap

# 3. Run API server
PYTHONPATH=. .venv/bin/python -m randomness_detection.api_server --host 0.0.0.0 --port 8765

# 4. Score a string
curl -s -X POST http://127.0.0.1:8765/score \
  -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

## Package Layout

```
randomness_detection/
├── randomness_detection/       # Python package
│   ├── api/                 # FastAPI app
│   ├── exclude/             # Fast exclusion + score cache
│   ├── scorer.py            # Core scoring API
│   ├── bootstrap.py         # Auto training pipeline
│   ├── inference_pool.py    # Parallel API inference
│   └── parallel.py          # Training parallelism
├── Docs/                    # This documentation
├── test_*.py                # Real integration tests
├── test_benchmark.py        # Real throughput benchmark
├── test_quality_benchmark.py  # Detection quality vs baselines
└── requirements.txt
```

## Support

- Interactive API docs: `http://localhost:8765/docs` (when server is running)
- OpenAPI schema: `http://localhost:8765/openapi.json`
