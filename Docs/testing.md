# Testing

All tests are **real integration tests** — no mocks, no fake data. They use the trained model, real English corpus, live API server, and cryptographic random tokens.

> For a detailed explanation of **how algorithm quality is measured** (the four-layer
> evaluation strategy, fair-comparison protocol, datasets, metrics, and pass criteria
> for each benchmark), see [Benchmark Methodology](benchmark-methodology.md).

## Test Suite

| Script | What it tests |
|--------|---------------|
| `test_exclude.py` | Exclusion rules, wildcards, score cache, API |
| `test_real_parallel.py` | Real corpus words, parallel API load, CPU usage |
| `test_cpu_full.py` | Training (50% CPU), inference, API CPU verification |
| `test_benchmark.py` | Real throughput benchmark (CLI, pool, exclude, API) |
| `test_quality_benchmark.py` | Detection quality vs freq/entropy/compression baselines |
| `run_real_tests.py` | Runs all tests sequentially |

## Prerequisites

Bootstrap must have run at least once:

```bash
PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap
```

This creates `.cache_test/` or `~/.cache/randomness_detection/` with the trained model.

## Run All Tests

```bash
cd randomness_detection
PYTHONUNBUFFERED=1 \
RANDOMNESS_PARALLEL_BACKEND=hybrid \
RANDOMNESS_INFERENCE_WORKERS=24 \
PYTHONPATH=. .venv/bin/python run_real_tests.py
```

Expected output:

```
ALL REAL TESTS PASSED
```

## Individual Tests

### Exclusion Tests

```bash
PYTHONPATH=. .venv/bin/python test_exclude.py
```

Tests:

- Unit: domain, suffix, prefix, glob matching
- API: add/remove exclusion rules
- Domain exclusion (`skipme.com` blocks subdomains)
- Suffix wildcard (`*.trusted.org`)
- Exact exclusion
- Score cache hit (natural word scored once, cached on second call)
- Batch mixed results (excluded + cached + scored)
- `/exclude/check` endpoint

Expected: `OVERALL: PASS`

### Real Parallel Test

```bash
RANDOMNESS_INFERENCE_WORKERS=24 \
PYTHONPATH=. .venv/bin/python test_real_parallel.py
```

Tests:

- 20,000 real words from `words_alpha.txt`
- 15-second sustained API load (singles + batches)
- System CPU monitoring via `/proc/stat`
- Worker process detection
- No mock strings

Expected: `OVERALL: PASS`

This is the **CPU-scaling proof**: it confirms the load actually fans out across the
configured cores. Sweep the worker count to see it scale (48-core machine):

| `RANDOMNESS_INFERENCE_WORKERS` | Live worker procs | System CPU peak |
|--------------------------------|-------------------|-----------------|
| 6  | 8  | 17.0% |
| 24 | 26 | 53.9% |
| 48 | 50 | 100%  |

Worker processes track the setting (the extra ~2 are `forkserver`/manager helpers) and
peak CPU rises toward full saturation — see
[Benchmark Methodology §4b](benchmark-methodology.md#4b-parallel-and-cpu-scaling-verification).

### Full CPU Verification

```bash
PYTHONPATH=. .venv/bin/python test_cpu_full.py
```

Three phases:

| Phase | Duration | Checks |
|-------|----------|--------|
| 1. Training | ~45s | 50% CPU, worker processes |
| 2. Inference | 12s | Process pool CPU, throughput |
| 3. API | 12s | Real corpus batch requests |

Expected: `OVERALL: ALL PASS`

**Note:** Phase 1 re-trains the model (`force=True`). Takes ~80 seconds total.

### Benchmark

```bash
RANDOMNESS_INFERENCE_WORKERS=24 \
PYTHONPATH=. .venv/bin/python test_benchmark.py
```

Measures CLI scoring, inference pool, exclude pre-filter (50K rules), and API single/batch throughput. Writes `benchmark_results.json`. Expected: `OVERALL: PASS`

See [Benchmarks in README](README.md#benchmarks) for reference numbers on a 48-core machine.

### Quality benchmark

```bash
PYTHONPATH=. .venv/bin/python test_quality_benchmark.py
```

Compares **randomness_detection** against freq, entropy, compression, avg3, max3, and the external tools **Mark Baggett freq.py** (git clone), **Fourmilab ent** (`apt install ent`), and a standalone DEFLATE CLI. Each method gets its own threshold tuned on a calibration split; metrics are on held-out test data. Writes `quality_benchmark_results.json`. Expected: `QUALITY CHECK: PASS`

### Robustness stress test (anti-band-aid guardrail)

```bash
PYTHONPATH=. .venv/bin/python test_robustness.py
```

Hand-curated **real-world** strings that are deliberately **not** produced by the training generator, so the result cannot be gamed by training-data tricks. It verifies the model generalizes — real identifiers, brands, hyphenated phrases, and digit-suffixed names must stay `natural`; UUIDs, SHAs, base64, API keys, and DGA-style labels must be flagged `random`. Expected: `ROBUSTNESS: PASS` (core_natural FP ≤ 10%, clear_random FN ≤ 5%). Short out-of-dictionary brands and concatenated word-salad are reported as diagnostics only.

### Real-world public-dataset test (production use case)

```bash
PYTHONPATH=. .venv/bin/python test_real_world_data.py
```

Downloads two **real public datasets** (cached under `.benchmark_tools/realworld/`) and runs the actual job the scorer exists for — separating legitimate domains from algorithmically-generated (DGA) ones:

- **Legit:** [Tranco top-1M](https://tranco-list.eu/) registrable labels.
- **DGA:** [andrewaeva/DGA](https://github.com/andrewaeva/DGA) — 800K+ real malware domains; families 1/2/3/5/6/8 are random-character, families 4/7 are dictionary-DGA (concatenated real words).

Reported (real data, last run): random-character DGA recall **95.4%**, clean solvable ROC-AUC **0.966** (legit vs random-char DGA after independent freqpy de-noising). The top-list "legit" label is intrinsically noisy — freqpy independently flags ~16% of it as random (CDN/hash/telemetry domains) — so a high raw FP is expected and the noise-robust criteria are used. Dictionary-DGA recall is now **42–65%** (families 4 & 7), lifted from a ~1–7% structural ceiling by the `word_count` feature. Expected: `REAL-WORLD: PASS` (random-char recall ≥ 95%, clean AUC ≥ 0.95).

## Test Data

| Data | Source | Mock? |
|------|--------|-------|
| English words | `words_alpha.txt` (downloaded corpus) | No |
| Random tokens | `secrets.token_hex()` | No (real crypto random) |
| API server | Real FastAPI + uvicorn subprocess | No |
| Model | Trained `ensemble.pkl` | No |
| Exclusion DB | Real SQLite per test run | No |

## Environment Variables for Tests

```bash
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export RANDOMNESS_PARALLEL_BACKEND=hybrid
export RANDOMNESS_INFERENCE_WORKERS=24
```

## Test Helpers

`test_helpers.py` provides shared utilities:

- `load_real_words(cache_dir)` — load corpus words
- `build_real_text_batch(words, size)` — mixed corpus + random batch
- `pick_natural_word(words)` — word with length ≥ 5
- `pick_random_token()` — `secrets.token_hex(16)`

## Logs

Tests write logs to the project root:

| File | Content |
|------|---------|
| `exclude_test_run.log` | Latest exclusion test output |
| `all_real_tests.log` | Full test suite output |
| `real_parallel_test.log` | Parallel test output |
| `cpu_full_test_fixed.log` | CPU verification output |

## Troubleshooting

### `Model cache missing, run bootstrap first`

```bash
PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap
```

### Port already in use

Tests use ports 8777, 8788, 8790. Kill stale servers:

```bash
pkill -f "randomness_detection.api_server"
```

### Test timeout

`test_cpu_full.py` has a 180-second per-phase timeout. If training takes longer, check CPU availability.

### Exclude test cache collision

`test_exclude.py` creates a fresh `exclude_test_<random>.db` per run to avoid score cache interference.

## Benchmarking Exclusion Speed

```bash
PYTHONPATH=. .venv/bin/python -c "
from pathlib import Path
from randomness_detection.exclude import ExcludeManager
import time, secrets

m = ExcludeManager.open(Path('.cache_test'), db_name='bench.db')
m.add_rules([(f'block{i}.example.com', 'domain') for i in range(50_000)])
samples = ['https://app.block25000.example.com'] + [secrets.token_hex(8) for _ in range(999)]
start = time.perf_counter()
for t in samples * 10: m.check_exclude(t)
print(f'{10000/(time.perf_counter()-start):,.0f} checks/sec')
m.close()
"
```

Expected: **60,000–100,000 checks/second** with 50K rules.

## CI Integration

```bash
#!/bin/bash
set -euo pipefail
cd randomness_detection
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export RANDOMNESS_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap
PYTHONPATH=. .venv/bin/python run_real_tests.py
```
