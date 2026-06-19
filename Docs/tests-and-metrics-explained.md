# Tests & Metrics — Plain-Language Guide

This page explains **what each test does** and **what the numbers mean** (F1, ROC-AUC, FP, FN, etc.).  
For the full technical protocol, see [Benchmark Methodology](benchmark-methodology.md).

---

## What the product measures

Every string gets a **score from 1 to 100**:

| Score | Meaning |
|-------|---------|
| **Low (≈ 1–30)** | Looks like natural language (English words, identifiers, phrases) |
| **Mid (≈ 31–59)** | Uncertain |
| **High (≈ 60–100)** | Looks random (hex, gibberish, machine tokens, many DGA domains) |

In tests we often treat **score ≥ 50** as “random” for pass/fail counting. Production labels use wider bands (`natural ≤ 30`, `likely_random ≥ 60`).

**Two classes in benchmarks:**

- **Natural (label 0)** — real words, brands, identifiers, legit domains  
- **Random (label 1)** — crypto tokens, UUIDs, malware DGA domains, gibberish  

---

## Metrics glossary

### Classification metrics (Quality benchmark)

| Metric | What it means | Good value |
|--------|---------------|------------|
| **Accuracy** | Fraction of all strings classified correctly | Higher is better (can be misleading if classes are imbalanced) |
| **Precision** | Of strings we called *random*, how many were truly random | Higher → fewer false alarms on natural text |
| **Recall** | Of all truly *random* strings, how many we caught | Higher → fewer random strings slip through |
| **F1** | Single score balancing precision and recall (harmonic mean). **1.0 = perfect**, 0 = useless | **≥ 0.95** is strong; README shows **1.000** for the product |
| **ROC-AUC** | How well the method *separates* natural vs random **across all possible thresholds**. Independent of where you draw the line. **0.5 = coin flip**, **1.0 = perfect separation** | **≥ 0.95** is strong; README shows **1.000** for the product |

**Why both F1 and ROC-AUC?**

- **F1** depends on the chosen threshold (e.g. score ≥ 50). It answers: “At our operating point, how good are we?”
- **ROC-AUC** answers: “Regardless of threshold, how separable are the two classes?” Useful when methods use different score scales (entropy vs ensemble).

### Error types (Robustness & real-world tests)

| Term | Full name | Meaning | Bad when… |
|------|-----------|---------|-----------|
| **FP** | False Positive | A **natural** string scored as random | Users see real words/domains flagged wrongly |
| **FN** | False Negative | A **random** string scored as natural | Random/malicious strings slip through |
| **FPR** | False Positive **Rate** | FP ÷ all natural strings (e.g. **8%** = 8 in 100 naturals wrong) | Too high → too many false alarms |
| **FNR** | False Negative **Rate** | FN ÷ all random strings (e.g. **0%** = no random missed) | Too high → security gap |

### Detection on real malware data

| Metric | What it means |
|--------|---------------|
| **Recall (DGA)** | Of real malware **DGA domains**, what fraction were scored as random (high score) |
| **Clean ROC-AUC** | Separation between legit domains and random-character DGA after removing noisy “legit” labels (CDN/hash domains). **Honest quality number on real data** |

### Speed metrics (Throughput benchmark)

| Metric | What it means |
|--------|---------------|
| **Throughput** | How many strings (or API requests) processed **per second** — higher is faster |
| **p50 (median latency)** | Half of requests finish faster than this (e.g. **p50 52 ms** for API) |
| **req/s** | API requests per second under load |

---

## Test scripts — what each one is

### Install smoke tests (run after `./install.sh`)

| Script | Purpose |
|--------|---------|
| `test_install_smoke.sh` | Isolated install: venv, model, CLI scoring, API `/health` + `/score` + batch, then stop |
| `test_oneline_install_smoke.sh` | Same as above but simulates `curl … \| bash` (git clone + install path) |

**Pass line:** `OVERALL: PASS`

---

### Layer 1 — Quality benchmark

**Script:** `test_quality_benchmark.py`

**Question:** Is our algorithm **better than known baselines** (freq, entropy, compression, external tools)?

**How it works (short):**

1. Builds a **held-out test set** (natural words/compounds vs random hex/url-safe/consonant strings).
2. Each method gets its **own best threshold** tuned on a calibration split (fair comparison).
3. Reports **F1**, **ROC-AUC**, FPR, FNR on the test split only.
4. Compares against **freqpy**, **ent**, **deflate_cli**, and internal signals.

**Pass:** `QUALITY CHECK: PASS` — product F1 beats all baselines and ROC-AUC ≥ 0.95.

**Example README numbers:**

| Method | F1 | ROC-AUC |
|--------|-----|---------|
| randomness_detection | 1.000 | 1.000 |
| freqpy (external) | 0.986 | 0.999 |
| ent alone | 0.756 | 0.739 |

---

### Layer 2 — Robustness stress test

**Script:** `test_robustness.py`

**Question:** Does the model work on **hand-curated real strings** it never saw in training? (Anti–“teaching to the test”.)

**Buckets:**

| Bucket | Examples | Pass criterion |
|--------|----------|----------------|
| `core_natural` | `getUserById`, `hello-world`, `stackoverflow` | **FPR ≤ 10%** |
| `clear_random` | UUIDs, SHAs, base64, mock API-key shapes, DGA-like labels | **FNR ≤ 5%** |
| `hard_natural` | `nvidia`, `figma` (short brands) | Diagnostic only |
| `adversarial_*` | Word-salad random, passphrases | Diagnostic only |

**Pass:** `ROBUSTNESS: PASS`

**Example README numbers:** core_natural FP **8%**, clear_random FN **0%**

---

### Layer 3 — Real-world DGA detection

**Script:** `test_real_world_data.py`

**Question:** On **real public datasets**, can we separate legit domains from malware **DGA** domains?

**Data:**

- **Legit:** [Tranco top-1M](https://tranco-list.eu/)
- **Malware DGA:** [andrewaeva/DGA](https://github.com/andrewaeva/DGA) (800K+ domains)

**Key metrics:**

| Metric | Meaning | Example result |
|--------|---------|----------------|
| Random-char DGA recall | % of character-random malware domains caught | **95.4%** |
| Clean ROC-AUC | Legit vs solvable DGA after noise cleanup | **0.966** |
| Dictionary-DGA recall | Word-salad DGAs (harder) | **42–65%** |

**Pass:** `REAL-WORLD: PASS` (recall ≥ 95%, clean AUC ≥ 0.95)

---

### Layer 4 — Speed / throughput

**Script:** `test_benchmark.py`

**Question:** Is the system **fast enough** for production load?

| Benchmark | What it measures | Example |
|-----------|------------------|---------|
| CLI batch | Scoring via command line | 219 texts/s |
| Inference pool | Internal worker pool | 876 texts/s |
| Exclude pre-filter | 50K exclusion rules | 81,646 checks/s |
| API `POST /score` | Single live HTTP requests | 211 req/s, p50 52 ms |
| API `POST /score/batch` | Batch API | 2,943 items/s |

**Pass:** `OVERALL: PASS`

---

### Integration & infrastructure tests

| Script | What it verifies |
|--------|------------------|
| `test_exclude.py` | Exclusion rules (domain/suffix/exact), score cache, live API |
| `test_real_parallel.py` | Parallel API load, CPU usage, real corpus words |
| `test_cpu_full.py` | Training/inference/API under sustained CPU load |
| `run_real_tests.py` | Runs exclude + parallel + cpu_full sequentially |
| `validate_and_train.py` | Full validate → train → score pipeline |
| `verify_cpu_usage.py` | CPU utilization during bootstrap |

---

## Quick reference — which number should I care about?

| Your goal | Look at |
|-----------|---------|
| “Is the algorithm good vs alternatives?” | **F1**, **ROC-AUC** in `test_quality_benchmark.py` |
| “Will it flag my real code/words wrongly?” | **FPR** in `test_robustness.py` (`core_natural`) |
| “Will random/malware slip through?” | **FNR** in robustness; **DGA recall** in real-world test |
| “Is it fast enough?” | **throughput**, **p50** in `test_benchmark.py` |
| “Did install work?” | `test_install_smoke.sh` / `test_oneline_install_smoke.sh` |

---

## How to run

```bash
# Install smoke tests
./test_install_smoke.sh
./test_oneline_install_smoke.sh

# Quality layers 1–4
PYTHONPATH=. .venv/bin/python test_quality_benchmark.py
PYTHONPATH=. .venv/bin/python test_robustness.py
PYTHONPATH=. .venv/bin/python test_real_world_data.py
PYTHONPATH=. .venv/bin/python test_benchmark.py

# Full integration suite
PYTHONPATH=. .venv/bin/python run_real_tests.py
```

See [Testing](testing.md) for prerequisites and expected output.
