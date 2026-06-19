# Testing

All tests are **real integration tests** — no mocks, no fake data. They use the trained model, real English corpus, live API server, and cryptographic random tokens.

> **Plain-language guide:** [Tests & Metrics Explained](tests-and-metrics-explained.md)  
> **Technical protocol:** [Benchmark Methodology](benchmark-methodology.md)

## Test Suite

| Script | What it tests |
|--------|---------------|
| `test_quality_benchmark.py` | Detection quality vs external/internal baselines |
| `test_robustness.py` | Hand-curated real-world stress test |
| `test_ensemble.py` | Combined smoke + quality + robustness + dictionary-DGA |
| `ablation_benchmark.py` | Feature-group ablation on the ensemble |
| `test_real_world_data.py` | Real Tranco + malware DGA datasets |
| `test_exclude.py` | Exclusion rules, wildcards, score cache, API |
| `test_real_parallel.py` | Parallel API load, CPU usage |
| `test_cpu_full.py` | Training, inference, API under CPU load |
| `test_benchmark.py` | Throughput (CLI, pool, exclude, API) |
| `run_real_tests.py` | Runs exclude + parallel + cpu_full sequentially |

## Prerequisites

Bootstrap must have run at least once:

```bash
./install.sh
# or
PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap
```

## Quality benchmark

```bash
PYTHONPATH=. .venv/bin/python test_quality_benchmark.py
```

Compares **randomness_detection** against freqpy, ent, deflate_cli, and internal baselines.  
Expected: `QUALITY CHECK: PASS` — F1 beats all baselines, ROC-AUC ≥ 0.95.

| Method | F1 | ROC-AUC |
|--------|-----|---------|
| randomness_detection | 1.000 | 1.000 |
| freqpy | 0.987 | 0.999 |

## Robustness

```bash
PYTHONPATH=. .venv/bin/python test_robustness.py
```

Expected: `ROBUSTNESS: PASS` — core_natural FP ≤ 10%, clear_random FN ≤ 5%.

| Metric | Result |
|--------|--------|
| core_natural FP | **5%** |
| clear_random FN | **0%** |

## Ensemble integration test

```bash
PYTHONPATH=. .venv/bin/python test_ensemble.py
```

Smoke, quality hold-out, robustness, and dictionary-DGA (4+ word salad) in one run.  
Expected: `ENSEMBLE TEST: PASS`

## Ablation

```bash
PYTHONPATH=. .venv/bin/python ablation_benchmark.py --quick
```

Expected: `ABLATION BENCHMARK: PASS`

## Output files

| File | Content |
|------|---------|
| `quality_benchmark_results.json` | Quality vs baselines |
| `ablation_benchmark_results.json` | Feature-group ablation |
| `benchmark_results.json` | Throughput |

See [Benchmarks in Docs/README.md](README.md#benchmarks) for reference numbers.
