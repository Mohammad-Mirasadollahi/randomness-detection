# Parallel Processing

The system supports **multiprocessing**, **threading**, and **hybrid** execution for both training and API inference.

## Backends

| Backend | Description | Best for |
|---------|-------------|----------|
| `process` | `ProcessPoolExecutor` only | CPU-bound scoring (bypasses GIL) |
| `thread` | `ThreadPoolExecutor` only | Light workloads, per-thread Scorer |
| `hybrid` | Both pools active (default) | Production API throughput |

Set via environment variable:

```bash
export RANDOMNESS_PARALLEL_BACKEND=hybrid   # process | thread | hybrid
```

## API Inference

### Architecture (hybrid mode)

```
HTTP request (asyncio)
    │
    ▼
FastAPI async handler
    │
    ├── Exclude/cache check (sync SQLite, no process pool)
    │
    ▼
ProcessPoolExecutor  ← actual scoring (freq + entropy + compression)
    │
ThreadPoolExecutor   ← concurrent request dispatch (asyncio layer)
```

### Worker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOMNESS_INFERENCE_WORKERS` | all CPUs | Process pool size |
| `RANDOMNESS_INFERENCE_THREADS` | same as workers | Thread pool size |
| `RANDOMNESS_INFERENCE_CPU_FRACTION` | `1.0` | Fraction of CPUs to allocate |
| `RANDOMNESS_UVICORN_WORKERS` | `1` | Uvicorn worker processes |

Example for a 48-core machine:

```bash
export RANDOMNESS_INFERENCE_WORKERS=24
export RANDOMNESS_INFERENCE_THREADS=24
export RANDOMNESS_PARALLEL_BACKEND=hybrid
export RANDOMNESS_UVICORN_WORKERS=1
```

With multiple uvicorn workers, inference workers are split:

```
inference_workers = (total_cpus * fraction) // uvicorn_workers
```

## Training Parallelism

Training uses `CPU_FRACTION=0.5` (50% of cores) by default.

| Phase | Parallelism |
|-------|-------------|
| Freq table build | Process + thread pools (hybrid) |
| Synthetic data generation | Process pool |
| Feature extraction | Process pool |
| sklearn GridSearchCV | joblib `n_jobs=workers` |

The training pool is **stopped before sklearn** spawns its own joblib workers to avoid deadlocks.

## Fork Safety

Process pools use `forkserver` context (falls back to `spawn`) to prevent deadlocks when creating pools after sklearn/joblib threads have been active.

`shutdown_joblib()` is called after training to release joblib worker processes before inference pools start.

## Performance Tuning

### High throughput API

```bash
export RANDOMNESS_PARALLEL_BACKEND=hybrid
export RANDOMNESS_INFERENCE_WORKERS=24
export RANDOMNESS_INFERENCE_THREADS=32
```

### CPU-constrained environment

```bash
export RANDOMNESS_INFERENCE_WORKERS=4
export RANDOMNESS_INFERENCE_THREADS=4
export RANDOMNESS_INFERENCE_CPU_FRACTION=0.25
```

### Single-core / debugging

```bash
export RANDOMNESS_PARALLEL_BACKEND=thread
export RANDOMNESS_INFERENCE_WORKERS=1
```

## Monitoring

Check `/health` for active configuration:

```json
{
  "parallel_backend": "hybrid",
  "inference_workers": 24,
  "inference_threads": 24
}
```

## BLAS Thread Control

Worker processes set these to prevent oversubscription:

```
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## Verified CPU Scaling

`test_real_parallel.py` proves that under sustained load the work is actually spread
across the configured number of cores — not run serially. It starts a real API
server, drives 15 s of mixed single + batch traffic with real corpus words, counts
the live worker processes (descendants of the server PID), and samples system-wide
CPU from `/proc/stat`.

Measured on a **48-core** machine (hybrid backend), sweeping the worker count:

| Configured workers | Live worker procs | System CPU peak | System CPU avg | Throughput* |
|--------------------|-------------------|-----------------|----------------|-------------|
| 6  | 8  | 17.0% | 4.9%  | 8.1 calls/s |
| 24 | 26 | 53.9% | 12.3% | 19.9 calls/s |
| 48 | 50 | 100%  | 22.5% | 19.2 calls/s |

What the numbers prove:

- **Processes scale with the setting.** 6 → 8, 24 → 26, 48 → 50 live workers (the
  extra ~2 are the `forkserver`/pool-manager helpers). When many requests arrive, the
  load fans out onto exactly as many cores as configured.
- **CPU utilization scales with cores.** Peak system CPU rises 17% → 54% → 100%; at 48
  workers on 48 cores the machine saturates under peak load — i.e. it is genuinely
  parallel, not serial.
- **Throughput plateaus past 24 here** because the test client's concurrency is capped
  at `min(32, workers × 2)` (32 for both 24 and 48), so 24 workers already saturate the
  32-way client load. CPU still climbs to 100% at 48 because batch items fan out across
  all workers.

\* "calls/s" counts a batch request (41 items) as one call, and ~half the calls are
batches, so the real per-item rate is several times higher. For pure per-item/req
throughput numbers see the [throughput benchmark](README.md#benchmarks).

See [Testing](testing.md#real-parallel-test) to reproduce this, and
[Benchmark Methodology](benchmark-methodology.md#4b-parallel-and-cpu-scaling-verification)
for the full method and pass criteria.
