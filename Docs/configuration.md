# Configuration

All configuration is via **environment variables**. No config file is required.

## Cache & Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOMNESS_CACHE_DIR` | `~/.cache/randomness_detection` | Model, corpus, and data cache |
| `RANDOMNESS_EXCLUDE_DB_PATH` | `$CACHE_DIR/exclude.db` | SQLite exclusion database |

## API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOMNESS_HOST` | `127.0.0.1` | Server bind host |
| `RANDOMNESS_PORT` | `8765` | Server bind port |
| `RANDOMNESS_LOG_LEVEL` | `info` | Uvicorn log level |
| `RANDOMNESS_UVICORN_WORKERS` | `1` | Number of uvicorn worker processes |

CLI flags override host/port/workers:

```bash
python -m randomness_detection.api_server --host 0.0.0.0 --port 8765 --workers 2
```

## Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOMNESS_API_KEY` | *(required)* | API key (min 32 chars) |
| `RANDOMNESS_ALLOW_NO_AUTH` | `false` | Disable auth (dev only) |

## Parallel Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOMNESS_PARALLEL_BACKEND` | `hybrid` | `process`, `thread`, or `hybrid` |
| `RANDOMNESS_INFERENCE_WORKERS` | all CPUs | Process pool worker count |
| `RANDOMNESS_INFERENCE_THREADS` | same as workers | Thread pool worker count |
| `RANDOMNESS_INFERENCE_CPU_FRACTION` | `1.0` | Fraction of CPUs for inference (0.1–1.0) |

See [Parallel Processing](parallel-processing.md) for details.

## Exclusion & Score Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOMNESS_EXCLUDE_ENABLED` | `true` | Enable exclusion rules |
| `RANDOMNESS_SKIP_CACHE_ENABLED` | `true` | Enable score cache skip |
| `RANDOMNESS_SKIP_SCORE_THRESHOLD` | `30` | Max cached score to skip re-inference |

See [Exclusion System](exclusion.md) for details.

## Training (Bootstrap)

These are code-level defaults in `config.py` (not env vars):

| Constant | Default | Description |
|----------|---------|-------------|
| `TRAIN_SAMPLES_PER_CLASS` | `50,000` | Training samples per class |
| `CPU_FRACTION` | `0.5` | Training CPU usage (50%) |
| `NATURAL_REAL_WORD_RATIO` | `0.7` | Fraction of real words in natural class |
| `BOOTSTRAP_VERSION` | `5` | Re-bootstrap when version increases |

## Example Production Config

```bash
# ~/.env or systemd EnvironmentFile

# Auth
RANDOMNESS_API_KEY="your-64-char-secret-key-here"

# Server
RANDOMNESS_HOST=0.0.0.0
RANDOMNESS_PORT=8765
RANDOMNESS_UVICORN_WORKERS=2

# Cache
RANDOMNESS_CACHE_DIR=/var/lib/randomness_detection

# Parallelism
RANDOMNESS_PARALLEL_BACKEND=hybrid
RANDOMNESS_INFERENCE_WORKERS=24
RANDOMNESS_INFERENCE_THREADS=24
RANDOMNESS_INFERENCE_CPU_FRACTION=0.5

# Exclusion
RANDOMNESS_EXCLUDE_ENABLED=true
RANDOMNESS_SKIP_CACHE_ENABLED=true
RANDOMNESS_SKIP_SCORE_THRESHOLD=30
```

## Python Path

All commands require the package on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/randomness_detection
```

Or install in development mode:

```bash
pip install -e .
```

## Code Defaults Reference

From `randomness_detection/config.py`:

```python
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "randomness_detection"
CPU_FRACTION = 0.5
INFERENCE_CPU_FRACTION = 1.0
PARALLEL_BACKEND = "hybrid"
EXCLUDE_ENABLED = True
SKIP_CACHE_ENABLED = True
SKIP_SCORE_THRESHOLD = 30
LABEL_LIKELY_RANDOM = 60
LABEL_NATURAL = 30
```
