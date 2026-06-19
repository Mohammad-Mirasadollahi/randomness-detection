# Getting Started

This guide walks you through installation, model bootstrap, and your first score.

## Requirements

- Python 3.10+
- Linux recommended (multiprocessing uses `forkserver` on Linux)
- Network access for first-time corpus download

## Installation

### Automated (recommended)

```bash
cd randomness_detection
chmod +x install.sh
./install.sh
```

This script will:

1. Create `.venv` virtualenv
2. Install Python dependencies and the package (editable)
3. Install the `randomness-detection` command on your `PATH`
4. Generate `.env` with API key and defaults
5. Bootstrap model (first run only, ~40–90s)
6. **Ask how to run the server:** `systemd` or `manual`
7. Start the API server

The CLI launcher is placed in `/usr/local/bin` (root) or `~/.local/bin` (non-root).
It loads `.env` automatically, so `randomness-detection` finds the trained model and
tuned settings from any directory without activating the venv.

Management:

```bash
./install.sh --guide       # full usage guide with examples
./install.sh --status      # check if server is running
./install.sh --stop        # stop server (manual and/or systemd)
./install.sh --systemd     # install as systemd service (skip prompt)
./install.sh --manual      # run as background process (skip prompt)
./install.sh --foreground  # manual foreground mode
./install.sh --no-start    # install only, don't start
```

### Run modes

| Mode | Description |
|------|-------------|
| **systemd** | Installs `randomness-detection.service`, enables auto-start on boot (requires root for system-wide unit) |
| **manual** | Runs via `nohup` in background, stores PID in `.run/api.pid` |

### Manual

```bash
cd randomness_detection
python3 -m venv .venv
.venv/bin/pip install -e .          # installs deps + the randomness-detection command
```

`pip install -e .` creates the `randomness-detection` and `randomness-detection-server`
console scripts in `.venv/bin/`. Activate the venv (`source .venv/bin/activate`) to use
them by name, or call `.venv/bin/randomness-detection` directly.

Dependencies:

| Package | Purpose |
|---------|---------|
| scikit-learn | Ensemble model, calibration, GridSearchCV |
| fastapi | REST API |
| uvicorn | ASGI server |
| pydantic | Request/response validation |

## First-Time Bootstrap

Bootstrap downloads English word lists, builds a frequency table, and trains the ensemble model. This runs automatically on first use, or you can force it:

```bash
export RANDOMNESS_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap
```

Default cache directory: `~/.cache/randomness_detection/`

Override with:

```bash
export RANDOMNESS_CACHE_DIR=/path/to/cache
```

### What Bootstrap Creates

| File | Description |
|------|-------------|
| `words_alpha.txt` | Downloaded English word corpus |
| `words.txt` | Additional word list |
| `english.freq` | Bigram frequency table |
| `ensemble.pkl` | Trained logistic regression pipeline |
| `metadata.json` | Training metrics and version info |

Bootstrap takes roughly **40–60 seconds** on a 48-core machine.

## Score from CLI

After install, use the `randomness-detection` command from anywhere:

```bash
# Single string
randomness-detection "hello"

# JSON output
randomness-detection "xK9#mQ2" --json

# Multiple words from file
randomness-detection -f words.txt
```

Equivalent module form (from the project root, no install needed):

```bash
PYTHONPATH=. .venv/bin/python -m randomness_detection "hello"
```

## Run the API Server

```bash
export RANDOMNESS_API_KEY="your-secret-key-at-least-32-characters-long"
export RANDOMNESS_INFERENCE_WORKERS=24
export RANDOMNESS_PARALLEL_BACKEND=hybrid

PYTHONPATH=. .venv/bin/python -m randomness_detection.api_server \
  --host 0.0.0.0 \
  --port 8765
```

### Health Check

```bash
curl http://127.0.0.1:8765/health
```

Example response:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_ready": true,
  "parallel_backend": "hybrid",
  "inference_workers": 24,
  "inference_threads": 24,
  "exclude_enabled": true,
  "skip_cache_enabled": true,
  "skip_score_threshold": 30,
  "exact_exclude_rules": 0,
  "wildcard_exclude_rules": 0,
  "score_cache_entries": 0
}
```

## First API Score

```bash
curl -s -X POST http://127.0.0.1:8765/score \
  -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "arboraceous"}' | jq
```

Example response:

```json
{
  "score": 1,
  "label": "natural",
  "confidence": "high",
  "breakdown": {
    "freq": 91,
    "entropy": 97,
    "compression": 100
  },
  "excluded": false,
  "cached": false,
  "skipped": false
}
```

## Next Steps

- [API Reference](api-reference.md) — full endpoint documentation
- [Exclusion System](exclusion.md) — skip domains and cached low-score items
- [Parallel Processing](parallel-processing.md) — tune CPU workers
- [Testing](testing.md) — run real integration tests
