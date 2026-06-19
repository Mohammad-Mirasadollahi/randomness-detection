# CLI Reference

After running `install.sh`, two commands are installed on your `PATH` and work from
any directory (no venv activation needed):

```bash
randomness-detection [OPTIONS] [TEXT]      # score strings
randomness-detection-server [OPTIONS]      # run the API server (foreground)
```

These are thin launchers that load the project's `.env` (so they automatically use
the trained-model cache and tuned settings) and then call the packaged console
scripts. They are defined as `console_scripts` entry points in `pyproject.toml`:

```toml
[project.scripts]
randomness-detection = "randomness_detection.__main__:main"
randomness-detection-server = "randomness_detection.api_server:main"
```

If you did not use `install.sh`, install the package to get the commands in your
environment's `bin/`:

```bash
pip install -e .          # creates randomness-detection in <venv>/bin
```

### Equivalent module form (no install)

Everything below also works without the installed command, from the project root:

```bash
PYTHONPATH=. .venv/bin/python -m randomness_detection [OPTIONS] [TEXT]
```

## Commands

### Score a Single String

```bash
randomness-detection "hello"
```

### Score Multiple Strings

Comma-separated:

```bash
randomness-detection "hello,world,test"
```

Multiline (stdin):

```bash
echo -e "hello\nworld" | randomness-detection
```

### Score from File

```bash
randomness-detection -f words.txt
```

File format: one word per line, or comma-separated.

## Options

| Option | Description |
|--------|-------------|
| `TEXT` | Optional input string |
| `-f, --file PATH` | Read words from file |
| `--cache-dir PATH` | Model cache directory (default: `$RANDOMNESS_CACHE_DIR`, else `~/.cache/randomness_detection`) |
| `--bootstrap` | Force re-download corpus and retrain |
| `--json` | Output JSON instead of human-readable text |
| `-h, --help` | Show help |

## Bootstrap Only

```bash
randomness-detection --bootstrap --json
```

Prints training metadata and exits (no scoring unless text is also provided).

## JSON Output

Single result:

```bash
randomness-detection "test" --json
```

```json
{
  "score": 1,
  "label": "natural",
  "confidence": "high",
  "breakdown": {
    "freq": 85,
    "entropy": 90,
    "compression": 95
  },
  "features": {
    "length": 4,
    "unique_chars": 4,
    "entropy_norm": 0.95
  }
}
```

Batch result:

```json
{
  "count": 2,
  "results": [
    {"text": "hello", "score": 1, "label": "natural", ...},
    {"text": "xK9mQ2", "score": 87, "label": "likely_random", ...}
  ]
}
```

## Bulk Exclusion Import

Import exclusion rules from a newline-delimited file:

```bash
PYTHONPATH=. .venv/bin/python -m randomness_detection.exclude_import \
  domains.txt \
  --type domain \
  --batch-size 10000 \
  --cache-dir ~/.cache/randomness_detection
```

| Option | Description |
|--------|-------------|
| `file` | Text file with one pattern per line (`#` comments ignored) |
| `--type` | Rule type: `exact`, `domain`, `suffix`, `prefix`, `glob`, `wildcard` |
| `--cache-dir` | Cache directory for `exclude.db` |
| `--batch-size` | Insert batch size (default: 10,000) |

## Environment Variables

The CLI respects:

| Variable | Effect |
|----------|--------|
| `RANDOMNESS_CACHE_DIR` | Override default cache path |

Note: exclusion and score cache are **API-only** features. The CLI uses the core `Scorer` directly without the exclusion layer.
