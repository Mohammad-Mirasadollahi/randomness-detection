# Exclusion System

The exclusion layer provides a **fast pre-inference filter** that skips scoring for known domains, patterns, and previously-seen low-score items. It uses **zero inference CPU** — checks run via SQLite lookups and in-memory trie matching.

## Overview

```
Request → Exclude check → Score cache check → Inference pool
              ↓                  ↓
         excluded:true      cached:true
         (no CPU)           (no CPU)
```

Typical check speed: **80,000+ lookups/second** with 50,000 rules (in-process, no inference).

## Rule Types

| Type | Pattern example | Matches |
|------|-----------------|---------|
| `domain` | `blocked.com` | `blocked.com`, `app.blocked.com`, `https://foo.blocked.com/path` |
| `suffix` | `*.cdn.example.net` | Any domain ending with `cdn.example.net` |
| `prefix` | `admin-` | Strings starting with `admin-` |
| `glob` | `test-user-*` | Shell-style glob patterns |
| `exact` | `cache-test-item` | Exact string match (case-insensitive) |
| `wildcard` | auto | Detects type from pattern shape |

### Domain Extraction

The system automatically extracts domains from:

- Plain domains: `example.com`
- URLs: `https://app.example.com/path`
- Emails: `user@example.com`
- Host paths: `app.example.com/login`

## Score Cache (Skip Re-check)

If a string was previously scored and the result is **at or below the threshold**, the cached result is returned without re-inference.

| Setting | Default | Description |
|---------|---------|-------------|
| `RANDOMNESS_SKIP_CACHE_ENABLED` | `true` | Enable score cache |
| `RANDOMNESS_SKIP_SCORE_THRESHOLD` | `30` | Max score to skip re-check |

**Example:** A word scoring `1` (natural) is cached. The next request returns instantly from cache.

Items scoring above the threshold (e.g. `likely_random` at 85) are cached in the database but **not** skipped — they will be re-scored on the next request.

## API Usage

### Add Rules

```bash
curl -X POST http://localhost:8765/exclude \
  -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "rules": [
      {"pattern": "blocked.com", "rule_type": "domain"},
      {"pattern": "*.trusted.org", "rule_type": "suffix"},
      {"pattern": "admin-", "rule_type": "prefix"}
    ]
  }'
```

### Remove Rules

```bash
curl -X DELETE http://localhost:8765/exclude \
  -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"patterns": ["blocked.com"]}'
```

### Check Without Scoring

```bash
curl -X POST http://localhost:8765/exclude/check \
  -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "https://app.blocked.com"}'
```

### Score with Exclusion

Exclusion is **on by default**. Disable per request:

```
POST /score?use_exclude=false&use_score_cache=false
```

## Bulk Import (Millions of Rules)

For large domain lists, use the CLI import tool:

```bash
# domains.txt — one pattern per line
# blocked1.com
# blocked2.com
# *.cdn.evil.net

PYTHONPATH=. .venv/bin/python -m randomness_detection.exclude_import \
  domains.txt \
  --type domain \
  --batch-size 10000
```

For API-based bulk import, call `POST /exclude` repeatedly with up to 10,000 rules per request.

## Storage Architecture

### SQLite (`exclude.db`)

| Table | Purpose | Scale |
|-------|---------|-------|
| `exact_rules` | Exact text and domain rules | Millions of rows |
| `wildcard_rules` | Suffix/prefix/glob patterns | Thousands–millions |
| `score_cache` | Previously scored results | Millions of rows |

SQLite settings:

- WAL journal mode
- Memory-mapped I/O (256 MB)
- Indexed primary keys for O(log n) lookups

### In-Memory Index

Wildcard rules are loaded into an in-memory structure at startup and on rule changes:

- **Suffix trie** (reversed domain labels) — for `*.domain.com` patterns
- **Prefix matcher** — sorted by length, longest first
- **Glob matcher** — `fnmatch` patterns

Exact/domain rules are queried directly from SQLite (batch `IN` queries for batch scoring).

## Response Fields

### Excluded

```json
{
  "score": 0,
  "label": "excluded",
  "excluded": true,
  "exclude_reason": "domain:blocked.com",
  "exclude_rule_type": "domain",
  "exclude_pattern": "blocked.com",
  "skipped": true,
  "skipped_reason": "excluded"
}
```

### Cache Hit

```json
{
  "score": 1,
  "label": "natural",
  "cached": true,
  "skipped": true,
  "skipped_reason": "score_cache_below_threshold"
}
```

## Batch Scoring Behavior

For `POST /score/batch` with 500 items:

1. All 500 items checked against exclusion rules in one SQLite batch query
2. Remaining items checked against score cache in one batch query
3. Only non-excluded, non-cached items sent to inference pool
4. New scores stored in cache after inference

This means a batch of mostly excluded domains incurs **zero inference CPU**.

## Configuration

```bash
export RANDOMNESS_EXCLUDE_ENABLED=true
export RANDOMNESS_EXCLUDE_DB_PATH=~/.cache/randomness_detection/exclude.db
export RANDOMNESS_SKIP_CACHE_ENABLED=true
export RANDOMNESS_SKIP_SCORE_THRESHOLD=30
```

## Performance Benchmarks

Tested on a 48-core Linux machine:

| Operation | Performance |
|-----------|-------------|
| 50,000 domain rules loaded | ~3 seconds |
| 10,000 exclude checks | ~122 ms (~82,000/s) |
| Single exclude check | < 0.1 ms |
| Excluded score via API | No inference workers used |

Run the benchmark yourself:

```bash
PYTHONPATH=. .venv/bin/python test_exclude.py
```

## Design Constraints

1. **No regex rules** — regex is intentionally excluded to prevent CPU-heavy pattern matching at scale
2. **Glob limited to fnmatch** — simple `*` and `?` only
3. **Case-insensitive** — all patterns and inputs are normalized to lowercase
4. **Thread-safe** — SQLite access is protected by a reentrant lock
