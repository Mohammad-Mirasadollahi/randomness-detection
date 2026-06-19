# API Reference

Base URL: `http://localhost:8765` (default)

Interactive docs: `/docs` (Swagger UI), `/redoc` (ReDoc)

## Authentication

All endpoints except `GET /health` require authentication. See [Authentication](authentication.md).

```
Authorization: Bearer <API_KEY>
# or
X-API-Key: <API_KEY>
```

---

## System

### `GET /health`

No authentication required.

**Response `200`:**

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

---

## Scoring

### `POST /score`

Score a single string.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `include_features` | bool | `false` | Include raw feature values |
| `use_exclude` | bool | `true` | Apply exclusion rules |
| `use_score_cache` | bool | `true` | Return cached score if ≤ threshold |

**Request body:**

```json
{
  "text": "hello"
}
```

| Field | Type | Limits |
|-------|------|--------|
| `text` | string | 1–4096 chars, no null bytes |

**Response `200`:**

```json
{
  "score": 1,
  "label": "natural",
  "confidence": "high",
  "breakdown": {"freq": 91, "entropy": 97, "compression": 100},
  "features": null,
  "excluded": false,
  "exclude_reason": null,
  "exclude_rule_type": null,
  "exclude_pattern": null,
  "cached": false,
  "skipped": false,
  "skipped_reason": null
}
```

**Excluded response example:**

```json
{
  "score": 0,
  "label": "excluded",
  "confidence": "high",
  "breakdown": {"freq": 0, "entropy": 0, "compression": 0},
  "excluded": true,
  "exclude_reason": "domain:blocked.com",
  "exclude_rule_type": "domain",
  "exclude_pattern": "blocked.com",
  "skipped": true,
  "skipped_reason": "excluded"
}
```

---

### `POST /score/batch`

Score up to 500 strings in one request.

**Query parameters:** same as `/score`

**Request body:**

```json
{
  "texts": ["hello", "https://blocked.com", "xK9mQ2"]
}
```

| Field | Type | Limits |
|-------|------|--------|
| `texts` | string[] | 1–500 items, each 1–1024 chars |

**Response `200`:**

```json
{
  "count": 3,
  "results": [
    {
      "text": "hello",
      "score": 1,
      "label": "natural",
      "confidence": "high",
      "breakdown": {"freq": 91, "entropy": 97, "compression": 100},
      "excluded": false,
      "cached": false,
      "skipped": false
    }
  ]
}
```

Batch requests apply exclusion and cache checks **per item**. Only non-excluded, non-cached items are sent to the inference pool.

---

## Exclusion Management

### `GET /exclude/stats`

Return exclusion and cache statistics.

**Response `200`:**

```json
{
  "enabled": true,
  "skip_cache_enabled": true,
  "skip_score_threshold": 30,
  "exact_rules": 50000,
  "wildcard_rules": 1200,
  "score_cache_entries": 340000,
  "wildcard_index_rules": 1200
}
```

---

### `POST /exclude`

Add exclusion rules (up to 10,000 per request).

**Request body:**

```json
{
  "rules": [
    {"pattern": "blocked.com", "rule_type": "domain"},
    {"pattern": "*.cdn.example.net", "rule_type": "suffix"},
    {"pattern": "admin-", "rule_type": "prefix"},
    {"pattern": "test-user-*", "rule_type": "glob"},
    {"pattern": "exact-string", "rule_type": "exact"}
  ]
}
```

| `rule_type` | Description |
|-------------|-------------|
| `domain` | Block domain and all subdomains |
| `suffix` | Block domains ending with pattern (e.g. `*.example.com`) |
| `prefix` | Block strings starting with prefix |
| `glob` | Shell-style glob (`*`, `?`) |
| `exact` | Exact string match |
| `wildcard` | Auto-detect type from pattern |

**Response `200`:**

```json
{
  "added": 5,
  "duplicates": 0,
  "exact_rules": 5,
  "wildcard_rules": 3
}
```

---

### `DELETE /exclude`

Remove rules by pattern.

**Request body:**

```json
{
  "patterns": ["blocked.com", "*.cdn.example.net"]
}
```

**Response `200`:**

```json
{
  "removed": 2,
  "exact_rules": 3,
  "wildcard_rules": 1
}
```

---

### `POST /exclude/check`

Check whether a string would be excluded or cache-skipped **without running inference**.

**Request body:**

```json
{
  "text": "https://app.blocked.com/login"
}
```

**Response `200`:**

```json
{
  "text": "https://app.blocked.com/login",
  "excluded": true,
  "exclude_reason": "domain:blocked.com",
  "exclude_rule_type": "domain",
  "exclude_pattern": "blocked.com",
  "cached": false,
  "cached_score": null,
  "would_skip": true
}
```

---

## Error Responses

| Status | Meaning |
|--------|---------|
| `401` | Invalid or missing API key |
| `422` | Validation error (bad payload) |
| `503` | Auth not configured |

```json
{
  "detail": "Invalid request payload."
}
```

## Limits

| Limit | Value |
|-------|-------|
| Single text max length | 4,096 chars |
| Batch size | 500 items |
| Batch item max length | 1,024 chars |
| Exclude rules per request | 10,000 |
| Pattern max length | 512 chars |

## Example: Full Workflow

```bash
API_KEY="your-key"
BASE="http://127.0.0.1:8765"

# Add exclusion
curl -s -X POST "$BASE/exclude" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"rules":[{"pattern":"skipme.com","rule_type":"domain"}]}'

# Score (will be excluded — no inference)
curl -s -X POST "$BASE/score" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"https://app.skipme.com"}'

# Score with exclusion disabled
curl -s -X POST "$BASE/score?use_exclude=false" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"https://app.skipme.com"}'
```
