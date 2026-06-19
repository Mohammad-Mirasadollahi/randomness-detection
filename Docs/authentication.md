# Authentication

The API uses **API key authentication** with constant-time comparison to prevent timing attacks.

## Setup

### Generate a Key

**Recommended (via install.sh):**

```bash
# Print a new key (for scripts)
./install.sh --gen-api-key

# Save to .env (creates file if missing; skips if valid key exists)
./install.sh --write-api-key

# Replace existing key (restart server after)
./install.sh --rotate-api-key
```

**Manual alternative:**

```bash
export RANDOMNESS_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

**Requirements:**

- Minimum **32 characters**
- Server refuses to start without a valid key (unless dev mode is enabled)

### Start Server

```bash
export RANDOMNESS_API_KEY="your-secret-key-at-least-32-characters-long"
PYTHONPATH=. .venv/bin/python -m randomness_detection.api_server
```

## Sending the Key

Two methods are supported:

### Bearer Token (recommended)

```http
Authorization: Bearer <API_KEY>
```

```bash
curl -H "Authorization: Bearer $RANDOMNESS_API_KEY" ...
```

### X-API-Key Header

```http
X-API-Key: <API_KEY>
```

```bash
curl -H "X-API-Key: $RANDOMNESS_API_KEY" ...
```

If both are provided, `Authorization: Bearer` takes precedence.

## Public Endpoints

| Endpoint | Auth required |
|----------|---------------|
| `GET /health` | No |
| All other endpoints | Yes |

## Development Mode

For local development only:

```bash
export RANDOMNESS_ALLOW_NO_AUTH=true
```

**Warning:** Do not use in production. All endpoints become accessible without a key.

## Error Responses

### `401 Unauthorized`

```json
{
  "detail": "Invalid or missing API key."
}
```

Causes:

- Missing header
- Wrong key
- Key length mismatch (compared in constant time)

### `503 Service Unavailable`

```json
{
  "detail": "Authentication is not configured."
}
```

Auth module failed to initialize.

## Security Notes

1. **Constant-time comparison** — uses `secrets.compare_digest()` to prevent timing side-channels
2. **Minimum key length** — 32 characters enforced at startup
3. **HTTPS** — terminate TLS at a reverse proxy (nginx, Caddy) in production
4. **Host filtering** — optional `RANDOMNESS_ALLOWED_HOSTS` env var
5. **CORS** — all browser origins are allowed (`Access-Control-Allow-Origin: *`)

## Optional Hardening

```bash
# Restrict allowed Host headers
export RANDOMNESS_ALLOWED_HOSTS="api.example.com,localhost"
```

## Server Startup Errors

```
RuntimeError: RANDOMNESS_API_KEY is required.
```

Set `RANDOMNESS_API_KEY` or `RANDOMNESS_ALLOW_NO_AUTH=true`.

```
RuntimeError: RANDOMNESS_API_KEY must be at least 32 characters.
```

Generate a longer key.
