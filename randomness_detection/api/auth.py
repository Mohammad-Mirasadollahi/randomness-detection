"""API key authentication with constant-time comparison."""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_SCHEME = HTTPBearer(auto_error=False)

_CONFIGURED_KEY: str | None = None
_AUTH_DISABLED = False


def load_auth_config() -> None:
    global _CONFIGURED_KEY, _AUTH_DISABLED

    allow_no_auth = os.environ.get("RANDOMNESS_ALLOW_NO_AUTH", "").lower() == "true"
    api_key = os.environ.get("RANDOMNESS_API_KEY", "").strip()

    if api_key:
        if len(api_key) < 32:
            raise RuntimeError("RANDOMNESS_API_KEY must be at least 32 characters.")
        _CONFIGURED_KEY = api_key
        _AUTH_DISABLED = False
        return

    if allow_no_auth:
        _CONFIGURED_KEY = None
        _AUTH_DISABLED = True
        return

    raise RuntimeError(
        "RANDOMNESS_API_KEY is required. "
        "Set a strong key (>=32 chars) or RANDOMNESS_ALLOW_NO_AUTH=true for local dev only."
    )


def _extract_provided_key(
    bearer: HTTPAuthorizationCredentials | None,
    header_key: str | None,
) -> str | None:
    if bearer is not None and bearer.scheme.lower() == "bearer":
        token = bearer.credentials.strip()
        return token or None
    if header_key:
        return header_key.strip() or None
    return None


def _keys_match(provided: str, expected: str) -> bool:
    provided_bytes = provided.encode("utf-8")
    expected_bytes = expected.encode("utf-8")
    if len(provided_bytes) != len(expected_bytes):
        # Compare against itself to keep timing similar without leaking length.
        secrets.compare_digest(expected_bytes, expected_bytes)
        return False
    return secrets.compare_digest(provided_bytes, expected_bytes)


async def require_api_key(
    bearer: Annotated[HTTPAuthorizationCredentials | None, Security(BEARER_SCHEME)],
    header_key: Annotated[str | None, Security(API_KEY_HEADER)],
) -> None:
    if _AUTH_DISABLED:
        return

    if _CONFIGURED_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    provided = _extract_provided_key(bearer, header_key)
    if provided is None or not _keys_match(provided, _CONFIGURED_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


Authenticated = Annotated[None, Depends(require_api_key)]
