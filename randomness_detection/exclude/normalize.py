"""Normalization helpers for exclusion keys."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

_DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"^[^@\s]+@([^@\s]+)$", re.IGNORECASE)


def normalize_text(value: str) -> str:
    return value.strip().lower()


def cache_key(value: str) -> str:
    normalized = normalize_text(value)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def extract_domain(value: str) -> str | None:
    """Extract a domain from plain domain, URL, or email."""
    text = normalize_text(value)
    if not text:
        return None

    email_match = _EMAIL_RE.match(text)
    if email_match:
        return email_match.group(1).rstrip(".")

    if "://" in text or text.startswith("//"):
        parsed = urlparse(text if "://" in text else f"//{text}")
        host = parsed.hostname
        return host.lower().rstrip(".") if host else None

    if "/" in text or "?" in text or "#" in text:
        parsed = urlparse(f"//{text}")
        host = parsed.hostname
        if host:
            return host.lower().rstrip(".")

    if text.startswith("www."):
        text = text[4:]

    if _DOMAIN_RE.match(text):
        return text.rstrip(".")

    return None


def normalize_domain_pattern(pattern: str) -> str:
    value = normalize_text(pattern)
    if value.startswith("*."):
        value = value[2:]
    if value.startswith("."):
        value = value[1:]
    return value.rstrip(".")


def reverse_domain(domain: str) -> str:
    parts = [part for part in domain.split(".") if part]
    return ".".join(reversed(parts))


def domain_ancestors(domain: str) -> list[str]:
    parts = [part for part in domain.split(".") if part]
    return [".".join(parts[index:]) for index in range(len(parts))]
