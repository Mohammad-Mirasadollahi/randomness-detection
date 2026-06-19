"""Download and merge safe public word corpora for training."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .config import WORD_SOURCES, WordSource


def _download_file(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "randomness-detection/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
            destination.write_bytes(payload)
            return len(payload)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _parse_plain_lines(content: str) -> list[str]:
    return [line.strip() for line in content.splitlines() if line.strip()]


def _parse_scrabble_lines(content: str) -> list[str]:
    words: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]
        if token.isalpha():
            words.append(token.lower())
    return words


def _parse_eff_lines(content: str) -> list[str]:
    words: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isalpha():
            words.append(parts[1].lower())
    return words


def _parse_source(source: WordSource, content: str) -> list[str]:
    if source.parse == "scrabble":
        return _parse_scrabble_lines(content)
    if source.parse == "eff":
        return _parse_eff_lines(content)
    return _parse_plain_lines(content)


def load_merged_words(
    cache_dir: Path,
    *,
    force_download: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Load and deduplicate words from all configured sources."""
    seen: set[str] = set()
    merged: list[str] = []

    def _log(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    for source in WORD_SOURCES:
        path = cache_dir / source.filename
        if force_download or not path.exists():
            _log(f"Downloading {source.label}...")
            nbytes = _download_file(source.url, path)
            _log(f"Saved {source.filename} ({nbytes / 1024:.1f} KB)")
        else:
            nbytes = path.stat().st_size
            _log(f"Using cached {source.filename} ({nbytes / 1024:.1f} KB)")

        words = _parse_source(source, path.read_text(encoding="utf-8"))
        added = 0
        for word in words:
            normalized = word.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(word)
            added += 1
        _log(f"  +{added:,} unique words from {source.filename} ({len(merged):,} total)")

    if not merged:
        raise RuntimeError("No words loaded from configured corpora.")
    return merged


def filter_words_for_freq(
    words: list[str],
    *,
    min_length: int,
    max_words: int,
) -> list[str]:
    eligible = [word for word in words if len(word) >= min_length and word.isalpha()]
    if max_words > 0:
        return eligible[:max_words]
    return eligible


def filter_words_for_training(words: list[str], *, min_length: int = 3) -> list[str]:
    return [
        word
        for word in words
        if min_length <= len(word) <= 48 and word.isalpha()
    ]
