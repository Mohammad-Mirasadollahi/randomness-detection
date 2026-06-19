"""Shared helpers for real integration tests (no mocks)."""

from __future__ import annotations

import random
import secrets
from pathlib import Path


def words_file(cache_dir: Path) -> Path:
    return cache_dir / "words_alpha.txt"


def load_real_words(cache_dir: Path, *, limit: int = 20_000) -> list[str]:
    """Load real English words downloaded by bootstrap."""
    path = words_file(cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run bootstrap first to download real words."
        )

    words: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            word = line.strip()
            if 3 <= len(word) <= 48 and word.isalpha():
                words.append(word)
            if len(words) >= limit:
                break

    if len(words) < 100:
        raise RuntimeError(f"Not enough real words in corpus ({len(words)}).")
    return words


def build_real_text_batch(
    words: list[str],
    size: int,
    *,
    seed: int | None = None,
) -> list[str]:
    """
    Build a batch of real scoring inputs:
    mostly corpus words, some compound words, some cryptographic random tokens.
    """
    rng = random.Random(seed)
    batch: list[str] = []
    for index in range(size):
        roll = index % 5
        if roll == 0:
            batch.append(secrets.token_hex(rng.randint(8, 24)))
        elif roll == 1:
            parts = [rng.choice(words) for _ in range(2)]
            batch.append(rng.choice(["", "-", "_"]).join(parts))
        else:
            batch.append(rng.choice(words))
    return batch


def pick_natural_word(words: list[str]) -> str:
    eligible = [word for word in words if len(word) >= 5]
    return random.choice(eligible or words)


def pick_random_token() -> str:
    return secrets.token_hex(16)
