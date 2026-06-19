"""Synthetic training sample generation (no external malicious datasets)."""

from __future__ import annotations

import base64
import random
import secrets
import string
import uuid

from .parallel import parallel_map, worker_count

CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"
DIGITS = "0123456789"
SEPARATORS = ("-", "_", ".")


def _pronounceable_random(length: int) -> str:
    """Word-like lowercase gibberish (alternating consonant/vowel).

    A pure high-entropy primitive that looks pronounceable like English but is not
    a real word — forces the model to rely on lexical structure, not vowel pattern.
    """
    chars: list[str] = []
    use_consonant = secrets.choice((True, False))
    for _ in range(length):
        source = CONSONANTS if use_consonant else VOWELS
        chars.append(secrets.choice(source))
        use_consonant = not use_consonant
    return "".join(chars)


def _apply_surface(parts: list[str], rng: random.Random) -> str:
    """Apply label-INDEPENDENT surface decoration: casing, separators, digits.

    The SAME decoration distribution is applied to both natural and random cores,
    so surface form (mixed case, '-'/'_'/'.', digit suffixes) carries no class
    information. This is the root-cause fix: the model can no longer use casing or
    separators as a shortcut and must judge the linguistic nature of the tokens.
    """
    scheme = rng.choice(("lower", "lower", "title", "upper", "camel", "mixed"))
    shaped: list[str] = []
    for index, part in enumerate(parts):
        if scheme == "lower":
            shaped.append(part.lower())
        elif scheme == "title":
            shaped.append(part.capitalize())
        elif scheme == "upper":
            shaped.append(part.upper())
        elif scheme == "camel":
            shaped.append(part.lower() if index == 0 else part.capitalize())
        else:  # mixed
            shaped.append(part.capitalize() if rng.random() < 0.5 else part.lower())

    if scheme == "camel":
        separator = ""
    elif len(shaped) == 1:
        separator = ""
    else:
        separator = rng.choice(("", *SEPARATORS))
    token = separator.join(shaped)

    if rng.random() < 0.25:
        digits = "".join(rng.choice(DIGITS) for _ in range(rng.randint(1, 4)))
        token = token + digits if rng.random() < 0.7 else digits + token
    return token


def _random_fragments(rng: random.Random) -> list[str]:
    """1-3 high-entropy alpha(+digit) fragments for identifier-shaped randoms."""
    count = rng.choice((1, 2, 2, 3))
    fragments: list[str] = []
    for _ in range(count):
        size = rng.randint(3, 8)
        kind = rng.random()
        if kind < 0.5:
            alphabet = string.ascii_lowercase
        elif kind < 0.8:
            alphabet = CONSONANTS
        else:
            alphabet = string.ascii_lowercase + string.digits
        fragments.append("".join(secrets.choice(alphabet) for _ in range(size)))
    return fragments


def _make_natural(eligible: list[str], rng: random.Random) -> str:
    """Natural core = 1-3 real dictionary words, then shared surface decoration.

    ~30% of samples also get a short non-dictionary affix (2-4 chars) to mimic
    real identifiers that mix words with abbreviations/tech tokens (api, js, id,
    http, v3). This populates the moderate lexical-coverage region (0.6-0.95) with
    the natural label so the model does not treat partial coverage as random.
    """
    count = rng.choice((1, 1, 1, 2, 2, 3))
    parts = [rng.choice(eligible) for _ in range(count)]
    # Add a short non-dictionary affix only when the real-word content dominates,
    # so coverage stays high (~0.7+). Otherwise affix-naturals would collide with
    # short pronounceable random tokens (e.g. 'tagirsog') and blur the boundary.
    if rng.random() < 0.3 and sum(len(part) for part in parts) >= 7:
        affix = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 3)))
        parts.insert(rng.randint(0, len(parts)), affix)
    return _apply_surface(parts, rng)


def _word_salad_random(eligible: list[str], rng: random.Random) -> str:
    """Concatenated dictionary word-salad (4-7 real words).

    Mimics dictionary-DGA / passphrase-style machine strings: high lexical coverage
    but an unnaturally high word count. Labeled random so the model learns that
    'many concatenated words' is machine-like, while short 1-3 word compounds
    (real names/identifiers) stay natural.
    """
    count = rng.randint(4, 7)
    parts = [rng.choice(eligible) for _ in range(count)]
    return _apply_surface(parts, rng)


def _make_random(eligible: list[str], rng: random.Random) -> str:
    """Random core: high-entropy token, decorated entropy fragments, or word-salad.

    A dedicated short-token branch (length 6-12) ensures short random/pronounceable
    strings — the hardest random-character DGA family — are well represented.
    """
    roll = rng.random()
    if roll < 0.32:
        return generate_random_string(rng.randint(6, 32))
    if roll < 0.50:
        return generate_random_string(rng.randint(6, 12))
    if roll < 0.80:
        return _apply_surface(_random_fragments(rng), rng)
    return _word_salad_random(eligible, rng)


def _generate_natural_batch(args: tuple[list[str], int, int]) -> list[str]:
    words, count, seed = args
    rng = random.Random(seed)
    eligible = [word for word in words if 3 <= len(word) <= 24 and word.isalpha()]
    if not eligible:
        eligible = words

    samples = [_make_natural(eligible, rng) for _ in range(count)]
    rng.shuffle(samples)
    return samples


def _generate_random_batch(args: tuple[list[str], int, int]) -> list[str]:
    words, count, seed = args
    rng = random.Random(seed)
    eligible = [word for word in words if 3 <= len(word) <= 24 and word.isalpha()]
    if not eligible:
        eligible = words
    return [_make_random(eligible, rng) for _ in range(count)]


def generate_random_string(
    length: int | None = None,
    *,
    min_len: int = 6,
    max_len: int = 32,
) -> str:
    """Pure high-entropy primitive (no surface decoration applied here)."""
    if length is None:
        length = random.randint(min_len, max_len)
    generators = (
        lambda: secrets.token_hex(length // 2 + length % 2)[:length],
        lambda: str(uuid.uuid4()).replace("-", "")[:length],
        lambda: "".join(
            secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length)
        ),
        lambda: secrets.token_urlsafe(length)[:length],
        lambda: "".join(secrets.choice(CONSONANTS) for _ in range(length)),
        lambda: base64.b64encode(secrets.token_bytes(length))[:length].decode("ascii"),
        lambda: "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
        ),
        lambda: _pronounceable_random(length),
    )
    return random.choice(generators)()


def build_labeled_dataset(
    words: list[str],
    samples_per_class: int,
    *,
    real_word_ratio: float = 0.7,
    seed: int = 42,
    cpu_fraction: float = 0.5,
) -> tuple[list[str], list[int]]:
    workers = worker_count(cpu_fraction)
    batch_count = max(workers * 4, 8)
    batch_size = max(1, samples_per_class // batch_count)

    natural_batches: list[tuple[list[str], int, int]] = []
    random_batches: list[tuple[list[str], int, int]] = []
    natural_remaining = samples_per_class
    random_remaining = samples_per_class
    batch_index = 0

    while natural_remaining > 0:
        size = min(batch_size, natural_remaining)
        natural_batches.append((words, size, seed + batch_index))
        natural_remaining -= size
        batch_index += 1

    while random_remaining > 0:
        size = min(batch_size, random_remaining)
        random_batches.append((words, size, seed + 10_000 + batch_index))
        random_remaining -= size
        batch_index += 1

    natural_parts = parallel_map(
        _generate_natural_batch,
        natural_batches,
        cpu_fraction=cpu_fraction,
        chunksize=1,
    )
    random_parts = parallel_map(
        _generate_random_batch,
        random_batches,
        cpu_fraction=cpu_fraction,
        chunksize=1,
    )

    natural_samples = [sample for batch in natural_parts for sample in batch]
    random_samples = [sample for batch in random_parts for sample in batch]

    natural_samples = natural_samples[:samples_per_class]
    random_samples = random_samples[:samples_per_class]

    texts = natural_samples + random_samples
    labels = [0] * len(natural_samples) + [1] * len(random_samples)
    combined = list(zip(texts, labels, strict=True))
    rng = random.Random(seed)
    rng.shuffle(combined)

    shuffled_texts, shuffled_labels = zip(*combined, strict=True)
    return list(shuffled_texts), list(shuffled_labels)
