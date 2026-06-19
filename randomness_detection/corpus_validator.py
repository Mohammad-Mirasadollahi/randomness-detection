"""Validate training corpora for safety and quality issues."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

HARD_FAIL_PATTERNS = (
    (re.compile(r"https?://", re.I), "url"),
    (re.compile(r"\d{1,3}(?:\.\d{1,3}){3}"), "ip_address"),
    (re.compile(r"[;|&`$<>]"), "shell_metachar"),
    (re.compile(r"[^\x00-\x7F]"), "non_ascii"),
)

INFO_PATTERNS = (
    (
        re.compile(r"^(?=.*[0-9+/])[A-Za-z0-9+/]{16,}={0,2}$"),
        "base64_like",
    ),
)

SUSPICIOUS_PATTERNS = HARD_FAIL_PATTERNS + INFO_PATTERNS


@dataclass
class CorpusReport:
    total_words: int
    unique_words_lower: int
    alpha_words: int
    eligible_for_freq: int
    eligible_for_training: int
    min_length: int
    max_length: int
    avg_length: float
    suspicious_counts: dict[str, int]
    suspicious_examples: dict[str, list[str]]
    empty_or_invalid: int
    passed: bool
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_words(
    words: list[str],
    *,
    min_length: int = 2,
    training_min_length: int = 3,
    training_max_length: int = 48,
    max_suspicious_ratio: float = 0.001,
) -> CorpusReport:
    suspicious_counts: dict[str, int] = {name: 0 for _, name in SUSPICIOUS_PATTERNS}
    suspicious_examples: dict[str, list[str]] = {name: [] for _, name in SUSPICIOUS_PATTERNS}
    hard_fail_names = {name for _, name in HARD_FAIL_PATTERNS}

    seen_lower: set[str] = set()
    alpha_words = 0
    eligible_for_freq = 0
    eligible_for_training = 0
    empty_or_invalid = 0
    lengths: list[int] = []
    issues: list[str] = []

    for word in words:
        stripped = word.strip()
        if not stripped:
            empty_or_invalid += 1
            continue

        seen_lower.add(stripped.lower())
        lengths.append(len(stripped))

        if stripped.isalpha():
            alpha_words += 1
            if len(stripped) >= min_length:
                eligible_for_freq += 1
            if training_min_length <= len(stripped) <= training_max_length:
                eligible_for_training += 1

        for pattern, name in SUSPICIOUS_PATTERNS:
            if pattern.search(stripped):
                suspicious_counts[name] += 1
                if len(suspicious_examples[name]) < 5:
                    suspicious_examples[name].append(stripped)

    total = len(words)
    hard_fail_total = sum(
        count for name, count in suspicious_counts.items() if name in hard_fail_names
    )
    suspicious_ratio = hard_fail_total / total if total else 0.0

    if total < 100_000:
        issues.append(f"Low word count: {total} (recommended >= 100,000)")
    if eligible_for_training < 50_000:
        issues.append(
            f"Low training-eligible words: {eligible_for_training} (recommended >= 50,000)"
        )
    if suspicious_ratio > max_suspicious_ratio:
        issues.append(
            "Suspicious pattern ratio "
            f"{suspicious_ratio:.4%} exceeds {max_suspicious_ratio:.4%}"
        )
    if suspicious_counts["url"] > 0 or suspicious_counts["ip_address"] > 0:
        issues.append("Found URL/IP-like entries in corpus")

    passed = not issues

    return CorpusReport(
        total_words=total,
        unique_words_lower=len(seen_lower),
        alpha_words=alpha_words,
        eligible_for_freq=eligible_for_freq,
        eligible_for_training=eligible_for_training,
        min_length=min(lengths) if lengths else 0,
        max_length=max(lengths) if lengths else 0,
        avg_length=sum(lengths) / len(lengths) if lengths else 0.0,
        suspicious_counts=suspicious_counts,
        suspicious_examples=suspicious_examples,
        empty_or_invalid=empty_or_invalid,
        passed=passed,
        issues=issues,
    )
