"""Character bigram frequency model aligned with Mark Baggett's freq.py."""

from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict


DEFAULT_IGNORE_CHARS = "\n\t\r~@#%^&*\"'/\\-+<>{}|$!:()[];?,="


class _FollowerCounts:
    __slots__ = ("_counts",)

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def add(self, follower: str, weight: int = 1) -> None:
        self._counts[follower] += weight

    def get(self, follower: str) -> int:
        return self._counts.get(follower, 0)

    def total(self) -> int:
        return sum(self._counts.values())

    def items(self):
        return self._counts.items()

    def to_dict(self) -> dict[str, int]:
        return dict(self._counts)


class FreqCounter:
    """Bigram frequency counter with two probability measures (freq.py-style)."""

    def __init__(
        self,
        ignore_chars: str = DEFAULT_IGNORE_CHARS,
        ignore_case: bool = True,
    ) -> None:
        self.ignore_chars = ignore_chars
        self.ignore_case = ignore_case
        self._table: DefaultDict[str, _FollowerCounts] = defaultdict(_FollowerCounts)
        self.lexicon: frozenset[str] = frozenset()

    def set_lexicon(self, words: list[str], *, min_length: int = 3, max_length: int = 18) -> None:
        """Store a lowercase dictionary used for lexical-coverage features."""
        self.lexicon = frozenset(
            word.lower()
            for word in words
            if min_length <= len(word) <= max_length and word.isalpha()
        )

    def tally_str(self, text: str, weight: int = 1) -> None:
        if self.ignore_case:
            text = text.lower()

        pairs = re.findall(r"..", text)
        pairs.extend(re.findall(r"..", text[1:]))
        for first, second in pairs:
            self._table[first].add(second, weight)

    def tally_words(self, words: list[str]) -> None:
        for word in words:
            self.tally_str(word)

    def probability(self, text: str) -> tuple[float, float]:
        """
        Return two naturalness scores (0-100), higher = more natural.

        measure1: average per-bigram conditional probability
        measure2: aggregate P(all bigrams) = sum(pair counts) / sum(first-char totals)
        """
        if len(text) < 2:
            return 0.0, 0.0

        if self.ignore_case:
            text = text.lower()

        pairs = re.findall(r"..", text)
        pairs.extend(re.findall(r"..", text[1:]))

        pair_probs: list[float] = []
        total_first = 0
        total_pair = 0

        for first, second in pairs:
            if first in self.ignore_chars or second in self.ignore_chars:
                continue

            pair_prob = self._pair_probability(first, second)
            pair_probs.append(pair_prob)

            first_total, pair_count = self._pair_counts(first, second)
            total_first += first_total
            total_pair += pair_count

        avg_prob = (sum(pair_probs) / len(pair_probs) * 100.0) if pair_probs else 0.0
        total_prob = (total_pair / total_first * 100.0) if total_first else 0.0
        return round(avg_prob, 4), round(total_prob, 4)

    def naturalness_score(self, text: str) -> float:
        """Combined naturalness score (0-100), higher = more natural."""
        m1, m2 = self.probability(text)
        return (m1 + m2) / 2.0

    def randomness_score(self, text: str) -> float:
        """Randomness score (0-100), higher = more random."""
        return max(0.0, min(100.0, 100.0 - self.naturalness_score(text)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "table": {
                first: followers.to_dict()
                for first, followers in self._table.items()
            },
            "ignore_chars": self.ignore_chars,
            "ignore_case": self.ignore_case,
            "lexicon": sorted(self.lexicon),
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        self.ignore_chars = payload["ignore_chars"]
        self.ignore_case = payload["ignore_case"]
        self.lexicon = frozenset(payload.get("lexicon", ()))
        self._table = defaultdict(_FollowerCounts)
        for first, followers in payload["table"].items():
            counts = _FollowerCounts()
            for second, count in followers.items():
                counts.add(second, count)
            self._table[first] = counts

    def merge_table(self, table: dict[str, dict[str, int]]) -> None:
        for first, followers in table.items():
            for second, count in followers.items():
                self._table[first].add(second, count)

    def export_table(self) -> dict[str, dict[str, int]]:
        return {first: followers.to_dict() for first, followers in self._table.items()}

    def _pair_counts(self, first: str, second: str) -> tuple[int, int]:
        if self.ignore_case:
            lower = self._table.get(first.lower(), _FollowerCounts())
            upper = self._table.get(first.upper(), _FollowerCounts())
            first_total = lower.total() + upper.total()
            pair_count = lower.get(second) + upper.get(second)
            ignored = self._ignored_follower_total(first.lower()) + self._ignored_follower_total(
                first.upper()
            )
            return max(first_total - ignored, 0), pair_count

        followers = self._table.get(first, _FollowerCounts())
        first_total = followers.total()
        ignored = self._ignored_follower_total(first)
        return max(first_total - ignored, 0), followers.get(second)

    def _pair_probability(self, first: str, second: str) -> float:
        first_total, pair_count = self._pair_counts(first, second)
        if first_total == 0:
            return 0.0
        return pair_count / first_total

    def _ignored_follower_total(self, first: str) -> int:
        followers = self._table.get(first, _FollowerCounts())
        return sum(followers.get(char) for char in self.ignore_chars)
