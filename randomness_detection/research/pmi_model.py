"""Word bigram PMI model for segmentation plausibility."""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

from .io_utils import atomic_pickle_dump
from .segmentation import segment_alpha_runs


class WordPMIModel:
    """
    Pointwise mutual information between adjacent dictionary words.

    Low average PMI across a segmentation suggests concatenated word-salad
    (dictionary DGA) rather than a natural compound or single word.
    """

    def __init__(self, min_count: int = 2, floor_pmi: float = -8.0) -> None:
        self.min_count = min_count
        self.floor_pmi = floor_pmi
        self._unigram: Counter[str] = Counter()
        self._bigram: Counter[tuple[str, str]] = Counter()
        self._total_bigrams = 0
        self._trained = False

    def train_from_words(self, words: list[str]) -> None:
        """Unigram counts from the corpus word list."""
        for word in words:
            token = word.lower().strip()
            if token.isalpha() and len(token) >= 3:
                self._unigram[token] += 1
        self._trained = True

    def train_bigrams_from_texts(self, texts: list[str], lexicon: frozenset[str]) -> None:
        """Collect adjacent word pairs from segmented natural training strings."""
        for text in texts:
            segments = segment_alpha_runs(text, lexicon)
            for left, right in zip(segments, segments[1:]):
                self._bigram[(left, right)] += 1
                self._total_bigrams += 1
        self._trained = True

    def pmi(self, left: str, right: str) -> float:
        if not self._trained or self._total_bigrams == 0:
            return self.floor_pmi

        pair_count = self._bigram.get((left, right), 0)
        if pair_count < self.min_count:
            return self.floor_pmi

        p_xy = pair_count / self._total_bigrams
        p_x = self._unigram.get(left, 0) / max(sum(self._unigram.values()), 1)
        p_y = self._unigram.get(right, 0) / max(sum(self._unigram.values()), 1)
        if p_x <= 0 or p_y <= 0 or p_xy <= 0:
            return self.floor_pmi
        return math.log2(p_xy / (p_x * p_y))

    def segmentation_pmi(self, text: str, lexicon: frozenset[str]) -> tuple[float, float, int]:
        """
        Return (mean_pmi, min_pmi, pair_count) over adjacent segmented words.

        When pair_count == 0, mean/min are set to floor_pmi.
        """
        segments = segment_alpha_runs(text, lexicon)
        if len(segments) < 2:
            return self.floor_pmi, self.floor_pmi, 0

        scores = [self.pmi(left, right) for left, right in zip(segments, segments[1:])]
        return sum(scores) / len(scores), min(scores), len(scores)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "min_count": self.min_count,
            "floor_pmi": self.floor_pmi,
            "unigram": dict(self._unigram),
            "bigram": {f"{a}\t{b}": count for (a, b), count in self._bigram.items()},
            "total_bigrams": self._total_bigrams,
            "trained": self._trained,
        }
        atomic_pickle_dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "WordPMIModel":
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        model = cls(
            min_count=payload.get("min_count", 2),
            floor_pmi=payload.get("floor_pmi", -8.0),
        )
        model._unigram = Counter(payload["unigram"])
        model._bigram = Counter()
        for key, count in payload["bigram"].items():
            left, right = key.split("\t", 1)
            model._bigram[(left, right)] = count
        model._total_bigrams = payload["total_bigrams"]
        model._trained = payload["trained"]
        return model
