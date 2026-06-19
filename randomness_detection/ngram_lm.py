"""Character n-gram language model with modified Kneser-Ney-style smoothing."""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import atomic_pickle_dump


class CharacterNgramLM:
    """
    Order-*n* character LM trained on natural-language strings.

    Perplexity and cross-entropy are the primary signals: natural text gets
    lower perplexity; random / word-salad strings get higher perplexity.
    """

    def __init__(self, order: int = 5, discount: float = 0.75, min_count: int = 2) -> None:
        if order < 2:
            raise ValueError("order must be >= 2")
        self.order = order
        self.discount = discount
        self.min_count = min_count
        self._counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._context_totals: Counter[tuple[str, ...]] = Counter()
        self._unigram: Counter[str] = Counter()
        self._vocab_size = 0
        self._trained = False

    def train(self, texts: list[str]) -> None:
        """Fit n-gram counts from lowercase training strings."""
        self._counts.clear()
        self._context_totals.clear()
        self._unigram.clear()

        for raw in texts:
            text = raw.lower().strip()
            if len(text) < 2:
                continue
            padded = ("^" * (self.order - 1)) + text + "$"
            chars = list(padded)
            for index in range(self.order - 1, len(chars)):
                context = tuple(chars[index - self.order + 1 : index])
                char = chars[index]
                self._counts[context][char] += 1
                self._context_totals[context] += 1
                self._unigram[char] += 1

        self._vocab_size = max(len(self._unigram), 1)
        self._trained = True

    def _char_prob(self, char: str, context: tuple[str, ...]) -> float:
        if not self._trained:
            return 1.0 / self._vocab_size

        counts = self._counts.get(context)
        if counts:
            total = self._context_totals[context]
            observed = counts.get(char, 0)
            if observed > 0:
                discounted = max(observed - self.discount, 0.0) / max(total, 1)
                if discounted > 0:
                    return discounted

        # Backoff to lower-order context
        if len(context) > 0:
            return self._char_prob(char, context[1:])

        # Unigram fallback with add-one smoothing
        return (self._unigram.get(char, 0) + 1.0) / (sum(self._unigram.values()) + self._vocab_size)

    def log_probability(self, text: str) -> float:
        """Sum of log P(char | context) for the string (higher = more natural)."""
        stripped = text.lower().strip()
        if len(stripped) < 1:
            return 0.0

        padded = ("^" * (self.order - 1)) + stripped + "$"
        chars = list(padded)
        total = 0.0
        for index in range(self.order - 1, len(chars)):
            context = tuple(chars[index - self.order + 1 : index])
            char = chars[index]
            prob = max(self._char_prob(char, context), 1e-12)
            total += math.log(prob)
        return total

    def cross_entropy(self, text: str) -> float:
        """Bits per character (higher = less language-like)."""
        stripped = text.lower().strip()
        length = max(len(stripped), 1)
        return -self.log_probability(text) / (length * math.log(2))

    def perplexity(self, text: str) -> float:
        """Standard perplexity; lower = more natural."""
        stripped = text.lower().strip()
        length = max(len(stripped), 1)
        return math.exp(-self.log_probability(text) / length)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "order": self.order,
            "discount": self.discount,
            "min_count": self.min_count,
            "counts": {ctx: dict(counter) for ctx, counter in self._counts.items()},
            "context_totals": dict(self._context_totals),
            "unigram": dict(self._unigram),
            "vocab_size": self._vocab_size,
            "trained": self._trained,
        }
        atomic_pickle_dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "CharacterNgramLM":
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        model = cls(
            order=payload["order"],
            discount=payload["discount"],
            min_count=payload.get("min_count", 2),
        )
        model._counts = defaultdict(Counter, {k: Counter(v) for k, v in payload["counts"].items()})
        model._context_totals = Counter(payload["context_totals"])
        model._unigram = Counter(payload["unigram"])
        model._vocab_size = payload["vocab_size"]
        model._trained = payload["trained"]
        return model
