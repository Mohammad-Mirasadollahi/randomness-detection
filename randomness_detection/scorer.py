"""Main scoring API."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap, is_bootstrapped, load_ensemble, load_freq_counter
from .config import (
    CONFIDENCE_STD_THRESHOLD,
    DEFAULT_CACHE_DIR,
    LABEL_LIKELY_RANDOM,
    LABEL_NATURAL,
)
from .features import breakdown_scores, extract_features


@dataclass
class ScoreResult:
    score: int
    label: str
    confidence: str
    breakdown: dict[str, int]
    features: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "breakdown": self.breakdown,
            "features": self.features,
        }


@dataclass
class BatchScoreResult:
    text: str
    result: ScoreResult

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, **self.result.to_dict()}


class Scorer:
    """Detect randomness in strings. Bootstraps automatically on first use."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        auto_bootstrap: bool = True,
        force_bootstrap: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self._freq_counter = None
        self._ensemble = None

        if force_bootstrap or (auto_bootstrap and not is_bootstrapped(self.cache_dir)):
            bootstrap(self.cache_dir, force=force_bootstrap)

        self._freq_counter = load_freq_counter(self.cache_dir)
        self._ensemble = load_ensemble(self.cache_dir)

    def score(self, text: str) -> ScoreResult:
        if self._freq_counter is None or self._ensemble is None:
            raise RuntimeError("Scorer is not initialized.")

        stripped = text.strip()
        if not stripped:
            return ScoreResult(
                score=1,
                label="natural",
                confidence="low",
                breakdown={"freq": 0, "entropy": 0, "compression": 0},
                features={"length": 0},
            )

        features = extract_features(stripped, self._freq_counter)
        probability = self._ensemble.predict_random_probability(features.as_list())
        score = max(1, min(100, int(round(probability * 100))))

        breakdown = breakdown_scores(features)
        breakdown_values = list(breakdown.values())
        confidence = (
            "high"
            if len(breakdown_values) > 1
            and statistics.pstdev(v / 100.0 for v in breakdown_values)
            < CONFIDENCE_STD_THRESHOLD
            else "low"
        )

        if score >= LABEL_LIKELY_RANDOM:
            label = "likely_random"
        elif score <= LABEL_NATURAL:
            label = "natural"
        else:
            label = "uncertain"

        if features.length < 4:
            confidence = "low"

        return ScoreResult(
            score=score,
            label=label,
            confidence=confidence,
            breakdown=breakdown,
            features=features.to_dict(),
        )

    def score_many(self, texts: list[str]) -> list[ScoreResult]:
        return [self.score(text) for text in texts]

    def score_batch(self, texts: list[str]) -> list[BatchScoreResult]:
        return [
            BatchScoreResult(text=text, result=self.score(text))
            for text in texts
            if text.strip()
        ]
