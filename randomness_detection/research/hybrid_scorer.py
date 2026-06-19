"""High-level scorer for the LRD-Hybrid research model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_CACHE_DIR, LABEL_LIKELY_RANDOM, LABEL_NATURAL
from ..freq_model import FreqCounter
from .hybrid_bootstrap import (
    HYBRID_ENSEMBLE_NAME,
    HYBRID_LM_NAME,
    HYBRID_PMI_NAME,
    is_hybrid_bootstrapped,
)
from .hybrid_features import breakdown_hybrid, extract_hybrid_features
from .hybrid_trainer import HybridEnsembleModel
from .ngram_lm import CharacterNgramLM
from .pmi_model import WordPMIModel


@dataclass
class HybridScoreResult:
    score: int
    label: str
    confidence: str
    breakdown: dict[str, int]
    features: dict[str, float]
    model: str = "lrd-hybrid"


class HybridScorer:
    """Score strings with the research-grade LRD-Hybrid ensemble."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        auto_bootstrap: bool = True,
        force_bootstrap: bool = False,
    ) -> None:
        from .hybrid_bootstrap import bootstrap_hybrid

        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)

        if force_bootstrap or (auto_bootstrap and not is_hybrid_bootstrapped(self.cache_dir)):
            bootstrap_hybrid(self.cache_dir, force=force_bootstrap)

        self._freq_counter = self._load_freq()
        self._language_model = CharacterNgramLM.load(self.cache_dir / HYBRID_LM_NAME)
        self._pmi_model = WordPMIModel.load(self.cache_dir / HYBRID_PMI_NAME)
        self._ensemble = HybridEnsembleModel.load(self.cache_dir / HYBRID_ENSEMBLE_NAME)

    def _load_freq(self) -> FreqCounter:
        from ..bootstrap import load_freq_counter

        return load_freq_counter(self.cache_dir)

    def score(self, text: str) -> HybridScoreResult:
        stripped = text.strip()
        if not stripped:
            return HybridScoreResult(
                score=1,
                label="natural",
                confidence="low",
                breakdown={"freq": 0, "entropy": 0, "compression": 0, "language_model": 0, "pmi": 0, "lexical": 0},
                features={"length": 0},
            )

        features = extract_hybrid_features(
            stripped,
            self._freq_counter,
            self._language_model,
            self._pmi_model,
        )
        probability = self._ensemble.predict_random_probability(features)
        score = max(1, min(100, int(round(probability * 100))))

        if score >= LABEL_LIKELY_RANDOM:
            label = "likely_random"
        elif score <= LABEL_NATURAL:
            label = "natural"
        else:
            label = "uncertain"

        breakdown = breakdown_hybrid(features)
        values = list(breakdown.values())
        spread = max(values) - min(values) if values else 0
        confidence = "high" if spread >= 25 else "low"
        if len(stripped) < 4:
            confidence = "low"

        feature_dict = features.as_dict()
        feature_dict["length"] = float(len(stripped))

        return HybridScoreResult(
            score=score,
            label=label,
            confidence=confidence,
            breakdown=breakdown,
            features=feature_dict,
        )
