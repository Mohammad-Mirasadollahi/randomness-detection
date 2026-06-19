"""Extended feature vector for the randomness detection ensemble."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .features import FeatureVector, extract_features
from .freq_model import FreqCounter
from .ngram_lm import CharacterNgramLM
from .pmi_model import WordPMIModel

# Feature groups for ablation studies (paper Table: ablation rows).
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "statistical": (
        "freq_randomness",
        "freq_avg_natural",
        "freq_total_natural",
        "entropy_normalized",
        "compression_ratio",
    ),
    "structural": (
        "length_normalized",
        "unique_char_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "is_base64_like",
        "vowel_ratio",
        "max_consonant_run_ratio",
    ),
    "lexical": (
        "lexical_coverage",
        "longest_word_ratio",
        "word_count",
    ),
    "language_model": (
        "lm_cross_entropy",
        "lm_log_prob_norm",
        "lm_perplexity_norm",
    ),
    "pmi": (
        "pmi_mean",
        "pmi_min",
        "pmi_pairs",
    ),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS.values() for name in group
)


@dataclass
class EnsembleFeatureVector:
    """All features used by the production ensemble."""

    # --- statistical (production signals) ---
    freq_randomness: float
    freq_avg_natural: float
    freq_total_natural: float
    entropy_normalized: float
    compression_ratio: float
    # --- structural ---
    length_normalized: float
    unique_char_ratio: float
    digit_ratio: float
    uppercase_ratio: float
    is_base64_like: float
    vowel_ratio: float
    max_consonant_run_ratio: float
    # --- lexical ---
    lexical_coverage: float
    longest_word_ratio: float
    word_count: float
    # --- language model (novel) ---
    lm_cross_entropy: float
    lm_log_prob_norm: float
    lm_perplexity_norm: float
    # --- PMI (novel) ---
    pmi_mean: float
    pmi_min: float
    pmi_pairs: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def as_list(self, *, active_groups: frozenset[str] | None = None) -> list[float]:
        data = self.as_dict()
        if active_groups is None:
            return [data[name] for name in FEATURE_NAMES]

        selected: list[float] = []
        for group, names in FEATURE_GROUPS.items():
            if group in active_groups:
                selected.extend(data[name] for name in names)
        return selected

    @classmethod
    def group_indices(cls, active_groups: frozenset[str]) -> list[int]:
        indices: list[int] = []
        offset = 0
        for group, names in FEATURE_GROUPS.items():
            size = len(names)
            if group in active_groups:
                indices.extend(range(offset, offset + size))
            offset += size
        return indices


def _normalize_lm_signals(text: str, lm: CharacterNgramLM) -> tuple[float, float, float]:
    """Return (cross_entropy, log_prob_norm, perplexity_norm) in roughly [0, 1]."""
    cross_entropy = lm.cross_entropy(text)
    log_prob = lm.log_probability(text)
    perplexity = lm.perplexity(text)

    # Map to [0, 1] with soft saturation — higher = more random-looking.
    ce_norm = min(cross_entropy / 8.0, 1.0)
    log_prob_norm = min(max(-log_prob / (max(len(text), 1) * 6.0), 0.0), 1.0)
    ppl_norm = min(math.log1p(perplexity) / 8.0, 1.0)  # type: ignore[name-defined]

    return ce_norm, log_prob_norm, ppl_norm


def extract_ensemble_features(
    text: str,
    freq_counter: FreqCounter,
    language_model: CharacterNgramLM,
    pmi_model: WordPMIModel,
) -> EnsembleFeatureVector:
    """Combine statistical, lexical, LM, and PMI signals."""
    base: FeatureVector = extract_features(text, freq_counter)
    lexicon = getattr(freq_counter, "lexicon", frozenset())

    ce_norm, log_prob_norm, ppl_norm = _normalize_lm_signals(text, language_model)
    pmi_mean, pmi_min, pair_count = pmi_model.segmentation_pmi(text, lexicon)

    # PMI: map [-8, 8] → [0, 1] where low PMI (word-salad) → high randomness signal
    pmi_mean_norm = min(max((8.0 - pmi_mean) / 16.0, 0.0), 1.0)
    pmi_min_norm = min(max((8.0 - pmi_min) / 16.0, 0.0), 1.0)
    pair_norm = min(pair_count / 8.0, 1.0)

    return EnsembleFeatureVector(
        freq_randomness=base.freq_randomness / 100.0,
        freq_avg_natural=base.freq_avg_natural / 100.0,
        freq_total_natural=base.freq_total_natural / 100.0,
        entropy_normalized=base.entropy_normalized,
        compression_ratio=base.compression_ratio,
        length_normalized=base.length_normalized,
        unique_char_ratio=base.unique_char_ratio,
        digit_ratio=base.digit_ratio,
        uppercase_ratio=base.uppercase_ratio,
        is_base64_like=base.is_base64_like,
        vowel_ratio=base.vowel_ratio,
        max_consonant_run_ratio=base.max_consonant_run_ratio,
        lexical_coverage=base.lexical_coverage,
        longest_word_ratio=base.longest_word_ratio,
        word_count=min(base.word_count / 12.0, 1.0),
        lm_cross_entropy=ce_norm,
        lm_log_prob_norm=log_prob_norm,
        lm_perplexity_norm=ppl_norm,
        pmi_mean=pmi_mean_norm,
        pmi_min=pmi_min_norm,
        pmi_pairs=pair_norm,
    )


def breakdown_ensemble(features: EnsembleFeatureVector) -> dict[str, int]:
    """Human-readable component scores (1–100) for API responses."""
    return {
        "freq": int(round(features.freq_randomness * 100)),
        "entropy": int(round(features.entropy_normalized * 100)),
        "compression": int(round(features.compression_ratio * 100)),
        "language_model": int(round(features.lm_perplexity_norm * 100)),
        "pmi": int(round(features.pmi_mean * 100)),
        "lexical": int(round((1.0 - features.lexical_coverage) * 100)),
    }
