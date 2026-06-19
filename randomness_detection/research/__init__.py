"""
LRD-Hybrid research module — Linguistic Randomness Detector.

Extends the production scorer with:
  - Character n-gram language model (perplexity / cross-entropy)
  - Word-segmentation PMI (dictionary co-occurrence plausibility)
  - Gradient-boosted calibrated ensemble (non-linear feature fusion)

Designed for reproducible ablation studies and paper-grade benchmarks.
"""

from .hybrid_bootstrap import bootstrap_hybrid, is_hybrid_bootstrapped, load_hybrid_scorer
from .hybrid_scorer import HybridScorer

__all__ = [
    "HybridScorer",
    "bootstrap_hybrid",
    "is_hybrid_bootstrapped",
    "load_hybrid_scorer",
]
