#!/usr/bin/env python3
"""
Integration tests for LRD-Hybrid research model.

Covers:
  1. Artifact presence + bootstrap
  2. Fixed-string smoke (natural vs random vs dictionary-DGA)
  3. Quality hold-out (same protocol as test_quality_benchmark.py)
  4. Robustness buckets (same curated data as test_robustness.py)

No mocks — uses trained LM, PMI, and ensemble on real corpus data.
"""

from __future__ import annotations

import os
import sys

from randomness_detection.config import TRAIN_SAMPLES_PER_CLASS
from randomness_detection.research.hybrid_bootstrap import (
    HYBRID_ENSEMBLE_NAME,
    HYBRID_LM_NAME,
    HYBRID_METADATA_NAME,
    HYBRID_PMI_NAME,
    bootstrap_hybrid,
    is_hybrid_bootstrapped,
)
from randomness_detection.research.hybrid_scorer import HybridScorer
from randomness_detection.scorer import Scorer
from test_quality_benchmark import (
    build_holdout_dataset,
    evaluate_method,
    resolve_cache_dir,
    split_calibration_test,
)
from test_helpers import load_real_words
from test_robustness import CLEAR_RANDOM, CORE_NATURAL, RANDOM_DECISION

# Dictionary-DGA: 4+ segmented words (primary paper target)
DICTIONARY_DGA = [
    "theirtheandaloneinto",
    "whenwherehowwhatwhy",
    "afterbeforeduringwhile",
    "youonehimoutnow",
]

# 3-word salad — diagnostic only (structurally similar to compounds)
DICTIONARY_DGA_HARD = [
    "boxcarmittenglow",
    "redfoxjumphigh",
]

# Must stay natural (score LOW)
MUST_BE_NATURAL = [
    "hello",
    "computer",
    "hello-world",
    "stackoverflow",
]

# Must be random (score HIGH)
MUST_BE_RANDOM = [
    "qzxwvbnmklpr",
    "550e8400-e29b-41d4-a716-446655440000",
    "a7f3k9x2m1p4",
]


def ensure_hybrid(cache_dir) -> None:
    min_samples = int(
        os.environ.get("RANDOMNESS_RESEARCH_TRAIN_SAMPLES", str(TRAIN_SAMPLES_PER_CLASS))
    )
    force = os.environ.get("RANDOMNESS_RESEARCH_FORCE", "").strip() in ("1", "true", "yes")
    if is_hybrid_bootstrapped(cache_dir) and not force:
        import json

        meta = json.loads((cache_dir / HYBRID_METADATA_NAME).read_text(encoding="utf-8"))
        if int(meta.get("samples_per_class", 0)) >= min_samples:
            return
        print(f"[test] Hybrid trained with {meta.get('samples_per_class')} samples; retraining with {min_samples} …")
    else:
        print(f"[test] Bootstrapping LRD-Hybrid ({min_samples} samples/class) …")
    bootstrap_hybrid(cache_dir, samples_per_class=min_samples, force=True, verbose=True)


def check_artifacts(cache_dir) -> None:
    for name in (HYBRID_LM_NAME, HYBRID_PMI_NAME, HYBRID_ENSEMBLE_NAME, HYBRID_METADATA_NAME):
        path = cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing hybrid artifact: {path}")


def smoke_fixed_strings(scorer: HybridScorer) -> list[str]:
    errors: list[str] = []
    for text in MUST_BE_NATURAL:
        score = scorer.score(text).score
        if score >= RANDOM_DECISION:
            errors.append(f"natural FP: {text!r} score={score}")
    for text in MUST_BE_RANDOM:
        score = scorer.score(text).score
        if score < RANDOM_DECISION:
            errors.append(f"random FN: {text!r} score={score}")
    for text in DICTIONARY_DGA:
        score = scorer.score(text).score
        if score < RANDOM_DECISION:
            errors.append(f"dictionary-DGA FN: {text!r} score={score}")
    return errors


def robustness_rates(scorer: HybridScorer) -> tuple[float, float]:
    core_fp = sum(
        1 for text in CORE_NATURAL if scorer.score(text).score >= RANDOM_DECISION
    ) / len(CORE_NATURAL)
    clear_fn = sum(
        1 for text in CLEAR_RANDOM if scorer.score(text).score < RANDOM_DECISION
    ) / len(CLEAR_RANDOM)
    return core_fp, clear_fn


def main() -> int:
    cache_dir = resolve_cache_dir()
    ensure_hybrid(cache_dir)
    check_artifacts(cache_dir)

    print("=" * 72)
    print("LRD-HYBRID INTEGRATION TESTS")
    print("=" * 72)
    print(f"Cache: {cache_dir}")

    hybrid = HybridScorer(cache_dir, auto_bootstrap=False)
    production = Scorer(cache_dir, auto_bootstrap=False)

    print("\n[1/4] Fixed-string smoke …")
    smoke_errors = smoke_fixed_strings(hybrid)
    for error in smoke_errors:
        print(f"  FAIL {error}")
    smoke_ok = not smoke_errors
    print(f"  {'PASS' if smoke_ok else 'FAIL'} ({len(smoke_errors)} errors)")

    print("\n[2/4] Robustness (curated real-world buckets) …")
    core_fp, clear_fn = robustness_rates(hybrid)
    robust_ok = core_fp <= 0.10 and clear_fn <= 0.05
    print(f"  core_natural FP={core_fp:.0%} (<=10%)")
    print(f"  clear_random FN={clear_fn:.0%} (<=5%)")
    print(f"  {'PASS' if robust_ok else 'FAIL'}")

    print("\n[3/4] Quality hold-out (same protocol as test_quality_benchmark) …")
    words = load_real_words(cache_dir, limit=100_000)
    all_samples = build_holdout_dataset(words)
    cal, test = split_calibration_test(all_samples)

    from test_quality_benchmark import MethodScorer, ensure_external_tools

    tools = ensure_external_tools(cache_dir)
    method_scorer = MethodScorer(cache_dir, tools)
    hybrid_metrics = evaluate_method("lrd_hybrid", cal, test, method_scorer)
    prod_metrics = evaluate_method("randomness_detection", cal, test, method_scorer)
    quality_ok = hybrid_metrics.f1 >= prod_metrics.f1 and hybrid_metrics.roc_auc >= 0.95
    print(
        f"  lrd_hybrid     F1={hybrid_metrics.f1:.3f}  AUC={hybrid_metrics.roc_auc:.3f}  "
        f"FPR={hybrid_metrics.false_positive_rate:.3f}  FNR={hybrid_metrics.false_negative_rate:.3f}"
    )
    print(
        f"  production     F1={prod_metrics.f1:.3f}  AUC={prod_metrics.roc_auc:.3f}"
    )
    print(f"  {'PASS' if quality_ok else 'FAIL'} (expect F1>={prod_metrics.f1:.3f}, AUC>=0.95)")

    print("\n[4/4] Dictionary-DGA recall (4+ word salad) …")
    dga_hits = sum(1 for text in DICTIONARY_DGA if hybrid.score(text).score >= RANDOM_DECISION)
    dga_recall = dga_hits / len(DICTIONARY_DGA)
    dga_ok = dga_recall >= 0.75
    print(f"  recall={dga_recall:.0%} ({dga_hits}/{len(DICTIONARY_DGA)})  threshold={RANDOM_DECISION}")
    hard_hits = sum(
        1 for text in DICTIONARY_DGA_HARD if hybrid.score(text).score >= RANDOM_DECISION
    )
    print(
        f"  diagnostic 3-word salad: {hard_hits}/{len(DICTIONARY_DGA_HARD)} "
        f"(not pass/fail — see Docs/research-lrd-hybrid.md)"
    )
    print(f"  {'PASS' if dga_ok else 'FAIL'} (expect >=75% on 4+ word salad)")

    print("\n" + "=" * 72)
    overall = smoke_ok and robust_ok and quality_ok and dga_ok
    print(f"RESEARCH HYBRID TEST: {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
