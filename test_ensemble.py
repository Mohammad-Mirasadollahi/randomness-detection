#!/usr/bin/env python3
"""
Integration tests for the full ensemble (LM + PMI + gradient boosting).

Covers smoke, robustness, quality hold-out, and dictionary-DGA recall.
Uses the same Scorer as production API and CLI.
"""

from __future__ import annotations

import sys

from randomness_detection.scorer import Scorer
from test_quality_benchmark import (
    build_holdout_dataset,
    evaluate_method,
    resolve_cache_dir,
    split_calibration_test,
)
from test_helpers import load_real_words
from test_robustness import CLEAR_RANDOM, CORE_NATURAL, RANDOM_DECISION

DICTIONARY_DGA = [
    "theirtheandaloneinto",
    "whenwherehowwhatwhy",
    "afterbeforeduringwhile",
    "youonehimoutnow",
]

DICTIONARY_DGA_HARD = [
    "boxcarmittenglow",
    "redfoxjumphigh",
]

MUST_BE_NATURAL = ["hello", "computer", "hello-world", "stackoverflow"]
MUST_BE_RANDOM = [
    "qzxwvbnmklpr",
    "550e8400-e29b-41d4-a716-446655440000",
    "a7f3k9x2m1p4",
]


def smoke_fixed_strings(scorer: Scorer) -> list[str]:
    errors: list[str] = []
    for text in MUST_BE_NATURAL:
        if scorer.score(text).score >= RANDOM_DECISION:
            errors.append(f"natural FP: {text!r}")
    for text in MUST_BE_RANDOM + DICTIONARY_DGA:
        if scorer.score(text).score < RANDOM_DECISION:
            errors.append(f"random FN: {text!r}")
    return errors


def robustness_rates(scorer: Scorer) -> tuple[float, float]:
    core_fp = sum(
        1 for text in CORE_NATURAL if scorer.score(text).score >= RANDOM_DECISION
    ) / len(CORE_NATURAL)
    clear_fn = sum(
        1 for text in CLEAR_RANDOM if scorer.score(text).score < RANDOM_DECISION
    ) / len(CLEAR_RANDOM)
    return core_fp, clear_fn


def main() -> int:
    cache_dir = resolve_cache_dir()
    scorer = Scorer(cache_dir=cache_dir, auto_bootstrap=False)

    print("=" * 72)
    print("ENSEMBLE INTEGRATION TESTS")
    print("=" * 72)
    print(f"Cache: {cache_dir}")

    print("\n[1/4] Fixed-string smoke …")
    smoke_errors = smoke_fixed_strings(scorer)
    for error in smoke_errors:
        print(f"  FAIL {error}")
    smoke_ok = not smoke_errors
    print(f"  {'PASS' if smoke_ok else 'FAIL'} ({len(smoke_errors)} errors)")

    print("\n[2/4] Robustness (curated real-world buckets) …")
    core_fp, clear_fn = robustness_rates(scorer)
    robust_ok = core_fp <= 0.10 and clear_fn <= 0.05
    print(f"  core_natural FP={core_fp:.0%} (<=10%)")
    print(f"  clear_random FN={clear_fn:.0%} (<=5%)")
    print(f"  {'PASS' if robust_ok else 'FAIL'}")

    print("\n[3/4] Quality hold-out …")
    words = load_real_words(cache_dir, limit=100_000)
    all_samples = build_holdout_dataset(words)
    cal, test = split_calibration_test(all_samples)
    from test_quality_benchmark import MethodScorer, ensure_external_tools

    tools = ensure_external_tools(cache_dir)
    method_scorer = MethodScorer(cache_dir, tools)
    metrics = evaluate_method("randomness_detection", cal, test, method_scorer)
    quality_ok = metrics.f1 >= 0.95 and metrics.roc_auc >= 0.95
    print(
        f"  F1={metrics.f1:.3f}  AUC={metrics.roc_auc:.3f}  "
        f"FPR={metrics.false_positive_rate:.3f}  FNR={metrics.false_negative_rate:.3f}"
    )
    print(f"  {'PASS' if quality_ok else 'FAIL'}")

    print("\n[4/4] Dictionary-DGA recall (4+ word salad) …")
    dga_hits = sum(1 for text in DICTIONARY_DGA if scorer.score(text).score >= RANDOM_DECISION)
    dga_recall = dga_hits / len(DICTIONARY_DGA)
    dga_ok = dga_recall >= 0.75
    hard_hits = sum(
        1 for text in DICTIONARY_DGA_HARD if scorer.score(text).score >= RANDOM_DECISION
    )
    print(f"  recall={dga_recall:.0%} ({dga_hits}/{len(DICTIONARY_DGA)})")
    print(f"  diagnostic 3-word salad: {hard_hits}/{len(DICTIONARY_DGA_HARD)}")
    print(f"  {'PASS' if dga_ok else 'FAIL'} (expect >=75%)")

    print("\n" + "=" * 72)
    overall = smoke_ok and robust_ok and quality_ok and dga_ok
    print(f"ENSEMBLE TEST: {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
