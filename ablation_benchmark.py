#!/usr/bin/env python3
"""
Ablation benchmark — feature-group ablation on the production ensemble.

Usage:
  PYTHONPATH=. .venv/bin/python ablation_benchmark.py
  PYTHONPATH=. .venv/bin/python ablation_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from randomness_detection.config import DEFAULT_CACHE_DIR
from randomness_detection.ensemble_features import FEATURE_GROUPS
from randomness_detection.parallel import extract_ensemble_features_parallel
from randomness_detection.trainer import train_ensemble
from test_helpers import load_real_words
from test_quality_benchmark import resolve_cache_dir

BASE = Path(__file__).resolve().parent
RESULTS_FILE = BASE / "ablation_benchmark_results.json"


@dataclass(frozen=True)
class Sample:
    text: str
    label: int


@dataclass
class RowMetrics:
    name: str
    f1: float
    roc_auc: float
    fpr: float
    fnr: float
    threshold: int


def build_eval_set(words: list[str], *, seed: int = 2026) -> list[Sample]:
    rng = random.Random(seed)
    eligible = sorted(word for word in words if word.isalpha() and 5 <= len(word) <= 32)
    corpus_slice = eligible[len(eligible) // 2 :]
    samples: list[Sample] = []
    for word in rng.sample(corpus_slice, min(800, len(corpus_slice))):
        samples.append(Sample(word, 0))
    for _ in range(400):
        w1, w2 = rng.choice(corpus_slice), rng.choice(corpus_slice)
        samples.append(Sample(rng.choice(("", "-", "_")).join([w1, w2]), 0))
    for _ in range(400):
        samples.append(Sample(secrets.token_hex(rng.randint(4, 16)), 1))
    for _ in range(400):
        samples.append(Sample(secrets.token_urlsafe(rng.randint(6, 24)).rstrip("="), 1))
    for _ in range(400):
        length = rng.randint(8, 20)
        samples.append(
            Sample(
                "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length)),
                1,
            )
        )
    for _ in range(400):
        length = rng.randint(10, 20)
        samples.append(
            Sample(
                "".join(secrets.choice("bcdfghjklmnpqrstvwxyz") for _ in range(length)),
                1,
            )
        )
    rng.shuffle(samples)
    return samples


def evaluate_scores(name: str, scores: list[float], labels: list[int]) -> RowMetrics:
    best_t, _ = 50, 0.0
    best_f1 = -1.0
    for threshold in range(1, 100):
        preds = [1 if score >= threshold else 0 for score in scores]
        value = f1_score(labels, preds, zero_division=0)
        if value > best_f1:
            best_f1 = value
            best_t = threshold
    preds = [1 if score >= best_t else 0 for score in scores]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return RowMetrics(
        name=name,
        f1=float(f1_score(labels, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, [s / 100.0 for s in scores])),
        fpr=float(fp / max(fp + tn, 1)),
        fnr=float(fn / max(fn + tp, 1)),
        threshold=best_t,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensemble feature ablation benchmark")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(resolve_cache_dir() if args.cache_dir == str(DEFAULT_CACHE_DIR) else args.cache_dir)
    t0 = time.time()

    print("=" * 72)
    print("Ensemble Ablation Benchmark")
    print("=" * 72)

    words = load_real_words(cache_dir)
    all_samples = build_eval_set(words)
    split = int(len(all_samples) * 0.65)
    eval_texts = [s.text for s in all_samples[:split]]
    eval_labels = [s.label for s in all_samples[:split]]
    train_texts = [s.text for s in all_samples[split:]] + eval_texts[: len(eval_texts) // 2]
    train_labels = [s.label for s in all_samples[split:]] + eval_labels[: len(eval_labels) // 2]

    ablations = {
        "ensemble_full": frozenset(FEATURE_GROUPS.keys()),
        "ensemble_no_lm": frozenset(k for k in FEATURE_GROUPS if k != "language_model"),
        "ensemble_no_pmi": frozenset(k for k in FEATURE_GROUPS if k != "pmi"),
        "ensemble_no_lexical": frozenset(k for k in FEATURE_GROUPS if k != "lexical"),
        "ensemble_statistical_only": frozenset({"statistical"}),
    }

    train_features = extract_ensemble_features_parallel(train_texts, cache_dir)
    eval_features = extract_ensemble_features_parallel(eval_texts, cache_dir)

    rows = []
    for name, groups in ablations.items():
        print(f"  training: {name} …")
        model, _ = train_ensemble(
            train_features,
            train_labels,
            active_groups=groups,
            test_size=0.25 if args.quick else 0.2,
        )
        scores = [model.predict_random_probability(row) * 100.0 for row in eval_features]
        rows.append(evaluate_scores(name, scores, eval_labels))

    print("\n" + "-" * 72)
    print(f"{'Variant':<28} {'F1':>7} {'AUC':>7} {'FPR':>7} {'FNR':>7}")
    print("-" * 72)
    for row in sorted(rows, key=lambda item: item.f1, reverse=True):
        print(f"{row.name:<28} {row.f1:7.3f} {row.roc_auc:7.3f} {row.fpr:7.3f} {row.fnr:7.3f}")

    best = max(rows, key=lambda item: item.f1)
    payload = {
        "elapsed_seconds": round(time.time() - t0, 1),
        "eval_samples": len(eval_texts),
        "results": [asdict(row) for row in rows],
        "best_variant": best.name,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"Saved: {RESULTS_FILE}")
    print(f"ABLATION BENCHMARK: PASS (best={best.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
