#!/usr/bin/env python3
"""
LRD-Hybrid research benchmark — ablation study + comparison vs production scorer.

Designed for paper tables:
  - Full hybrid vs production logistic ensemble
  - Ablation: remove LM, PMI, lexical, statistical groups
  - Metrics: F1, ROC-AUC, FPR, FNR on held-out quality split

Usage:
  PYTHONPATH=. .venv/bin/python research_benchmark.py
  PYTHONPATH=. .venv/bin/python research_benchmark.py --quick
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

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from randomness_detection.config import DEFAULT_CACHE_DIR, TRAIN_SAMPLES_PER_CLASS
from randomness_detection.research.hybrid_bootstrap import (
    bootstrap_hybrid,
    extract_hybrid_features_parallel,
    is_hybrid_bootstrapped,
)
from randomness_detection.research.hybrid_features import FEATURE_GROUPS
from randomness_detection.research.hybrid_trainer import train_hybrid_ensemble
from randomness_detection.scorer import Scorer
from test_helpers import load_real_words

BASE = Path(__file__).resolve().parent
RESULTS_FILE = BASE / "research_benchmark_results.json"


@dataclass(frozen=True)
class Sample:
    text: str
    label: int
    category: str


@dataclass
class RowMetrics:
    name: str
    f1: float
    roc_auc: float
    fpr: float
    fnr: float
    threshold: int


def build_eval_set(words: list[str], *, seed: int = 2026) -> list[Sample]:
    """Same protocol as test_quality_benchmark.py (reproducible hold-out)."""
    rng = random.Random(seed)
    eligible = sorted(
        word for word in words if word.isalpha() and 5 <= len(word) <= 32
    )
    half = len(eligible) // 2
    corpus_slice = eligible[half:]

    samples: list[Sample] = []
    for word in rng.sample(corpus_slice, min(800, len(corpus_slice))):
        samples.append(Sample(word, 0, "natural_word"))

    compounds = 0
    while compounds < 400:
        w1, w2 = rng.choice(corpus_slice), rng.choice(corpus_slice)
        sep = rng.choice(("", "-", "_"))
        samples.append(Sample(f"{w1}{sep}{w2}", 0, "natural_compound"))
        compounds += 1

    def rand_hex(n: int) -> str:
        return secrets.token_hex(n // 2)

    categories = [
        ("random_hex", lambda: rand_hex(rng.randint(8, 32))),
        ("random_urlsafe", lambda: secrets.token_urlsafe(rng.randint(6, 24)).rstrip("=")),
        (
            "random_alnum",
            lambda: "".join(
                secrets.choice(string.ascii_lowercase + string.digits)
                for _ in range(rng.randint(8, 24))
            ),
        ),
        (
            "random_consonant",
            lambda: "".join(
                secrets.choice("bcdfghjklmnpqrstvwxyz")
                for _ in range(rng.randint(8, 20))
            ),
        ),
    ]
    for category, generator in categories:
        for _ in range(400):
            samples.append(Sample(generator(), 1, category))

    rng.shuffle(samples)
    return samples


def best_threshold(scores: list[float], labels: list[int]) -> tuple[int, float]:
    best_t = 50
    best_f1 = -1.0
    for threshold in range(1, 100):
        preds = [1 if score >= threshold else 0 for score in scores]
        value = f1_score(labels, preds, zero_division=0)
        if value > best_f1:
            best_f1 = value
            best_t = threshold
    return best_t, best_f1


def evaluate_scores(
    name: str,
    scores: list[float],
    labels: list[int],
) -> RowMetrics:
    threshold, _ = best_threshold(scores, labels)
    preds = [1 if score >= threshold else 0 for score in scores]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return RowMetrics(
        name=name,
        f1=float(f1_score(labels, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, scores)),
        fpr=float(fpr),
        fnr=float(fnr),
        threshold=threshold,
    )


def run_ablation(
    cache_dir: Path,
    texts: list[str],
    labels: list[int],
    *,
    train_texts: list[str],
    train_labels: list[int],
    quick: bool,
) -> list[RowMetrics]:
    rows: list[RowMetrics] = []

    ablations: dict[str, frozenset[str]] = {
        "lrd_hybrid_full": frozenset(FEATURE_GROUPS.keys()),
        "lrd_hybrid_no_lm": frozenset(k for k in FEATURE_GROUPS if k != "language_model"),
        "lrd_hybrid_no_pmi": frozenset(k for k in FEATURE_GROUPS if k != "pmi"),
        "lrd_hybrid_no_lexical": frozenset(k for k in FEATURE_GROUPS if k != "lexical"),
        "lrd_hybrid_statistical_only": frozenset({"statistical"}),
    }

    train_features = extract_hybrid_features_parallel(train_texts, cache_dir)
    eval_features = extract_hybrid_features_parallel(texts, cache_dir)

    for name, groups in ablations.items():
        print(f"  training ablation: {name} …")
        model, _ = train_hybrid_ensemble(
            train_features,
            train_labels,
            active_groups=groups,
            test_size=0.25 if quick else 0.2,
        )
        scores = [
            model.predict_random_probability(row) * 100.0 for row in eval_features
        ]
        rows.append(evaluate_scores(name, scores, labels))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="LRD-Hybrid ablation benchmark")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Model cache directory",
    )
    parser.add_argument("--quick", action="store_true", help="Smaller train split, faster run")
    parser.add_argument("--force-train", action="store_true", help="Retrain hybrid artifacts")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    t0 = time.time()

    print("=" * 72)
    print("LRD-Hybrid Research Benchmark")
    print("=" * 72)

    if not is_hybrid_bootstrapped(cache_dir) or args.force_train:
        print("\n[1/4] Training LRD-Hybrid …")
        samples = TRAIN_SAMPLES_PER_CLASS // 4 if args.quick else TRAIN_SAMPLES_PER_CLASS
        bootstrap_hybrid(cache_dir, samples_per_class=samples, force=True, verbose=True)
    else:
        print("\n[1/4] Using existing LRD-Hybrid artifacts")

    print("\n[2/4] Building evaluation set …")
    words = load_real_words(cache_dir)
    all_samples = build_eval_set(words)
    split = int(len(all_samples) * 0.65)
    eval_samples = all_samples[:split]
    cal_samples = all_samples[split:]

    eval_texts = [sample.text for sample in eval_samples]
    eval_labels = [sample.label for sample in eval_samples]
    cal_texts = [sample.text for sample in cal_samples]
    cal_labels = [sample.label for sample in cal_samples]

    print(f"  eval={len(eval_samples)}  calibration={len(cal_samples)}")

    print("\n[3/4] Production baseline …")
    production = Scorer(cache_dir=cache_dir, auto_bootstrap=False)
    prod_scores = [production.score(text).score for text in eval_texts]
    prod_row = evaluate_scores("production_lr", prod_scores, eval_labels)

    print("\n[4/4] Ablation study …")
    ablation_rows = run_ablation(
        cache_dir,
        eval_texts,
        eval_labels,
        train_texts=cal_texts + eval_texts[: len(eval_texts) // 2],
        train_labels=cal_labels + eval_labels[: len(eval_labels) // 2],
        quick=args.quick,
    )

    all_rows = [prod_row, *ablation_rows]
    print("\n" + "-" * 72)
    print(f"{'Method':<28} {'F1':>7} {'AUC':>7} {'FPR':>7} {'FNR':>7} {'Thr':>5}")
    print("-" * 72)
    for row in sorted(all_rows, key=lambda item: item.f1, reverse=True):
        print(
            f"{row.name:<28} {row.f1:7.3f} {row.roc_auc:7.3f} "
            f"{row.fpr:7.3f} {row.fnr:7.3f} {row.threshold:5d}"
        )

    best = max(all_rows, key=lambda item: item.f1)
    hybrid_full = next((row for row in all_rows if row.name == "lrd_hybrid_full"), None)
    passed = (
        hybrid_full is not None
        and hybrid_full.f1 >= prod_row.f1
        and hybrid_full.roc_auc >= 0.95
    )

    payload = {
        "elapsed_seconds": round(time.time() - t0, 1),
        "eval_samples": len(eval_samples),
        "results": [asdict(row) for row in all_rows],
        "best_method": best.name,
        "passed": passed,
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"Results saved to {RESULTS_FILE}")
    print(f"RESEARCH BENCHMARK: {'PASS' if passed else 'FAIL'}  (best={best.name})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
