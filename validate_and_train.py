#!/usr/bin/env python3
"""Validate corpus, train with parallel CPU, and run scoring tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from randomness_detection.bootstrap import bootstrap, load_ensemble, load_freq_counter
from randomness_detection.config import CPU_FRACTION, TRAIN_SAMPLES_PER_CLASS
from randomness_detection.corpora import filter_words_for_training, load_merged_words
from randomness_detection.corpus_validator import validate_words
from randomness_detection.features import extract_features
from randomness_detection.parallel import worker_count
from randomness_detection.scorer import Scorer


CACHE_DIR = Path(__file__).resolve().parent / ".cache_test"
EXPECTED_WORKERS = worker_count(CPU_FRACTION)


def sample_process_tree_cpu(pid: int) -> float:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"ps -o %cpu= --ppid {pid} 2>/dev/null; ps -p {pid} -o %cpu= 2>/dev/null",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    total = 0.0
    for token in result.stdout.split():
        try:
            total += float(token)
        except ValueError:
            continue
    return total


def sample_total_cpu() -> float:
    with open("/proc/stat", encoding="utf-8") as handle:
        parts = handle.readline().split()[1:]
    values = list(map(int, parts))
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def monitor_cpu_during(pid: int, seconds: float = 8.0) -> dict[str, float]:
    idle1, total1 = sample_total_cpu()
    samples: list[float] = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        samples.append(sample_process_cpu(pid))
        time.sleep(0.5)

    idle2, total2 = sample_total_cpu()
    total_delta = total2 - total1
    idle_delta = idle2 - idle1
    system_usage = 0.0
    if total_delta > 0:
        system_usage = (1.0 - (idle_delta / total_delta)) * 100.0

    process_peak = max(samples) if samples else 0.0
    process_avg = sum(samples) / len(samples) if samples else 0.0
    return {
        "process_peak_cpu_percent": process_peak,
        "process_avg_cpu_percent": process_avg,
        "system_cpu_percent": system_usage,
        "samples": float(len(samples)),
    }


def run_corpus_validation(cache_dir: Path) -> dict:
    words = load_merged_words(cache_dir)
    training_words = filter_words_for_training(words, min_length=3)
    report = validate_words(training_words, min_length=3, training_min_length=3)
    return {
        "merged_words": len(words),
        "training_words": len(training_words),
        "report": report.to_dict(),
    }


def run_scoring_tests(cache_dir: Path) -> list[dict]:
    scorer = Scorer(cache_dir=cache_dir, auto_bootstrap=False)
    cases = [
        ("hello", "natural", 40),
        ("beautiful", "natural", 40),
        ("google", "natural", 40),
        ("xk9f2m8q1p4v7n2", "likely_random", 60),
        ("a3f9b2c1d8e7f0a1", "likely_random", 60),
        ("qxzqvmnrwtpl", "likely_random", 60),
    ]
    results = []
    for text, expected_label, threshold in cases:
        result = scorer.score(text)
        passed = (
            result.label == expected_label
            if expected_label != "likely_random"
            else result.score >= threshold
        )
        results.append(
            {
                "text": text,
                "expected": expected_label,
                "score": result.score,
                "label": result.label,
                "passed": passed,
            }
        )
    return results


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpus = os.cpu_count() or 1

    print("=== Corpus Validation ===")
    corpus = run_corpus_validation(CACHE_DIR)
    print(json.dumps(corpus, indent=2))
    if not corpus["report"]["passed"]:
        print("FAIL: corpus validation failed", file=sys.stderr)
        return 1

    print("\n=== Training (50% CPU) ===")
    print(f"System CPUs: {cpus}")
    print(f"Configured workers: {EXPECTED_WORKERS} ({CPU_FRACTION:.0%} of CPUs)")

    start = time.time()
    pid = os.getpid()
    cpu_stats: dict[str, float] = {
        "process_peak_cpu_percent": 0.0,
        "process_avg_cpu_percent": 0.0,
        "samples": 0.0,
    }

    import threading

    stop_event = threading.Event()
    samples: list[float] = []

    def monitor() -> None:
        while not stop_event.is_set():
            samples.append(sample_process_tree_cpu(pid))
            time.sleep(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    metadata = bootstrap(CACHE_DIR, force=True)

    stop_event.set()
    monitor_thread.join(timeout=2)
    elapsed = time.time() - start

    if samples:
        cpu_stats["process_peak_cpu_percent"] = max(samples)
        cpu_stats["process_avg_cpu_percent"] = sum(samples) / len(samples)
        cpu_stats["samples"] = float(len(samples))

    print(f"Training completed in {elapsed:.1f}s")
    print(json.dumps(metadata, indent=2))

    workers_used = metadata.get("cpu_workers", metadata.get("metrics", {}).get("cpu_workers"))
    cpu_ok = workers_used == EXPECTED_WORKERS
    cpu_used_ok = cpu_stats["process_peak_cpu_percent"] >= (EXPECTED_WORKERS * 30.0)

    print("\n=== CPU Verification ===")
    print(json.dumps({"expected_workers": EXPECTED_WORKERS, "workers_used": workers_used, **cpu_stats}, indent=2))
    print(f"Worker count OK: {cpu_ok}")
    print(f"CPU usage detected OK: {cpu_used_ok}")

    print("\n=== Scoring Tests ===")
    test_results = run_scoring_tests(CACHE_DIR)
    print(json.dumps(test_results, indent=2))
    tests_passed = all(item["passed"] for item in test_results)

    print("\n=== Summary ===")
    print(f"Words for training: {corpus['training_words']:,}")
    print(f"Recommended minimum: 50,000 -> {'OK' if corpus['training_words'] >= 50_000 else 'LOW'}")
    print(f"Training samples/class: {TRAIN_SAMPLES_PER_CLASS:,}")
    print(f"Corpus validation: PASS")
    print(f"Parallel workers: {'PASS' if cpu_ok else 'FAIL'}")
    print(f"CPU utilization: {'PASS' if cpu_used_ok else 'FAIL'}")
    print(f"Scoring tests: {'PASS' if tests_passed else 'FAIL'}")

    if not (cpu_ok and cpu_used_ok and tests_passed):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
