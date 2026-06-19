#!/usr/bin/env python3
"""
Quality benchmark — real held-out evaluation with per-method threshold calibration.

External tools (downloaded / installed automatically):
  - freqpy            Mark Baggett freq.py — cloned from GitHub, trained on words_alpha.txt
  - ent               Fourmilab ent — apt install, Shannon entropy via CLI
  - deflate_cli       Standalone DEFLATE ratio script in benchmark_tools/ (subprocess)

Internal baselines (same signal families, implemented in this project):
  - freq              Bigram frequency randomness score
  - entropy           Normalized Shannon entropy (internal)
  - compression       Raw DEFLATE ratio (internal)
  - avg3 / max3       Combinations of internal signals

Product:
  - randomness_detection  Production ensemble scorer

No mocks. Each method gets its own threshold on a calibration split; metrics on held-out test.
"""

from __future__ import annotations

import importlib.util
import json
import random
import secrets
import shutil
import string
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from randomness_detection.scorer import Scorer
from test_helpers import load_real_words

BASE = Path(__file__).resolve().parent
RESULTS_FILE = BASE / "quality_benchmark_results.json"
BENCHMARK_TOOLS = BASE / ".benchmark_tools"
BUNDLED_TOOLS = BASE / "benchmark_tools"
FREQPY_REPO = "https://github.com/MarkBaggett/freq.git"
FREQPY_DIR = BENCHMARK_TOOLS / "freq"
FREQPY_SCRIPT = FREQPY_DIR / "freq.py"
FREQPY_TABLE = BENCHMARK_TOOLS / "markbaggett.freq"

METHOD_SPECS: tuple[tuple[str, str, str], ...] = (
    ("randomness_detection", "product", "Production ensemble (LM + PMI + HistGradientBoosting)"),
    ("freqpy", "external", "Mark Baggett freq.py — https://github.com/MarkBaggett/freq"),
    ("ent", "external", "Fourmilab ent — apt package, Shannon entropy CLI"),
    ("deflate_cli", "external", "Standalone DEFLATE ratio CLI (benchmark_tools/deflate_score.py)"),
    ("freq", "internal", "Internal bigram frequency randomness score"),
    ("entropy", "internal", "Internal normalized Shannon entropy"),
    ("compression", "internal", "Internal raw DEFLATE compression ratio"),
    ("avg3", "internal", "Mean of internal freq + entropy + compression"),
    ("max3", "internal", "Max of internal freq + entropy + compression"),
)

METHODS = tuple(name for name, _, _ in METHOD_SPECS)


@dataclass(frozen=True)
class Sample:
    text: str
    label: int  # 0 = natural, 1 = random
    category: str


@dataclass
class MethodMetrics:
    method: str
    source: str
    description: str
    samples: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_positive_rate: float
    false_negative_rate: float
    natural_accuracy: float
    random_accuracy: float
    score_natural_mean: float
    score_random_mean: float
    confusion: dict[str, int]


@dataclass
class ExternalTools:
    freqpy_script: Path
    freqpy_table: Path
    ent_binary: str
    deflate_cli: Path


def method_meta(method: str) -> tuple[str, str]:
    for name, source, description in METHOD_SPECS:
        if name == method:
            return source, description
    raise KeyError(method)


def resolve_cache_dir() -> Path:
    import os

    env = os.environ.get("RANDOMNESS_CACHE_DIR", "").strip()
    if env:
        path = Path(env)
        if (path / "ensemble.pkl").exists():
            return path
    for candidate in (BASE / ".cache", BASE / ".cache_test"):
        if (candidate / "ensemble.pkl").exists():
            return candidate
    raise FileNotFoundError(
        "Trained model not found. Run: PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap"
    )


def build_holdout_dataset(words: list[str], *, seed: int = 2026) -> list[Sample]:
    rng = random.Random(seed)
    eligible = sorted({word for word in words if 5 <= len(word) <= 32 and word.isalpha()})
    if len(eligible) < 2_000:
        raise RuntimeError(f"Not enough eligible corpus words ({len(eligible)}).")

    holdout = eligible[len(eligible) // 2 :]
    samples: list[Sample] = []

    for _ in range(800):
        samples.append(Sample(rng.choice(holdout), 0, "corpus_word"))
    for _ in range(400):
        parts = [rng.choice(holdout) for _ in range(2)]
        samples.append(
            Sample(rng.choice(["", "-", "_"]).join(parts), 0, "compound_word")
        )
    for _ in range(400):
        length = rng.randint(8, 24)
        samples.append(Sample(secrets.token_hex(length // 2 + length % 2)[:length], 1, "hex"))
    for _ in range(400):
        length = rng.randint(10, 28)
        samples.append(Sample(secrets.token_urlsafe(length)[:length], 1, "urlsafe"))
    for _ in range(400):
        length = rng.randint(8, 20)
        samples.append(
            Sample(
                "".join(
                    secrets.choice(string.ascii_lowercase + string.digits)
                    for _ in range(length)
                ),
                1,
                "alnum",
            )
        )
    for _ in range(400):
        length = rng.randint(10, 22)
        consonants = "bcdfghjklmnpqrstvwxyz"
        samples.append(
            Sample("".join(secrets.choice(consonants) for _ in range(length)), 1, "consonant")
        )

    rng.shuffle(samples)
    return samples


def split_calibration_test(
    samples: list[Sample], *, cal_fraction: float = 0.35, seed: int = 2026
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    cal_size = int(len(samples) * cal_fraction)
    cal_idx = set(indices[:cal_size])
    cal = [samples[i] for i in range(len(samples)) if i in cal_idx]
    test = [samples[i] for i in range(len(samples)) if i not in cal_idx]
    return cal, test


def ensure_ent_binary() -> str:
    path = shutil.which("ent")
    if path:
        return path
    print("[benchmark] Installing Fourmilab ent via apt-get ...")
    subprocess.run(
        ["apt-get", "update", "-qq"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["apt-get", "install", "-y", "ent"],
        check=True,
        capture_output=True,
        text=True,
    )
    path = shutil.which("ent")
    if not path:
        raise RuntimeError("ent is not available after apt-get install ent")
    return path


DEFLATE_CLI = BUNDLED_TOOLS / "deflate_score.py"


def ensure_deflate_cli() -> Path:
    if not DEFLATE_CLI.is_file():
        raise RuntimeError(
            f"Missing bundled benchmark utility: {DEFLATE_CLI}"
        )
    return DEFLATE_CLI


def ensure_freqpy_installed(cache_dir: Path) -> None:
    BENCHMARK_TOOLS.mkdir(parents=True, exist_ok=True)
    if not FREQPY_DIR.exists():
        print(f"[benchmark] Cloning freq.py from {FREQPY_REPO} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", FREQPY_REPO, str(FREQPY_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )

    if not FREQPY_SCRIPT.is_file():
        raise RuntimeError(f"freq.py not found at {FREQPY_SCRIPT}")

    words_file = cache_dir / "words_alpha.txt"
    if not words_file.is_file():
        raise FileNotFoundError(f"Corpus file missing: {words_file}")

    table_mtime = FREQPY_TABLE.stat().st_mtime if FREQPY_TABLE.exists() else 0
    corpus_mtime = words_file.stat().st_mtime
    if FREQPY_TABLE.exists() and table_mtime >= corpus_mtime:
        return

    print(f"[benchmark] Building Mark Baggett frequency table from {words_file.name} ...")
    if FREQPY_TABLE.exists():
        FREQPY_TABLE.unlink()
    subprocess.run(
        [sys.executable, str(FREQPY_SCRIPT), "--create", str(FREQPY_TABLE)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(FREQPY_SCRIPT),
            "--normalfile",
            str(words_file),
            str(FREQPY_TABLE),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_external_tools(cache_dir: Path) -> ExternalTools:
    ensure_freqpy_installed(cache_dir)
    ent_bin = ensure_ent_binary()
    deflate_cli = ensure_deflate_cli()
    return ExternalTools(
        freqpy_script=FREQPY_SCRIPT,
        freqpy_table=FREQPY_TABLE,
        ent_binary=ent_bin,
        deflate_cli=deflate_cli,
    )


def load_freqpy_counter():
    spec = importlib.util.spec_from_file_location("markbaggett_freq", FREQPY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import freq.py from {FREQPY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counter = module.FreqCounter()
    counter.load(str(FREQPY_TABLE))
    return counter


def ent_randomness_score(text: str, ent_binary: str) -> float:
    proc = subprocess.run(
        [ent_binary, "-t"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    line = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1]
    parts = line.split(",")
    entropy_bpc = float(parts[2])
    return max(0.0, min(100.0, (entropy_bpc / 8.0) * 100.0))


def deflate_cli_randomness_score(text: str, deflate_cli: Path) -> float:
    proc = subprocess.run(
        [sys.executable, str(deflate_cli)],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    return float(proc.stdout.decode("utf-8").strip())


class MethodScorer:
    def __init__(self, cache_dir: Path, tools: ExternalTools) -> None:
        self.tools = tools
        self.cache_dir = cache_dir
        self.scorer = Scorer(cache_dir=cache_dir, auto_bootstrap=False)
        self.freq_counter = self.scorer._freq_counter
        if self.freq_counter is None:
            raise RuntimeError("Freq counter not loaded")
        self._freqpy = load_freqpy_counter()

    def score(self, method: str, text: str) -> float:
        from randomness_detection.features import breakdown_scores, extract_features

        if method == "freqpy":
            m1, m2 = self._freqpy.probability(text)
            naturalness = (float(m1) + float(m2)) / 2.0
            return max(0.0, min(100.0, 100.0 - naturalness))

        if method == "ent":
            return ent_randomness_score(text, self.tools.ent_binary)

        if method == "deflate_cli":
            return deflate_cli_randomness_score(text, self.tools.deflate_cli)

        if method == "randomness_detection":
            return float(self.scorer.score(text).score)

        features = extract_features(text, self.freq_counter)
        breakdown = breakdown_scores(features)
        freq = float(breakdown["freq"])
        entropy = float(breakdown["entropy"])
        compression = float(breakdown["compression"])

        if method == "freq":
            return freq
        if method == "entropy":
            return entropy
        if method == "compression":
            return compression
        if method == "avg3":
            return (freq + entropy + compression) / 3.0
        if method == "max3":
            return float(max(freq, entropy, compression))
        raise ValueError(f"unknown method: {method}")


def verify_external_tools(scorer: MethodScorer) -> None:
    """Smoke test: external tools must run and discriminate natural vs random."""
    natural = "arboraceous"
    token = secrets.token_hex(32)
    checks = [
        ("freqpy", scorer.score("freqpy", natural), scorer.score("freqpy", token)),
        ("ent", scorer.score("ent", natural), scorer.score("ent", token)),
    ]
    compressible = "the " * 20
    random_long = secrets.token_hex(64)
    checks.append(
        (
            "deflate_cli",
            scorer.score("deflate_cli", compressible),
            scorer.score("deflate_cli", random_long),
        )
    )
    print("[benchmark] External tool smoke tests:")
    for name, natural_score, random_score in checks:
        ok = random_score > natural_score
        status = "OK" if ok else "FAIL"
        print(
            f"  {status} {name}: natural={natural_score:.1f} random={random_score:.1f} "
            f"(expect random > natural)"
        )
        if not ok:
            raise RuntimeError(f"External tool {name} failed smoke test")


def score_samples(method: str, samples: list[Sample], scorer: MethodScorer) -> list[float]:
    return [scorer.score(method, sample.text) for sample in samples]


def find_best_threshold(y_true: list[int], scores: list[float]) -> float:
    best_threshold = 50.0
    best_f1 = -1.0
    for threshold in range(1, 100):
        preds = [1 if score >= threshold else 0 for score in scores]
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def predict(scores: list[float], threshold: float) -> list[int]:
    return [1 if score >= threshold else 0 for score in scores]


def evaluate_method(
    method: str,
    cal: list[Sample],
    test: list[Sample],
    scorer: MethodScorer,
) -> MethodMetrics:
    source, description = method_meta(method)
    y_cal = [sample.label for sample in cal]
    y_test = [sample.label for sample in test]

    cal_scores = score_samples(method, cal, scorer)
    test_scores = score_samples(method, test, scorer)

    threshold = find_best_threshold(y_cal, cal_scores)
    y_pred = predict(test_scores, threshold)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    natural_scores = [score for score, label in zip(test_scores, y_test, strict=True) if label == 0]
    random_scores = [score for score, label in zip(test_scores, y_test, strict=True) if label == 1]
    natural_total = len(natural_scores)
    random_total = len(random_scores)

    natural_correct = sum(
        1 for pred, label in zip(y_pred, y_test, strict=True) if label == 0 and pred == 0
    )
    random_correct = sum(
        1 for pred, label in zip(y_pred, y_test, strict=True) if label == 1 and pred == 1
    )

    return MethodMetrics(
        method=method,
        source=source,
        description=description,
        samples=len(test),
        threshold=threshold,
        accuracy=round(accuracy_score(y_test, y_pred), 4),
        precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_test, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_test, y_pred, zero_division=0), 4),
        roc_auc=round(roc_auc_score(y_test, [s / 100.0 for s in test_scores]), 4),
        false_positive_rate=round(fp / natural_total, 4) if natural_total else 0.0,
        false_negative_rate=round(fn / random_total, 4) if random_total else 0.0,
        natural_accuracy=round(natural_correct / natural_total, 4) if natural_total else 0.0,
        random_accuracy=round(random_correct / random_total, 4) if random_total else 0.0,
        score_natural_mean=round(sum(natural_scores) / natural_total, 2) if natural_total else 0.0,
        score_random_mean=round(sum(random_scores) / random_total, 2) if random_total else 0.0,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )


def print_table(results: list[MethodMetrics]) -> None:
    print("\n" + "=" * 118)
    print("QUALITY BENCHMARK — real tools, calibrated thresholds, held-out test set")
    print("=" * 118)
    print(
        f"{'Method':<22} {'Src':<8} {'Thr':>5} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>8} "
        f"{'FPR':>8} {'FNR':>8} {'Natμ':>7} {'Rndμ':>7}"
    )
    print("-" * 118)
    for row in sorted(results, key=lambda item: item.f1, reverse=True):
        print(
            f"{row.method:<22} {row.source:<8} {row.threshold:>5.0f} {row.accuracy:>9.3f} "
            f"{row.f1:>8.3f} {row.roc_auc:>8.3f} {row.false_positive_rate:>8.3f} "
            f"{row.false_negative_rate:>8.3f} {row.score_natural_mean:>7.1f} {row.score_random_mean:>7.1f}"
        )
    print("=" * 118)
    print("Src: external = downloaded/installed tool | internal = baseline signal | product = randomness_detection")
    print("Thr = per-method threshold tuned on calibration split (max F1)")


def main() -> int:
    cache_dir = resolve_cache_dir()
    tools = ensure_external_tools(cache_dir)

    words = load_real_words(cache_dir, limit=100_000)
    all_samples = build_holdout_dataset(words)
    cal, test = split_calibration_test(all_samples)
    scorer = MethodScorer(cache_dir, tools)
    verify_external_tools(scorer)

    print("=" * 72)
    print("RANDOMNESS DETECTION — QUALITY BENCHMARK (REAL TOOLS)")
    print("=" * 72)
    print(f"Cache:            {cache_dir}")
    print(f"freq.py:          {tools.freqpy_script}")
    print(f"freq.py table:    {tools.freqpy_table}")
    print(f"ent binary:       {tools.ent_binary}")
    print(f"deflate_cli:      {tools.deflate_cli}")
    print(f"Total samples:    {len(all_samples)}")
    print(f"Calibration:      {len(cal)}  |  Test: {len(test)}")
    print(f"Test natural:     {sum(s.label == 0 for s in test)}")
    print(f"Test random:      {sum(s.label == 1 for s in test)}")

    start = time.perf_counter()
    results: list[MethodMetrics] = []
    for index, method in enumerate(METHODS, start=1):
        source, _ = method_meta(method)
        print(f"[{index}/{len(METHODS)}] Evaluating {method} ({source}) ...")
        results.append(evaluate_method(method, cal, test, scorer))
    elapsed = time.perf_counter() - start

    print_table(results)

    product = next(row for row in results if row.method == "randomness_detection")
    ranked = sorted(results, key=lambda row: row.f1, reverse=True)
    product_rank = next(
        index + 1 for index, row in enumerate(ranked) if row.method == "randomness_detection"
    )
    print(f"\nBest F1 on test set: {ranked[0].method} ({ranked[0].f1:.3f})")
    print(f"randomness_detection rank: {product_rank}/{len(results)}")

    payload = {
        "meta": {
            "cache_dir": str(cache_dir),
            "external_tools": {
                "freqpy_script": str(tools.freqpy_script),
                "freqpy_table": str(tools.freqpy_table),
                "ent_binary": tools.ent_binary,
                "deflate_cli": str(tools.deflate_cli),
            },
            "total_samples": len(all_samples),
            "calibration_samples": len(cal),
            "test_samples": len(test),
            "elapsed_sec": round(elapsed, 2),
            "methodology": (
                "Real external tools installed/downloaded; per-method threshold on calibration "
                "split (max F1); metrics on held-out test; no mocks"
            ),
        },
        "method_specs": [
            {"method": name, "source": source, "description": desc}
            for name, source, desc in METHOD_SPECS
        ],
        "results": [asdict(row) for row in results],
        "generated_at_unix": int(time.time()),
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {RESULTS_FILE}")

    baselines = [row for row in results if row.method != "randomness_detection"]
    if not all(product.f1 >= row.f1 for row in baselines) or product.roc_auc < 0.95:
        print(
            f"QUALITY CHECK: FAIL — randomness_detection F1={product.f1:.3f} "
            f"ROC-AUC={product.roc_auc:.3f}",
            file=sys.stderr,
        )
        return 1

    print(
        f"QUALITY CHECK: PASS — randomness_detection F1={product.f1:.3f}, "
        f"ROC-AUC={product.roc_auc:.3f}, beats all baselines"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
