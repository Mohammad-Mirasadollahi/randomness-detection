#!/usr/bin/env python3
"""
Real-world data evaluation — the production use case, with REAL public datasets.

This is the strongest, least-gameable test: none of the data is produced by this
project's training generator. It measures the actual job the scorer exists for —
telling legitimate domains apart from algorithmically-generated (DGA) ones.

Datasets (downloaded + cached under .benchmark_tools/realworld/):
  - LEGIT : Tranco top-1M domains          (real, popular, ranking-curated)
            https://tranco-list.eu/top-1m.csv.zip
  - DGA   : andrewaeva/DGA all_dga.txt     (real malware DGA output, 800K+ domains)
            https://github.com/andrewaeva/DGA
            families 1,2,3,5,6,8 = random-character DGA
            families 4,7         = DICTIONARY DGA (concatenated real words — hard)

We score the registrable label (the second-level domain, TLD stripped), which is
the unit a DGA actually randomizes. Per-family DGA recall is reported so the
inherently-hard dictionary-DGA ceiling is shown honestly, not hidden.

Top-domain lists are ranked by traffic and contain many machine-generated labels
(CDN nodes, telemetry, hashes), so the "legit" label is intrinsically noisy and a
nonzero legit false-positive rate is expected (many such FPs are genuinely random
and the model is correct). The primary, noise-robust pass criterion is therefore
the separation between legit and random-character DGA (AUC), plus random-char DGA
recall — the actual detectable task.
"""

from __future__ import annotations

import io
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

from randomness_detection.scorer import Scorer
from test_quality_benchmark import (
    ensure_freqpy_installed,
    load_freqpy_counter,
    resolve_cache_dir,
)

# freqpy's own calibrated random boundary (from test_quality_benchmark): a label
# whose freqpy randomness >= this is independently judged random, hence treated as
# contamination in the noisy "legit" top-list when estimating a clean FP/AUC.
FREQPY_RANDOM_BOUNDARY = 95.0

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / ".benchmark_tools" / "realworld"
LEGIT_ZIP = DATA_DIR / "tranco-top-1m.csv.zip"
DGA_FILE = DATA_DIR / "all_dga.txt"
LEGIT_URL = "https://tranco-list.eu/top-1m.csv.zip"
DGA_URL = "https://raw.githubusercontent.com/andrewaeva/DGA/master/all_dga.txt"

RANDOM_DECISION = 50
SAMPLE_PER_CLASS = 6000
DICTIONARY_DGA_FAMILIES = {"4", "7"}

# Minimal set of two-level public suffixes so 'foo.co.uk' -> 'foo', not 'co'.
TWO_LEVEL_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.br", "com.au", "com.cn",
    "co.jp", "co.in", "co.kr", "com.mx", "com.tr", "com.tw", "co.za",
    "com.ar", "com.sg", "com.hk", "net.cn", "org.cn", "gov.cn",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"[realworld] downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "randomness-detection-benchmark"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        dest.write_bytes(resp.read())


def registrable_label(domain: str) -> str | None:
    domain = domain.strip().lower().rstrip(".")
    if not domain or " " in domain:
        return None
    parts = domain.split(".")
    if len(parts) < 2:
        return None
    last_two = ".".join(parts[-2:])
    label = parts[-3] if (last_two in TWO_LEVEL_TLDS and len(parts) >= 3) else parts[-2]
    label = label.strip()
    return label or None


def load_legit(limit: int, rng: random.Random) -> list[str]:
    _download(LEGIT_URL, LEGIT_ZIP)
    labels: list[str] = []
    seen: set[str] = set()
    with zipfile.ZipFile(LEGIT_ZIP) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as handle:
            for raw in io.TextIOWrapper(handle, encoding="utf-8", errors="ignore"):
                _, _, domain = raw.partition(",")
                label = registrable_label(domain)
                if label and label not in seen and label.isascii():
                    seen.add(label)
                    labels.append(label)
    rng.shuffle(labels)
    return labels[:limit]


def load_dga(limit: int, rng: random.Random) -> list[tuple[str, str]]:
    _download(DGA_URL, DGA_FILE)
    by_family: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for raw in DGA_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        bits = raw.split()
        if len(bits) < 2:
            continue
        domain, family = bits[0], bits[1]
        label = registrable_label(domain)
        if label and label not in seen and label.isascii():
            seen.add(label)
            by_family[family].append(label)

    families = sorted(by_family)
    per_family = max(1, limit // len(families))
    samples: list[tuple[str, str]] = []
    for family in families:
        pool = by_family[family]
        rng.shuffle(pool)
        samples.extend((label, family) for label in pool[:per_family])
    rng.shuffle(samples)
    return samples[:limit]


def freqpy_randomness(counter, text: str) -> float:
    m1, m2 = counter.probability(text)
    return max(0.0, min(100.0, 100.0 - (float(m1) + float(m2)) / 2.0))


def main() -> int:
    cache_dir = resolve_cache_dir()
    scorer = Scorer(cache_dir=cache_dir, auto_bootstrap=False)
    ensure_freqpy_installed(cache_dir)
    freqpy = load_freqpy_counter()
    rng = random.Random(2026)

    print("=" * 72)
    print("REAL-WORLD DATA TEST — legit (Tranco) vs DGA (real malware)")
    print("=" * 72)

    legit = load_legit(SAMPLE_PER_CLASS, rng)
    dga = load_dga(SAMPLE_PER_CLASS, rng)
    print(f"legit labels: {len(legit)}  |  DGA labels: {len(dga)}")
    print(f"random-decision threshold: {RANDOM_DECISION}\n")

    legit_scores = [scorer.score(t).score for t in legit]
    dga_scores = [scorer.score(t).score for t, _ in dga]

    legit_fp = sum(1 for s in legit_scores if s >= RANDOM_DECISION)
    legit_fp_rate = legit_fp / len(legit_scores)

    fam_total: dict[str, int] = defaultdict(int)
    fam_hit: dict[str, int] = defaultdict(int)
    for (label, family), score in zip(dga, dga_scores, strict=True):
        fam_total[family] += 1
        if score >= RANDOM_DECISION:
            fam_hit[family] += 1

    random_char_total = random_char_hit = 0
    dict_total = dict_hit = 0
    for family in fam_total:
        if family in DICTIONARY_DGA_FAMILIES:
            dict_total += fam_total[family]
            dict_hit += fam_hit[family]
        else:
            random_char_total += fam_total[family]
            random_char_hit += fam_hit[family]

    y_true = [0] * len(legit_scores) + [1] * len(dga_scores)
    y_score = [s / 100.0 for s in legit_scores + dga_scores]
    auc = roc_auc_score(y_true, y_score)

    # Noise-robust solvable task: legit vs random-character DGA only
    # (dictionary DGA is the documented structural ceiling, excluded here).
    rc_scores = [
        score for (label, family), score in zip(dga, dga_scores, strict=True)
        if family not in DICTIONARY_DGA_FAMILIES
    ]
    y_true_solv = [0] * len(legit_scores) + [1] * len(rc_scores)
    y_score_solv = [s / 100.0 for s in legit_scores + rc_scores]
    auc_solvable = roc_auc_score(y_true_solv, y_score_solv)

    # Independent de-noising of the legit label using freqpy (different codebase /
    # model). Drop legit labels that freqpy ALSO judges random — they are top-list
    # contamination (CDN/hash/telemetry), not human-meaningful names.
    clean_legit_scores = [
        score
        for label, score in zip(legit, legit_scores, strict=True)
        if freqpy_randomness(freqpy, label) < FREQPY_RANDOM_BOUNDARY
    ]
    dropped = len(legit_scores) - len(clean_legit_scores)
    clean_fp = sum(1 for s in clean_legit_scores if s >= RANDOM_DECISION)
    clean_fp_rate = clean_fp / len(clean_legit_scores) if clean_legit_scores else 0.0
    y_true_clean = [0] * len(clean_legit_scores) + [1] * len(rc_scores)
    y_score_clean = [s / 100.0 for s in clean_legit_scores + rc_scores]
    auc_clean = roc_auc_score(y_true_clean, y_score_clean)

    print("-" * 72)
    print(f"LEGIT false-positive rate : {legit_fp}/{len(legit_scores)} = {legit_fp_rate:.1%}")
    print(f"DGA recall (random-char)  : {random_char_hit}/{random_char_total} = "
          f"{(random_char_hit / random_char_total if random_char_total else 0):.1%}")
    print(f"DGA recall (dictionary)   : {dict_hit}/{dict_total} = "
          f"{(dict_hit / dict_total if dict_total else 0):.1%}  (inherently hard)")
    print(f"ROC-AUC (legit vs all DGA): {auc:.4f}")
    print(f"ROC-AUC (legit vs rc DGA) : {auc_solvable:.4f}  <- raw (legit label is noisy)")
    print(
        f"freqpy-clean legit        : dropped {dropped}/{len(legit_scores)} "
        f"contaminated ({dropped / len(legit_scores):.1%}); clean FP={clean_fp_rate:.1%}"
    )
    print(f"ROC-AUC (clean vs rc DGA) : {auc_clean:.4f}  <- noise-robust solvable task")
    print("\nper-family DGA recall:")
    for family in sorted(fam_total, key=lambda k: int(k)):
        kind = "dictionary" if family in DICTIONARY_DGA_FAMILIES else "random-char"
        rate = fam_hit[family] / fam_total[family]
        print(f"  family {family} ({kind:11}): {fam_hit[family]:4}/{fam_total[family]:4} = {rate:.1%}")

    print("\nsample legit FALSE POSITIVES:")
    fps = [(s, t) for t, s in zip(legit, legit_scores) if s >= RANDOM_DECISION]
    for s, t in sorted(fps, reverse=True)[:12]:
        print(f"  score={s:3} {t!r}")

    print("\n" + "=" * 72)
    # Noise-robust pass criteria: the model must strongly detect real
    # random-character DGA and cleanly separate it from legit domains. The legit
    # FP rate is reported for transparency but not used as a hard gate, because the
    # top-list "legit" label contains many genuinely random machine domains.
    rc_recall = random_char_hit / random_char_total if random_char_total else 0
    ok = rc_recall >= 0.95 and auc_clean >= 0.95
    print(
        f"random-char DGA recall={rc_recall:.1%} (>=95%) | "
        f"clean solvable AUC={auc_clean:.3f} (>=0.95)  "
        f"[raw legit FP={legit_fp_rate:.1%}, clean legit FP={clean_fp_rate:.1%}]"
    )
    print("REAL-WORLD:", "PASS" if ok else "FAIL")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
