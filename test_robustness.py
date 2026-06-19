#!/usr/bin/env python3
"""
Independent robustness stress test — anti-band-aid guardrail.

CRITICAL: every string here is HAND-CURATED real-world data. None of it is produced
by randomness_detection/synthetic.py, so this set cannot be "gamed" by training-data
tricks. It exists to catch root-cause regressions that the synthetic quality benchmark
(which shares its generator family with training) is structurally blind to.

It directly probes the two failure modes that the lexical-coverage feature and
hard-negative training could introduce:
  - FP: real human-meaningful strings (brands, identifiers, digits, separators,
        mixed case) wrongly flagged as random.
  - FN: machine-generated high-entropy tokens wrongly accepted as natural.

Pass criteria (printed at the end):
  - core_natural  FP rate <= 10%
  - clear_random  FN rate <= 5%
Hard / ambiguous buckets are reported as diagnostics only (no pass/fail), because
short out-of-dictionary brands vs short pronounceable randoms are not reliably
separable from structure alone.
"""

from __future__ import annotations

import sys

from randomness_detection.scorer import Scorer
from test_quality_benchmark import resolve_cache_dir

RANDOM_DECISION = 50  # score >= this => model calls it random

# --- NATURAL: human-meaningful. Should score LOW (natural). ---------------------
CORE_NATURAL = [
    # plain dictionary words
    "computer", "elephant", "language", "mountain", "developer", "keyboard",
    # long real brands / sites (in or near dictionary via segmentation)
    "stackoverflow", "microservice", "wikipedia", "javascript", "typescript",
    "playstation", "screenshot", "framework", "database", "kubernetes",
    # word compounds with separators
    "hello-world", "the-quick-brown-fox", "user-profile-page", "open-source",
    "hello_world", "data_pipeline", "machine_learning", "create_user_account",
    # mixed-case identifiers built from real words
    "getUserById", "parseHttpRequest", "createUserAccount", "HttpRequestHandler",
    "MaxBufferSize", "JsonParser", "BackgroundWorker",
    # words + digits (years, versions, ids)
    "report2024", "server01", "covid19", "route66", "version2",
    "annual-report-2024", "release_v3", "user_account_42",
]

# --- RANDOM: machine-generated high-entropy. Should score HIGH (random). --------
CLEAR_RANDOM = [
    # real UUIDs
    "550e8400-e29b-41d4-a716-446655440000",
    "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    # git SHAs / hex digests
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4",
    "356a192b7913b04c54574d18c28d46e6395428ab",
    "9e52c65b9077eabb175d493eab2eb390",
    # base64 of random bytes
    "Tm90QVJlYWxXb3JkSnVzdEJ5dGVz", "f4Zk9QwL2mXpR7vT",
    # API-key-like surface forms (synthetic — not real provider tokens)
    "paymock_x7k2p9qzab_not_a_real_secret",
    "cloudmock_access_key_not_real_001",
    "vcsmock_personal_token_not_real_001",
    # DGA-like / random domain labels
    "x7k2p9qzab", "qzxwvbnmklpr", "vbnmqwrtzx", "kjhgfdsapo",
    # random alnum / token fragments
    "a8Xk2Pq9Lm", "Z9x4Kw2Qp7", "Xr4Tg8Bn2Lq", "wW 82-8JwH4v" .replace(" ", ""),
]

# --- HARD NATURAL (diagnostic): short out-of-dictionary brands ------------------
HARD_NATURAL = [
    "nvidia", "tiktok", "spotify", "airbnb", "figma", "nginx", "redis",
    "docker", "django", "ubuntu", "github", "google", "youtube", "netflix",
]

# --- ADVERSARIAL RANDOM (diagnostic): randoms that partly segment to words ------
ADVERSARIAL_RANDOM = [
    "boxcarmittenglow", "redfoxjumphigh", "tincanlamphat",  # concatenated words
    "l33th4x0rpwn", "p4ssw0rdz", "h7llowor1d",              # leetspeak
]

# --- ADVERSARIAL NATURAL (diagnostic): passphrases of real words ---------------
ADVERSARIAL_NATURAL = [
    "correcthorsebatterystaple", "trustthebridgehome", "applebananacherry",
]


def evaluate(scorer: Scorer, label: str, strings: list[str], expect_random: bool):
    wrong = []
    for text in strings:
        score = scorer.score(text).score
        is_random = score >= RANDOM_DECISION
        if is_random != expect_random:
            wrong.append((text, score))
    rate = len(wrong) / len(strings) if strings else 0.0
    kind = "FN" if expect_random else "FP"
    print(f"\n[{label}] {kind} rate: {len(wrong)}/{len(strings)} = {rate:.0%}")
    for text, score in wrong:
        print(f"    {kind} score={score:3} {text!r}")
    return rate


def main() -> int:
    cache_dir = resolve_cache_dir()
    scorer = Scorer(cache_dir=cache_dir, auto_bootstrap=False)
    print("=" * 72)
    print("INDEPENDENT ROBUSTNESS STRESS TEST (curated real-world data)")
    print("=" * 72)
    print(f"Cache: {cache_dir}  |  random-decision threshold: {RANDOM_DECISION}")

    core_fp = evaluate(scorer, "core_natural", CORE_NATURAL, expect_random=False)
    clear_fn = evaluate(scorer, "clear_random", CLEAR_RANDOM, expect_random=True)

    print("\n" + "-" * 72)
    print("DIAGNOSTIC BUCKETS (not pass/fail — inherently hard):")
    evaluate(scorer, "hard_natural (short OOV brands)", HARD_NATURAL, expect_random=False)
    evaluate(scorer, "adversarial_random (word-like)", ADVERSARIAL_RANDOM, expect_random=True)
    evaluate(scorer, "adversarial_natural (passphrases)", ADVERSARIAL_NATURAL, expect_random=False)

    print("\n" + "=" * 72)
    ok = core_fp <= 0.10 and clear_fn <= 0.05
    print(
        f"core_natural FP={core_fp:.0%} (<=10%)  |  clear_random FN={clear_fn:.0%} (<=5%)"
    )
    print("ROBUSTNESS:", "PASS" if ok else "FAIL")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
