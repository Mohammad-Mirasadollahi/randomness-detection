#!/usr/bin/env python3
"""Real integration tests for exclusion and score-cache fast path."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / ".cache_test"
PORT = 8790


def api_request(
    method: str,
    path: str,
    api_key: str,
    payload: dict | None = None,
    *,
    query: str = "",
) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{PORT}{path}{query}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode())


def start_server(api_key: str, exclude_db: Path) -> subprocess.Popen:
    subprocess.run(["pkill", "-f", f"randomness_detection.api_server.*{PORT}"], check=False)
    time.sleep(1)
    env = os.environ.copy()
    env["RANDOMNESS_API_KEY"] = api_key
    env["RANDOMNESS_CACHE_DIR"] = str(CACHE_DIR)
    env["RANDOMNESS_EXCLUDE_DB_PATH"] = str(exclude_db)
    env["RANDOMNESS_EXCLUDE_ENABLED"] = "true"
    env["RANDOMNESS_SKIP_CACHE_ENABLED"] = "true"
    env["RANDOMNESS_SKIP_SCORE_THRESHOLD"] = "30"
    env["RANDOMNESS_INFERENCE_WORKERS"] = "4"
    env["PYTHONPATH"] = str(BASE)
    return subprocess.Popen(
        [str(BASE / ".venv/bin/python"), "-m", "randomness_detection.api_server", "--host", "127.0.0.1", "--port", str(PORT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_health() -> dict:
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def test_manager_unit() -> None:
    from randomness_detection.exclude import ExcludeManager

    db_path = CACHE_DIR / "exclude_unit.db"
    if db_path.exists():
        db_path.unlink()

    manager = ExcludeManager.open(CACHE_DIR, db_name="exclude_unit.db")
    try:
        manager.add_rules(
            [
                ("blocked.com", "domain"),
                ("*.cdn.example.net", "suffix"),
                ("admin-", "prefix"),
                ("test-user-*", "glob"),
            ]
        )

        assert manager.check_exclude("https://foo.blocked.com/path") is not None
        assert manager.check_exclude("media.cdn.example.net") is not None
        assert manager.check_exclude("admin-panel") is not None
        assert manager.check_exclude("test-user-42") is not None
        assert manager.check_exclude("hello-world") is None

        manager.store_score(
            "cached-natural-word",
            {
                "score": 8,
                "label": "natural",
                "confidence": "high",
                "breakdown": {"freq": 5, "entropy": 6, "compression": 7},
            },
        )
        cached = manager.get_cached_score("cached-natural-word")
        assert cached is not None
        assert cached.score == 8
        assert manager.get_cached_score(secrets.token_hex(16)) is None
    finally:
        manager.close()


def main() -> int:
    print("=" * 64)
    print("EXCLUDE FEATURE — REAL TESTS")
    print("=" * 64)

    if not (CACHE_DIR / "ensemble.pkl").exists():
        print("ERROR: model cache missing, run bootstrap first", file=sys.stderr)
        return 1

    print("Unit checks...", flush=True)
    test_manager_unit()
    print("Unit checks: PASS", flush=True)

    api_key = secrets.token_urlsafe(48)
    exclude_db = CACHE_DIR / f"exclude_test_{secrets.token_hex(4)}.db"
    if exclude_db.exists():
        exclude_db.unlink()

    server = start_server(api_key, exclude_db)
    try:
        health = wait_health()
        print(f"Health: {health}")

        _, add_result = api_request(
            "POST",
            "/exclude",
            api_key,
            {
                "rules": [
                    {"pattern": "skipme.com", "rule_type": "domain"},
                    {"pattern": "*.trusted.org", "rule_type": "suffix"},
                    {"pattern": "cache-test-item", "rule_type": "exact"},
                ]
            },
        )
        print(f"Exclude add: {add_result}")

        _, excluded = api_request(
            "POST",
            "/score",
            api_key,
            {"text": "https://app.skipme.com/login"},
        )
        assert excluded["excluded"] is True
        assert excluded["label"] == "excluded"
        assert excluded["skipped"] is True
        print(f"Excluded domain score: {excluded}")

        _, suffix_excluded = api_request(
            "POST",
            "/score",
            api_key,
            {"text": "cdn.trusted.org"},
        )
        assert suffix_excluded["excluded"] is True
        print(f"Excluded suffix score: {suffix_excluded}")

        _, first_score = api_request(
            "POST",
            "/score",
            api_key,
            {"text": "cache-test-item"},
        )
        assert first_score["excluded"] is True
        print(f"Exact excluded: {first_score}")

        token = secrets.token_hex(16)
        natural_word = "arboraceous"
        _, scored = api_request("POST", "/score", api_key, {"text": natural_word}, query="?use_exclude=false")
        assert scored["excluded"] is False
        assert scored["cached"] is False
        assert scored["score"] <= 30, f"expected low score for cache test, got {scored['score']}"

        _, cached = api_request("POST", "/score", api_key, {"text": natural_word}, query="?use_exclude=false")
        assert cached["cached"] is True
        assert cached["skipped"] is True
        assert cached["score"] == scored["score"]
        print(f"Score cache hit: {cached}")

        _, random_scored = api_request("POST", "/score", api_key, {"text": token})
        assert random_scored["excluded"] is False

        _, batch = api_request(
            "POST",
            "/score/batch",
            api_key,
            {
                "texts": [
                    "foo.skipme.com",
                    natural_word,
                    secrets.token_hex(12),
                ]
            },
        )
        assert batch["count"] == 3
        assert batch["results"][0]["excluded"] is True
        assert batch["results"][1]["cached"] is True
        print(f"Batch mixed results OK: {[item['label'] for item in batch['results']]}")

        _, check = api_request(
            "POST",
            "/exclude/check",
            api_key,
            {"text": "foo.skipme.com"},
        )
        assert check["excluded"] is True
        assert check["would_skip"] is True
        print(f"Exclude check: {check}")

        print("=" * 64)
        print("OVERALL: PASS")
        return 0
    finally:
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
