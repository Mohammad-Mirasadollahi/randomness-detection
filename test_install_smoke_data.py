#!/usr/bin/env python3
"""Validate install.sh output with real scoring samples (CLI + API)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Fixed samples — not generated at runtime so results are stable across runs.
NATURAL_SAMPLES = [
    ("hello", 0, 40),
    ("computer", 0, 40),
    ("open-source", 0, 45),
    ("stackoverflow", 0, 50),
]

RANDOM_SAMPLES = [
    ("qzxwvbnmklpr", 60, 100),
    ("f83a9c2e1b004d7a6e5f0123456789ab", 60, 100),
    ("xK9mQ2pL", 55, 100),
]

BATCH_SAMPLES = [name for name, _, _ in NATURAL_SAMPLES[:2]] + [
    name for name, _, _ in RANDOM_SAMPLES[:2]
]


def _fail(message: str) -> None:
    print(f"[install-smoke-data] FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def _pass(message: str) -> None:
    print(f"[install-smoke-data] PASS: {message}", file=sys.stderr)


def _load_api_key(env_file: Path) -> str:
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("RANDOMNESS_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if len(key) >= 32:
                return key
    _fail(f"RANDOMNESS_API_KEY not found in {env_file}")


def _parse_cli_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        _fail(f"CLI --json output is not valid JSON: {exc}\n{text}")


def _check_score(name: str, score: int, lo: int, hi: int, channel: str) -> None:
    if not (lo <= score <= hi):
        _fail(
            f"{channel} '{name}': score={score} outside expected range [{lo}, {hi}]"
        )


def _check_label(name: str, label: str, expect: str, channel: str) -> None:
    if label != expect:
        _fail(f"{channel} '{name}': label={label!r}, expected {expect!r}")


def run_cli_tests(work_dir: Path) -> None:
    cli = work_dir / ".venv/bin/randomness-detection"
    if not cli.is_file():
        _fail(f"CLI not found: {cli}")

    env = os.environ.copy()
    env_file = work_dir / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()

    for name, lo, hi in NATURAL_SAMPLES:
        proc = subprocess.run(
            [str(cli), name, "--json"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            _fail(f"CLI exited {proc.returncode} for '{name}': {proc.stderr.strip()}")
        data = _parse_cli_json(proc.stdout)
        score = int(data["score"])
        _check_score(name, score, lo, hi, "CLI natural")
        _check_label(name, data.get("label", ""), "natural", "CLI")

    for name, lo, hi in RANDOM_SAMPLES:
        proc = subprocess.run(
            [str(cli), name, "--json"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            _fail(f"CLI exited {proc.returncode} for '{name}': {proc.stderr.strip()}")
        data = _parse_cli_json(proc.stdout)
        score = int(data["score"])
        _check_score(name, score, lo, hi, "CLI random")
        label = data.get("label", "")
        if label not in {"likely_random", "random", "uncertain"}:
            _fail(f"CLI random '{name}': unexpected label {label!r}")

    _pass(f"CLI scored {len(NATURAL_SAMPLES)} natural + {len(RANDOM_SAMPLES)} random samples")


def _api_json(method: str, url: str, api_key: str = "", payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        _fail(f"HTTP {exc.code} {url}: {body}")
    except urllib.error.URLError as exc:
        _fail(f"Request failed {url}: {exc}")


def run_api_tests(work_dir: Path, port: int) -> None:
    env_file = work_dir / ".env"
    api_key = _load_api_key(env_file)
    base = f"http://127.0.0.1:{port}"

    health = _api_json("GET", f"{base}/health", api_key="")
    if health.get("status") != "ok":
        _fail(f"/health status={health.get('status')!r}")
    if not health.get("model_ready", False):
        _fail("/health model_ready is false")
    _pass(f"/health ok (model_ready={health.get('model_ready')})")

    for name, lo, hi in NATURAL_SAMPLES:
        data = _api_json("POST", f"{base}/score", api_key, {"text": name})
        score = int(data["score"])
        _check_score(name, score, lo, hi, "API natural")

    for name, lo, hi in RANDOM_SAMPLES:
        data = _api_json("POST", f"{base}/score", api_key, {"text": name})
        score = int(data["score"])
        _check_score(name, score, lo, hi, "API random")

    batch = _api_json("POST", f"{base}/score/batch", api_key, {"texts": BATCH_SAMPLES})
    results = batch.get("results", [])
    if len(results) != len(BATCH_SAMPLES):
        _fail(f"/score/batch returned {len(results)} results, expected {len(BATCH_SAMPLES)}")

    for item, (name, lo, hi) in zip(results, [
        *[(n, lo, hi) for n, lo, hi in NATURAL_SAMPLES[:2]],
        *[(n, lo, hi) for n, lo, hi in RANDOM_SAMPLES[:2]],
    ], strict=True):
        text = item.get("text", item.get("input", ""))
        score = int(item["score"])
        _check_score(text or name, score, lo, hi, "API batch")

    _pass(
        f"API scored {len(NATURAL_SAMPLES)} natural + {len(RANDOM_SAMPLES)} random "
        f"+ batch({len(BATCH_SAMPLES)})"
    )


def verify_model_artifacts(work_dir: Path) -> None:
    cache = work_dir / ".cache"
    for name in ("ensemble.pkl", "english.freq", "metadata.json"):
        path = cache / name
        if not path.is_file():
            _fail(f"missing model artifact after install: {path}")
    meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if not meta.get("metrics"):
        _fail("metadata.json has no training metrics")
    _pass(f"model artifacts present (version={meta.get('version', '?')})")


def main() -> int:
    parser = argparse.ArgumentParser(description="install.sh smoke data checks")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("artifacts", "cli", "api", "all"),
        default="all",
    )
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()

    if args.mode in {"artifacts", "all"}:
        verify_model_artifacts(work_dir)
    if args.mode in {"cli", "all"}:
        run_cli_tests(work_dir)
    if args.mode in {"api", "all"}:
        if args.port <= 0:
            _fail("--port is required for API mode")
        run_api_tests(work_dir, args.port)

    print("[install-smoke-data] OVERALL: PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
