#!/usr/bin/env python3
"""Run all real integration tests (no mocks)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(script: str) -> int:
    import os

    print("\n" + "=" * 72)
    print(f"RUNNING: {script}")
    print("=" * 72)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path(__file__).parent))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("RANDOMNESS_PARALLEL_BACKEND", "hybrid")
    result = subprocess.run(
        [str(Path(__file__).parent / ".venv/bin/python"), script],
        cwd=Path(__file__).parent,
        env=env,
    )
    return result.returncode


def main() -> int:
    tests = [
        "test_real_parallel.py",
        "test_exclude.py",
        "test_cpu_full.py",
    ]
    failed = []
    for name in tests:
        code = run(name)
        if code != 0:
            failed.append(name)
    print("\n" + "=" * 72)
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("ALL REAL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
