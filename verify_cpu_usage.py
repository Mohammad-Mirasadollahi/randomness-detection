#!/usr/bin/env python3
"""Verify parallel training actually uses ~50% of CPU cores."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from randomness_detection.bootstrap import bootstrap
from randomness_detection.config import CPU_FRACTION
from randomness_detection.parallel import worker_count


def sample_system_cpu() -> tuple[int, int]:
    with open("/proc/stat", encoding="utf-8") as handle:
        parts = handle.readline().split()[1:]
    values = list(map(int, parts))
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def count_child_processes(pid: int) -> int:
    result = subprocess.run(
        ["pgrep", "-P", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if not result.stdout.strip():
        return 0
    return len(result.stdout.strip().splitlines())


def main() -> int:
    cache_dir = Path(__file__).resolve().parent / ".cache_cpu_test"
    cpus = os.cpu_count() or 1
    expected_workers = worker_count(CPU_FRACTION)
    pid = os.getpid()

    cpu_samples: list[float] = []
    child_samples: list[int] = []
    stop = threading.Event()

    def monitor() -> None:
        idle1, total1 = sample_system_cpu()
        time.sleep(0.5)
        while not stop.is_set():
            idle2, total2 = sample_system_cpu()
            total_delta = total2 - total1
            idle_delta = idle2 - idle1
            if total_delta > 0:
                usage = (1.0 - idle_delta / total_delta) * 100.0
                cpu_samples.append(usage)
            child_samples.append(count_child_processes(pid))
            idle1, total1 = idle2, total2
            time.sleep(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.time()
    bootstrap(cache_dir, force=True)
    stop.set()
    thread.join(timeout=2)
    elapsed = time.time() - start

    peak_cpu = max(cpu_samples) if cpu_samples else 0.0
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    peak_children = max(child_samples) if child_samples else 0
    target_cpu = cpus * CPU_FRACTION * 100.0 / cpus  # fraction as percent of machine

    print(f"System CPUs: {cpus}")
    print(f"Expected workers: {expected_workers}")
    print(f"Peak child processes: {peak_children}")
    print(f"Peak system CPU: {peak_cpu:.1f}%")
    print(f"Avg system CPU during train: {avg_cpu:.1f}%")
    print(f"Target (~{CPU_FRACTION:.0%}): {CPU_FRACTION * 100:.0f}%")
    print(f"Elapsed: {elapsed:.1f}s")

    workers_ok = peak_children >= max(1, int(expected_workers * 0.75))
    cpu_ok = peak_cpu >= CPU_FRACTION * 100.0 * 0.75

    print(f"Worker processes OK: {workers_ok}")
    print(f"CPU utilization OK: {cpu_ok}")

    if not (workers_ok and cpu_ok):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
