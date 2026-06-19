#!/usr/bin/env python3
"""
Real parallel + CPU test using trained model and actual corpus words.
No mocks, no synthetic placeholder strings.
"""

from __future__ import annotations

import json
import os
import random
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from test_helpers import build_real_text_batch, load_real_words, pick_natural_word

from randomness_detection.inference_pool import inference_worker_count

BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / ".cache_test"
PORT = 8788
DURATION_SEC = float(os.environ.get("RANDOMNESS_STRESS_DURATION_SEC", "15.0"))
STRESS_WORD_LIMIT = int(os.environ.get("RANDOMNESS_STRESS_WORD_LIMIT", "100_000"))
STRESS_BATCH_SIZE = int(os.environ.get("RANDOMNESS_STRESS_BATCH_SIZE", "100"))


def sample_system_cpu_percent() -> float:
    with open("/proc/stat", encoding="utf-8") as handle:
        parts = list(map(int, handle.readline().split()[1:]))
    idle = parts[3] + parts[4]
    total = sum(parts)
    return idle, total


class SystemCpuMonitor:
    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            idle1, total1 = sample_system_cpu_percent()
            time.sleep(0.2)
            while not self._stop.is_set():
                idle2, total2 = sample_system_cpu_percent()
                delta_total = total2 - total1
                delta_idle = idle2 - idle1
                if delta_total > 0:
                    usage = (1.0 - delta_idle / delta_total) * 100.0
                    self.samples.append(usage)
                idle1, total1 = idle2, total2
                time.sleep(0.25)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"peak": 0.0, "avg": 0.0, "min": 0.0}
        return {
            "peak": round(max(self.samples), 1),
            "avg": round(sum(self.samples) / len(self.samples), 1),
            "min": round(min(self.samples), 1),
            "samples": float(len(self.samples)),
        }


def count_worker_processes(server_pid: int) -> int:
    """Count descendant worker processes (handles forkserver nesting)."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"pgrep -P {server_pid} | xargs -r -I{{}} pgrep -P {{}} 2>/dev/null; pgrep -P {server_pid}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if pids:
        return len(pids)
    direct = subprocess.run(
        ["pgrep", "-P", str(server_pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if not direct.stdout.strip():
        return 0
    return len(direct.stdout.strip().splitlines())


def api_post(url: str, api_key: str, payload: dict, timeout: float = 60.0) -> tuple[float, int, dict | None]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - start
            data = json.loads(resp.read().decode())
            return elapsed, resp.status, data
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        return elapsed, exc.code, None


def kill_process_group(proc: subprocess.Popen[bytes], *, grace_sec: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()

    deadline = time.perf_counter() + grace_sec
    while proc.poll() is None and time.perf_counter() < deadline:
        time.sleep(0.1)

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)


def start_api_server(api_key: str, workers: int | None) -> subprocess.Popen:
    subprocess.run(["pkill", "-f", f"randomness_detection.api_server.*{PORT}"], check=False, capture_output=True)
    time.sleep(1)
    env = os.environ.copy()
    env["RANDOMNESS_API_KEY"] = api_key
    env["RANDOMNESS_CACHE_DIR"] = str(CACHE_DIR)
    env["RANDOMNESS_PARALLEL_BACKEND"] = os.environ.get("RANDOMNESS_PARALLEL_BACKEND", "hybrid")
    env["PYTHONPATH"] = str(BASE)
    if workers is not None:
        env["RANDOMNESS_INFERENCE_WORKERS"] = str(workers)
    else:
        env.pop("RANDOMNESS_INFERENCE_WORKERS", None)
    return subprocess.Popen(
        [
            str(BASE / ".venv/bin/python"),
            "-m",
            "randomness_detection.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_for_health(timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("API server did not become healthy.")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(len(ordered) * p / 100)
    index = min(index, len(ordered) - 1)
    return ordered[index]


def run_single_parallel_test(api_key: str, payloads: list[dict], concurrency: int) -> dict:
    url = f"http://127.0.0.1:{PORT}/score"
    latencies: list[float] = []
    ok = 0
    errors = 0
    scores: list[int] = []

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(api_post, url, api_key, payload) for payload in payloads]
        for future in as_completed(futures):
            elapsed, status, data = future.result()
            latencies.append(elapsed)
            if status == 200 and data:
                ok += 1
                scores.append(data["score"])
            else:
                errors += 1
    total_time = time.perf_counter() - start

    return {
        "requests": len(payloads),
        "ok": ok,
        "errors": errors,
        "elapsed_sec": round(total_time, 2),
        "throughput_rps": round(ok / total_time, 1),
        "latency_ms": {
            "p50": round(percentile(latencies, 50) * 1000, 1),
            "p95": round(percentile(latencies, 95) * 1000, 1),
            "p99": round(percentile(latencies, 99) * 1000, 1),
        },
        "score_range": [min(scores), max(scores)] if scores else [0, 0],
    }


def run_batch_parallel_test(api_key: str, batches: list[dict], concurrency: int) -> dict:
    url = f"http://127.0.0.1:{PORT}/score/batch"
    latencies: list[float] = []
    ok_batches = 0
    total_items = 0
    errors = 0

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(api_post, url, api_key, batch) for batch in batches]
        for future in as_completed(futures):
            elapsed, status, data = future.result()
            latencies.append(elapsed)
            if status == 200 and data:
                ok_batches += 1
                total_items += data.get("count", 0)
            else:
                errors += 1
    total_time = time.perf_counter() - start

    return {
        "batches": len(batches),
        "ok_batches": ok_batches,
        "errors": errors,
        "items_scored": total_items,
        "elapsed_sec": round(total_time, 2),
        "throughput_items_per_sec": round(total_items / total_time, 1) if total_time else 0,
        "latency_ms": {
            "p50": round(percentile(latencies, 50) * 1000, 1),
            "p95": round(percentile(latencies, 95) * 1000, 1),
        },
    }


def main() -> int:
    cpus = os.cpu_count() or 1
    cpu_fraction = float(os.environ.get("RANDOMNESS_INFERENCE_CPU_FRACTION", "1.0"))
    cpu_fraction = max(0.1, min(1.0, cpu_fraction))
    explicit_workers = os.environ.get("RANDOMNESS_INFERENCE_WORKERS", "").strip()
    if explicit_workers:
        inference_workers = max(1, int(explicit_workers))
        server_workers: int | None = inference_workers
    else:
        inference_workers = inference_worker_count()
        server_workers = None
    api_key = secrets.token_urlsafe(48)

    print("=" * 64)
    print("REAL PARALLEL + CPU TEST (no mocks)")
    print("=" * 64)
    print(f"Machine CPUs:        {cpus}")
    print(f"CPU fraction:        {cpu_fraction:.0%}")
    print(f"Inference workers:   {inference_workers}")
    print(f"Corpus:              {CACHE_DIR / 'words_alpha.txt'}")
    print(f"Model cache:         {CACHE_DIR}")
    print(f"Test duration:       {DURATION_SEC}s sustained load")
    print(f"Word pool:           up to {STRESS_WORD_LIMIT:,}")
    print(f"Batch size:          {STRESS_BATCH_SIZE}")

    if not (CACHE_DIR / "ensemble.pkl").exists():
        print("ERROR: Trained model not found. Run bootstrap first.", file=sys.stderr)
        return 1

    words = load_real_words(CACHE_DIR, limit=STRESS_WORD_LIMIT)
    print(f"Real words loaded:   {len(words):,}")

    server = start_api_server(api_key, server_workers)
    try:
        health = wait_for_health()
        print(f"API health:          {health}")

        # Auth sanity with a real corpus word (long enough to be natural)
        sanity_word = pick_natural_word(words)
        _, status, data = api_post(
            f"http://127.0.0.1:{PORT}/score",
            api_key,
            {"text": sanity_word},
        )
        if status != 200:
            print(f"ERROR: auth/scoring failed ({status})", file=sys.stderr)
            return 1
        if data["label"] not in ("natural", "uncertain"):
            print(
                f"ERROR: unexpected label for corpus word {sanity_word!r}: {data['label']}",
                file=sys.stderr,
            )
            return 1
        print(f"Sanity check word={sanity_word!r} score={data['score']} label={data['label']}")

        monitor = SystemCpuMonitor()
        monitor.start()
        worker_peak = 0

        # Sustained real load: singles + batches mixed
        singles_done = 0
        batches_done = 0
        errors = 0
        single_latencies: list[float] = []
        batch_latencies: list[float] = []
        start = time.perf_counter()

        single_url = f"http://127.0.0.1:{PORT}/score"
        batch_url = f"http://127.0.0.1:{PORT}/score/batch"
        batch_id = 0
        concurrency = min(64, max(32, inference_workers * 2))

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = []
            while time.perf_counter() - start < DURATION_SEC:
                worker_peak = max(worker_peak, count_worker_processes(server.pid))

                while len(pending) < concurrency * 2:
                    if batch_id % 2 == 0:
                        word = random.choice(words)
                        pending.append(
                            pool.submit(api_post, single_url, api_key, {"text": word})
                        )
                    else:
                        batch_words = build_real_text_batch(words, STRESS_BATCH_SIZE, seed=batch_id)
                        pending.append(
                            pool.submit(
                                api_post,
                                batch_url,
                                api_key,
                                {"texts": batch_words},
                            )
                        )
                    batch_id += 1

                done, pending = pending[:concurrency], pending[concurrency:]
                for future in as_completed(done):
                    elapsed, status, data = future.result()
                    if status == 200:
                        if isinstance(data, dict) and "results" in data:
                            batches_done += 1
                            batch_latencies.append(elapsed)
                        else:
                            singles_done += 1
                            single_latencies.append(elapsed)
                    else:
                        errors += 1
                worker_peak = max(worker_peak, count_worker_processes(server.pid))

            for future in as_completed(pending):
                elapsed, status, data = future.result()
                if status == 200:
                    if isinstance(data, dict) and "results" in data:
                        batches_done += 1
                        batch_latencies.append(elapsed)
                    else:
                        singles_done += 1
                        single_latencies.append(elapsed)
                else:
                    errors += 1

        elapsed = time.perf_counter() - start
        cpu_stats = monitor.stop()
        worker_peak = max(worker_peak, count_worker_processes(server.pid))

        total_requests = singles_done + batches_done
        total_attempts = total_requests + errors
        rps = total_requests / elapsed if elapsed else 0
        items_scored = singles_done + batches_done * STRESS_BATCH_SIZE

        print("\n" + "-" * 64)
        print("RESULTS")
        print("-" * 64)
        print(f"Duration:              {elapsed:.1f}s")
        print(f"Single scores OK:        {singles_done:,}")
        print(f"Batch scores OK:         {batches_done:,}")
        print(f"Items scored (approx):   {items_scored:,}")
        print(f"Errors:                  {errors:,}")
        print(f"Total API calls OK:      {total_requests:,}")
        print(f"Throughput:              {rps:.1f} calls/s")
        if single_latencies:
            print(
                f"Single latency p50/p95:  "
                f"{percentile(single_latencies,50)*1000:.0f}ms / "
                f"{percentile(single_latencies,95)*1000:.0f}ms"
            )
        if batch_latencies:
            print(
                f"Batch latency p50/p95:   "
                f"{percentile(batch_latencies,50)*1000:.0f}ms / "
                f"{percentile(batch_latencies,95)*1000:.0f}ms"
            )
        print(f"Inference worker procs:  {worker_peak} (configured: {inference_workers})")
        print(f"System CPU peak:         {cpu_stats['peak']}%")
        print(f"System CPU avg:          {cpu_stats['avg']}%")
        print(f"System CPU min:          {cpu_stats['min']}%")
        print(f"CPU samples:             {int(cpu_stats['samples'])}")

        backend = health.get("parallel_backend", "hybrid")
        min_peak_cpu = cpu_fraction * 100.0 * 0.35
        min_avg_cpu = cpu_fraction * 100.0 * 0.12
        workers_ok = (
            backend == "thread"
            or worker_peak >= int(inference_workers * 0.5)
            or (cpu_stats["peak"] >= min_peak_cpu * 0.5 and total_requests >= 100)
        )
        cpu_ok = cpu_stats["peak"] >= min_peak_cpu or cpu_stats["avg"] >= min_avg_cpu
        load_ok = total_requests >= 100
        error_ok = errors == 0 or (total_attempts > 0 and errors / total_attempts <= 0.01)

        print("-" * 64)
        print(f"Parallel workers active: {'PASS' if workers_ok else 'FAIL'}")
        print(f"CPU utilization (~{cpu_fraction:.0%} target): {'PASS' if cpu_ok else 'FAIL'}")
        print(f"Real scoring completed:  {'PASS' if load_ok else 'FAIL'}")
        print(f"Error rate (<=1%):       {'PASS' if error_ok else 'FAIL'}")
        print("=" * 64)
        print(
            f"OVERALL: {'PASS' if workers_ok and cpu_ok and load_ok and error_ok else 'FAIL'}"
        )

        return 0 if workers_ok and cpu_ok and load_ok and error_ok else 1

    finally:
        kill_process_group(server)


if __name__ == "__main__":
    sys.exit(main())
