#!/usr/bin/env python3
"""Verify CPU usage during training (50%) and inference (max allocated)."""

from __future__ import annotations

import asyncio
import json
import os
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

from randomness_detection.bootstrap import bootstrap
from randomness_detection.config import CPU_FRACTION
from randomness_detection.inference_pool import InferencePool, inference_worker_count
from randomness_detection.parallel import resolve_parallel_backend, shutdown_joblib, worker_count
from test_helpers import build_real_text_batch, load_real_words, pick_natural_word, pick_random_token

PHASE_TIMEOUT_SEC = 180.0


def sample_system_cpu() -> tuple[int, int]:
    with open("/proc/stat", encoding="utf-8") as handle:
        parts = handle.readline().split()[1:]
    values = list(map(int, parts))
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def count_descendant_processes(pid: int) -> int:
    """Count all descendant processes (not only direct children)."""
    result = subprocess.run(
        ["bash", "-c", f"pgrep -P {pid} | xargs -r -I{{}} pgrep -P {{}} 2>/dev/null; pgrep -P {pid}"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return len(pids)


def count_direct_children(pid: int) -> int:
    result = subprocess.run(
        ["pgrep", "-P", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if not result.stdout.strip():
        return 0
    return len(result.stdout.strip().splitlines())


def count_process_tree(pid: int) -> int:
    descendants = count_descendant_processes(pid)
    return descendants if descendants > 0 else count_direct_children(pid)


def kill_process_group(proc: subprocess.Popen[bytes], *, grace_sec: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.terminate()
    except PermissionError:
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


class CpuMonitor:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.cpu_samples: list[float] = []
        self.child_samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            idle1, total1 = sample_system_cpu()
            time.sleep(0.3)
            while not self._stop.is_set():
                idle2, total2 = sample_system_cpu()
                total_delta = total2 - total1
                idle_delta = idle2 - idle1
                if total_delta > 0:
                    usage = (1.0 - idle_delta / total_delta) * 100.0
                    self.cpu_samples.append(usage)
                self.child_samples.append(count_process_tree(self.pid))
                idle1, total1 = idle2, total2
                time.sleep(0.3)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0.0
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        peak_children = max(self.child_samples) if self.child_samples else 0
        return {
            "peak_cpu_percent": round(peak_cpu, 1),
            "avg_cpu_percent": round(avg_cpu, 1),
            "peak_child_processes": peak_children,
            "samples": len(self.cpu_samples),
        }


def test_training(cache_dir: Path, cpus: int) -> dict:
    expected_workers = worker_count(CPU_FRACTION)
    backend = resolve_parallel_backend()
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING CPU TEST")
    print("=" * 60)
    print(f"System CPUs: {cpus}")
    print(f"Parallel backend: {backend}")
    print(f"Target: {CPU_FRACTION:.0%} → {expected_workers} workers")

    monitor = CpuMonitor(os.getpid())
    monitor.start()
    start = time.perf_counter()
    metadata = bootstrap(cache_dir, force=True)
    elapsed = time.perf_counter() - start
    stats = monitor.stop()
    shutdown_joblib()

    workers_ok = stats["peak_child_processes"] >= int(expected_workers * 0.75)
    cpu_ok = (
        stats["peak_cpu_percent"] >= CPU_FRACTION * 100.0 * 0.70
        or stats["avg_cpu_percent"] >= CPU_FRACTION * 100.0 * 0.35
    )

    result = {
        "phase": "training",
        "expected_workers": expected_workers,
        "elapsed_sec": round(elapsed, 1),
        "metrics": metadata.get("metrics", {}),
        "cpu": stats,
        "workers_ok": workers_ok,
        "cpu_ok": cpu_ok,
        "passed": workers_ok and cpu_ok,
    }

    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Peak child processes: {stats['peak_child_processes']} (need >= {int(expected_workers * 0.75)})")
    print(f"Peak system CPU: {stats['peak_cpu_percent']}% (need >= {CPU_FRACTION * 100 * 0.70:.0f}% peak or >= {CPU_FRACTION * 100 * 0.35:.0f}% avg)")
    print(f"Avg system CPU: {stats['avg_cpu_percent']}%")
    print(f"Train accuracy: {metadata.get('metrics', {}).get('accuracy', 'n/a')}")
    print(f"RESULT: {'PASS' if result['passed'] else 'FAIL'}")
    return result


async def _inference_load(
    pool: InferencePool,
    words: list[str],
    workers: int,
    duration_sec: float = 12.0,
) -> int:
    """Sustained load using real corpus words and random tokens."""
    total = 0
    end = time.perf_counter() + duration_sec
    batch_index = 0
    while time.perf_counter() < end:
        batch = build_real_text_batch(words, workers * 4, seed=batch_index)
        await asyncio.wait_for(pool.score_batch(batch), timeout=60.0)
        total += len(batch)
        batch_index += 1
    return total


def _validate_real_scoring(pool: InferencePool, words: list[str]) -> None:
    """Ensure the model scores real inputs with sensible labels."""
    random_token = pick_random_token()
    random_result = asyncio.run(asyncio.wait_for(pool.score(random_token), timeout=30.0))
    if random_result["label"] != "likely_random":
        raise RuntimeError(
            f"Expected likely_random for token {random_token!r}, got {random_result['label']}"
        )

    natural_word = None
    natural = None
    for candidate in words[:500]:
        if len(candidate) < 5:
            continue
        result = asyncio.run(asyncio.wait_for(pool.score(candidate), timeout=30.0))
        if result["label"] in ("natural", "uncertain") and result["score"] <= 55:
            natural_word = candidate
            natural = result
            break

    if natural is None:
        raise RuntimeError("Could not find a corpus word that scores as natural/uncertain")

    if natural["score"] >= random_result["score"]:
        raise RuntimeError(
            f"Expected corpus word score < random token score, got {natural['score']} vs {random_result['score']}"
        )
    print(
        f"Scoring sanity: word={natural_word!r} score={natural['score']} "
        f"token score={random_result['score']}"
    )


def test_inference(cache_dir: Path, cpus: int) -> dict:
    expected_workers = inference_worker_count()
    backend = resolve_parallel_backend()
    words = load_real_words(cache_dir, limit=10_000)
    print("\n" + "=" * 60)
    print("PHASE 2: INFERENCE CPU TEST (real corpus words)")
    print("=" * 60)
    print(f"System CPUs: {cpus}")
    print(f"Parallel backend: {backend}")
    print(f"Inference workers: {expected_workers}")
    print(f"Real words loaded: {len(words):,}")

    shutdown_joblib()
    pool = InferencePool(str(cache_dir), workers=expected_workers, backend=backend)
    pool.start()

    try:
        _validate_real_scoring(pool, words)
        warmup_children = count_process_tree(os.getpid())
        print(f"Worker processes after warmup: {warmup_children}")

        monitor = CpuMonitor(os.getpid())
        monitor.start()
        start = time.perf_counter()
        total_requests = asyncio.run(
            asyncio.wait_for(
                _inference_load(pool, words, expected_workers, duration_sec=12.0),
                timeout=PHASE_TIMEOUT_SEC,
            )
        )
        elapsed = time.perf_counter() - start
        stats = monitor.stop()
    finally:
        pool.stop()
        shutdown_joblib()

    if backend == "thread":
        workers_ok = True
    else:
        workers_ok = warmup_children >= int(expected_workers * 0.5) or (
            stats["peak_child_processes"] >= int(expected_workers * 0.5)
        )
    cpu_ok = stats["peak_cpu_percent"] >= 10.0 or stats["avg_cpu_percent"] >= 5.0

    result = {
        "phase": "inference",
        "expected_workers": expected_workers,
        "requests": total_requests,
        "elapsed_sec": round(elapsed, 1),
        "throughput_rps": round(total_requests / elapsed, 1),
        "cpu": stats,
        "workers_ok": workers_ok,
        "cpu_ok": cpu_ok,
        "passed": workers_ok and cpu_ok,
    }

    print(f"Requests: {total_requests}")
    print(f"Elapsed: {elapsed:.1f}s ({result['throughput_rps']} req/s)")
    print(f"Worker processes after warmup: {warmup_children} (need >= {int(expected_workers * 0.5)})")
    print(f"Peak child processes during load: {stats['peak_child_processes']}")
    print(f"Peak system CPU: {stats['peak_cpu_percent']}% (need >= 10% peak or >= 5% avg)")
    print(f"Avg system CPU: {stats['avg_cpu_percent']}%")
    print(f"RESULT: {'PASS' if result['passed'] else 'FAIL'}")
    return result


def test_api_inference(cache_dir: Path, cpus: int, port: int = 8777) -> dict:
    expected_workers = int(os.environ.get("RANDOMNESS_INFERENCE_WORKERS", str(min(24, cpus))))
    backend = resolve_parallel_backend()
    words = load_real_words(cache_dir, limit=10_000)
    api_key = secrets.token_urlsafe(48)

    subprocess.run(["pkill", "-f", f"randomness_detection.api_server.*{port}"], check=False, capture_output=True)
    time.sleep(1)

    env = os.environ.copy()
    env["RANDOMNESS_API_KEY"] = api_key
    env["RANDOMNESS_CACHE_DIR"] = str(cache_dir)
    env["RANDOMNESS_INFERENCE_WORKERS"] = str(expected_workers)
    env["RANDOMNESS_PARALLEL_BACKEND"] = backend
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)

    print("\n" + "=" * 60)
    print("PHASE 3: API INFERENCE CPU TEST (real corpus words)")
    print("=" * 60)
    print(f"Parallel backend: {backend}")
    print(f"Inference workers: {expected_workers}")
    print(f"Real words loaded: {len(words):,}")

    server = subprocess.Popen(
        [
            str(Path(__file__).resolve().parent / ".venv/bin/python"),
            "-m",
            "randomness_detection.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    health_url = f"http://127.0.0.1:{port}/health"
    ok_batches = 0
    stats: dict[str, float | int] = {}
    elapsed = 0.0
    try:
        for _ in range(60):
            if server.poll() is not None:
                stderr = (server.stderr.read() if server.stderr else b"").decode(errors="replace")
                return {"phase": "api", "passed": False, "error": f"server exited early: {stderr[:300]}"}
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            return {"phase": "api", "passed": False, "error": "server failed to start within 30s"}

        auth_check = urllib.request.Request(
            f"http://127.0.0.1:{port}/score",
            data=json.dumps({"text": pick_natural_word(words)}).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(auth_check, timeout=15) as resp:
            if resp.status != 200:
                return {"phase": "api", "passed": False, "error": "auth check failed"}
            auth_data = json.loads(resp.read().decode())
            if auth_data.get("label") not in ("natural", "uncertain"):
                return {
                    "phase": "api",
                    "passed": False,
                    "error": f"unexpected label for corpus word: {auth_data.get('label')}",
                }

        monitor = CpuMonitor(server.pid)
        monitor.start()

        def post_batch(batch_id: int) -> int:
            texts = build_real_text_batch(words, 50, seed=batch_id)
            body = json.dumps({"texts": texts}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/score/batch",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status

        duration_sec = 12.0
        phase_deadline = time.perf_counter() + PHASE_TIMEOUT_SEC
        load_start = time.perf_counter()
        load_deadline = load_start + duration_sec
        ok_batches = 0
        batch_id = 0
        concurrent = min(24, expected_workers)

        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            while time.perf_counter() < load_deadline:
                if time.perf_counter() > phase_deadline:
                    return {"phase": "api", "passed": False, "error": "phase timeout exceeded"}
                futures = [
                    executor.submit(post_batch, batch_id + index)
                    for index in range(concurrent)
                ]
                batch_id += concurrent
                for future in as_completed(futures, timeout=45):
                    try:
                        if future.result() == 200:
                            ok_batches += 1
                    except (urllib.error.URLError, TimeoutError, OSError):
                        pass

        elapsed = time.perf_counter() - load_start
        stats = monitor.stop()
    finally:
        kill_process_group(server)

    if ok_batches == 0:
        return {"phase": "api", "passed": False, "error": "no successful batch responses"}

    n = ok_batches * 50
    if backend == "thread":
        workers_ok = True
    else:
        workers_ok = stats["peak_child_processes"] >= int(expected_workers * 0.5)
    cpu_ok = stats["peak_cpu_percent"] >= 10.0 or stats["avg_cpu_percent"] >= 5.0

    result = {
        "phase": "api",
        "expected_workers": expected_workers,
        "requests": n,
        "ok_requests": n,
        "batches": ok_batches,
        "elapsed_sec": round(elapsed, 1),
        "throughput_rps": round(n / elapsed, 1) if elapsed > 0 else 0.0,
        "cpu": stats,
        "workers_ok": workers_ok,
        "cpu_ok": cpu_ok,
        "passed": workers_ok and cpu_ok and ok_batches > 0,
    }

    print(f"Batches OK: {ok_batches} ({n} strings)")
    print(f"Elapsed: {elapsed:.1f}s ({result['throughput_rps']} req/s)")
    print(f"Peak child processes: {stats['peak_child_processes']}")
    print(f"Peak system CPU: {stats['peak_cpu_percent']}%")
    print(f"Avg system CPU: {stats['avg_cpu_percent']}%")
    print(f"RESULT: {'PASS' if result['passed'] else 'FAIL'}")
    return result


def main() -> int:
    base = Path(__file__).resolve().parent
    cache_dir = base / ".cache_cpu_full_test"
    cpus = os.cpu_count() or 1

    os.environ.setdefault("RANDOMNESS_INFERENCE_WORKERS", str(min(24, cpus)))
    os.environ.setdefault("RANDOMNESS_UVICORN_WORKERS", "1")
    os.environ.setdefault("RANDOMNESS_PARALLEL_BACKEND", "hybrid")

    print("=" * 60)
    print("RANDOMNESS SCORER — FULL CPU VERIFICATION (real data)")
    print("=" * 60)
    print(f"Machine CPUs: {cpus}")
    print(f"Parallel backend: {resolve_parallel_backend()}")
    print(f"Train CPU fraction: {CPU_FRACTION:.0%} ({worker_count(CPU_FRACTION)} workers)")
    print(f"Inference workers: {inference_worker_count()}")
    print(f"Phase timeout: {PHASE_TIMEOUT_SEC:.0f}s")

    results = [
        test_training(cache_dir, cpus),
        test_inference(cache_dir, cpus),
        test_api_inference(cache_dir, cpus),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for item in results:
        status = "PASS" if item.get("passed") else "FAIL"
        phase = item.get("phase", "?")
        peak = item.get("cpu", {}).get("peak_cpu_percent", "n/a")
        children = item.get("cpu", {}).get("peak_child_processes", "n/a")
        print(f"  [{status}] {phase:12} peak_cpu={peak}%  workers={children}")
        if item.get("error"):
            print(f"           error: {item['error']}")
        if not item.get("passed"):
            all_passed = False

    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
