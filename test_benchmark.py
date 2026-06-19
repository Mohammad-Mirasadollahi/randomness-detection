#!/usr/bin/env python3
"""Real performance benchmarks — trained model, corpus words, live API (no mocks)."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from randomness_detection.exclude import ExcludeManager
from randomness_detection.inference_pool import InferencePool, inference_worker_count
from randomness_detection.parallel import resolve_parallel_backend, shutdown_joblib
from randomness_detection.scorer import Scorer
from test_helpers import build_real_text_batch, load_real_words

BASE = Path(__file__).resolve().parent
PORT = 8799
RESULTS_FILE = BASE / "benchmark_results.json"


def resolve_cache_dir() -> Path:
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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * p / 100), len(ordered) - 1)
    return ordered[index]


@dataclass
class BenchmarkResult:
    name: str
    unit: str
    throughput: float
    samples: int
    elapsed_sec: float
    extra: dict[str, float | int | str]


def bench_cli_scoring(cache_dir: Path, texts: list[str]) -> BenchmarkResult:
    scorer = Scorer(cache_dir=str(cache_dir), auto_bootstrap=False)
    start = time.perf_counter()
    results = scorer.score_batch(texts)
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        name="CLI batch scoring",
        unit="texts/s",
        throughput=round(len(results) / elapsed, 1),
        samples=len(texts),
        elapsed_sec=round(elapsed, 3),
        extra={"backend": "single-process Scorer"},
    )


async def _pool_load(pool: InferencePool, texts: list[str], duration_sec: float) -> int:
    total = 0
    end = time.perf_counter() + duration_sec
    batch_size = max(32, inference_worker_count() * 4)
    index = 0
    while time.perf_counter() < end:
        offset = (index * batch_size) % len(texts)
        batch = texts[offset : offset + batch_size]
        if len(batch) < batch_size:
            batch = texts[:batch_size]
        await pool.score_batch(batch)
        total += len(batch)
        index += 1
    return total


def bench_inference_pool(cache_dir: Path, texts: list[str], *, duration_sec: float = 10.0) -> BenchmarkResult:
    backend = resolve_parallel_backend()
    workers = inference_worker_count()
    shutdown_joblib()
    pool = InferencePool(str(cache_dir), workers=workers, backend=backend)
    pool.start()
    try:
        start = time.perf_counter()
        total = asyncio.run(_pool_load(pool, texts, duration_sec))
        elapsed = time.perf_counter() - start
    finally:
        pool.stop()
        shutdown_joblib()
    return BenchmarkResult(
        name="Inference pool",
        unit="texts/s",
        throughput=round(total / elapsed, 1) if elapsed else 0.0,
        samples=total,
        elapsed_sec=round(elapsed, 3),
        extra={"backend": backend, "workers": workers},
    )


def bench_exclude(cache_dir: Path, *, rule_count: int = 50_000, check_count: int = 10_000) -> BenchmarkResult:
    db_name = f"bench_{secrets.token_hex(4)}.db"
    manager = ExcludeManager.open(cache_dir, db_name=db_name)
    try:
        rules = [(f"block{i}.example.com", "domain") for i in range(rule_count)]
        manager.add_rules(rules)
        samples = [f"https://app.block{rule_count // 2}.example.com/path"]
        samples += [secrets.token_hex(8) for _ in range(min(check_count, 999) - 1)]
        while len(samples) < check_count:
            samples.extend(samples[: check_count - len(samples)])
        samples = samples[:check_count]

        start = time.perf_counter()
        hits = 0
        for text in samples:
            if manager.check_exclude(text) is not None:
                hits += 1
        elapsed = time.perf_counter() - start
    finally:
        manager.close()
        db_path = cache_dir / db_name
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

    return BenchmarkResult(
        name="Exclude pre-filter",
        unit="checks/s",
        throughput=round(check_count / elapsed, 1) if elapsed else 0.0,
        samples=check_count,
        elapsed_sec=round(elapsed, 3),
        extra={"rules": rule_count, "hits": hits},
    )


def api_post(url: str, api_key: str, payload: dict) -> tuple[float, int]:
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
            return time.perf_counter() - start, resp.status
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - start, exc.code


def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)


def start_api_server(api_key: str, cache_dir: Path, workers: int) -> subprocess.Popen:
    subprocess.run(
        ["pkill", "-f", f"randomness_detection.api_server.*{PORT}"],
        check=False,
        capture_output=True,
    )
    time.sleep(0.5)
    env = os.environ.copy()
    env["RANDOMNESS_API_KEY"] = api_key
    env["RANDOMNESS_CACHE_DIR"] = str(cache_dir)
    env["RANDOMNESS_INFERENCE_WORKERS"] = str(workers)
    env["RANDOMNESS_PARALLEL_BACKEND"] = os.environ.get("RANDOMNESS_PARALLEL_BACKEND", "hybrid")
    env["PYTHONPATH"] = str(BASE)
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


def wait_for_health(timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("API server did not become healthy.")


def bench_api_singles(api_key: str, texts: list[str], *, requests: int, concurrency: int) -> BenchmarkResult:
    url = f"http://127.0.0.1:{PORT}/score"
    payloads = [{"text": texts[i % len(texts)]} for i in range(requests)]
    latencies: list[float] = []
    ok = 0

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(api_post, url, api_key, payload) for payload in payloads]
        for future in as_completed(futures):
            elapsed, status = future.result()
            latencies.append(elapsed)
            if status == 200:
                ok += 1
    total = time.perf_counter() - start

    return BenchmarkResult(
        name="API POST /score",
        unit="req/s",
        throughput=round(ok / total, 1) if total else 0.0,
        samples=ok,
        elapsed_sec=round(total, 3),
        extra={
            "concurrency": concurrency,
            "p50_ms": round(percentile(latencies, 50) * 1000, 1),
            "p95_ms": round(percentile(latencies, 95) * 1000, 1),
            "errors": requests - ok,
        },
    )


def bench_api_batches(
    api_key: str,
    texts: list[str],
    *,
    batches: int,
    batch_size: int,
    concurrency: int,
) -> BenchmarkResult:
    url = f"http://127.0.0.1:{PORT}/score/batch"
    batch_payloads = []
    for batch_id in range(batches):
        batch = build_real_text_batch(texts, batch_size, seed=batch_id)
        batch_payloads.append({"texts": batch})

    latencies: list[float] = []
    items = 0
    ok = 0

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(api_post, url, api_key, payload) for payload in batch_payloads]
        for future in as_completed(futures):
            elapsed, status = future.result()
            latencies.append(elapsed)
            if status == 200:
                ok += 1
                items += batch_size
    total = time.perf_counter() - start

    return BenchmarkResult(
        name="API POST /score/batch",
        unit="items/s",
        throughput=round(items / total, 1) if total else 0.0,
        samples=items,
        elapsed_sec=round(total, 3),
        extra={
            "batches": batches,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "p50_ms": round(percentile(latencies, 50) * 1000, 1),
            "p95_ms": round(percentile(latencies, 95) * 1000, 1),
            "errors": batches - ok,
        },
    )


def print_report(meta: dict, results: list[BenchmarkResult]) -> None:
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS (real model, real corpus, no mocks)")
    print("=" * 72)
    for key, value in meta.items():
        print(f"  {key}: {value}")
    print("-" * 72)
    print(f"{'Benchmark':<28} {'Throughput':>14} {'Samples':>10} {'Time':>8}")
    print("-" * 72)
    for row in results:
        print(
            f"{row.name:<28} {row.throughput:>10,.1f} {row.unit:<3} {row.samples:>10,} {row.elapsed_sec:>7.2f}s"
        )
        for extra_key, extra_val in row.extra.items():
            if extra_key.endswith("_ms"):
                print(f"    {extra_key}: {extra_val}")
    print("=" * 72)


def main() -> int:
    cpus = os.cpu_count() or 1
    workers = inference_worker_count()
    backend = resolve_parallel_backend()
    cache_dir = resolve_cache_dir()
    words = load_real_words(cache_dir, limit=15_000)
    texts = build_real_text_batch(words, 4_000, seed=42)
    api_key = secrets.token_urlsafe(48)
    concurrency = min(32, max(8, workers))

    meta = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cpus": cpus,
        "inference_workers": workers,
        "parallel_backend": backend,
        "cache_dir": str(cache_dir),
        "corpus_words": len(words),
    }

    print("=" * 72)
    print("RANDOMNESS DETECTION — REAL BENCHMARK")
    print("=" * 72)
    for key, value in meta.items():
        print(f"  {key}: {value}")

    results: list[BenchmarkResult] = []

    print("\n[1/5] CLI batch scoring...")
    results.append(bench_cli_scoring(cache_dir, texts[:2_000]))

    print("[2/5] Inference pool (10s sustained)...")
    results.append(bench_inference_pool(cache_dir, texts))

    print("[3/5] Exclude pre-filter (50K domain rules)...")
    results.append(bench_exclude(cache_dir))

    server = start_api_server(api_key, cache_dir, workers)
    try:
        wait_for_health()
        print("[4/5] API single-score throughput...")
        results.append(
            bench_api_singles(api_key, texts, requests=800, concurrency=concurrency)
        )
        print("[5/5] API batch throughput...")
        results.append(
            bench_api_batches(
                api_key,
                words,
                batches=40,
                batch_size=40,
                concurrency=max(4, concurrency // 2),
            )
        )
    finally:
        kill_process_group(server)

    print_report(meta, results)

    payload = {
        "meta": meta,
        "results": [asdict(row) for row in results],
        "generated_at_unix": int(time.time()),
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {RESULTS_FILE}")

    # Sanity: all benchmarks must complete with non-zero throughput
    failed = [row.name for row in results if row.throughput <= 0 or row.samples <= 0]
    if failed:
        print(f"FAILED benchmarks: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
