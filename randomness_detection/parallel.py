"""Parallel execution with process and/or thread pools."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from pathlib import Path
from typing import Callable, Literal, TypeVar

from .config import PARALLEL_BACKEND

T = TypeVar("T")
R = TypeVar("R")

ParallelBackend = Literal["process", "thread", "hybrid"]

_SESSION_PROCESS_EXECUTOR: ProcessPoolExecutor | None = None
_SESSION_THREAD_EXECUTOR: ThreadPoolExecutor | None = None
_ACTIVE_PROCESS_WORKERS = 0
_ACTIVE_THREAD_WORKERS = 0
_ACTIVE_BACKEND: ParallelBackend = "process"


def resolve_parallel_backend() -> ParallelBackend:
    raw = os.environ.get("RANDOMNESS_PARALLEL_BACKEND", PARALLEL_BACKEND).strip().lower()
    if raw in ("process", "thread", "hybrid"):
        return raw  # type: ignore[return-value]
    return "hybrid"


def worker_count(cpu_fraction: float = 0.5) -> int:
    cpus = os.cpu_count() or 1
    return max(1, int(cpus * cpu_fraction))


def configure_worker_env() -> None:
    """Prevent BLAS/OpenMP oversubscription inside worker processes."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def process_pool_context() -> mp.context.BaseContext:
    """
    Return a fork-safe multiprocessing context.

  Uses forkserver on Linux to avoid deadlocks when creating process pools after
  sklearn/joblib or thread pools have been active in the parent process.
    """
    for method in ("forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return mp.get_context()


def shutdown_joblib() -> None:
    """Release sklearn/joblib worker processes before starting our own pools."""
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
    try:
        from joblib.externals.loky.process_executor import _process_executor_cache

        _process_executor_cache.clear()
    except Exception:
        pass


def active_workers() -> int:
    return _ACTIVE_PROCESS_WORKERS


def active_thread_workers() -> int:
    return _ACTIVE_THREAD_WORKERS


def active_backend() -> ParallelBackend:
    return _ACTIVE_BACKEND


def start_parallel(
    cpu_fraction: float = 0.5,
    *,
    backend: ParallelBackend | None = None,
) -> dict[str, int | str]:
    """Start reusable process and/or thread pools for the training phase."""
    global _SESSION_PROCESS_EXECUTOR, _SESSION_THREAD_EXECUTOR
    global _ACTIVE_PROCESS_WORKERS, _ACTIVE_THREAD_WORKERS, _ACTIVE_BACKEND

    stop_parallel()
    configure_worker_env()

    selected = backend or resolve_parallel_backend()
    _ACTIVE_BACKEND = selected
    process_workers = worker_count(cpu_fraction)
    thread_workers = max(1, process_workers)

    if selected in ("process", "hybrid"):
        _ACTIVE_PROCESS_WORKERS = process_workers
        ctx = process_pool_context()
        _SESSION_PROCESS_EXECUTOR = ProcessPoolExecutor(
            max_workers=process_workers,
            mp_context=ctx,
            initializer=configure_worker_env,
        )

    if selected in ("thread", "hybrid"):
        _ACTIVE_THREAD_WORKERS = thread_workers
        _SESSION_THREAD_EXECUTOR = ThreadPoolExecutor(
            max_workers=thread_workers,
            thread_name_prefix="rs-train",
        )

    return {
        "backend": selected,
        "process_workers": _ACTIVE_PROCESS_WORKERS,
        "thread_workers": _ACTIVE_THREAD_WORKERS,
    }


def stop_parallel() -> None:
    global _SESSION_PROCESS_EXECUTOR, _SESSION_THREAD_EXECUTOR
    global _ACTIVE_PROCESS_WORKERS, _ACTIVE_THREAD_WORKERS, _ACTIVE_BACKEND

    if _SESSION_PROCESS_EXECUTOR is not None:
        _SESSION_PROCESS_EXECUTOR.shutdown(wait=True)
        _SESSION_PROCESS_EXECUTOR = None
    if _SESSION_THREAD_EXECUTOR is not None:
        _SESSION_THREAD_EXECUTOR.shutdown(wait=True)
        _SESSION_THREAD_EXECUTOR = None
    _ACTIVE_PROCESS_WORKERS = 0
    _ACTIVE_THREAD_WORKERS = 0
    _ACTIVE_BACKEND = "process"


def _map_process(
    func: Callable[[T], R],
    items: list[T],
    *,
    workers: int,
    chunksize: int,
) -> list[R]:
    if not items:
        return []
    if workers <= 1:
        return [func(item) for item in items]

    if _SESSION_PROCESS_EXECUTOR is not None:
        return list(_SESSION_PROCESS_EXECUTOR.map(func, items, chunksize=chunksize))

    ctx = process_pool_context()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=configure_worker_env,
    ) as executor:
        return list(executor.map(func, items, chunksize=chunksize))


def _map_thread(
    func: Callable[[T], R],
    items: list[T],
    *,
    workers: int,
    chunksize: int,
) -> list[R]:
    if not items:
        return []
    if workers <= 1:
        return [func(item) for item in items]

    if _SESSION_THREAD_EXECUTOR is not None:
        return list(_SESSION_THREAD_EXECUTOR.map(func, items, chunksize=chunksize))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rs-map") as executor:
        return list(executor.map(func, items, chunksize=chunksize))


def _map_hybrid(
    func: Callable[[T], R],
    items: list[T],
    *,
    workers: int,
    chunksize: int,
) -> list[R]:
    if not items:
        return []
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    chunk_size = max(chunksize, max(1, len(items) // (workers * 2)))
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def process_chunk(chunk: list[T]) -> list[R]:
        if _SESSION_PROCESS_EXECUTOR is not None and len(chunk) > 1:
            return list(_SESSION_PROCESS_EXECUTOR.map(func, chunk))
        return [func(item) for item in chunk]

    if _SESSION_THREAD_EXECUTOR is not None:
        futures = [_SESSION_THREAD_EXECUTOR.submit(process_chunk, chunk) for chunk in chunks]
        return [item for future in futures for item in future.result()]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rs-hybrid") as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        return [item for future in futures for item in future.result()]


def _map(
    func: Callable[[T], R],
    items: list[T],
    *,
    workers: int,
    chunksize: int,
    backend: ParallelBackend | None = None,
) -> list[R]:
    selected = backend or _ACTIVE_BACKEND or resolve_parallel_backend()
    if selected == "thread":
        return _map_thread(func, items, workers=workers, chunksize=chunksize)
    if selected == "hybrid":
        return _map_hybrid(func, items, workers=workers, chunksize=chunksize)
    return _map_process(func, items, workers=workers, chunksize=chunksize)


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    *,
    cpu_fraction: float = 0.5,
    chunksize: int = 64,
    backend: ParallelBackend | None = None,
) -> list[R]:
    selected = backend or _ACTIVE_BACKEND or resolve_parallel_backend()
    workers = _ACTIVE_PROCESS_WORKERS or _ACTIVE_THREAD_WORKERS or worker_count(cpu_fraction)
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    return _map(func, items, workers=workers, chunksize=chunksize, backend=selected)


def _extract_chunk(args: tuple[list[str], str]) -> list[list[float]]:
    texts, freq_path = args
    configure_worker_env()
    from .features import extract_features
    from .freq_model import FreqCounter

    counter = FreqCounter()
    counter.load(freq_path)
    return [extract_features(text, counter).as_list() for text in texts]


def _extract_ensemble_chunk_vectors(args: tuple[list[str], str]):
    texts, cache_dir = args
    configure_worker_env()
    from .bootstrap import load_freq_counter, load_language_model, load_pmi_model
    from .ensemble_features import extract_ensemble_features

    freq = load_freq_counter(cache_dir)
    lm = load_language_model(cache_dir)
    pmi = load_pmi_model(cache_dir)
    return [extract_ensemble_features(text, freq, lm, pmi) for text in texts]


def extract_ensemble_features_parallel(
    texts: list[str],
    cache_dir: str | Path,
    *,
    cpu_fraction: float = 0.5,
    chunksize: int = 64,
) -> list:
    cache_dir = Path(cache_dir)
    workers = _ACTIVE_PROCESS_WORKERS or _ACTIVE_THREAD_WORKERS or worker_count(cpu_fraction)
    if workers <= 1 or len(texts) <= 1:
        from .bootstrap import load_freq_counter, load_language_model, load_pmi_model
        from .ensemble_features import extract_ensemble_features

        freq = load_freq_counter(cache_dir)
        lm = load_language_model(cache_dir)
        pmi = load_pmi_model(cache_dir)
        return [extract_ensemble_features(text, freq, lm, pmi) for text in texts]

    chunk_size = max(chunksize, max(1, len(texts) // (workers * 4)))
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    jobs = [(chunk, str(cache_dir)) for chunk in chunks]
    parts = _map(_extract_ensemble_chunk_vectors, jobs, workers=workers, chunksize=1)
    return [row for part in parts for row in part]


def extract_features_parallel(
    texts: list[str],
    freq_path: str,
    *,
    cpu_fraction: float = 0.5,
    chunksize: int = 64,
) -> list[list[float]]:
    workers = _ACTIVE_PROCESS_WORKERS or _ACTIVE_THREAD_WORKERS or worker_count(cpu_fraction)
    if workers <= 1:
        from .features import extract_features
        from .freq_model import FreqCounter

        counter = FreqCounter()
        counter.load(freq_path)
        return [extract_features(text, counter).as_list() for text in texts]

    chunk_size = max(chunksize, max(1, len(texts) // (workers * 4)))
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    jobs = [(chunk, freq_path) for chunk in chunks]
    parts = _map(_extract_chunk, jobs, workers=workers, chunksize=1)
    return [row for part in parts for row in part]


def _tally_chunk(chunk: list[str]) -> dict[str, dict[str, int]]:
    configure_worker_env()
    from .freq_model import FreqCounter

    partial_counter = FreqCounter()
    partial_counter.tally_words(chunk)
    return partial_counter.export_table()


def tally_words_parallel(
    words: list[str],
    freq_path: str | Path,
    *,
    cpu_fraction: float = 0.5,
) -> None:
    from .freq_model import FreqCounter

    freq_path = Path(freq_path)
    workers = _ACTIVE_PROCESS_WORKERS or _ACTIVE_THREAD_WORKERS or worker_count(cpu_fraction)

    if workers <= 1 or len(words) < 1_000:
        counter = FreqCounter()
        counter.tally_words(words)
        counter.save(freq_path)
        return

    chunk_size = max(500, len(words) // workers)
    chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]
    partial_tables = _map(_tally_chunk, chunks, workers=workers, chunksize=1)

    counter = FreqCounter()
    for table in partial_tables:
        counter.merge_table(table)
    counter.save(freq_path)
