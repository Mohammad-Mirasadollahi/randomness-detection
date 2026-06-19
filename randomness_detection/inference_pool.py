"""High-throughput parallel inference pool for API scoring."""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from .parallel import (
    ParallelBackend,
    configure_worker_env,
    process_pool_context,
    resolve_parallel_backend,
)
from .scorer import Scorer

_WORKER_SCORER: Scorer | None = None
_THREAD_LOCAL = threading.local()


def inference_worker_count() -> int:
    """
    Resolve process worker count for this API process.

    Honors RANDOMNESS_INFERENCE_WORKERS or splits CPUs across uvicorn workers
    using RANDOMNESS_INFERENCE_CPU_FRACTION (default 1.0 = all allocated CPUs).
    """
    explicit = os.environ.get("RANDOMNESS_INFERENCE_WORKERS", "").strip()
    if explicit:
        return max(1, int(explicit))

    total_cpus = os.cpu_count() or 1
    uvicorn_workers = max(1, int(os.environ.get("RANDOMNESS_UVICORN_WORKERS", "1")))
    fraction = float(os.environ.get("RANDOMNESS_INFERENCE_CPU_FRACTION", "1.0"))
    fraction = max(0.1, min(1.0, fraction))

    allocated = max(1, int(total_cpus * fraction))
    return max(1, allocated // uvicorn_workers)


def inference_thread_count() -> int:
    """Resolve thread worker count (used in thread/hybrid modes)."""
    explicit = os.environ.get("RANDOMNESS_INFERENCE_THREADS", "").strip()
    if explicit:
        return max(1, int(explicit))
    return inference_worker_count()


def _init_process_worker(cache_dir: str) -> None:
    global _WORKER_SCORER
    configure_worker_env()
    _WORKER_SCORER = Scorer(cache_dir=cache_dir, auto_bootstrap=False)


def _score_text_process(text: str) -> dict[str, Any]:
    if _WORKER_SCORER is None:
        raise RuntimeError("Process inference worker is not initialized.")
    return _WORKER_SCORER.score(text).to_dict()


def _init_thread_worker(cache_dir: str) -> None:
    configure_worker_env()
    _THREAD_LOCAL.scorer = Scorer(cache_dir=cache_dir, auto_bootstrap=False)


def _score_text_thread(text: str) -> dict[str, Any]:
    scorer = getattr(_THREAD_LOCAL, "scorer", None)
    if scorer is None:
        raise RuntimeError("Thread inference worker is not initialized.")
    return scorer.score(text).to_dict()


class InferencePool:
    """
    Parallel inference pool supporting process, thread, or hybrid backends.

    - process: ProcessPoolExecutor for CPU-bound scoring (bypasses GIL)
    - thread: ThreadPoolExecutor with per-thread Scorer instances
    - hybrid: asyncio threads handle concurrent requests; process pool scores batches
    """

    def __init__(
        self,
        cache_dir: str,
        *,
        workers: int | None = None,
        threads: int | None = None,
        backend: ParallelBackend | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.backend = backend or resolve_parallel_backend()
        self.process_workers = workers if workers is not None else inference_worker_count()
        self.thread_workers = threads if threads is not None else inference_thread_count()
        self._process_executor: ProcessPoolExecutor | None = None
        self._thread_executor: ThreadPoolExecutor | None = None

    @property
    def workers(self) -> int:
        """Backward-compatible alias for process worker count."""
        return self.process_workers

    def start(self) -> None:
        if self._process_executor is not None or self._thread_executor is not None:
            return

        configure_worker_env()

        if self.backend in ("process", "hybrid"):
            ctx = process_pool_context()
            self._process_executor = ProcessPoolExecutor(
                max_workers=self.process_workers,
                mp_context=ctx,
                initializer=_init_process_worker,
                initargs=(self.cache_dir,),
            )

        if self.backend in ("thread", "hybrid"):
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.thread_workers,
                thread_name_prefix="rs-infer",
                initializer=_init_thread_worker,
                initargs=(self.cache_dir,),
            )

    def stop(self) -> None:
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=True, cancel_futures=True)
            self._process_executor = None
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=True, cancel_futures=True)
            self._thread_executor = None

    def _score_fn(self) -> Any:
        if self.backend == "thread":
            return _score_text_thread
        return _score_text_process

    def _map_texts_process(self, texts: list[str]) -> list[dict[str, Any]]:
        executor = self._process_executor
        if executor is None:
            raise RuntimeError("Process inference pool is not started.")
        chunksize = max(1, min(64, len(texts) // (self.process_workers * 2) or 1))
        return list(executor.map(_score_text_process, texts, chunksize=chunksize))

    def _map_texts_thread(self, texts: list[str]) -> list[dict[str, Any]]:
        executor = self._thread_executor
        if executor is None:
            raise RuntimeError("Thread inference pool is not started.")
        chunksize = max(1, min(64, len(texts) // (self.thread_workers * 2) or 1))
        return list(executor.map(_score_text_thread, texts, chunksize=chunksize))

    def _map_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []

        if self.backend == "thread":
            return self._map_texts_thread(texts)
        # process and hybrid: score on process pool (threads handle API concurrency)
        return self._map_texts_process(texts)

    async def score(self, text: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()

        if self.backend == "thread":
            executor = self._thread_executor
        else:
            executor = self._process_executor

        if executor is None:
            raise RuntimeError("Inference pool is not started.")
        return await loop.run_in_executor(executor, self._score_fn(), text)

    async def score_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        if self.backend == "thread":
            loop = asyncio.get_running_loop()
            executor = self._thread_executor
            if executor is None:
                raise RuntimeError("Thread inference pool is not started.")
            return await loop.run_in_executor(executor, self._map_texts_thread, texts)
        return await asyncio.to_thread(self._map_texts_process, texts)
