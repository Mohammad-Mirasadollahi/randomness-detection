"""Progress logging for the bootstrap / training pipeline."""

from __future__ import annotations

import sys
import time
from typing import TextIO


class BootstrapProgress:
    """Step timer with live stderr logs for install and CLI bootstrap."""

    TOTAL_STEPS = 8

    def __init__(self, *, enabled: bool = True, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self._stream = stream or sys.stderr
        self._started = time.monotonic()
        self._step_started = self._started

    def _elapsed(self, since: float | None = None) -> str:
        seconds = int(time.monotonic() - (since if since is not None else self._started))
        minutes, secs = divmod(max(seconds, 0), 60)
        return f"{minutes}:{secs:02d}"

    def banner(self) -> None:
        if not self.enabled:
            return
        print(
            "\n"
            "============================================================\n"
            " Randomness Detection — model training\n"
            "============================================================\n"
            " Downloads public English word lists, trains locally on\n"
            " your CPU. No data leaves your machine.\n"
            "============================================================",
            file=self._stream,
            flush=True,
        )

    def sources(self, sources: list[tuple[str, str]]) -> None:
        if not self.enabled:
            return
        print("[bootstrap] Data sources:", file=self._stream, flush=True)
        for index, (label, url) in enumerate(sources, start=1):
            print(f"  {index}. {label}", file=self._stream, flush=True)
            print(f"     {url}", file=self._stream, flush=True)

    def step(self, step: int, message: str) -> None:
        if not self.enabled:
            return
        self._step_started = time.monotonic()
        print(
            f"[bootstrap {self._elapsed()}] [{step}/{self.TOTAL_STEPS}] {message}",
            file=self._stream,
            flush=True,
        )

    def detail(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"[bootstrap {self._elapsed()}]   {message}", file=self._stream, flush=True)

    def step_done(self, message: str = "done") -> None:
        if not self.enabled:
            return
        print(
            f"[bootstrap {self._elapsed()}]   {message} "
            f"(step took {self._elapsed(self._step_started)})",
            file=self._stream,
            flush=True,
        )

    def finished(self, metadata: dict) -> None:
        if not self.enabled:
            return
        metrics = metadata.get("metrics", {})
        print(
            f"\n[bootstrap {self._elapsed()}] Training complete in {self._elapsed()}\n"
            f"  Words loaded:     {metadata.get('total_words_loaded', '?')}\n"
            f"  Training samples: {metadata.get('samples_per_class', '?')} per class\n"
            f"  Test accuracy:    {metrics.get('accuracy', '?')}\n"
            f"  Test ROC-AUC:     {metrics.get('auc', '?')}\n"
            f"  Cache directory:  {metadata.get('cache_dir', '?')}",
            file=self._stream,
            flush=True,
        )

    def skipped(self, reason: str) -> None:
        if not self.enabled:
            return
        print(
            f"[bootstrap {self._elapsed()}] Skipped — {reason}",
            file=self._stream,
            flush=True,
        )
