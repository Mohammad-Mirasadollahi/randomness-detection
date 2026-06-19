"""Atomic pickle I/O for model artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path


def atomic_pickle_dump(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
