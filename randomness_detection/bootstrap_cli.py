"""CLI entry point for verbose model training (used by install.sh)."""

from __future__ import annotations

import argparse
import os
import sys

from .bootstrap import bootstrap
from .config import DEFAULT_CACHE_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download public word corpora and train the randomness detection model.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("RANDOMNESS_CACHE_DIR", str(DEFAULT_CACHE_DIR)),
        help="Directory for cached models and training data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download corpora and retrain even if a model already exists",
    )
    args = parser.parse_args(argv)

    try:
        bootstrap(args.cache_dir, force=args.force, verbose=True)
    except Exception as exc:
        print(f"[bootstrap] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
