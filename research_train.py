#!/usr/bin/env python3
"""
Train the LRD-Hybrid research model (paper-grade ensemble).

Usage:
  PYTHONPATH=. .venv/bin/python research_train.py
  PYTHONPATH=. .venv/bin/python research_train.py --force --verbose
  PYTHONPATH=. .venv/bin/python research_train.py --samples 10000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from randomness_detection.config import DEFAULT_CACHE_DIR, TRAIN_SAMPLES_PER_CLASS
from randomness_detection.research.hybrid_bootstrap import bootstrap_hybrid


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LRD-Hybrid research ensemble")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("RANDOMNESS_CACHE_DIR", str(DEFAULT_CACHE_DIR)),
        help="Model cache directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=TRAIN_SAMPLES_PER_CLASS,
        help="Training samples per class",
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if artifacts exist")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    args = parser.parse_args()

    metadata = bootstrap_hybrid(
        args.cache_dir,
        samples_per_class=args.samples,
        force=args.force,
        verbose=args.verbose,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
