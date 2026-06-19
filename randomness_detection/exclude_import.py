"""Bulk import exclusion rules from a newline-delimited file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .exclude import ExcludeManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import exclusion rules from a text file.")
    parser.add_argument("file", help="Newline-delimited patterns")
    parser.add_argument(
        "--type",
        default="wildcard",
        choices=["exact", "domain", "suffix", "prefix", "glob", "wildcard"],
        help="Rule type for all patterns",
    )
    parser.add_argument("--cache-dir", default=None, help="Model/exclude cache directory")
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    manager = ExcludeManager.open(args.cache_dir)
    try:
        batch: list[tuple[str, str]] = []
        total_added = 0
        total_dupes = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                pattern = line.strip()
                if not pattern or pattern.startswith("#"):
                    continue
                batch.append((pattern, args.type))
                if len(batch) >= args.batch_size:
                    result = manager.add_rules(batch)
                    total_added += result["added"]
                    total_dupes += result["duplicates"]
                    batch.clear()
                    print(f"Imported so far: added={total_added:,} duplicates={total_dupes:,}")
        if batch:
            result = manager.add_rules(batch)
            total_added += result["added"]
            total_dupes += result["duplicates"]
        stats = manager.stats()
        print(
            f"Done. added={total_added:,} duplicates={total_dupes:,} "
            f"exact_rules={stats['exact_rules']:,} wildcard_rules={stats['wildcard_rules']:,}"
        )
    finally:
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
