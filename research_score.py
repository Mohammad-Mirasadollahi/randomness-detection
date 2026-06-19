#!/usr/bin/env python3
"""Score strings with the LRD-Hybrid research model."""

from __future__ import annotations

import argparse
import json
import sys

from randomness_detection.research import HybridScorer


def main() -> int:
    parser = argparse.ArgumentParser(description="Score text with LRD-Hybrid")
    parser.add_argument("text", nargs="?", help="String to score")
    parser.add_argument("-f", "--file", help="File with one string per line")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide text or --file")

    scorer = HybridScorer()
    lines: list[str] = []
    if args.file:
        lines.extend(line.strip() for line in open(args.file, encoding="utf-8") if line.strip())
    if args.text:
        lines.append(args.text)

    for line in lines:
        result = scorer.score(line)
        if args.json:
            print(json.dumps({"text": line, **result.__dict__}, ensure_ascii=False))
        else:
            print(f"{result.score:3d}  {result.label:14s}  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
