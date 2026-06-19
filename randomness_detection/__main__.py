"""CLI entry point for randomness_detection."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .bootstrap import bootstrap
from .config import DEFAULT_CACHE_DIR
from .input_parser import parse_word_list, parse_word_list_from_file
from .scorer import BatchScoreResult, ScoreResult, Scorer


def _collect_words(args: argparse.Namespace) -> list[str]:
    if args.file:
        return parse_word_list_from_file(args.file)

    raw_input = args.text
    if raw_input is None and not sys.stdin.isatty():
        raw_input = sys.stdin.read()

    return parse_word_list(raw_input) if raw_input else []


def _print_batch_results(results: list[BatchScoreResult], *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {"count": len(results), "results": [r.to_dict() for r in results]},
                indent=2,
            )
        )
        return

    print(f"{'TEXT':<30} {'SCORE':>5}  {'LABEL':<15} CONF")
    print("-" * 62)
    for item in results:
        print(
            f"{item.text[:30]:<30} {item.result.score:>5}  "
            f"{item.result.label:<15} {item.result.confidence}"
        )


def _print_single_result(result: ScoreResult, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"Score: {result.score}")
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence}")
    print(
        "Breakdown: "
        f"freq={result.breakdown['freq']}, "
        f"entropy={result.breakdown['entropy']}, "
        f"compression={result.breakdown['compression']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score how random a string looks (1-100, higher = more random).",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Single string, comma-separated words, or multiline word list",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Read words from file (one per line or comma-separated)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("RANDOMNESS_CACHE_DIR", str(DEFAULT_CACHE_DIR)),
        help=(
            "Directory for cached models and training data "
            "(default: $RANDOMNESS_CACHE_DIR or ~/.cache/randomness_detection)"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Force re-download words and retrain models",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    args = parser.parse_args(argv)

    if args.bootstrap:
        metadata = bootstrap(args.cache_dir, force=True, verbose=True)
        if args.json:
            print(json.dumps({"bootstrapped": True, "metadata": metadata}, indent=2))
        else:
            print("Bootstrap complete.")
            print(json.dumps(metadata, indent=2))
        if not args.text and not args.file and sys.stdin.isatty():
            return 0

    words = _collect_words(args)
    if not words:
        parser.print_help()
        return 1

    scorer = Scorer(cache_dir=args.cache_dir, force_bootstrap=args.bootstrap)

    if len(words) == 1:
        _print_single_result(scorer.score(words[0]), as_json=args.json)
        return 0

    _print_batch_results(scorer.score_batch(words), as_json=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
