#!/usr/bin/env python3
"""Standalone DEFLATE compression-ratio scorer for quality benchmarks.

Invoked via subprocess — not part of randomness_detection.features.
Uses raw DEFLATE (wbits=-15) so short strings are scored without gzip framing overhead.
"""
from __future__ import annotations

import sys
import zlib


def main() -> int:
    text = sys.stdin.read()
    raw = text.encode("utf-8")
    if len(raw) < 2:
        print("50.0")
        return 0
    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    ratio = len(compressed) / len(raw)
    print(min(ratio, 1.0) * 100.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
