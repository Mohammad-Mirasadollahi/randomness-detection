"""Run the FastAPI server."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Randomness Detection API server.")
    parser.add_argument("--host", default=os.environ.get("RANDOMNESS_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RANDOMNESS_PORT", "8765")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RANDOMNESS_UVICORN_WORKERS", "1")),
        help="Number of uvicorn worker processes (set via RANDOMNESS_UVICORN_WORKERS)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only, forces workers=1)",
    )
    args = parser.parse_args(argv)

    workers = 1 if args.reload else max(1, args.workers)
    os.environ["RANDOMNESS_UVICORN_WORKERS"] = str(workers)

    uvicorn.run(
        "randomness_detection.api.app:app",
        host=args.host,
        port=args.port,
        workers=workers,
        reload=args.reload,
        log_level=os.environ.get("RANDOMNESS_LOG_LEVEL", "info"),
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
