"""FastAPI HTTP interface for randomness_detection."""

from .app import app, create_app

__all__ = ["app", "create_app"]
