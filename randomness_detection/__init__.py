"""Randomness detection using freq analysis, entropy, and compression."""

from .input_parser import parse_word_list, parse_word_list_from_file
from .scorer import BatchScoreResult, ScoreResult, Scorer

__all__ = [
    "Scorer",
    "ScoreResult",
    "BatchScoreResult",
    "parse_word_list",
    "parse_word_list_from_file",
]
__version__ = "1.0.0"
__license__ = "MIT"
