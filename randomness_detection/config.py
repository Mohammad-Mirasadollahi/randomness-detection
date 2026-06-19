"""Configuration and constants for randomness_detection."""

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "randomness_detection"


@dataclass(frozen=True)
class WordSource:
    """Public corpus file downloaded during bootstrap."""

    filename: str
    url: str
    label: str
    parse: str = "plain"  # plain | scrabble | eff


# Natural-language training corpora (one download per source, merged locally).
WORD_SOURCES: list[WordSource] = [
    WordSource(
        "words_alpha.txt",
        "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
        "dwyl/english-words — alphabetic dictionary (~370K words)",
    ),
    WordSource(
        "words.txt",
        "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt",
        "dwyl/english-words — full word list incl. contractions",
    ),
    WordSource(
        "google-10000-english.txt",
        "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt",
        "first20hours/google-10000-english — 10K most common English words",
    ),
    WordSource(
        "scrabble-dictionary.txt",
        "https://raw.githubusercontent.com/redbo/scrabble/master/dictionary.txt",
        "redbo/scrabble — North American Scrabble dictionary",
        parse="scrabble",
    ),
    WordSource(
        "eff-large-wordlist.txt",
        "https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt",
        "EFF — Diceware large wordlist (7,776 words)",
        parse="eff",
    ),
]

FREQ_TABLE_NAME = "english.freq"
ENSEMBLE_MODEL_NAME = "ensemble.pkl"
METADATA_NAME = "metadata.json"

# Training defaults (safe pipeline: english words + synthetic random only)
TRAIN_SAMPLES_PER_CLASS = 50_000
FREQ_MIN_WORD_LENGTH = 2
FREQ_MAX_WORDS = 0  # 0 = use all eligible words
NATURAL_REAL_WORD_RATIO = 0.7
BOOTSTRAP_VERSION = 18
CPU_FRACTION = 0.5

# Inference (API) defaults — use maximum allocated CPU unless overridden
INFERENCE_CPU_FRACTION = 1.0

# Parallel execution: process | thread | hybrid (both pools active)
PARALLEL_BACKEND = "hybrid"

# Exclusion / score-cache fast path (no inference CPU)
EXCLUDE_ENABLED = True
EXCLUDE_DB_NAME = "exclude.db"
SKIP_CACHE_ENABLED = True
SKIP_SCORE_THRESHOLD = 30

# Score thresholds (1-100 scale, higher = more random)
LABEL_LIKELY_RANDOM = 60
LABEL_NATURAL = 30
CONFIDENCE_STD_THRESHOLD = 0.15
