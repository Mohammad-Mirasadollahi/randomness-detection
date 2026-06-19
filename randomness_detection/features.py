"""Feature extraction: freq, entropy, compression, and metadata."""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import asdict, dataclass

from .freq_model import FreqCounter

BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]+=*$")
ALPHA_RUN_PATTERN = re.compile(r"[a-z]+")
VOWELS = frozenset("aeiou")
LEXICON_MIN_WORD = 3
LEXICON_MAX_WORD = 18


@dataclass
class FeatureVector:
    freq_avg_natural: float
    freq_total_natural: float
    freq_randomness: float
    entropy: float
    entropy_normalized: float
    compression_ratio: float
    length: int
    length_normalized: float
    unique_char_ratio: float
    digit_ratio: float
    uppercase_ratio: float
    is_base64_like: float
    lexical_coverage: float
    longest_word_ratio: float
    vowel_ratio: float
    max_consonant_run_ratio: float
    word_count: float

    def as_list(self) -> list[float]:
        return [
            self.freq_randomness / 100.0,
            self.freq_avg_natural / 100.0,
            self.freq_total_natural / 100.0,
            self.entropy_normalized,
            self.compression_ratio,
            self.length_normalized,
            self.unique_char_ratio,
            self.digit_ratio,
            self.uppercase_ratio,
            self.is_base64_like,
            self.lexical_coverage,
            self.longest_word_ratio,
            self.vowel_ratio,
            self.max_consonant_run_ratio,
            self.word_count,
        ]

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def shannon_entropy(text: str) -> float:
    """Shannon entropy in bits per character."""
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def normalized_entropy(text: str) -> tuple[float, float]:
    """
    Return (raw entropy, normalized 0-1).

    Normalized by log2(unique_characters) — the theoretical maximum
    for the observed alphabet size.
    """
    if not text:
        return 0.0, 0.0

    raw = shannon_entropy(text)
    unique = len(set(text))
    if unique <= 1:
        return raw, 0.0

    max_entropy = math.log2(unique)
    return raw, min(raw / max_entropy, 1.0)


def compression_ratio(text: str) -> float:
    """
    Normalized compression ratio in [0, 1].

    Uses raw DEFLATE (wbits=-15) to avoid zlib header overhead on
    short strings — standard practice for Kolmogorov-complexity estimation.
  """
    if not text:
        return 1.0

    raw = text.encode("utf-8", errors="ignore")
    if len(raw) < 2:
        return 0.5

    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    ratio = len(compressed) / len(raw)
    return min(ratio, 1.0)


def _segment_coverage(token: str, lexicon: frozenset[str]) -> tuple[int, int, int]:
    """
    DP word-segmentation over a single lowercase alpha token.

    Returns (max_chars_covered_by_dictionary_words, longest_single_word_length,
    word_count). word_count is the *minimal* number of dictionary words that
    achieves the maximal coverage — so a string that is one real word counts as 1,
    while concatenated word-salad (dictionary DGA) counts as many.
    Only dictionary words with length in [LEXICON_MIN_WORD, LEXICON_MAX_WORD] count.
    """
    n = len(token)
    if n < LEXICON_MIN_WORD or not lexicon:
        return 0, 0, 0

    cov = [0] * (n + 1)
    words = [0] * (n + 1)
    longest = 0
    for end in range(1, n + 1):
        cov[end] = cov[end - 1]  # leave char (end-1) uncovered
        words[end] = words[end - 1]
        lo = max(0, end - LEXICON_MAX_WORD)
        for start in range(lo, end - LEXICON_MIN_WORD + 1):
            segment = token[start:end]
            if segment in lexicon:
                seg_len = end - start
                candidate_cov = cov[start] + seg_len
                candidate_words = words[start] + 1
                if candidate_cov > cov[end] or (
                    candidate_cov == cov[end] and candidate_words < words[end]
                ):
                    cov[end] = candidate_cov
                    words[end] = candidate_words
                if seg_len > longest:
                    longest = seg_len
    return cov[n], longest, words[n]


def lexical_metrics(text: str, lexicon: frozenset[str]) -> tuple[float, float, int]:
    """Return (coverage, longest_word_ratio, word_count) over alphabetic runs.

    Digits and separators both act as boundaries, so non-adjacent letters are
    never merged into spurious dictionary words (e.g. inside hex tokens).
    word_count is the total number of dictionary words across runs — the key
    signal that separates short legitimate compounds (1-3 words) from
    concatenated dictionary word-salad / dictionary DGA (4+ words).
    """
    runs = ALPHA_RUN_PATTERN.findall(text.lower())
    alpha_total = sum(len(run) for run in runs)
    if alpha_total == 0:
        return 0.0, 0.0, 0

    covered = 0
    longest = 0
    word_count = 0
    for run in runs:
        run_covered, run_longest, run_words = _segment_coverage(run, lexicon)
        covered += run_covered
        longest = max(longest, run_longest)
        word_count += run_words

    coverage = covered / alpha_total
    longest_ratio = longest / alpha_total
    return min(coverage, 1.0), min(longest_ratio, 1.0), word_count


def _structure_ratios(text: str) -> tuple[float, float]:
    """Return (vowel_ratio, max_consonant_run_ratio) over alphabetic characters."""
    letters = [char for char in text.lower() if char.isalpha()]
    if not letters:
        return 0.0, 0.0

    vowels = sum(char in VOWELS for char in letters)
    max_run = 0
    run = 0
    for char in letters:
        if char not in VOWELS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return vowels / len(letters), max_run / len(letters)


def extract_features(text: str, freq_counter: FreqCounter) -> FeatureVector:
    stripped = text.strip()
    length = len(stripped)

    freq_avg, freq_total = freq_counter.probability(stripped)
    freq_natural = (freq_avg + freq_total) / 2.0
    freq_randomness = max(0.0, min(100.0, 100.0 - freq_natural))

    entropy, entropy_normalized = normalized_entropy(stripped)
    ratio_normalized = compression_ratio(stripped)

    unique_char_ratio = len(set(stripped)) / length if length else 0.0
    digit_ratio = sum(char.isdigit() for char in stripped) / length if length else 0.0
    uppercase_ratio = sum(char.isupper() for char in stripped) / length if length else 0.0

    base64_like = 1.0 if length >= 8 and BASE64_PATTERN.match(stripped) else 0.0
    length_normalized = min(length / 64.0, 1.0)

    lexicon = getattr(freq_counter, "lexicon", frozenset())
    coverage, longest_word_ratio, word_count = lexical_metrics(stripped, lexicon)
    vowel_ratio, max_consonant_run_ratio = _structure_ratios(stripped)
    # Raw count, not a capped ratio: the StandardScaler in the model pipeline
    # standardizes it, and an uncapped count keeps the signal monotonic for any
    # number of concatenated words (1-word names through 20-word salad).

    return FeatureVector(
        freq_avg_natural=freq_avg,
        freq_total_natural=freq_total,
        freq_randomness=freq_randomness,
        entropy=entropy,
        entropy_normalized=entropy_normalized,
        compression_ratio=ratio_normalized,
        length=length,
        length_normalized=length_normalized,
        unique_char_ratio=unique_char_ratio,
        digit_ratio=digit_ratio,
        uppercase_ratio=uppercase_ratio,
        is_base64_like=base64_like,
        lexical_coverage=coverage,
        longest_word_ratio=longest_word_ratio,
        vowel_ratio=vowel_ratio,
        max_consonant_run_ratio=max_consonant_run_ratio,
        word_count=float(word_count),
    )


def breakdown_scores(features: FeatureVector) -> dict[str, int]:
    """Per-method randomness scores on a 1-100 scale."""
    return {
        "freq": int(round(features.freq_randomness)),
        "entropy": int(round(features.entropy_normalized * 100)),
        "compression": int(round(features.compression_ratio * 100)),
    }
