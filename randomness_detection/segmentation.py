"""Dictionary word segmentation (shared by PMI and lexical features)."""

from __future__ import annotations

LEXICON_MIN_WORD = 3
LEXICON_MAX_WORD = 18


def segment_token(token: str, lexicon: frozenset[str]) -> list[str]:
    """
    Minimal dictionary segmentation for a lowercase alpha token.

    Returns the word list achieving maximal coverage with fewest words
    (same DP objective as features._segment_coverage).
    """
    n = len(token)
    if n < LEXICON_MIN_WORD or not lexicon:
        return []

    cov = [0] * (n + 1)
    words = [0] * (n + 1)
    back: list[list[str] | None] = [None] * (n + 1)
    back[0] = []

    for end in range(1, n + 1):
        cov[end] = cov[end - 1]
        words[end] = words[end - 1]
        back[end] = back[end - 1]
        lo = max(0, end - LEXICON_MAX_WORD)
        for start in range(lo, end - LEXICON_MIN_WORD + 1):
            segment = token[start:end]
            if segment not in lexicon:
                continue
            seg_len = end - start
            candidate_cov = cov[start] + seg_len
            candidate_words = words[start] + 1
            if candidate_cov > cov[end] or (
                candidate_cov == cov[end] and candidate_words < words[end]
            ):
                cov[end] = candidate_cov
                words[end] = candidate_words
                back[end] = (back[start] or []) + [segment]

    return back[n] or []


def segment_alpha_runs(text: str, lexicon: frozenset[str]) -> list[str]:
    """Segment all alphabetic runs in *text* and return concatenated word list."""
    import re

    words: list[str] = []
    for run in re.findall(r"[a-z]+", text.lower()):
        words.extend(segment_token(run, lexicon))
    return words
