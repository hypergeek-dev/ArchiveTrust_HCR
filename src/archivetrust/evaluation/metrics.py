"""Text-accuracy metric primitives (release WS7/WS8).

One definition per metric, versioned, deterministic, no provider-name special casing. These are
the shared primitives ground-truth evaluation uses; operational analytics (`learning/analytics`)
measure different quantities (survival, effort) and deliberately do not duplicate these.
"""

from __future__ import annotations

import unicodedata
from pydantic import BaseModel, ConfigDict

METRICS_VERSION = 1
"""Recorded in every evaluation output. Bump when any definition below changes meaning."""


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse all whitespace runs to single spaces, stripped. No case
    folding: casing is real information a transcription can get wrong."""
    collapsed = " ".join(unicodedata.normalize("NFC", text).split())
    return collapsed


def levenshtein(reference: str, hypothesis: str) -> int:
    """Plain O(len*len) edit distance — annotation fields are short; no dependency needed."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_char != hyp_char),  # substitution
                )
            )
        previous = current
    return previous[-1]


class TextComparison(BaseModel):
    """One reference/hypothesis comparison, all four text metrics at once."""

    model_config = ConfigDict(frozen=True)

    exact_match: bool
    character_error_rate: float
    """Edit distance / reference length, after `normalize_text` on both sides. Can exceed 1.0
    when the hypothesis is much longer than the reference (standard CER behavior, not clamped)."""
    word_error_rate: float
    normalized_similarity: float
    """1 - distance/max(len) in [0, 1] — a symmetric similarity for ranking candidates."""


def compare_text(reference: str, hypothesis: str) -> TextComparison:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    char_distance = levenshtein(ref, hyp)
    ref_words = ref.split(" ") if ref else []
    hyp_words = hyp.split(" ") if hyp else []
    word_distance = _word_levenshtein(ref_words, hyp_words)
    max_len = max(len(ref), len(hyp))
    return TextComparison(
        exact_match=ref == hyp,
        character_error_rate=(char_distance / len(ref)) if ref else (1.0 if hyp else 0.0),
        word_error_rate=(word_distance / len(ref_words)) if ref_words else (1.0 if hyp_words else 0.0),
        normalized_similarity=1.0 - (char_distance / max_len) if max_len else 1.0,
    )


def _word_levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word))
            )
        previous = current
    return previous[-1]
