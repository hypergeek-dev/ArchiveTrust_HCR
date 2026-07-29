"""Recognition metrics (docs/htr-migration-plan.md Stage 9's "recognition metrics").

Extends, never reimplements, `evaluation/metrics.py`'s retained primitives: `levenshtein` and
`normalize_text` are imported directly, and `compare_text` is what "normalized" CER/WER below
actually calls. This module adds three things `evaluation/metrics.py` does not have:

1. A **raw** (non-normalized) CER/WER, kept as a genuinely separate computation from the
   **normalized** one `compare_text` already provides -- "raw" and "normalized" are two different,
   both-real numbers, never one overwriting the other (see `RecognitionMetrics` below, which
   carries both simultaneously).
2. An edit-operation-*classifying* Levenshtein variant (`classify_char_edits`/
   `classify_word_edits`) -- `evaluation.metrics.levenshtein` only returns a distance (a count),
   it cannot say whether a given edit was an insertion, deletion, or substitution. This is
   implemented here as a standard DP-table backtrace, and cross-checked in tests against
   `evaluation.metrics.levenshtein` (`counts.total_edits == levenshtein(reference, hypothesis)`
   is an invariant, not a coincidence) so it is provably consistent with the retained primitive,
   not a divergent reimplementation.
3. Line/word-level accuracy metrics (`exact_line_accuracy`, `exact_word_accuracy`) built on top of
   (1) and (2).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from archivetrust.evaluation.metrics import compare_text, levenshtein, normalize_text
from archivetrust.htr.evaluation import definitions
from archivetrust.htr.experiment.models import MetricResult


class EditOpCounts(BaseModel):
    """The classification `evaluation.metrics.levenshtein` cannot produce: how many of the edits
    between `reference` and `hypothesis` were matches, substitutions, insertions, or deletions.
    `matches + substitutions + deletions == len(reference-in-whatever-unit)` and
    `matches + substitutions + insertions == len(hypothesis-in-whatever-unit)` always hold -- both
    are asserted in tests, not just claimed here.
    """

    model_config = ConfigDict(frozen=True)

    matches: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def total_edits(self) -> int:
        """Equal to `evaluation.metrics.levenshtein(reference, hypothesis)` -- the cross-check
        invariant with the retained primitive (verified in tests, not just asserted here)."""
        return self.substitutions + self.insertions + self.deletions

    @property
    def reference_units(self) -> int:
        return self.matches + self.substitutions + self.deletions


def _classify_edits(reference: Sequence, hypothesis: Sequence) -> EditOpCounts:
    """Standard edit-distance DP table + backtrace, generic over any sequence of hashable/
    comparable units (characters or word tokens -- `classify_char_edits`/`classify_word_edits`
    below are thin wrappers specializing the unit). O(len(reference) * len(hypothesis)); fine for
    line-level and even page-level HTR text, matching `evaluation.metrics.levenshtein`'s own
    documented complexity trade-off ("annotation fields are short; no dependency needed").
    """
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return EditOpCounts(matches=0, substitutions=0, insertions=m, deletions=0)
    if m == 0:
        return EditOpCounts(matches=0, substitutions=0, insertions=0, deletions=n)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ref_unit = reference[i - 1]
        for j in range(1, m + 1):
            if ref_unit == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    matches = substitutions = insertions = deletions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            matches += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return EditOpCounts(matches=matches, substitutions=substitutions, insertions=insertions, deletions=deletions)


def classify_char_edits(reference: str, hypothesis: str) -> EditOpCounts:
    """Character-level edit-operation classification, on the exact strings passed in -- callers
    decide whether to pass raw or `normalize_text`-normalized strings; this function does not
    normalize on its own."""
    return _classify_edits(reference, hypothesis)


def classify_word_edits(reference_words: Sequence[str], hypothesis_words: Sequence[str]) -> EditOpCounts:
    """Word-token-level edit-operation classification. Callers supply already-tokenized word
    sequences (e.g. `text.split()`) -- this function does not tokenize on its own."""
    return _classify_edits(list(reference_words), list(hypothesis_words))


class RecognitionMetrics(BaseModel):
    """Raw and normalized CER/WER together, as genuinely distinct values -- never one collapsing
    into the other. Plus the edit-operation breakdown (computed on the normalized text, since that
    is the more meaningful "what did the method actually get wrong" view once whitespace/Unicode-
    form noise is removed)."""

    model_config = ConfigDict(frozen=True)

    reference: str
    hypothesis: str
    character_error_rate_raw: float
    character_error_rate_normalized: float
    word_error_rate_raw: float
    word_error_rate_normalized: float
    exact_match_raw: bool
    exact_match_normalized: bool
    char_edits_normalized: EditOpCounts
    word_edits_normalized: EditOpCounts
    exact_word_accuracy: float


def _raw_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    distance = _classify_edits(ref_words, hyp_words).total_edits
    return distance / len(ref_words)


def compute_recognition_metrics(reference: str, hypothesis: str) -> RecognitionMetrics:
    """The one entry point computing every recognition-metric number for one reference/hypothesis
    pair (one line, or one page's linearized text -- this function is granularity-agnostic).
    """
    raw_distance = levenshtein(reference, hypothesis)
    character_error_rate_raw = (
        raw_distance / len(reference) if reference else (1.0 if hypothesis else 0.0)
    )
    normalized_comparison = compare_text(reference, hypothesis)

    normalized_ref = normalize_text(reference)
    normalized_hyp = normalize_text(hypothesis)
    char_edits_normalized = classify_char_edits(normalized_ref, normalized_hyp)
    word_edits_normalized = classify_word_edits(normalized_ref.split(), normalized_hyp.split())

    ref_word_count = word_edits_normalized.reference_units
    exact_word_accuracy = (
        1.0
        - (word_edits_normalized.substitutions + word_edits_normalized.deletions) / ref_word_count
        if ref_word_count
        else (0.0 if word_edits_normalized.insertions else 1.0)
    )

    return RecognitionMetrics(
        reference=reference,
        hypothesis=hypothesis,
        character_error_rate_raw=character_error_rate_raw,
        character_error_rate_normalized=normalized_comparison.character_error_rate,
        word_error_rate_raw=_raw_wer(reference, hypothesis),
        word_error_rate_normalized=normalized_comparison.word_error_rate,
        exact_match_raw=reference == hypothesis,
        exact_match_normalized=normalized_comparison.exact_match,
        char_edits_normalized=char_edits_normalized,
        word_edits_normalized=word_edits_normalized,
        exact_word_accuracy=exact_word_accuracy,
    )


def exact_line_accuracy(reference_lines: Sequence[str], hypothesis_lines: Sequence[str]) -> float:
    """Index-aligned exact-line accuracy, after `normalize_text` on each line. Assumes the caller
    has already aligned reference and hypothesis lines (e.g. by shared `TextLine.reading_order_
    index`) -- pads the shorter sequence with unmatchable sentinels so a missing/extra line always
    counts as a miss rather than silently truncating the comparison."""
    total = max(len(reference_lines), len(hypothesis_lines))
    if total == 0:
        return 1.0
    matches = 0
    for i in range(total):
        ref_line = normalize_text(reference_lines[i]) if i < len(reference_lines) else None
        hyp_line = normalize_text(hypothesis_lines[i]) if i < len(hypothesis_lines) else None
        if ref_line is not None and ref_line == hyp_line:
            matches += 1
    return matches / total


def recognition_metric_results(
    reference: str, hypothesis: str, *, method_run_id: str
) -> tuple[MetricResult, ...]:
    """Builds every recognition `MetricResult` for one (reference, hypothesis) pair -- the
    engine's public entry point for producing stored, versioned results (as opposed to
    `compute_recognition_metrics`, which returns the raw computed numbers for a caller that wants
    them directly, e.g. for further classification in `failures.py`)."""
    metrics = compute_recognition_metrics(reference, hypothesis)
    make = lambda definition, value: MetricResult.create(
        metric_definition_id=definition.metric_definition_id, method_run_id=method_run_id, value=value
    )
    return (
        make(definitions.CHARACTER_ERROR_RATE_RAW, metrics.character_error_rate_raw),
        make(definitions.CHARACTER_ERROR_RATE_NORMALIZED, metrics.character_error_rate_normalized),
        make(definitions.WORD_ERROR_RATE_RAW, metrics.word_error_rate_raw),
        make(definitions.WORD_ERROR_RATE_NORMALIZED, metrics.word_error_rate_normalized),
        make(definitions.EXACT_WORD_ACCURACY, metrics.exact_word_accuracy),
        make(definitions.CHARACTER_INSERTIONS, float(metrics.char_edits_normalized.insertions)),
        make(definitions.CHARACTER_DELETIONS, float(metrics.char_edits_normalized.deletions)),
        make(definitions.CHARACTER_SUBSTITUTIONS, float(metrics.char_edits_normalized.substitutions)),
        make(definitions.WORD_INSERTIONS, float(metrics.word_edits_normalized.insertions)),
        make(definitions.WORD_DELETIONS, float(metrics.word_edits_normalized.deletions)),
        make(definitions.WORD_SUBSTITUTIONS, float(metrics.word_edits_normalized.substitutions)),
    )
