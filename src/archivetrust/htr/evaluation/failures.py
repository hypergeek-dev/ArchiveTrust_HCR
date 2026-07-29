"""Reliability metrics / `FailureRecord` classification (docs/htr-migration-plan.md Stage 9).

Two distinct responsibilities, kept in one module because they share the "never silently exclude
a failure" discipline:

1. `classify_reliability` -- looks at one method run's raw/parsed/normalized transcription triple
   plus ground truth (and, when the adapter itself already reported a failure, that failure
   directly) and returns zero or more `FailureRecord`s. **Method failure passthrough**: when the
   adapter already produced a failure (SATRN's `cuda_oom`/`malformed_input`, Transkribus's
   `malformed_xml`, etc. -- see `tests/htr/evaluation/test_failures.py`, which reuses the real
   `build_failure_record` outputs from `providers/satrn/adapter.py`/`providers/transkribus/
   adapter.py`'s own test fixtures rather than inventing synthetic failures), this function passes
   that `FailureRecord` through **unchanged** -- it does not reclassify or second-guess a failure
   the method itself already reported (Constitution Article 6: Full Exposure -- the adapter's own
   category is the ground truth for *why the run failed*, this module only adds *reliability*
   findings for runs that technically succeeded but are still suspect).

2. `aggregate_method_run_metrics` -- the explicit aggregation function the brief requires:
   "must NOT silently exclude failures from aggregate metrics ... implement an explicit
   aggregation function that includes failed runs (e.g. as zero-score or clearly-flagged
   entries)". Mirrors `evaluation/evaluate.py::_summaries`' own `mean_cer_all` convention exactly
   ("Misses scored as CER 1.0 — the honest all-in number; both are reported") rather than
   inventing a new aggregation philosophy: every method's aggregate reports both a
   succeeded-only mean and an all-runs-including-failures mean side by side, plus a
   `failure_categories` breakdown that is never empty when failures exist.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.evaluation.metrics import compare_text
from archivetrust.htr.evaluation.recognition import classify_word_edits
from archivetrust.htr.experiment.models import FailureRecord

_MALFORMED_MARKERS = ("<s>", "</s>", "<pad>", "�")
"""Markers that should never survive into *parsed*/*normalized* text -- `</s><s>...</s>` is
Florence-2's genuinely raw decoder special-token wrapping (see `providers/florence2_htr/
adapter.py`'s "Raw decoder output" -- real, expected on `raw_output`), so seeing it on the parsed/
normalized stage means a parsing step that should have stripped it did not."""


class ReliabilityFlag(str, Enum):
    HALLUCINATED_TEXT = "hallucinated_text"
    OMITTED_TEXT = "omitted_text"
    REPEATED_TEXT = "repeated_text"
    UNSUPPORTED_NORMALIZATION = "unsupported_normalization"
    INVENTED_LINE_BREAKS = "invented_line_breaks"
    LOST_LINE_BREAKS = "lost_line_breaks"
    READING_ORDER_ERROR = "reading_order_error"
    OUTPUT_TRUNCATION = "output_truncation"
    EMPTY_OUTPUT = "empty_output"
    MALFORMED_OUTPUT = "malformed_output"
    CONFIDENCE_CALIBRATION_DISAGREEMENT = "confidence_calibration_disagreement"


def _best_available_text(*candidates: str | None) -> str | None:
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def classify_reliability(
    *,
    method_run_id: str,
    reference_text: str | None,
    raw_text: str | None,
    parsed_text: str | None,
    normalized_text: str | None,
    reported_confidence: float | None = None,
    adapter_failure: FailureRecord | None = None,
    confidence_disagreement_threshold: float = 0.35,
    repeated_run_threshold: int = 3,
    truncation_length_ratio: float = 0.6,
    truncation_prefix_similarity: float = 0.75,
) -> tuple[FailureRecord, ...]:
    """Returns zero or more `FailureRecord`s. Never raises on any combination of `None` inputs --
    checks that need a field they were not given simply do not fire, they do not crash the whole
    classification (a partially-populated result triple must still be classifiable)."""
    if adapter_failure is not None:
        return (adapter_failure,)  # method failure passthrough -- unchanged, not reclassified

    hypothesis = _best_available_text(normalized_text, parsed_text, raw_text)
    flags: list[tuple[ReliabilityFlag, str]] = []

    if hypothesis is None or hypothesis.strip() == "":
        flags.append((ReliabilityFlag.EMPTY_OUTPUT, "recognize() succeeded but produced no text"))
        return tuple(
            FailureRecord.create(method_run_id=method_run_id, reason=reason, category=flag.value)
            for flag, reason in flags
        )

    malformed_check_text = _best_available_text(normalized_text, parsed_text)
    if malformed_check_text is not None and any(marker in malformed_check_text for marker in _MALFORMED_MARKERS):
        flags.append(
            (
                ReliabilityFlag.MALFORMED_OUTPUT,
                f"parsed/normalized text still contains a raw decoder marker: {malformed_check_text!r}",
            )
        )

    if normalized_text is not None and parsed_text is not None:
        expected = unicodedata.normalize("NFC", parsed_text).strip()
        if normalized_text != expected:
            flags.append(
                (
                    ReliabilityFlag.UNSUPPORTED_NORMALIZATION,
                    "normalized text is not explainable by NFC + outer-whitespace-strip alone "
                    f"(the only sanctioned normalization): expected {expected!r}, got "
                    f"{normalized_text!r}",
                )
            )

    if reference_text is not None:
        ref_line_count = reference_text.count("\n") + 1
        hyp_line_count = hypothesis.count("\n") + 1
        if hyp_line_count > ref_line_count:
            flags.append(
                (
                    ReliabilityFlag.INVENTED_LINE_BREAKS,
                    f"hypothesis has {hyp_line_count} lines, reference has {ref_line_count}",
                )
            )
        elif hyp_line_count < ref_line_count:
            flags.append(
                (
                    ReliabilityFlag.LOST_LINE_BREAKS,
                    f"hypothesis has {hyp_line_count} lines, reference has {ref_line_count}",
                )
            )

        ref_words = reference_text.split()
        hyp_words = hypothesis.split()
        if ref_words:
            word_edits = classify_word_edits(ref_words, hyp_words)
            ref_count = word_edits.reference_units
            if ref_count and word_edits.insertions / ref_count > 0.5 and len(hyp_words) > len(ref_words) * 1.3:
                flags.append(
                    (
                        ReliabilityFlag.HALLUCINATED_TEXT,
                        f"{word_edits.insertions} inserted words against {ref_count} reference "
                        "words, hypothesis substantially longer than reference",
                    )
                )
            if ref_count and word_edits.deletions / ref_count > 0.5:
                flags.append(
                    (
                        ReliabilityFlag.OMITTED_TEXT,
                        f"{word_edits.deletions} deleted words out of {ref_count} reference words",
                    )
                )

            sequential_wer = compare_text(reference_text, hypothesis).word_error_rate
            bag_reference = " ".join(sorted(w.casefold() for w in ref_words))
            bag_hypothesis = " ".join(sorted(w.casefold() for w in hyp_words))
            bag_wer = compare_text(bag_reference, bag_hypothesis).word_error_rate
            if sequential_wer - bag_wer > 0.3 and bag_wer < 0.5:
                flags.append(
                    (
                        ReliabilityFlag.READING_ORDER_ERROR,
                        f"sequential WER {sequential_wer:.3f} far exceeds bag-of-words WER "
                        f"{bag_wer:.3f} -- largely the right words, likely in the wrong order",
                    )
                )

        if (
            len(hypothesis) < len(reference_text) * truncation_length_ratio
            and len(reference_text) > 0
        ):
            prefix = reference_text[: len(hypothesis)]
            similarity = compare_text(prefix, hypothesis).normalized_similarity
            if similarity >= truncation_prefix_similarity:
                flags.append(
                    (
                        ReliabilityFlag.OUTPUT_TRUNCATION,
                        f"hypothesis ({len(hypothesis)} chars) closely matches only the first "
                        f"{len(hypothesis)} chars of a {len(reference_text)}-char reference "
                        f"(prefix similarity {similarity:.3f})",
                    )
                )

        if reported_confidence is not None:
            actual = compare_text(reference_text, hypothesis).normalized_similarity
            disagreement = abs(reported_confidence - actual)
            if disagreement > confidence_disagreement_threshold:
                flags.append(
                    (
                        ReliabilityFlag.CONFIDENCE_CALIBRATION_DISAGREEMENT,
                        f"reported confidence {reported_confidence:.3f} vs. measured similarity "
                        f"{actual:.3f} against ground truth (disagreement {disagreement:.3f} > "
                        f"threshold {confidence_disagreement_threshold})",
                    )
                )

    repeated = _detect_repeated_run(hypothesis, minimum_run=repeated_run_threshold)
    if repeated is not None:
        flags.append(
            (
                ReliabilityFlag.REPEATED_TEXT,
                f"word {repeated!r} repeats {repeated_run_threshold}+ times consecutively",
            )
        )

    return tuple(
        FailureRecord.create(method_run_id=method_run_id, reason=reason, category=flag.value)
        for flag, reason in flags
    )


def _detect_repeated_run(text: str, *, minimum_run: int) -> str | None:
    """Finds the first word that repeats `minimum_run` or more times consecutively (case-
    insensitive) -- a known VLM failure mode (degenerate repetition loops). Returns the repeated
    word, or `None` if no such run exists."""
    words = [w.casefold() for w in text.split()]
    run_word = None
    run_length = 0
    for word in words:
        if word == run_word:
            run_length += 1
        else:
            run_word, run_length = word, 1
        if run_length >= minimum_run:
            return run_word
    return None


class MethodRunEvaluationEntry(BaseModel):
    """One method run's evaluable outcome -- the input row `aggregate_method_run_metrics` groups
    and aggregates. A caller builds one of these per `MethodRun` regardless of whether it
    succeeded or failed; a failed run has `character_error_rate_normalized=None` and a non-`None`
    `failure_category`, a succeeded run the reverse -- both are legal, exactly one pattern per row
    is not enforced here (aggregation reads whichever fields are present)."""

    model_config = ConfigDict(frozen=True)

    method_run_id: str
    method_id: str
    outcome: str
    """`"succeeded" | "failed" | "no_output"`, mirroring `MethodRun.outcome`."""
    character_error_rate_normalized: float | None = None
    failure_category: str | None = None


class MethodAggregateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    method_id: str
    total_runs: int
    succeeded_runs: int
    failed_runs: int
    mean_cer_normalized_succeeded_only: float | None
    mean_cer_normalized_all_including_failures: float | None
    """Failed runs scored as CER 1.0 -- mirrors `evaluation/evaluate.py::_summaries`'
    `mean_cer_all` convention exactly. `None` only when `total_runs == 0`."""
    failure_categories: dict[str, int]
    """Never silently empty when `failed_runs > 0` -- every failed run's category is counted here
    (an untyped/unknown category is counted under the literal string it carried, never dropped)."""


def aggregate_method_run_metrics(
    entries: Sequence[MethodRunEvaluationEntry],
) -> tuple[MethodAggregateMetrics, ...]:
    """The brief's required explicit-inclusion aggregation: every entry -- succeeded or failed --
    is counted in `total_runs`, and `mean_cer_normalized_all_including_failures` folds every
    failed run in as CER 1.0 rather than dropping it from the denominator. See
    `tests/htr/evaluation/test_failures.py::test_aggregate_includes_failed_runs_visibly_not_dropped`
    for the specific test proving a failure changes the reported all-in mean and appears in
    `failure_categories`, i.e. is visible, not silently excluded.
    """
    by_method: dict[str, list[MethodRunEvaluationEntry]] = {}
    for entry in entries:
        by_method.setdefault(entry.method_id, []).append(entry)

    aggregates: list[MethodAggregateMetrics] = []
    for method_id in sorted(by_method):
        rows = by_method[method_id]
        succeeded = [r for r in rows if r.character_error_rate_normalized is not None]
        failed = [r for r in rows if r.character_error_rate_normalized is None]
        cer_succeeded = [r.character_error_rate_normalized for r in succeeded]
        cer_all = cer_succeeded + [1.0] * len(failed)

        failure_categories: dict[str, int] = {}
        for row in failed:
            key = row.failure_category or "unknown"
            failure_categories[key] = failure_categories.get(key, 0) + 1

        aggregates.append(
            MethodAggregateMetrics(
                method_id=method_id,
                total_runs=len(rows),
                succeeded_runs=len(succeeded),
                failed_runs=len(failed),
                mean_cer_normalized_succeeded_only=(
                    sum(cer_succeeded) / len(cer_succeeded) if cer_succeeded else None
                ),
                mean_cer_normalized_all_including_failures=(
                    sum(cer_all) / len(cer_all) if cer_all else None
                ),
                failure_categories=failure_categories,
            )
        )
    return tuple(aggregates)
