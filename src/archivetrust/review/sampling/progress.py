"""Correlates the sampling log against submitted corrections to compute per-pattern calibration
progress (Milestone 11, Phases 4/5) -- the one place that implements the "calibration evidence
must come from unbiased sampling" design decision: only `HumanCorrectionSubmitted` events whose
`correction_id` is attached to a `ReviewIntent.CALIBRATION` `SamplingDecision` count.

Lives in `review/sampling/` rather than `domain/calibration/` because it depends on
`SamplingDecision` (a review-layer concept, tracking *why* a slot was selected) -- `domain/
calibration/` stays scoped to statistics that only need measured population and review outcomes.
"""

from __future__ import annotations

from collections.abc import Callable

from archivetrust.domain.calibration.pattern import CalibrationPattern
from archivetrust.domain.calibration.progress import CalibrationProgress, compute_calibration_progress
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.review.sampling.discovery import by_classification_and_type
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.log import SamplingLogSink


def compute_calibration_progress_map(
    index: TelemetryIndex,
    sampling_log: SamplingLogSink,
    *,
    pattern_source: Callable[[TelemetryIndex], tuple[CalibrationPattern, ...]] = by_classification_and_type,
) -> dict[str, CalibrationProgress]:
    """One `CalibrationProgress` per pattern discovered by `pattern_source`, counting only
    calibration-intent reviews whose resulting correction has been correlated back to the
    `SamplingDecision` that produced it.
    """
    patterns = pattern_source(index)
    correction_by_id = {c.correction_id: c for c in index.corrections_submitted}

    reviews_completed: dict[str, int] = {}
    corrections_recorded: dict[str, int] = {}
    for decision in sampling_log.all_decisions():
        if decision.intent != ReviewIntent.CALIBRATION or decision.correction_id is None:
            continue
        correction = correction_by_id.get(decision.correction_id)
        if correction is None:
            continue
        reviews_completed[decision.pattern] = reviews_completed.get(decision.pattern, 0) + 1
        if correction.action != "accept":
            corrections_recorded[decision.pattern] = corrections_recorded.get(decision.pattern, 0) + 1

    return {
        pattern.name: compute_calibration_progress(
            pattern,
            reviews_completed=reviews_completed.get(pattern.name, 0),
            corrections_recorded=corrections_recorded.get(pattern.name, 0),
        )
        for pattern in patterns
    }
