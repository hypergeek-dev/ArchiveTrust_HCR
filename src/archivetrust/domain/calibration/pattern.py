"""Calibration patterns: the grain confidence calibration is measured at (Milestone 11, Phase 4).

`CalibrationPattern` uses the same grain `review/triage.py::review_reason_for` already routes
on -- `(ComparisonClassification, ObservationType)`. Only the pure shape lives here (domain layer:
zero dependency on `learning`/`review`, per `tests/domain/test_dependency_direction.py` and
`tests/review/test_separation.py`'s enforced boundary); discovering patterns from a live
`TelemetryIndex` (which does need `review.triage.review_reason_for` to know what's queued) lives in
`review/sampling/discovery.py` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from archivetrust.domain.canonical.observation import CanonicalObservation


@dataclass(frozen=True)
class CalibrationPattern:
    """One identifiable, independently-calibratable slice of the corpus."""

    name: str
    population: int
    queued_for_review: int


def classification_type_key(canonical: CanonicalObservation) -> str:
    """The primary pattern grain as a standalone key function -- reused by
    `review/sampling/strategies.py` and `review/sampling/discovery.py` so a sampled item's pattern
    name always matches the one its `CalibrationProgress` was computed under."""
    return f"{canonical.comparison_confidence.classification.value}/{canonical.observation_type.value}"
