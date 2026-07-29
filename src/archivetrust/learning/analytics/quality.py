"""Quality Center analyzers (ROADMAP_V2.md S10).

Long-term operational intelligence over the same telemetry the Processing Center watches in real
time, but aggregated across documents and time. This module implements the observation-type
success view (S10: "which Canonical Observation payload types most often survive comparison
unedited, and which are most often corrected") and the aggregate human-effort rollup (S10: "human
effort, aggregated"), both as pure functions -- reproducible from stored telemetry (GP 7), always
carrying the underlying counts so no insight is an unexplainable black box (S10).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.feedback.models import CorrectionAction
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.learning.review.session import ReviewOutcome, ReviewSession
from archivetrust.learning.source import TelemetrySource


class ObservationTypeQuality(BaseModel):
    """How well one ObservationType fares through comparison and human review.

    A correction whose action is `ACCEPT` is a human *confirmation*, not a defect, and is counted
    as a survival, never against the type -- conflating "a human looked and agreed" with "a human
    had to fix it" would invert the very signal this metric exists to produce. Every other
    correction action (edit, reject, merge, split, flag) counts the slot as corrected.
    """

    model_config = ConfigDict(frozen=True)

    observation_type: ObservationType
    slot_count: int
    """Distinct semantic slots (latest version per slot) of this type."""
    corrected_slot_count: int
    """Slots that received at least one non-ACCEPT correction."""
    confirmed_slot_count: int
    """Slots that received at least one ACCEPT correction (human-confirmed survival)."""

    @property
    def survival_rate(self) -> float | None:
        """Fraction of slots not corrected. None when there are no slots of this type, so an
        absent population is never reported as perfect quality."""
        if self.slot_count == 0:
            return None
        return 1.0 - (self.corrected_slot_count / self.slot_count)


def observation_type_quality(source: TelemetrySource) -> tuple[ObservationTypeQuality, ...]:
    """Per-ObservationType survival/correction counts, sorted by survival rate ascending (the
    types most in need of attention first), ties broken by type name for determinism (GP 7).
    """
    index = TelemetryIndex.build(source)

    slots_by_type: dict[ObservationType, set[str]] = {}
    for canonical in index.latest_canonicals():
        slots_by_type.setdefault(canonical.observation_type, set()).add(canonical.semantic_slot_id)

    corrected_slots_by_type: dict[ObservationType, set[str]] = {}
    confirmed_slots_by_type: dict[ObservationType, set[str]] = {}
    for correction in index.corrections_submitted:
        target = index.canonical_by_id.get(correction.target_canonical_observation_id)
        if target is None:
            # Correction against a Canonical Observation with no recorded decision event: skip
            # rather than attribute it to a guessed type (GP 7 reproducibility depends on never
            # inventing the missing join).
            continue
        bucket = (
            confirmed_slots_by_type
            if correction.action == CorrectionAction.ACCEPT.value
            else corrected_slots_by_type
        )
        bucket.setdefault(target.observation_type, set()).add(target.semantic_slot_id)

    results = [
        ObservationTypeQuality(
            observation_type=obs_type,
            slot_count=len(slots),
            corrected_slot_count=len(corrected_slots_by_type.get(obs_type, set())),
            confirmed_slot_count=len(confirmed_slots_by_type.get(obs_type, set())),
        )
        for obs_type, slots in slots_by_type.items()
    ]
    results.sort(
        key=lambda q: (q.survival_rate if q.survival_rate is not None else 1.0, q.observation_type.value)
    )
    return tuple(results)


class HumanEffortSummary(BaseModel):
    """Aggregate human effort over a set of review sessions (ROADMAP_V2.md S10).

    Durations are summed only over sessions that actually recorded them -- an incomplete review
    (no duration) contributes to `session_count` but not to the duration totals, so effort is never
    understated by dividing by reviews that never finished nor overstated by treating a missing
    duration as zero.
    """

    model_config = ConfigDict(frozen=True)

    session_count: int
    accepted_count: int
    edited_count: int
    incomplete_count: int
    total_review_duration: float
    total_edit_duration: float

    @property
    def edit_rate(self) -> float | None:
        """Fraction of resolved reviews that required a manual edit -- the concrete measure behind
        Guiding Principle 9 ("editing is the exception"). None when no review resolved."""
        resolved = self.accepted_count + self.edited_count
        if resolved == 0:
            return None
        return self.edited_count / resolved


def human_effort_summary(sessions: tuple[ReviewSession, ...]) -> HumanEffortSummary:
    """Roll up review sessions into aggregate effort. Pure over its input (GP 7)."""
    accepted = sum(1 for s in sessions if s.outcome is ReviewOutcome.ACCEPTED)
    edited = sum(1 for s in sessions if s.outcome is ReviewOutcome.EDITED)
    incomplete = sum(1 for s in sessions if s.outcome is ReviewOutcome.INCOMPLETE)
    total_review = sum(s.review_duration for s in sessions if s.review_duration is not None)
    total_edit = sum(s.edit_duration for s in sessions if s.edit_duration is not None)
    return HumanEffortSummary(
        session_count=len(sessions),
        accepted_count=accepted,
        edited_count=edited,
        incomplete_count=incomplete,
        total_review_duration=total_review,
        total_edit_duration=total_edit,
    )
