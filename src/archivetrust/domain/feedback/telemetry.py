"""Bridges Human Feedback domain objects into telemetry (ROADMAP.md S12; Constitution Article 15).

Emits `HumanCorrectionSubmitted` (the raw proposal), `HumanCorrectionApplied` (the resulting
superseding `CanonicalObservation` -- consumed by `Journal` exactly like `CanonicalDecisionCreated`,
per `application/journal.py`'s existing handling), `ConfidenceChanged` (level=CANONICAL, the
concrete mechanism Confidence Evolution updates through, per ROADMAP.md S5.10), and, when a
`DatasetCandidate` is supplied, `DatasetCandidateCreated`.
"""

from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.feedback.models import DatasetCandidate, HumanCorrection
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    ConfidenceChanged,
    ConfidenceLevel,
    DatasetCandidateCreated,
    HumanCorrectionApplied,
    HumanCorrectionSubmitted,
    TelemetryEvent,
)


def human_correction_events(
    *,
    correction: HumanCorrection,
    original: CanonicalObservation,
    resulting: CanonicalObservation,
    document_ref: str,
    policy: FeedbackPolicy,
    dataset_candidate: DatasetCandidate | None = None,
    reviewer_ref: str | None = None,
    submitted_at: str | None = None,
    review_duration_seconds: float | None = None,
) -> tuple[TelemetryEvent, ...]:
    events: list[TelemetryEvent] = [
        HumanCorrectionSubmitted(
            event_id=new_id("event"),
            document_ref=document_ref,
            correction_id=correction.correction_id,
            target_canonical_observation_id=correction.target_canonical_observation_id,
            category=correction.category.value,
            action=correction.action.value,
            raw_ai_output=correction.raw_ai_output,
            raw_corrected_output=correction.raw_corrected_output,
            rationale=correction.rationale,
            reviewer_ref=reviewer_ref,
            submitted_at=submitted_at,
            review_duration_seconds=review_duration_seconds,
        ),
        HumanCorrectionApplied(
            event_id=new_id("event"),
            document_ref=document_ref,
            correction_id=correction.correction_id,
            resulting_canonical_observation=resulting,
        ),
    ]
    if resulting.canonical_confidence is not None:
        events.append(
            ConfidenceChanged(
                event_id=new_id("event"),
                document_ref=document_ref,
                subject_id=resulting.canonical_observation_id,
                level=ConfidenceLevel.CANONICAL,
                previous_value=original.canonical_confidence.value if original.canonical_confidence else None,
                new_value=resulting.canonical_confidence.value,
                reason=resulting.canonical_confidence.derivation,
                feedback_policy_version=policy.feedback_policy_version,
            )
        )
    if dataset_candidate is not None:
        events.append(
            DatasetCandidateCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                candidate_id=dataset_candidate.candidate_id,
                correction_id=dataset_candidate.correction_id,
                description=dataset_candidate.description,
                archive_object_ref=dataset_candidate.archive_object_ref,
            )
        )
    return tuple(events)
