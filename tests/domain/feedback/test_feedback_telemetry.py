from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.feedback.engine import apply_human_correction
from archivetrust.domain.feedback.models import CorrectionAction, CorrectionCategory, DatasetCandidate, HumanCorrection
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.feedback.telemetry import human_correction_events
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.telemetry.events import (
    ConfidenceChanged,
    DatasetCandidateCreated,
    HumanCorrectionApplied,
    HumanCorrectionSubmitted,
)


def _policy() -> FeedbackPolicy:
    return FeedbackPolicy(feedback_policy_version=2)


def _setup() -> tuple[Observation, CanonicalObservation, HumanCorrection]:
    ev = Evidence.create(provider="docling", provider_version="1.0", raw_output="Chaptr 1", processing_stage=ProcessingStage.OCR)
    observation = Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chaptr 1", level=1), evidence=(ev,))
    original = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1", payload=observation.payload, contributing_observations=(observation,),
        comparison_confidence=ComparisonConfidence(classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"),
        clustering_basis="test", reconciliation_basis="single-source acceptance", reconciliation_sequence=0,
    )
    correction = HumanCorrection(
        correction_id="correction-1",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="Chaptr 1",
        raw_corrected_output="Chapter 1",
        rationale="OCR misread",
    )
    return observation, original, correction


def test_human_correction_events_emits_submitted_applied_and_confidence_changed():
    observation, original, correction = _setup()
    resulting = apply_human_correction(original, correction, (observation,), _policy())

    events = human_correction_events(
        correction=correction, original=original, resulting=resulting, document_ref="doc-1", policy=_policy()
    )

    submitted = [e for e in events if isinstance(e, HumanCorrectionSubmitted)]
    applied = [e for e in events if isinstance(e, HumanCorrectionApplied)]
    confidence_changed = [e for e in events if isinstance(e, ConfidenceChanged)]
    assert len(submitted) == 1
    assert submitted[0].action == "edit"
    assert submitted[0].raw_corrected_output == "Chapter 1"
    assert len(applied) == 1
    assert applied[0].resulting_canonical_observation == resulting
    assert len(confidence_changed) == 1
    assert confidence_changed[0].previous_value is None  # original had no canonical_confidence yet
    assert confidence_changed[0].new_value == resulting.canonical_confidence.value
    assert confidence_changed[0].feedback_policy_version == 2


def test_dataset_candidate_event_emitted_only_when_provided():
    observation, original, correction = _setup()
    resulting = apply_human_correction(original, correction, (observation,), _policy())

    without_candidate = human_correction_events(
        correction=correction, original=original, resulting=resulting, document_ref="doc-1", policy=_policy()
    )
    assert not [e for e in without_candidate if isinstance(e, DatasetCandidateCreated)]

    candidate = DatasetCandidate(
        candidate_id="candidate-1", correction_id=correction.correction_id, archive_object_ref="doc-1",
        description="transcription correction",
    )
    with_candidate = human_correction_events(
        correction=correction, original=original, resulting=resulting, document_ref="doc-1", policy=_policy(),
        dataset_candidate=candidate,
    )
    dataset_events = [e for e in with_candidate if isinstance(e, DatasetCandidateCreated)]
    assert len(dataset_events) == 1
    assert dataset_events[0].candidate_id == "candidate-1"
