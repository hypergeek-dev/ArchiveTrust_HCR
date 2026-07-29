"""ROADMAP.md Milestone 6 acceptance criterion: "Feedback submission emits correct telemetry
(HumanCorrectionApplied, DatasetCandidateCreated) and updates Confidence Evolution, queryable via
replay."
"""

from __future__ import annotations

from archivetrust.application.journal import Journal
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.feedback.engine import apply_human_correction
from archivetrust.domain.feedback.models import CorrectionAction, CorrectionCategory, DatasetCandidate, HumanCorrection
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.feedback.telemetry import human_correction_events
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.telemetry.events import CanonicalDecisionCreated
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink

DOCUMENT_REF = "archive-object-feedback-1"


def test_human_correction_supersession_and_confidence_evolution_survive_replay():
    sink = InMemoryTelemetrySink()

    ev = Evidence.create(provider="docling", provider_version="1.0", raw_output="Chaptr 1", processing_stage=ProcessingStage.OCR)
    observation = Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chaptr 1", level=1), evidence=(ev,))
    original = CanonicalObservation.reconcile(
        semantic_slot_id="slot-1", payload=observation.payload, contributing_observations=(observation,),
        comparison_confidence=ComparisonConfidence(classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"),
        clustering_basis="test", reconciliation_basis="single-source acceptance", reconciliation_sequence=0,
    )
    sink.append(CanonicalDecisionCreated(event_id="event-1", document_ref=DOCUMENT_REF, canonical_observation=original))

    correction = HumanCorrection(
        correction_id="correction-1",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="Chaptr 1",
        raw_corrected_output="Chapter 1",
        rationale="OCR misread",
    )
    policy = FeedbackPolicy(feedback_policy_version=1)
    resulting = apply_human_correction(original, correction, (observation,), policy)
    dataset_candidate = DatasetCandidate(
        candidate_id="candidate-1", correction_id=correction.correction_id, archive_object_ref=DOCUMENT_REF,
        description="transcription correction",
    )
    for event in human_correction_events(
        correction=correction, original=original, resulting=resulting, document_ref=DOCUMENT_REF, policy=policy,
        dataset_candidate=dataset_candidate,
    ):
        sink.append(event)

    state = Journal().replay(sink.events_for_document(DOCUMENT_REF))

    history = state.canonical_observation_history("slot-1")
    assert history == (original, resulting)

    evolution = state.confidence_evolution(resulting.canonical_observation_id)
    assert len(evolution) == 1
    assert evolution[0].new_value == resulting.canonical_confidence.value
    assert evolution[0].feedback_policy_version == 1

    latest = state.canonical_observation_as_of("slot-1", resulting.reconciliation_sequence)
    assert latest == resulting
    assert latest.payload.text == "Chapter 1"
