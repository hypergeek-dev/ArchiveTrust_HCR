from __future__ import annotations

from archivetrust.application.current_state import CurrentStateService
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.current_state import project_current_document
from archivetrust.domain.feedback.engine import apply_human_correction
from archivetrust.domain.feedback.models import (
    CorrectionAction,
    CorrectionCategory,
    HumanCorrection,
)
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.feedback.telemetry import human_correction_events
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import CanonicalDocumentCreated
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink

from tests.review._helpers import emit_document_snapshot, emit_slot, heading


def _correct_without_reassembly(sink: InMemoryTelemetrySink):
    original = emit_slot(
        sink,
        document_ref="doc_1",
        canonical_payload=heading("Chapter I"),
        provider_payloads=(("source_a", heading("Chapter I")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    emit_document_snapshot(sink, document_ref="doc_1")
    correction = HumanCorrection(
        correction_id="correction_1",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="Chapter I",
        raw_corrected_output="Chapter 1",
    )
    observations = tuple(
        event.observation
        for event in sink.events_for_document("doc_1")
        if event.kind.value == "ObservationCreated"
    )
    resulting = apply_human_correction(
        original,
        correction,
        observations,
        FeedbackPolicy(feedback_policy_version=7),
    )
    for event in human_correction_events(
        correction=correction,
        original=original,
        resulting=resulting,
        document_ref="doc_1",
        policy=FeedbackPolicy(feedback_policy_version=7),
    ):
        sink.append(event)
    return original, resulting


def test_projection_marks_corrected_truth_stale_until_document_advances() -> None:
    sink = InMemoryTelemetrySink()
    original, resulting = _correct_without_reassembly(sink)

    stale = project_current_document(sink.events_for_document("doc_1"))

    assert stale.slot(original.semantic_slot_id).current == resulting
    assert stale.slot(original.semantic_slot_id).corrected is True
    assert stale.slot(original.semantic_slot_id).history == (original, resulting)
    assert stale.effective_contained_observations == (resulting.canonical_observation_id,)
    assert stale.requires_reassembly is True
    assert stale.export_eligible is False
    assert stale.policy_versions.feedback_version == 7

    previous = stale.latest_canonical_document
    advanced = previous.model_copy(
        update={
            "document_snapshot_id": new_id("document_snapshot"),
            "contained_observations": stale.effective_contained_observations,
            "document_version": previous.document_version + 1,
            "reassembly_trigger": "human_correction:correction_1",
            "supersedes": previous.document_snapshot_id,
        }
    )
    sink.append(
        CanonicalDocumentCreated(
            event_id=new_id("event"),
            document_ref="doc_1",
            canonical_document=advanced,
        )
    )

    current = project_current_document(sink.events_for_document("doc_1"))
    assert current.latest_canonical_document == advanced
    assert current.requires_reassembly is False
    assert current.export_eligible is True
    assert current.canonical_document_history == (previous, advanced)


def test_current_state_service_invalidates_cache_after_append() -> None:
    sink = InMemoryTelemetrySink()
    original = emit_slot(
        sink,
        document_ref="doc_1",
        canonical_payload=heading("Original"),
        provider_payloads=(("source_a", heading("Original")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    emit_document_snapshot(sink, document_ref="doc_1")
    service = CurrentStateService(sink)
    before = service.document("doc_1")

    _, resulting = _correct_without_reassembly_for_existing(sink, original)
    after = service.document("doc_1")

    assert before.slot(original.semantic_slot_id).current == original
    assert after.slot(original.semantic_slot_id).current == resulting
    assert after is not before


def _correct_without_reassembly_for_existing(sink, original):
    correction = HumanCorrection(
        correction_id="correction_existing",
        target_canonical_observation_id=original.canonical_observation_id,
        category=CorrectionCategory.TRANSCRIPTION_ERROR,
        action=CorrectionAction.EDIT,
        raw_ai_output="Original",
        raw_corrected_output="Corrected",
    )
    observations = tuple(
        event.observation
        for event in sink.events_for_document("doc_1")
        if event.kind.value == "ObservationCreated"
    )
    resulting = apply_human_correction(
        original,
        correction,
        observations,
        FeedbackPolicy(feedback_policy_version=1),
    )
    for event in human_correction_events(
        correction=correction,
        original=original,
        resulting=resulting,
        document_ref="doc_1",
        policy=FeedbackPolicy(feedback_policy_version=1),
    ):
        sink.append(event)
    return original, resulting
