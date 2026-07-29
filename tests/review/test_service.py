from __future__ import annotations

import pytest

from archivetrust.application.journal import Journal
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.feedback.models import CorrectionAction, CorrectionCategory
from archivetrust.domain.telemetry.events import (
    CanonicalDocumentCreated,
    HumanCorrectionApplied,
    HumanCorrectionSubmitted,
    ReviewOutcome,
    ReviewOutcomeRecorded,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.learning.review.interaction import ReviewInteraction, ReviewInteractionKind
from archivetrust.review.packet import DisclosureTier, ReviewReason
from archivetrust.review.service import ReviewAction, ReviewService

from tests.review._helpers import emit_document_snapshot, emit_slot, heading

DOC = "doc_1"
ARCHIVE = "archive_object_1"


def _service(sink: InMemoryTelemetrySink) -> ReviewService:
    if not any(isinstance(event, CanonicalDocumentCreated) for event in sink.all_events()):
        emit_document_snapshot(sink, document_ref=DOC, archive_object_ref=ARCHIVE)
    return ReviewService(
        telemetry_source=sink,
        telemetry_sink=sink,
        interaction_sink=InMemoryReviewInteractionSink(),
    )


def test_open_document_queues_only_uncertain_slots_in_order() -> None:
    sink = InMemoryTelemetrySink()
    # A corroborated, high-confidence slot: must NOT be queued.
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Certain"),
        provider_payloads=(("docling", heading("Certain")), ("qwen", heading("Certain"))),
        classification=ComparisonClassification.CORROBORATED,
        canonical_confidence=0.95,
    )
    # A contested slot: queued first (most urgent).
    contested = emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter I"))),
        classification=ComparisonClassification.CONTESTED,
    )
    # A single-source slot: queued last.
    single = emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Only me"),
        provider_payloads=(("qwen", heading("Only me")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )

    packets = _service(sink).open_document(document_ref=DOC, archive_object_ref=ARCHIVE)

    assert len(packets) == 2  # the corroborated/high-confidence slot is excluded
    assert packets[0].canonical_observation_id == contested.canonical_observation_id
    assert packets[0].review_reason is ReviewReason.SOURCES_DISAGREE
    assert packets[0].disclosure_tier is DisclosureTier.TIER_2_MODERATE
    assert packets[1].canonical_observation_id == single.canonical_observation_id
    assert packets[1].review_reason is ReviewReason.SINGLE_SOURCE


def test_packet_carries_candidates_evidence_geometry_and_agreement() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter I"))),
        classification=ComparisonClassification.CONTESTED,
    )
    (packet,) = _service(sink).open_document(document_ref=DOC, archive_object_ref=ARCHIVE)

    assert packet.current_value == "Chapter 1"
    assert {c.value for c in packet.candidates} == {"Chapter 1", "Chapter I"}
    assert packet.agreement.classification is ComparisonClassification.CONTESTED
    assert packet.agreement.independent_source_count == 2
    assert packet.primary_bounding_box is not None  # drives auto-navigation (UX-INV-1)
    assert packet.archive_object_ref == ARCHIVE


def test_accept_produces_superseding_observation_and_replayable_telemetry() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter I"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)

    result = service.submit_decision(packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    assert result.resulting_canonical_observation is not None
    resulting = result.resulting_canonical_observation
    # Supersession, never erasure (Constitution Article 15).
    assert resulting.supersedes == packet.canonical_observation_id
    assert resulting.reconciliation_sequence == 1
    assert resulting.human_correction_ref == result.correction_id
    # The decision is replayable from stored telemetry alone (HR-4).
    applied = [e for e in sink.all_events() if isinstance(e, HumanCorrectionApplied)]
    assert len(applied) == 1
    assert applied[0].resulting_canonical_observation.canonical_observation_id == (
        resulting.canonical_observation_id
    )
    assert result.resulting_canonical_document is not None
    document_events = [
        e for e in result.emitted_events if isinstance(e, CanonicalDocumentCreated)
    ]
    assert len(document_events) == 1
    document = document_events[0].canonical_document
    assert document == result.resulting_canonical_document
    assert document.contained_observations == (resulting.canonical_observation_id,)
    assert document.supersedes is not None
    assert document.reassembly_trigger == f"human_correction:{result.correction_id}"
    applied_index = result.emitted_events.index(applied[0])
    document_index = result.emitted_events.index(document_events[0])
    outcome_index = next(
        index
        for index, event in enumerate(result.emitted_events)
        if isinstance(event, ReviewOutcomeRecorded)
    )
    assert applied_index < document_index < outcome_index


def test_rejects_stale_packet_after_correction() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter I"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    service.submit_decision(packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    with pytest.raises(ValueError, match="Stale review packet"):
        service.submit_decision(packet=packet, action=ReviewAction.ACCEPT_PROVIDER)


def test_manual_edit_changes_value_and_replays() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter I"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)

    result = service.submit_decision(
        packet=packet, action=ReviewAction.MANUAL_EDIT, corrected_output="Chapter One"
    )

    assert result.resulting_canonical_observation.payload.text == "Chapter One"
    # Replay from telemetry reconstructs the corrected value as the slot's latest version.
    replayed = Journal().replay(sink.events_for_document(DOC))
    latest = replayed.canonical_observation_history(packet.semantic_slot_id)[-1]
    assert latest.payload.text == "Chapter One"
    assert latest.human_correction_ref == result.correction_id


def test_manual_edit_requires_corrected_output() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    with pytest.raises(ValueError, match="corrected_output"):
        service.submit_decision(packet=packet, action=ReviewAction.MANUAL_EDIT)


def test_skip_produces_no_correction() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    before = len(tuple(sink.all_events()))

    result = service.submit_decision(packet=packet, action=ReviewAction.SKIP)

    assert result.resulting_canonical_observation is None
    assert result.correction_id is None
    # Nothing canonical is emitted (§10.7) -- but SKIP is a human decision, not silence
    # (Constitution Article 30): exactly one ReviewOutcomeRecorded(outcome=DEFERRED) is appended.
    assert len(tuple(sink.all_events())) == before + 1
    outcome_event = next(e for e in result.emitted_events if isinstance(e, ReviewOutcomeRecorded))
    assert outcome_event.outcome == ReviewOutcome.DEFERRED
    assert outcome_event.correction_id is None
    assert outcome_event.semantic_slot_id == packet.semantic_slot_id


def test_accept_maps_to_accept_correction_action() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    result = service.submit_decision(packet=packet, action=ReviewAction.ACCEPT_PROVIDER)
    assert result.resulting_canonical_observation.reconciliation_basis.startswith(
        f"human correction applied: {CorrectionAction.ACCEPT.value}"
    )


def test_different_things_records_first_class_category_and_action() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = _service(sink)
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)

    result = service.submit_decision(
        packet=packet,
        action=ReviewAction.DIFFERENT_THINGS,
        rationale="left margin note vs section heading",
    )

    submitted = [e for e in result.emitted_events if isinstance(e, HumanCorrectionSubmitted)]
    assert len(submitted) == 1
    assert submitted[0].action == CorrectionAction.DIFFERENT_THINGS.value
    assert submitted[0].category == CorrectionCategory.DIFFERENT_THINGS.value
    assert submitted[0].rationale == "left margin note vs section heading"
    outcomes = [e for e in result.emitted_events if isinstance(e, ReviewOutcomeRecorded)]
    assert outcomes[0].action == ReviewAction.DIFFERENT_THINGS.value
    assert outcomes[0].outcome == ReviewOutcome.DIFFERENT_THINGS


def test_passive_interaction_recorded_on_its_own_stream() -> None:
    sink = InMemoryTelemetrySink()
    interactions = InMemoryReviewInteractionSink()
    service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=interactions
    )
    service.record_interaction(
        ReviewInteraction(
            interaction_id="i1",
            review_id="r1",
            target_canonical_observation_id="c1",
            reviewer_ref="reviewer_a",
            kind=ReviewInteractionKind.REVIEW_OPENED,
            timestamp=0.0,
        )
    )
    # Learning Platform stream, never the Trust Engine's frozen event set (LP-3).
    assert len(tuple(interactions.all_interactions())) == 1
    assert len(tuple(sink.all_events())) == 0
