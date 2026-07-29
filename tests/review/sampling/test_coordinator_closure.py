"""F4 lifecycle wiring: the coordinator is the operational boundary that dispatches and closes
review packets. These tests exercise the real runtime path (`open_packet` -> `submit_decision`)
against the durable stream, not the closure helpers in isolation (`tests/review/test_closure.py`
already covers those).
"""

from __future__ import annotations

import pytest

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.telemetry.events import (
    CanonicalDocumentCreated,
    HumanCorrectionApplied,
    ReviewOutcome,
    ReviewOutcomeRecorded,
    ReviewPacketClosed,
    ReviewPacketClosureKind,
    ReviewPacketCreated,
    ReviewPacketDispatched,
    ReviewPacketOpened,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.review.sampling.coordinator import AdaptiveReviewCoordinator
from archivetrust.review.sampling.log import InMemorySamplingLogSink
from archivetrust.review.service import ReviewAction, ReviewService
from tests.review._helpers import emit_document_snapshot, emit_slot, heading


def _sink() -> InMemoryTelemetrySink:
    sink = InMemoryTelemetrySink()
    for i in range(2):
        emit_slot(
            sink,
            document_ref="doc1",
            canonical_payload=heading(f"single {i}"),
            provider_payloads=(("docling", heading(f"single {i}")),),
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        )
    emit_document_snapshot(sink, document_ref="doc1", archive_object_ref="obj1")
    return sink


def _coordinator(sink: InMemoryTelemetrySink) -> AdaptiveReviewCoordinator:
    service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )
    return AdaptiveReviewCoordinator(
        review_service=service,
        telemetry_source=sink,
        sampling_log=InMemorySamplingLogSink(),
        telemetry_sink=sink,
        reviewer_ref="reviewer-1",
    )


def _events_of(sink: InMemoryTelemetrySink, kind: type) -> list:
    return [e for e in sink.all_events() if isinstance(e, kind)]


def test_open_packet_dispatches_once_and_is_idempotent():
    sink = _sink()
    coordinator = _coordinator(sink)
    entry = coordinator.build_queue(k=0)[0]

    coordinator.open_packet(entry, archive_object_ref="obj1")
    coordinator.open_packet(entry, archive_object_ref="obj1")  # re-open, same session

    dispatches = _events_of(sink, ReviewPacketDispatched)
    assert len(dispatches) == 1
    assert dispatches[0].reviewer_ref == "reviewer-1"
    assert dispatches[0].review_intent == entry.intent.value
    assert len(_events_of(sink, ReviewPacketCreated)) == 1
    assert len(_events_of(sink, ReviewPacketOpened)) == 1
    created = _events_of(sink, ReviewPacketCreated)[0]
    opened = _events_of(sink, ReviewPacketOpened)[0]
    assert created.packet_id == dispatches[0].packet_id == opened.packet_id
    assert opened.reviewer_ref == "reviewer-1"


def test_decision_closes_packet_with_outcome_reference():
    sink = _sink()
    coordinator = _coordinator(sink)
    entry = coordinator.build_queue(k=0)[0]
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    result = coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    closures = _events_of(sink, ReviewPacketClosed)
    assert len(closures) == 1
    closure = closures[0]
    assert closure.closure_kind is ReviewPacketClosureKind.ACCEPTED
    assert closure.correction_id == result.correction_id
    assert closure.outcome_id is not None
    assert closure.packet_id == _events_of(sink, ReviewPacketDispatched)[0].packet_id
    assert coordinator.open_dispatch_for(entry) is None
    assert coordinator.closure_for(entry) is not None
    outcome = next(
        event
        for event in _events_of(sink, ReviewOutcomeRecorded)
        if event.outcome_id == closure.outcome_id
    )
    assert outcome.outcome is ReviewOutcome.ACCEPTED
    assert outcome.reviewer_ref == "reviewer-1"
    assert outcome.review_duration_seconds is not None

    events = list(sink.all_events())
    indices = [
        next(index for index, event in enumerate(events) if isinstance(event, event_type))
        for event_type in (
            ReviewPacketCreated,
            ReviewPacketDispatched,
            ReviewPacketOpened,
            HumanCorrectionApplied,
        )
    ]
    indices.append(
        next(
            index
            for index, event in enumerate(events)
            if index > indices[-1] and isinstance(event, CanonicalDocumentCreated)
        )
    )
    indices.append(events.index(outcome))
    indices.append(events.index(closure))
    assert indices == sorted(indices)


def test_skip_closes_as_deferred_without_correction():
    sink = _sink()
    coordinator = _coordinator(sink)
    entry = coordinator.build_queue(k=0)[0]
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.SKIP)

    (closure,) = _events_of(sink, ReviewPacketClosed)
    assert closure.closure_kind is ReviewPacketClosureKind.DEFERRED
    assert closure.correction_id is None
    assert closure.outcome_id is not None


def test_repeat_submission_does_not_duplicate_closure():
    sink = _sink()
    coordinator = _coordinator(sink)
    entry = coordinator.build_queue(k=0)[0]
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER)
    # A second submission against the same (now closed) packet is rejected and cannot emit a
    # duplicate correction, document snapshot, outcome, or closure.
    with pytest.raises(ValueError, match="Stale review packet"):
        coordinator.submit_decision(
            entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER
        )

    assert len(_events_of(sink, ReviewPacketClosed)) == 1


def test_failed_decision_records_terminal_failure_without_canonical_change():
    sink = _sink()
    coordinator = _coordinator(sink)
    entry = coordinator.build_queue(k=0)[0]
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    with pytest.raises(ValueError):
        # MANUAL_EDIT without corrected_output fails inside ReviewService before any write.
        coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.MANUAL_EDIT)

    (closure,) = _events_of(sink, ReviewPacketClosed)
    assert closure.closure_kind is ReviewPacketClosureKind.FAILED_DURING_APPLICATION
    assert closure.outcome_id is not None
    assert closure.correction_id is None
    assert coordinator.open_dispatch_for(entry) is None
    outcome = next(
        event
        for event in _events_of(sink, ReviewOutcomeRecorded)
        if event.outcome_id == closure.outcome_id
    )
    assert outcome.outcome is ReviewOutcome.FAILED_DURING_APPLICATION


def test_deferred_slot_reopened_later_gets_a_new_packet():
    sink = _sink()
    coordinator = _coordinator(sink)
    entry = coordinator.build_queue(k=0)[0]
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")
    coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.SKIP)

    coordinator.open_packet(entry, archive_object_ref="obj1")

    dispatches = _events_of(sink, ReviewPacketDispatched)
    assert len(dispatches) == 2
    assert dispatches[0].packet_id != dispatches[1].packet_id


def test_legacy_history_without_dispatch_is_untouched_and_closure_skipped():
    """A decision path with no sink wired (pre-F4 history) records outcomes but no lifecycle
    events; a later closure-aware coordinator must read that history as-is and never fabricate
    dispatches or closures for it.
    """
    sink = _sink()
    legacy_service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )
    legacy = AdaptiveReviewCoordinator(
        review_service=legacy_service, telemetry_source=sink, sampling_log=InMemorySamplingLogSink()
    )
    entry = legacy.build_queue(k=0)[0]
    packet = legacy.open_packet(entry, archive_object_ref="obj1")
    legacy.submit_decision(entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    assert _events_of(sink, ReviewPacketDispatched) == []
    assert _events_of(sink, ReviewPacketClosed) == []

    coordinator = _coordinator(sink)
    remaining = coordinator.build_queue(k=0)
    # The decided slot left the queue (existing triage behavior); closure-aware reads see the
    # legacy decision as history with no packet lifecycle, not as an open or closed packet.
    assert all(e.semantic_slot_id != entry.semantic_slot_id for e in remaining)
    assert coordinator.closure_for(entry) is None
