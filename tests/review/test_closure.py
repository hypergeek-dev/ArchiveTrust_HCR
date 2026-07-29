from __future__ import annotations

from datetime import datetime, timedelta, timezone

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.telemetry.events import ReviewPacketClosureKind
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.presentation.read_model.core import CoreAggregate
from archivetrust.presentation.read_model.projections.quality import review_decision_stats
from archivetrust.review.closure import (
    close_review_packet,
    dispatch_review_packet,
    review_closure_status,
)
from archivetrust.review.service import ReviewService
from tests.review._helpers import emit_slot, heading


DOC = "doc_1"
ARCHIVE = "archive_object_1"


def _packet(sink: InMemoryTelemetrySink):
    emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    service = ReviewService(
        telemetry_source=sink,
        telemetry_sink=sink,
        interaction_sink=InMemoryReviewInteractionSink(),
    )
    (packet,) = service.open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    return packet


def test_dispatched_packet_is_open_until_closed() -> None:
    sink = InMemoryTelemetrySink()
    packet = _packet(sink)

    dispatch = dispatch_review_packet(sink=sink, packet=packet, reviewer_ref="reviewer-a")

    status = review_closure_status(tuple(sink.all_events()), horizon=timedelta(days=7))
    assert status.dispatched_count == 1
    assert status.open_count == 1
    assert status.closed_count == 0
    assert status.open_packets[0].packet_id == dispatch.packet_id


def test_closing_a_dispatched_packet_removes_it_from_open_status() -> None:
    sink = InMemoryTelemetrySink()
    packet = _packet(sink)
    dispatch = dispatch_review_packet(sink=sink, packet=packet)

    close_review_packet(
        sink=sink,
        dispatch=dispatch,
        closure_kind=ReviewPacketClosureKind.RESOLVED,
        reason="reviewer accepted provider reading",
        correction_id="correction-1",
    )

    status = review_closure_status(tuple(sink.all_events()), horizon=timedelta(days=7))
    assert status.open_count == 0
    assert status.closed_count == 1
    assert status.closed_packets[0].closure_kind == ReviewPacketClosureKind.RESOLVED
    assert status.closed_packets[0].reason == "reviewer accepted provider reading"


def test_expired_and_withdrawn_are_terminal_closure_states() -> None:
    sink = InMemoryTelemetrySink()
    packet = _packet(sink)
    expired = dispatch_review_packet(sink=sink, packet=packet)
    withdrawn = dispatch_review_packet(sink=sink, packet=packet)

    close_review_packet(sink=sink, dispatch=expired, closure_kind=ReviewPacketClosureKind.EXPIRED)
    close_review_packet(sink=sink, dispatch=withdrawn, closure_kind=ReviewPacketClosureKind.WITHDRAWN)

    status = review_closure_status(tuple(sink.all_events()), horizon=timedelta(days=7))
    assert status.open_count == 0
    assert {packet.closure_kind for packet in status.closed_packets} == {
        ReviewPacketClosureKind.EXPIRED,
        ReviewPacketClosureKind.WITHDRAWN,
    }


def test_first_terminal_closure_wins_for_duplicate_closure_events() -> None:
    sink = InMemoryTelemetrySink()
    packet = _packet(sink)
    dispatch = dispatch_review_packet(sink=sink, packet=packet)

    close_review_packet(sink=sink, dispatch=dispatch, closure_kind=ReviewPacketClosureKind.DEFERRED)
    close_review_packet(sink=sink, dispatch=dispatch, closure_kind=ReviewPacketClosureKind.EXPIRED)

    status = review_closure_status(tuple(sink.all_events()), horizon=timedelta(days=7))
    assert status.closed_count == 1
    assert status.closed_packets[0].closure_kind == ReviewPacketClosureKind.DEFERRED


def test_oldest_open_age_and_aged_open_count_are_derived_from_dispatch_time() -> None:
    sink = InMemoryTelemetrySink()
    packet = _packet(sink)
    dispatch = dispatch_review_packet(sink=sink, packet=packet)
    old_dispatch = dispatch.model_copy(update={"recorded_at": "2026-01-01T00:00:00+00:00"})
    events = tuple(
        old_dispatch if event.event_id == dispatch.event_id else event
        for event in sink.all_events()
    )

    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    status = review_closure_status(events, horizon=timedelta(days=7), now=now)

    assert status.aged_open_count == 1
    assert status.oldest_open_age_seconds == 9 * 24 * 60 * 60


def test_review_closure_counts_surface_in_quality_projection() -> None:
    sink = InMemoryTelemetrySink()
    packet = _packet(sink)
    dispatch = dispatch_review_packet(sink=sink, packet=packet)
    close_review_packet(sink=sink, dispatch=dispatch, closure_kind=ReviewPacketClosureKind.DEFERRED)

    aggregate = CoreAggregate()
    aggregate.update(tuple(sink.all_events()))
    stats = review_decision_stats(aggregate)

    assert stats.packets_dispatched == 1
    assert stats.packets_closed == 1
    assert stats.open_dispatched_packets == 0
    assert stats.packet_closures_by_kind == (("deferred", 1),)
