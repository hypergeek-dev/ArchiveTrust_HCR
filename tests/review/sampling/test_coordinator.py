from __future__ import annotations

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.review.sampling.coordinator import AdaptiveReviewCoordinator
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.log import InMemorySamplingLogSink
from archivetrust.review.service import ReviewAction, ReviewService
from archivetrust.review.triage import triage_review_queue
from archivetrust.application.journal import Journal
from tests.review._helpers import emit_document_snapshot, emit_slot, heading


def _sink() -> InMemoryTelemetrySink:
    sink = InMemoryTelemetrySink()
    for i in range(3):
        emit_slot(
            sink,
            document_ref="doc1",
            canonical_payload=heading(f"single {i}"),
            provider_payloads=(("docling", heading(f"single {i}")),),
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        )
    emit_document_snapshot(sink, document_ref="doc1", archive_object_ref="obj1")
    return sink


def _coordinator(sink: InMemoryTelemetrySink) -> tuple[AdaptiveReviewCoordinator, InMemorySamplingLogSink]:
    service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )
    log = InMemorySamplingLogSink()
    coordinator = AdaptiveReviewCoordinator(review_service=service, telemetry_source=sink, sampling_log=log)
    return coordinator, log


def test_build_queue_with_k_zero_matches_operational_triage_exactly():
    sink = _sink()
    coordinator, _log = _coordinator(sink)
    state = Journal().replay(sink.events_for_document("doc1"))
    direct = triage_review_queue(state)

    queue = coordinator.build_queue(k=0)

    assert [e.semantic_slot_id for e in queue] == [i.semantic_slot_id for i in direct]
    assert all(e.intent == ReviewIntent.OPERATIONAL for e in queue)


def test_build_queue_with_calibration_sampling_adds_tagged_items():
    sink = _sink()
    coordinator, log = _coordinator(sink)

    queue = coordinator.build_queue(intent=ReviewIntent.CALIBRATION, strategy_name="random_sampling", k=2, seed=1)

    calibration_items = [e for e in queue if e.intent == ReviewIntent.CALIBRATION]
    assert len(calibration_items) == 2
    # Sampling decisions must be logged before any review happens.
    assert len(list(log.all_decisions())) == 2


def test_submit_decision_correlates_correction_to_sampling_log_for_calibration_entries():
    sink = _sink()
    coordinator, log = _coordinator(sink)

    queue = coordinator.build_queue(intent=ReviewIntent.CALIBRATION, strategy_name="random_sampling", k=1, seed=1)
    entry = next(e for e in queue if e.intent == ReviewIntent.CALIBRATION)
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    result = coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    assert result.correction_id is not None
    decisions = [d for d in log.all_decisions() if d.semantic_slot_id == entry.semantic_slot_id]
    assert decisions and decisions[0].correction_id == result.correction_id


def test_submit_decision_for_operational_entry_does_not_touch_sampling_log():
    sink = _sink()
    coordinator, log = _coordinator(sink)

    queue = coordinator.build_queue(k=0)
    entry = queue[0]
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    assert list(log.all_decisions()) == []


def test_open_packet_for_calibration_entry_carries_honest_natural_reason():
    sink = _sink()
    coordinator, _log = _coordinator(sink)
    queue = coordinator.build_queue(intent=ReviewIntent.CALIBRATION, strategy_name="random_sampling", k=1, seed=1)
    entry = next(e for e in queue if e.intent == ReviewIntent.CALIBRATION)

    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    # All fixture slots are UNCORROBORATED_SINGLE_SOURCE -> natural reason is SINGLE_SOURCE.
    from archivetrust.review.packet import ReviewReason

    assert packet.review_reason == ReviewReason.SINGLE_SOURCE
