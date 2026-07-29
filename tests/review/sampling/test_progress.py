from __future__ import annotations

from datetime import datetime, timezone

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.review.sampling.coordinator import AdaptiveReviewCoordinator
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.log import InMemorySamplingLogSink, SamplingDecision
from archivetrust.review.sampling.progress import compute_calibration_progress_map
from archivetrust.review.service import ReviewAction, ReviewService
from archivetrust.learning.analytics.index import TelemetryIndex
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


def test_operational_corrections_never_count_toward_calibration_progress():
    """Core design decision: operational review is a biased, deterministic subsample and must
    never be counted as calibration evidence, even though it produces real
    `HumanCorrectionSubmitted` events."""
    sink = _sink()
    service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )
    packets = service.open_document(document_ref="doc1", archive_object_ref="obj1")
    for packet in packets:
        service.submit_decision(packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    index = TelemetryIndex.build(sink)
    log = InMemorySamplingLogSink()  # empty: no calibration sampling ever happened
    progress = compute_calibration_progress_map(index, log)

    assert all(p.reviews_completed == 0 for p in progress.values())


def test_calibration_correction_is_counted_only_once_logged_and_correlated():
    sink = _sink()
    service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )
    log = InMemorySamplingLogSink()
    coordinator = AdaptiveReviewCoordinator(review_service=service, telemetry_source=sink, sampling_log=log)

    queue = coordinator.build_queue(intent=ReviewIntent.CALIBRATION, strategy_name="random_sampling", k=1, seed=1)
    entry = next(e for e in queue if e.intent == ReviewIntent.CALIBRATION)
    packet = coordinator.open_packet(entry, archive_object_ref="obj1")

    index_before = TelemetryIndex.build(sink)
    progress_before = compute_calibration_progress_map(index_before, log)
    assert progress_before["uncorroborated_single_source/heading"].reviews_completed == 0

    coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.REJECT)

    index_after = TelemetryIndex.build(sink)
    progress_after = compute_calibration_progress_map(index_after, log)
    pattern_progress = progress_after["uncorroborated_single_source/heading"]
    assert pattern_progress.reviews_completed == 1
    assert pattern_progress.corrections_recorded == 1  # REJECT is not "accept" -> a correction


def test_decisions_without_correction_id_are_ignored():
    log = InMemorySamplingLogSink()
    log.append(
        SamplingDecision(
            semantic_slot_id="slot_1",
            document_ref="doc1",
            intent=ReviewIntent.CALIBRATION,
            strategy_name="random_sampling",
            pattern="uncorroborated_single_source/heading",
            explanation="test",
            sampled_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    sink = _sink()
    index = TelemetryIndex.build(sink)
    progress = compute_calibration_progress_map(index, log)
    assert progress["uncorroborated_single_source/heading"].reviews_completed == 0
