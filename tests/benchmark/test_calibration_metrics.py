from __future__ import annotations

from archivetrust.benchmark.aggregate import calibration_metrics
from archivetrust.benchmark.metrics import MetricStatus
from archivetrust.benchmark.report import calibration_section
from archivetrust.benchmark.run import BenchmarkMetadata, BenchmarkRun
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.review.sampling.coordinator import AdaptiveReviewCoordinator
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.log import InMemorySamplingLogSink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.review.service import ReviewAction, ReviewService
from tests.benchmark.test_benchmark import build_run
from tests.review._helpers import emit_document_snapshot, emit_slot, heading


def _sink() -> InMemoryTelemetrySink:
    sink = InMemoryTelemetrySink()
    for i in range(10):
        emit_slot(
            sink,
            document_ref="doc1",
            canonical_payload=heading(f"single {i}"),
            provider_payloads=(("docling", heading(f"single {i}")),),
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        )
    emit_document_snapshot(sink, document_ref="doc1", archive_object_ref="obj1")
    return sink


def test_calibration_metrics_honest_with_no_calibration_reviews():
    sink = _sink()
    log = InMemorySamplingLogSink()

    metrics = calibration_metrics(sink, log)

    assert metrics.patterns_total.value == 1
    assert metrics.calibration_reviews_total.value == 0
    assert metrics.patterns_with_calibration_evidence.value == 0
    assert metrics.blind_spots_total.value == 1
    assert metrics.blind_spot_corpus_share.value == 1.0


def test_calibration_metrics_reflect_recorded_calibration_reviews():
    sink = _sink()
    log = InMemorySamplingLogSink()
    service = ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )
    coordinator = AdaptiveReviewCoordinator(review_service=service, telemetry_source=sink, sampling_log=log)

    queue = coordinator.build_queue(intent=ReviewIntent.CALIBRATION, strategy_name="random_sampling", k=3, seed=1)
    for entry in [e for e in queue if e.intent == ReviewIntent.CALIBRATION]:
        packet = coordinator.open_packet(entry, archive_object_ref="obj1")
        coordinator.submit_decision(entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER)

    metrics = calibration_metrics(sink, log)

    assert metrics.calibration_reviews_total.value == 3
    assert metrics.patterns_with_calibration_evidence.value == 1


def test_benchmark_run_without_calibration_field_still_loads_and_renders():
    """Backward compatibility: a run persisted before Milestone 11 has `calibration=None`."""
    run = build_run()
    assert run.calibration is None
    assert calibration_section(run) is None


def test_benchmark_run_with_calibration_renders_a_section():
    sink = _sink()
    log = InMemorySamplingLogSink()
    run = build_run()
    run_with_calibration = run.model_copy(
        update={"calibration": calibration_metrics(sink, log)}
    )
    section = calibration_section(run_with_calibration)
    assert section is not None
    assert "Confidence Calibration" in section


def test_calibration_metrics_survive_json_round_trip():
    sink = _sink()
    log = InMemorySamplingLogSink()
    metrics = calibration_metrics(sink, log)
    run = build_run().model_copy(update={"calibration": metrics})

    dumped = run.model_dump_json()
    reloaded = BenchmarkRun.model_validate_json(dumped)

    assert reloaded.calibration is not None
    assert reloaded.calibration.patterns_total.status == MetricStatus.MEASURED
