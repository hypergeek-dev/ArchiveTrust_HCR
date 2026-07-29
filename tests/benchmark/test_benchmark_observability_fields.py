"""Tests proving the Observability milestone's new `ProviderObservationAttempted` fields are
automatically consumed by the existing benchmark aggregators (2026-07-13) — no aggregator
signature changed, so these fields transitioning a metric from Unavailable to Measured/Derived is
purely a function of the telemetry now carrying them.
"""

from __future__ import annotations

from archivetrust.application.progress import ProcessingProgressEvent, ProcessingProgressKind
from archivetrust.benchmark.aggregate import dataset_metrics, operational_metrics, provider_runtime_metrics
from archivetrust.benchmark.metrics import MetricStatus
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    ProviderFailureCategory,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink


def _attempt(
    *,
    document_ref: str = "d1",
    provider_id: str = "tesseract_layoutparser",
    outcome: ProviderInvocationOutcome = ProviderInvocationOutcome.PRODUCED_OBSERVATIONS,
    duration_ms: int | None = None,
    retry_count: int | None = 0,
    timeout: bool | None = None,
    failure_category: ProviderFailureCategory | None = None,
    pages_processed: int | None = None,
    cold_start: bool | None = None,
) -> ProviderObservationAttempted:
    return ProviderObservationAttempted(
        event_id=new_id("event"), document_ref=document_ref, provider_id=provider_id,
        provider_version="1.0", invocation_id=new_id("invocation"), outcome=outcome,
        observation_count=1 if outcome == ProviderInvocationOutcome.PRODUCED_OBSERVATIONS else 0,
        duration_ms=duration_ms, retry_count=retry_count, timeout=timeout,
        failure_category=failure_category, pages_processed=pages_processed, cold_start=cold_start,
    )


def test_runtime_stats_transition_to_measured_once_duration_is_recorded() -> None:
    sink = InMemoryTelemetrySink()
    for ms, pages in ((100, 1), (200, 1), (300, 1), (1000, 1)):
        sink.append(_attempt(duration_ms=ms, pages_processed=pages))

    metrics = provider_runtime_metrics(sink)
    tesseract = next(m for m in metrics if m.provider_id == "tesseract_layoutparser")

    assert tesseract.avg_runtime_per_document_seconds.status == MetricStatus.MEASURED
    assert tesseract.avg_runtime_per_document_seconds.value == 0.4  # (0.1+0.2+0.3+1.0)/4
    assert tesseract.median_runtime_seconds.status == MetricStatus.MEASURED
    assert tesseract.median_runtime_seconds.value == 0.25
    assert tesseract.peak_runtime_seconds.value == 1.0
    assert tesseract.p95_runtime_seconds.status == MetricStatus.MEASURED

    assert tesseract.avg_runtime_per_page_seconds.status == MetricStatus.MEASURED
    assert tesseract.throughput_documents_per_hour.status == MetricStatus.DERIVED
    assert tesseract.throughput_documents_per_hour.value == 3600 / 0.4
    assert tesseract.throughput_pages_per_hour.status == MetricStatus.DERIVED


def test_cold_start_invocations_are_excluded_from_steady_state_timing() -> None:
    """Runtime Warm-Up Verification (2026-07-14): the 2026-07-14 benchmark's first PaddleOCR-VL
    invocation carried `cold_start=True` and a `duration_ms` that included `warm_up()`'s cost --
    `provider_runtime_metrics` must not fold that one-time cost into the same avg/median/p95/
    throughput figures as every steady-state invocation."""
    sink = InMemoryTelemetrySink()
    sink.append(_attempt(duration_ms=7540, pages_processed=1, cold_start=True))
    for ms in (2183, 3003, 3253, 3311, 4290):
        sink.append(_attempt(duration_ms=ms, pages_processed=1, cold_start=False))

    metrics = provider_runtime_metrics(sink)
    tesseract = next(m for m in metrics if m.provider_id == "tesseract_layoutparser")

    steady_state_ms = [2183, 3003, 3253, 3311, 4290]
    expected_avg = (sum(steady_state_ms) / len(steady_state_ms)) / 1000
    assert tesseract.avg_runtime_per_document_seconds.status == MetricStatus.MEASURED
    assert tesseract.avg_runtime_per_document_seconds.value == expected_avg
    assert tesseract.peak_runtime_seconds.value == max(steady_state_ms) / 1000  # 7540ms excluded
    assert "cold-start" in tesseract.avg_runtime_per_document_seconds.note


def test_runtime_metrics_stay_unavailable_when_every_invocation_was_a_cold_start() -> None:
    sink = InMemoryTelemetrySink()
    sink.append(_attempt(duration_ms=7540, pages_processed=1, cold_start=True))

    metrics = provider_runtime_metrics(sink)
    tesseract = next(m for m in metrics if m.provider_id == "tesseract_layoutparser")

    assert tesseract.avg_runtime_per_document_seconds.status == MetricStatus.UNAVAILABLE
    assert "cold start" in tesseract.avg_runtime_per_document_seconds.note


def test_retry_count_is_measured_as_zero_not_unavailable() -> None:
    sink = InMemoryTelemetrySink()
    sink.append(_attempt(duration_ms=50, retry_count=0))
    metrics = provider_runtime_metrics(sink)
    tesseract = next(m for m in metrics if m.provider_id == "tesseract_layoutparser")
    assert tesseract.retry_count.status == MetricStatus.MEASURED
    assert tesseract.retry_count.value == 0


def test_crash_rate_is_derived_from_exception_failure_category() -> None:
    sink = InMemoryTelemetrySink()
    sink.append(_attempt(duration_ms=50, outcome=ProviderInvocationOutcome.PRODUCED_OBSERVATIONS))
    sink.append(
        _attempt(
            duration_ms=10, outcome=ProviderInvocationOutcome.FAILED,
            failure_category=ProviderFailureCategory.EXCEPTION, timeout=False,
        )
    )
    metrics = provider_runtime_metrics(sink)
    tesseract = next(m for m in metrics if m.provider_id == "tesseract_layoutparser")
    assert tesseract.crash_rate.status == MetricStatus.DERIVED
    assert tesseract.crash_rate.value == 0.5


def test_runtime_metrics_stay_unavailable_when_no_duration_recorded() -> None:
    sink = InMemoryTelemetrySink()
    sink.append(_attempt(duration_ms=None, retry_count=None))  # historical-shaped event
    metrics = provider_runtime_metrics(sink)
    tesseract = next(m for m in metrics if m.provider_id == "tesseract_layoutparser")
    assert tesseract.avg_runtime_per_document_seconds.status == MetricStatus.UNAVAILABLE
    assert tesseract.retry_count.status == MetricStatus.UNAVAILABLE  # retry_count also None here


def test_operational_timeouts_transitions_to_measured() -> None:
    sink = InMemoryTelemetrySink()
    sink.append(
        _attempt(
            outcome=ProviderInvocationOutcome.FAILED, duration_ms=5000, timeout=True,
            failure_category=ProviderFailureCategory.TIMEOUT,
        )
    )
    sink.append(
        _attempt(
            outcome=ProviderInvocationOutcome.FAILED, duration_ms=10, timeout=False,
            failure_category=ProviderFailureCategory.PROVIDER_ERROR,
        )
    )
    metrics = operational_metrics(sink)
    assert metrics.timeouts.status == MetricStatus.MEASURED
    assert metrics.timeouts.value == 1


def test_operational_timeouts_stays_unavailable_with_no_failures() -> None:
    sink = InMemoryTelemetrySink()
    sink.append(_attempt(outcome=ProviderInvocationOutcome.PRODUCED_OBSERVATIONS, duration_ms=10))
    metrics = operational_metrics(sink)
    assert metrics.timeouts.status == MetricStatus.UNAVAILABLE


def test_operational_timeouts_is_derived_lower_bound_with_unlabeled_historical_failures() -> None:
    sink = InMemoryTelemetrySink()
    # A historical FAILED event predating the `timeout` field (None, not False).
    sink.append(_attempt(outcome=ProviderInvocationOutcome.FAILED, duration_ms=10, timeout=None))
    sink.append(
        _attempt(outcome=ProviderInvocationOutcome.FAILED, duration_ms=5000, timeout=True,
                  failure_category=ProviderFailureCategory.TIMEOUT)
    )
    metrics = operational_metrics(sink)
    assert metrics.timeouts.status == MetricStatus.DERIVED
    assert metrics.timeouts.value == 1
    assert "lower bound" in metrics.timeouts.note


def test_dataset_average_document_processing_seconds_is_measured() -> None:
    sink = InMemoryTelemetrySink()
    progress = (
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_COMPLETED, run_id="r1",
            recorded_at="2026-07-13T10:00:00+00:00", duration_seconds=12.0,
        ),
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_COMPLETED, run_id="r1",
            recorded_at="2026-07-13T10:01:00+00:00", duration_seconds=8.0,
        ),
    )
    metrics = dataset_metrics(sink, progress_events=progress)
    assert metrics.average_document_processing_seconds.status == MetricStatus.MEASURED
    assert metrics.average_document_processing_seconds.value == 10.0


def test_dataset_average_document_processing_seconds_unavailable_without_progress_stream() -> None:
    sink = InMemoryTelemetrySink()
    metrics = dataset_metrics(sink)
    assert metrics.average_document_processing_seconds.status == MetricStatus.UNAVAILABLE
