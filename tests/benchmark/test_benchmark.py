"""Tests for the Benchmark & Measurement Framework (2026-07-13).

Builds a small synthetic corpus using the same telemetry-fixture pattern
`tests/learning/_helpers.py` already establishes (real event shapes, not hand-rolled stand-ins),
plus a processing-progress stream and a runtime-telemetry stream, and exercises every aggregator,
the run/store/compare machinery, report rendering, and CSV/JSON export.
"""

from __future__ import annotations

import json

from archivetrust.application.progress import ProcessingProgressEvent, ProcessingProgressKind
from archivetrust.benchmark.aggregate import (
    dataset_metrics,
    human_review_metrics,
    observation_type_comparison,
    operational_metrics,
    provider_quality_metrics,
    provider_runtime_metrics,
    trust_metrics,
)
from archivetrust.benchmark.export import to_csv, to_json
from archivetrust.benchmark.groups import DatasetMetrics
from archivetrust.benchmark.metrics import MetricStatus
from archivetrust.benchmark.report import (
    executive_summary,
    full_report,
    observation_type_table,
    provider_comparison_table,
    public_benchmark_text,
)
from archivetrust.benchmark.run import BenchmarkMetadata, BenchmarkRun, BenchmarkRunStore, compare_runs
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.feedback.models import CorrectionAction
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads.page import PagePayload
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    ObservationCreated,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.runtime.runtime_telemetry import RuntimeTelemetryEvent, RuntimeTelemetryKind

from tests.learning._helpers import emit_correction, emit_evidence_rejection, emit_slot, heading


def _page(sink: InMemoryTelemetrySink, *, document_ref: str, provider: str, page_number: int) -> None:
    payload = PagePayload(page_number=page_number)
    evidence = Evidence.create(
        provider=provider, provider_version="1.0", raw_output=str(page_number),
        processing_stage=ProcessingStage.RAW, page=page_number,
    )
    observation = Observation.from_evidence(
        provider_id=provider, provider_version="1.0", payload=payload, evidence=(evidence,)
    )
    sink.append(
        ProviderObservationAttempted(
            event_id=new_id("event"), document_ref=document_ref, provider_id=provider,
            provider_version="1.0", invocation_id=new_id("invocation"),
            outcome=ProviderInvocationOutcome.PRODUCED_OBSERVATIONS, observation_count=1,
        )
    )
    sink.append(
        ObservationCreated(
            event_id=new_id("event"), document_ref=document_ref,
            invocation_id=new_id("invocation"), observation=observation,
        )
    )


def build_corpus() -> InMemoryTelemetrySink:
    sink = InMemoryTelemetrySink()
    # Document 1: 2 pages, one corroborated heading (docling+tesseract), one contested heading.
    _page(sink, document_ref="d1", provider="docling", page_number=1)
    _page(sink, document_ref="d1", provider="docling", page_number=2)
    corroborated = emit_slot(
        sink, document_ref="d1", payload=heading("Title"), providers=("docling", "tesseract")
    )
    emit_correction(sink, document_ref="d1", target=corroborated, action=CorrectionAction.ACCEPT.value)

    # Document 2: single page, single-source uncorroborated slot -> triggers review.
    _page(sink, document_ref="d2", provider="docling", page_number=1)
    single_source = emit_slot(sink, document_ref="d2", payload=heading("Solo"), providers=("docling",))
    emit_correction(sink, document_ref="d2", target=single_source, action=CorrectionAction.EDIT.value)
    emit_evidence_rejection(sink, document_ref="d2", provider="tesseract")

    return sink


def build_progress_events(run_id: str = "run-1") -> tuple[ProcessingProgressEvent, ...]:
    return (
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED, run_id=run_id,
            recorded_at="2026-07-13T10:00:00+00:00", provider_ids=("docling", "tesseract"),
            pending_count=2,
        ),
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_FINISHED, run_id=run_id,
            recorded_at="2026-07-13T10:05:00+00:00", documents_done=2, pending_count=0,
        ),
    )


def build_runtime_events() -> tuple[RuntimeTelemetryEvent, ...]:
    return (
        RuntimeTelemetryEvent(
            kind=RuntimeTelemetryKind.WARMUP_TIME, recorded_at=0.0, model_id="docling", seconds=3.5
        ),
        RuntimeTelemetryEvent(
            kind=RuntimeTelemetryKind.GPU_ALLOCATED, recorded_at=0.0, model_id="docling",
            reserved_bytes=1_000_000, total_bytes=8_000_000_000,
        ),
    )


def build_run() -> BenchmarkRun:
    sink = build_corpus()
    progress = build_progress_events()
    runtime = build_runtime_events()
    return BenchmarkRun(
        metadata=BenchmarkMetadata(run_id="run-1", run_at="2026-07-13T10:05:01+00:00", dataset_id="synthetic-v1"),
        dataset=dataset_metrics(sink, progress_events=progress),
        runtime=provider_runtime_metrics(sink, runtime_events=runtime),
        quality=provider_quality_metrics(sink),
        observation_types=observation_type_comparison(sink),
        trust=trust_metrics(sink),
        human_review=human_review_metrics(sink),
        operational=operational_metrics(sink, runtime_events=runtime, progress_events=progress),
    )


# -- Dataset metrics -----------------------------------------------------------------------------


def test_dataset_metrics_counts_documents_and_pages() -> None:
    sink = build_corpus()
    metrics = dataset_metrics(sink, progress_events=build_progress_events())
    assert metrics.documents_processed.value == 2
    assert metrics.documents_processed.status == MetricStatus.MEASURED
    assert metrics.pages_processed.value == 3  # 2 + 1
    assert metrics.average_pages_per_document.value == 1.5
    assert metrics.processing_duration_seconds.status == MetricStatus.MEASURED
    assert metrics.processing_duration_seconds.value == 300.0


def test_dataset_metrics_honest_when_no_progress_stream() -> None:
    sink = build_corpus()
    metrics = dataset_metrics(sink)
    assert metrics.processing_duration_seconds.status == MetricStatus.UNAVAILABLE
    assert metrics.document_types.status == MetricStatus.UNAVAILABLE


def test_document_types_is_estimated_when_mime_types_supplied() -> None:
    sink = build_corpus()
    metrics = dataset_metrics(sink, document_mime_types={"d1": "application/pdf", "d2": "application/pdf"})
    assert metrics.document_types.status == MetricStatus.ESTIMATED
    assert metrics.document_types.value == {"application/pdf": 2}


# -- Provider runtime metrics ---------------------------------------------------------------------


def test_runtime_metrics_never_fabricate_per_invocation_timing() -> None:
    sink = build_corpus()
    metrics = provider_runtime_metrics(sink, runtime_events=build_runtime_events())
    docling = next(m for m in metrics if m.provider_id == "docling")
    assert docling.avg_runtime_per_document_seconds.status == MetricStatus.UNAVAILABLE
    assert docling.cpu_utilization_percent.status == MetricStatus.UNAVAILABLE
    assert docling.retry_count.status == MetricStatus.UNAVAILABLE
    # But what IS recorded (warmup, GPU bytes) is surfaced, matched by model_id.
    assert docling.cold_startup_seconds.status == MetricStatus.MEASURED
    assert docling.cold_startup_seconds.value == 3.5
    assert docling.peak_gpu_memory_bytes.value == 1_000_000


def test_failure_rate_is_derived_from_provider_health() -> None:
    sink = InMemoryTelemetrySink()
    emit_slot(sink, document_ref="d1", payload=heading("A"), providers=("docling",))
    sink.append(
        ProviderObservationAttempted(
            event_id=new_id("e"), document_ref="d1", provider_id="docling", provider_version="1.0",
            invocation_id=new_id("i"), outcome=ProviderInvocationOutcome.FAILED, failure_reason="boom",
        )
    )
    metrics = provider_runtime_metrics(sink)
    docling = next(m for m in metrics if m.provider_id == "docling")
    assert docling.failure_rate.status == MetricStatus.DERIVED
    assert docling.failure_rate.value == 0.5  # 1 failed of 2 attempts (1 from emit_slot, 1 explicit)


# -- Provider quality metrics ---------------------------------------------------------------------


def test_quality_metrics_agreement_and_review_trigger_rate() -> None:
    sink = build_corpus()
    metrics = provider_quality_metrics(sink)
    docling = next(m for m in metrics if m.provider_id == "docling")
    # docling contributed to 2 slots: 1 CORROBORATED (d1 heading), 1 UNCORROBORATED (d2 heading).
    assert docling.agreement_with_canonical.value == 0.5
    assert docling.unique_observations_contributed.value == 1
    # The single-source d2 slot triggers review (SINGLE_SOURCE is queued for HEADING by default policy).
    assert docling.review_trigger_rate.value == 0.5
    tesseract = next(m for m in metrics if m.provider_id == "tesseract")
    assert tesseract.observations_rejected.value == 1


def test_quality_metrics_never_fabricate_confidence_when_absent() -> None:
    sink = InMemoryTelemetrySink()
    metrics = provider_quality_metrics(sink)
    assert metrics == ()


# -- Observation-type comparison -------------------------------------------------------------------


def test_observation_type_comparison_identifies_best_contributor() -> None:
    sink = build_corpus()
    rows = observation_type_comparison(sink)
    heading_row = next(r for r in rows if r.observation_type == "heading")
    assert heading_row.best_provider.status == MetricStatus.DERIVED
    assert "docling" in heading_row.best_provider.value


# -- Trust metrics -----------------------------------------------------------------------------------


def test_trust_metrics_agreement_distribution_and_density() -> None:
    sink = build_corpus()
    metrics = trust_metrics(sink)
    assert metrics.agreement_distribution.value["corroborated"] == 1
    assert metrics.agreement_distribution.value["uncorroborated_single_source"] == 1
    assert metrics.average_providers_agreeing.value == 2.0
    assert metrics.evidence_density.status == MetricStatus.DERIVED


# -- Human review metrics -----------------------------------------------------------------------------


def test_human_review_metrics_reduction_vs_baseline() -> None:
    sink = build_corpus()
    metrics = human_review_metrics(sink)
    # Only the single-source d2 heading is queued under default policy (heading is a reviewed type).
    assert metrics.review_items_total.value == 1
    assert metrics.documents_entering_review.value == 1
    assert metrics.human_corrections.value == 2
    assert metrics.human_overrides.value == 1  # the EDIT on d2; the d1 ACCEPT is not an override
    # Baseline (v1) would queue the same single-source slot too (heading has no type gate in v1) ->
    # reduction is 0 here since both policies agree on this small corpus; assert it's computed, not absent.
    assert metrics.review_reduction_vs_baseline.status == MetricStatus.DERIVED


def test_average_review_time_uses_recorded_durations() -> None:
    sink = InMemoryTelemetrySink()
    canonical = emit_slot(sink, document_ref="d1", payload=heading("A"), providers=("docling",))
    from archivetrust.domain.telemetry.events import HumanCorrectionSubmitted

    sink.append(
        HumanCorrectionSubmitted(
            event_id=new_id("e"), document_ref="d1", correction_id=new_id("c"),
            target_canonical_observation_id=canonical.canonical_observation_id,
            category="transcription_error", action="edit", raw_ai_output="A",
            raw_corrected_output="B", review_duration_seconds=12.5,
        )
    )
    metrics = human_review_metrics(sink)
    assert metrics.average_review_time_seconds.status == MetricStatus.MEASURED
    assert metrics.average_review_time_seconds.value == 12.5


# -- Operational metrics -----------------------------------------------------------------------------


def test_operational_metrics_completion_and_completeness() -> None:
    sink = build_corpus()
    metrics = operational_metrics(
        sink, runtime_events=build_runtime_events(), progress_events=build_progress_events()
    )
    assert metrics.pipeline_completion_rate.value == 1.0
    assert metrics.resource_exhaustion_events.status == MetricStatus.UNAVAILABLE
    assert metrics.timeouts.status == MetricStatus.UNAVAILABLE


# -- Run / Store / Compare -----------------------------------------------------------------------------


def test_benchmark_run_persists_and_reloads(tmp_path) -> None:
    run = build_run()
    store = BenchmarkRunStore(tmp_path / "benchmarks.jsonl")
    store.append(run)

    reloaded_store = BenchmarkRunStore(tmp_path / "benchmarks.jsonl")
    assert reloaded_store.get("run-1") == run
    assert reloaded_store.for_dataset("synthetic-v1") == (run,)
    assert reloaded_store.get("does-not-exist") is None


def test_compare_runs_reports_shared_metrics_only() -> None:
    before = build_run()
    after = build_run().model_copy(
        update={"metadata": before.metadata.model_copy(update={"run_id": "run-2"})}
    )
    deltas = compare_runs(before, after)
    assert len(deltas) > 0
    by_path = {d.path: d for d in deltas}
    assert "trust.average_canonical_confidence" in by_path
    delta = by_path["trust.average_canonical_confidence"]
    assert delta.before == delta.after  # identical corpora -> identical metric


# -- Report rendering ---------------------------------------------------------------------------------


def test_reports_render_without_fabricating_unavailable_metrics() -> None:
    run = build_run()
    summary = executive_summary(run)
    assert "Documents processed" in summary

    table = provider_comparison_table(run)
    assert "docling" in table
    assert "unavailable" in table  # avg runtime / pages-per-hour are genuinely unavailable

    type_table = observation_type_table(run)
    assert "heading" in type_table

    text = full_report(run)
    assert "Reliability" in text and "Trust" in text and "Performance" in text

    public = public_benchmark_text(run)
    assert "ArchiveTrust Benchmark" in public
    assert "synthetic-v1" in public
    # No document/archive identifiers ever appear in the public text.
    assert "d1" not in public and "d2" not in public


# -- Export --------------------------------------------------------------------------------------------


def test_export_json_round_trips_through_pydantic() -> None:
    run = build_run()
    payload = json.loads(to_json(run))
    assert payload["metadata"]["run_id"] == "run-1"
    assert BenchmarkRun.model_validate(payload) == run


def test_export_csv_has_one_row_per_metric_leaf() -> None:
    run = build_run()
    csv_text = to_csv(run)
    lines = csv_text.strip().splitlines()
    assert lines[0] == "metric,value,status,unit,note"
    assert len(lines) > 20  # dozens of metric leaves across six groups
    assert any("trust.average_canonical_confidence" in line for line in lines)
