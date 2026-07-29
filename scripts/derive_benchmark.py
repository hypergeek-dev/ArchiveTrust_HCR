"""Derives Benchmark #1 from the completed portion of the 2026-07-13/14 overnight run
(workspace `46e94f15d73347c2be79f1527e646c11`, 478/1000 documents completed before the process
crashed on document #479 — see `docs/PRODUCTION_INCIDENT_2026-07-14.md`).

Read-only over `archivetrust_data/` (preserved forensic evidence — never modified, never
reprocessed). Every figure comes from `benchmark/aggregate.py`'s existing, already-tested
aggregation functions; this script only wires real telemetry into them and renders the results.
No provider is invoked, no document is reprocessed, no value is estimated beyond what those
functions already mark as ESTIMATED/DERIVED.

Run: `python scripts/derive_benchmark.py`. Writes into `benchmarks/` (not `archivetrust_data/`).
"""

from __future__ import annotations

import csv
import io
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

WORKSPACE_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
TOP_LEVEL_TELEMETRY_DIR = REPO_ROOT / "archivetrust_data" / "telemetry"
OUTPUT_DIR = REPO_ROOT / "benchmarks"
DATASET_ID = "archive_test-1000-doc-corpus-2026-07-13"
RUN_ID = "benchmark_1_2026-07-13_interrupted"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    from archivetrust.acquisition.events import ArchiveObjectRegistered, FileAcquisitionTelemetrySink
    from archivetrust.application.progress import FileProcessingProgressSink, ProcessingProgressKind
    from archivetrust.benchmark.aggregate import (
        calibration_metrics,
        dataset_metrics,
        human_review_metrics,
        observation_type_comparison,
        operational_metrics,
        provider_quality_metrics,
        provider_runtime_metrics,
        trust_metrics,
    )
    from archivetrust.benchmark.export import to_csv, to_json
    from archivetrust.benchmark.report import full_report, public_benchmark_text
    from archivetrust.benchmark.run import BenchmarkMetadata, BenchmarkRun, BenchmarkRunStore, utc_now_iso
    from archivetrust.domain.telemetry.events import ProviderInvocationOutcome
    from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
    from archivetrust.review.sampling.log import InMemorySamplingLogSink
    from archivetrust.runtime.runtime_telemetry import FileRuntimeTelemetrySink

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "exports").mkdir(parents=True, exist_ok=True)

    _log("Loading domain telemetry (events.jsonl, ~977MB) — this takes about a minute...")
    telemetry = FileTelemetrySink(WORKSPACE_DIR / "telemetry" / "events.jsonl")
    all_events = list(telemetry.all_events())
    _log(f"Loaded {len(all_events)} domain events.")

    _log("Loading processing progress (processing.jsonl)...")
    progress = FileProcessingProgressSink(WORKSPACE_DIR / "telemetry" / "processing.jsonl")
    progress_events = progress.events()
    _log(f"Loaded {len(progress_events)} processing-progress events.")

    _log("Loading acquisition telemetry (acquisition.jsonl) for document mime types...")
    acquisition = FileAcquisitionTelemetrySink(WORKSPACE_DIR / "telemetry" / "acquisition.jsonl")
    document_mime_types = {
        e.archive_object.id: e.archive_object.mime_type
        for e in acquisition.all_events()
        if isinstance(e, ArchiveObjectRegistered)
    }
    _log(f"Loaded {len(document_mime_types)} registered archive objects.")

    _log("Loading runtime telemetry (top-level runtime_events.jsonl)...")
    runtime_sink = FileRuntimeTelemetrySink(TOP_LEVEL_TELEMETRY_DIR / "runtime_events.jsonl")
    runtime_events = runtime_sink.events()
    _log(f"Loaded {len(runtime_events)} runtime events (process-wide, includes prior sessions).")

    # -- Scope verification: this run's events.jsonl must already contain only the 478 completed
    # documents (document #479 recorded zero domain events before the crash, confirmed in the
    # 2026-07-14 incident report) -- verified here, not assumed.
    documents_with_events = {e.document_ref for e in all_events}
    completed_ids = {
        e.archive_object_id
        for e in progress_events
        if e.kind == ProcessingProgressKind.DOCUMENT_COMPLETED
    }
    started_ids = {
        e.archive_object_id
        for e in progress_events
        if e.kind == ProcessingProgressKind.DOCUMENT_STARTED
    }
    interrupted_ids = started_ids - completed_ids
    _log(
        f"Scope check: {len(documents_with_events)} documents in events.jsonl, "
        f"{len(completed_ids)} DocumentProcessingCompleted, {len(interrupted_ids)} interrupted "
        f"(started, never completed): {sorted(interrupted_ids)}"
    )
    assert documents_with_events == completed_ids, (
        "events.jsonl scope does not match completed-document set — investigate before trusting "
        "any metric below; this script must never silently include interrupted work."
    )

    _log("Computing dataset metrics...")
    dataset = dataset_metrics(telemetry, progress_events=progress_events, document_mime_types=document_mime_types)

    _log("Computing provider runtime metrics...")
    runtime = provider_runtime_metrics(telemetry, runtime_events=runtime_events)

    _log("Computing provider quality metrics...")
    quality = provider_quality_metrics(telemetry)

    _log("Computing observation-type comparison...")
    observation_types = observation_type_comparison(telemetry)

    _log("Computing trust metrics...")
    trust = trust_metrics(telemetry)

    _log("Computing human review metrics (no interaction sink recorded for this run)...")
    human_review = human_review_metrics(telemetry)

    _log("Computing operational metrics...")
    operational = operational_metrics(telemetry, runtime_events=runtime_events, progress_events=progress_events)

    _log("Computing calibration metrics (Milestone 11) -- read-only, same as every other metric "
         "here; no sampling log exists yet for this workspace (no calibration-intent review has "
         "ever run against it), so this honestly reports zero calibration evidence, matching the "
         "existing sufficiency report...")
    calibration = calibration_metrics(telemetry, InMemorySamplingLogSink())

    run = BenchmarkRun(
        metadata=BenchmarkMetadata(
            run_id=RUN_ID,
            run_at=utc_now_iso(),
            dataset_id=DATASET_ID,
        ),
        dataset=dataset,
        runtime=runtime,
        quality=quality,
        observation_types=observation_types,
        trust=trust,
        human_review=human_review,
        operational=operational,
        calibration=calibration,
    )

    _log("Persisting as Benchmark #1...")
    store = BenchmarkRunStore(OUTPUT_DIR / "benchmark_runs.jsonl")
    store.append(run)

    _log("Rendering reports...")
    (OUTPUT_DIR / "reports" / "FULL_REPORT.md").write_text(full_report(run), encoding="utf-8")
    (OUTPUT_DIR / "reports" / "PUBLIC_REPORT.md").write_text(public_benchmark_text(run), encoding="utf-8")

    _log("Exporting JSON/CSV...")
    (OUTPUT_DIR / "exports" / "benchmark_1.json").write_text(to_json(run), encoding="utf-8")
    (OUTPUT_DIR / "exports" / "benchmark_1.csv").write_text(to_csv(run), encoding="utf-8")

    _log("Computing degradation analysis (not part of the existing framework — new, additive)...")
    degradation = _degradation_analysis(progress_events, all_events)
    (OUTPUT_DIR / "exports" / "degradation_by_document.csv").write_text(degradation["by_document_csv"], encoding="utf-8")
    (OUTPUT_DIR / "exports" / "degradation_by_hour.csv").write_text(degradation["by_hour_csv"], encoding="utf-8")
    (OUTPUT_DIR / "exports" / "failure_timeline.csv").write_text(degradation["failure_timeline_csv"], encoding="utf-8")
    (OUTPUT_DIR / "degradation_summary.json").write_text(
        json.dumps(degradation["summary"], indent=2), encoding="utf-8"
    )

    _log("Done.")
    _log(f"Documents in this benchmark: {dataset.documents_processed.value}")
    _log(f"Interrupted (excluded): {sorted(interrupted_ids)}")


def _degradation_analysis(progress_events, all_events) -> dict:
    """Throughput-over-time analysis (not an existing `benchmark/aggregate.py` group — this
    investigation's own addition). Uses only `DocumentProcessingCompleted.duration_seconds`/
    `recorded_at` (measured, real) and `ProviderObservationAttempted` failure records (measured,
    real). No estimation: a document with no recorded duration is skipped, not interpolated.
    """
    from archivetrust.application.progress import ProcessingProgressKind
    from archivetrust.domain.telemetry.events import ProviderInvocationOutcome
    from datetime import datetime, timezone

    completions = [
        e for e in progress_events
        if e.kind == ProcessingProgressKind.DOCUMENT_COMPLETED and e.duration_seconds is not None
    ]

    by_document_rows = [["document_number", "documents_done", "duration_seconds", "recorded_at"]]
    for i, e in enumerate(completions, start=1):
        by_document_rows.append([i, e.documents_done, f"{e.duration_seconds:.3f}", e.recorded_at])

    buf = io.StringIO()
    csv.writer(buf).writerows(by_document_rows)
    by_document_csv = buf.getvalue()

    by_hour: dict[str, list[float]] = defaultdict(list)
    for e in completions:
        ts = datetime.fromisoformat(e.recorded_at)
        hour_bucket = ts.strftime("%Y-%m-%dT%H:00")
        by_hour[hour_bucket].append(e.duration_seconds)

    by_hour_rows = [["hour_utc", "documents_completed", "avg_duration_seconds", "documents_per_hour"]]
    for hour in sorted(by_hour):
        durations = by_hour[hour]
        by_hour_rows.append([hour, len(durations), f"{statistics.mean(durations):.2f}", len(durations)])
    buf = io.StringIO()
    csv.writer(buf).writerows(by_hour_rows)
    by_hour_csv = buf.getvalue()

    failed = [
        e for e in all_events
        if getattr(e, "outcome", None) == ProviderInvocationOutcome.FAILED
    ]
    failure_rows = [["recorded_at", "document_ref", "provider_id", "failure_reason", "document_number_at_failure"]]
    started_order = [
        e.archive_object_id for e in progress_events
        if e.kind == ProcessingProgressKind.DOCUMENT_STARTED
    ]
    doc_number_by_ref = {ref: i + 1 for i, ref in enumerate(started_order)}
    for e in sorted(failed, key=lambda x: x.recorded_at):
        failure_rows.append([
            e.recorded_at, e.document_ref, e.provider_id,
            (e.failure_reason or "")[:200], doc_number_by_ref.get(e.document_ref, ""),
        ])
    buf = io.StringIO()
    csv.writer(buf).writerows(failure_rows)
    failure_timeline_csv = buf.getvalue()

    # First-half vs second-half comparison -- the smallest honest trend statement possible without
    # fitting a model to noisy per-document data (this investigation's own "do not speculate about
    # causes" constraint; a raw split needs no model at all).
    n = len(completions)
    trend_summary: dict = {"documents_with_measured_duration": n}
    if n >= 10:
        half = n // 2
        first_half = [e.duration_seconds for e in completions[:half]]
        second_half = [e.duration_seconds for e in completions[half:]]
        trend_summary["first_half_avg_seconds"] = round(statistics.mean(first_half), 2)
        trend_summary["second_half_avg_seconds"] = round(statistics.mean(second_half), 2)
        trend_summary["change_percent"] = round(
            100 * (statistics.mean(second_half) - statistics.mean(first_half)) / statistics.mean(first_half), 1
        )
    else:
        trend_summary["note"] = "fewer than 10 documents with measured duration; no split computed"

    trend_summary["total_failed_invocations"] = len(failed)
    bad_alloc_failures = [e for e in failed if e.failure_reason and "bad_alloc" in e.failure_reason]
    trend_summary["bad_alloc_invocations"] = len(bad_alloc_failures)
    if bad_alloc_failures:
        first = min(bad_alloc_failures, key=lambda e: e.recorded_at)
        trend_summary["first_bad_alloc_document_number"] = doc_number_by_ref.get(first.document_ref)

    return {
        "by_document_csv": by_document_csv,
        "by_hour_csv": by_hour_csv,
        "failure_timeline_csv": failure_timeline_csv,
        "summary": trend_summary,
    }


if __name__ == "__main__":
    main()
