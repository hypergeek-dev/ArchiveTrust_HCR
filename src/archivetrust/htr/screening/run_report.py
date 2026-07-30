"""Regenerates a reliability run's report from its durable telemetry log and nothing else.

Written as a pure function of the event stream, deliberately: if a report could only be produced by
the process that did the work, then a run stopped and resumed across three sessions could never
produce one, and the durable log would not actually be the source of truth this harness claims it
is. Every number below is read back out of `events.jsonl`.

**No accuracy metric is computed and none may be added.** This corpus has no ground truth. What is
reported is execution outcome, timing, output shape, the reference-free heuristics from
`htr/screening/reliability_signals.py`, and the segmentation confound -- restated here because this
file produces the numbers a reader will quote.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from archivetrust.domain.telemetry.events import (
    ExperimentRunCompleted,
    ExperimentRunStarted,
    InputCropCreated,
    MethodRunCompleted,
    MethodRunStarted,
    NormalizedMethodResultRecorded,
    ParsedMethodResultRecorded,
    RawMethodResultRecorded,
    ReliabilityIssueClassified,
    ReproducibilityManifestRecorded,
    SegmentationRunCompleted,
)
from archivetrust.htr.screening.reliability_signals import (
    HEURISTIC_CATALOG,
    RELIABILITY_HEURISTICS_VERSION,
    ReliabilityThresholds,
    length_outlier_bounds,
    repeated_output_summary,
    timing_summary,
)
from archivetrust.htr.screening.run_state import CONFIGURATION_MANIFEST_KEY, RunPaths

NO_ACCURACY_STATEMENT = (
    "No CER, WER or accuracy is computed or reported. This corpus has no ground truth and none is "
    "planned; only reference-free technical-reliability signals are recorded."
)

CONFOUND_RESTATEMENT = (
    "The line detector is a Florence-2-family model, so one screened method's own model family "
    "controls the input the other is judged on. Accepted and flagged; not neutralized here."
)


def _seconds_between(started: str | None, completed: str | None) -> float | None:
    if not started or not completed:
        return None
    try:
        return (
            datetime.fromisoformat(completed) - datetime.fromisoformat(started)
        ).total_seconds()
    except ValueError:
        return None


def build_report(events: Iterable, *, run_id: str) -> dict:
    """The whole report, from the stream alone."""
    configuration: dict | None = None
    configuration_hash: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    crops: dict[str, dict] = {}
    method_runs: dict[str, dict] = {}
    texts: dict[str, dict[str, str]] = {}
    issues: list[dict] = []
    segmentation: list[dict] = []
    event_kinds: Counter = Counter()

    for event in events:
        event_kinds[event.kind.value if hasattr(event.kind, "value") else str(event.kind)] += 1
        if isinstance(event, ExperimentRunStarted):
            started_at = event.experiment_run.started_at
        elif isinstance(event, ExperimentRunCompleted):
            completed_at = event.experiment_run.completed_at
        elif isinstance(event, ReproducibilityManifestRecorded):
            payload = event.manifest.software_environment.get(CONFIGURATION_MANIFEST_KEY)
            if payload is not None:
                configuration = payload
                configuration_hash = event.manifest.pipeline_configuration_hash
        elif isinstance(event, InputCropCreated):
            crop = event.input_crop
            crops[crop.crop_id] = {
                "crop_id": crop.crop_id,
                "hash": crop.hash,
                "width": crop.width,
                "height": crop.height,
                "byte_size": crop.byte_size,
            }
        elif isinstance(event, SegmentationRunCompleted):
            segmentation.append(
                {
                    "segmentation_run_id": event.segmentation_run_id,
                    "page_id": event.page_id,
                    "adapter": event.segmentation_adapter_name,
                    "regions": len(event.region_ids),
                    "text_lines": len(event.text_line_ids),
                    "input_crops": len(event.input_crop_ids),
                }
            )
        elif isinstance(event, MethodRunStarted):
            run = event.method_run
            method_runs[run.method_run_id] = {
                "method_run_id": run.method_run_id,
                "method_id": run.method_id,
                "model_version_id": run.model_version_id,
                "input_crop_id": run.input_crop_id,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "wall_seconds": _seconds_between(run.started_at, run.completed_at),
                "outcome": run.outcome,
                "failure_reason": None,
            }
        elif isinstance(event, MethodRunCompleted):
            record = method_runs.setdefault(
                event.method_run_id,
                {"method_run_id": event.method_run_id, "method_id": event.method_id},
            )
            record["outcome"] = event.outcome
            record["failure_reason"] = event.failure_reason
        elif isinstance(event, RawMethodResultRecorded):
            texts.setdefault(event.method_run_id, {})["raw"] = event.text
        elif isinstance(event, ParsedMethodResultRecorded):
            texts.setdefault(event.method_run_id, {})["parsed"] = event.text
        elif isinstance(event, NormalizedMethodResultRecorded):
            texts.setdefault(event.method_run_id, {})["normalized"] = event.text
        elif isinstance(event, ReliabilityIssueClassified):
            issues.append(
                {
                    "method_run_id": event.method_run_id,
                    "classification": event.classification,
                    "detail": event.detail,
                }
            )

    by_method: dict[str, list[dict]] = {}
    for record in method_runs.values():
        stage = texts.get(record["method_run_id"], {})
        record["text"] = stage.get("normalized") or stage.get("parsed") or stage.get("raw")
        record["output_length"] = len(record["text"]) if record.get("text") is not None else None
        record["input_hash"] = (
            crops.get(record.get("input_crop_id") or "", {}).get("hash")
        )
        by_method.setdefault(record["method_id"], []).append(record)

    issue_counts_by_method: dict[str, Counter] = {}
    run_by_id = {record["method_run_id"]: record for record in method_runs.values()}
    for issue in issues:
        owner = run_by_id.get(issue["method_run_id"])
        method = owner["method_id"] if owner else "segmentation"
        issue_counts_by_method.setdefault(method, Counter())[issue["classification"]] += 1

    per_method: dict[str, dict] = {}
    for method, records in sorted(by_method.items()):
        succeeded = [r for r in records if r.get("outcome") == "succeeded"]
        observations = [
            (r["input_hash"], r["text"])
            for r in succeeded
            if r.get("input_hash") and r.get("text") is not None
        ]
        lengths = [r["output_length"] for r in succeeded if r.get("output_length") is not None]
        timings = [r["wall_seconds"] for r in records if r.get("wall_seconds") is not None]
        per_method[method] = {
            "runs_recorded": len(records),
            "succeeded": len(succeeded),
            "failed": len(records) - len(succeeded),
            "timing_seconds": timing_summary(timings),
            "output_length_outlier_bounds": length_outlier_bounds(lengths),
            "repeated_output_across_inputs": repeated_output_summary(observations).model_dump(
                mode="json"
            ),
            "heuristic_flag_counts": dict(sorted(issue_counts_by_method.get(method, {}).items())),
        }

    return {
        "schema": "archivetrust.reliability_run_report.v1",
        "run_id": run_id,
        "generated_from": "durable telemetry only (events.jsonl)",
        "no_accuracy_metric": NO_ACCURACY_STATEMENT,
        "confound": CONFOUND_RESTATEMENT,
        "reliability_heuristics_version": RELIABILITY_HEURISTICS_VERSION,
        "reliability_thresholds": ReliabilityThresholds().model_dump(mode="json"),
        "heuristic_catalog": [
            {
                "heuristic_id": entry.heuristic_id,
                "scope": entry.scope,
                "detects": entry.detects,
                "does_not_mean": entry.does_not_mean,
            }
            for entry in HEURISTIC_CATALOG
        ],
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "started_at": started_at,
        "completed_at": completed_at,
        "sealed": completed_at is not None,
        "telemetry_event_counts": dict(sorted(event_kinds.items())),
        "segmentation": {
            "pages_segmented": len(segmentation),
            "total_lines_detected": sum(entry["text_lines"] for entry in segmentation),
            "total_crops_written": sum(entry["input_crops"] for entry in segmentation),
            "pages": segmentation,
        },
        "per_method": per_method,
        "method_runs": sorted(method_runs.values(), key=lambda r: r["method_run_id"]),
        "reliability_issues": issues,
    }


def write_report(paths: RunPaths) -> Path:
    """Regenerates and writes `<run>/reliability-run-report.json`, reading only the durable log."""
    from archivetrust.infrastructure.storage.telemetry_sink import (  # noqa: PLC0415
        FileTelemetrySink,
    )

    sink = FileTelemetrySink(paths.events, blob_dir=paths.blobs)
    report = build_report(sink.all_events(), run_id=paths.run_id)
    paths.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths.report
