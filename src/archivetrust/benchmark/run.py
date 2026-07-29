"""Benchmark runs — the persisted, comparable unit (Benchmark & Measurement Framework, 2026-07-13).

A `BenchmarkRun` is a snapshot of every metric group plus the metadata needed to make the run
*comparable* to another one later: software version, provider versions, runtime version, dataset
identity, date, hardware, configuration. `BenchmarkRunStore` persists runs the same way every other
durable stream in this codebase persists — append-only JSON Lines, one record per line, the file is
the single source of truth (`infrastructure.storage.telemetry_sink.FileTelemetrySink`,
`application.progress.FileProcessingProgressSink` — same shape, same discipline).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.benchmark.groups import (
    CalibrationMetrics,
    DatasetMetrics,
    HumanReviewMetrics,
    ObservationTypeMetrics,
    OperationalMetrics,
    ProviderQualityMetrics,
    ProviderRuntimeMetrics,
    TrustMetrics,
)


class BenchmarkMetadata(BaseModel):
    """Everything needed to tell two runs apart, or to know when two runs *should* be comparable
    and when they shouldn't (a run against a different dataset, or a different reconciliation
    policy version, is not a regression — it is a different experiment)."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_at: str
    """UTC ISO-8601, when the benchmark itself was computed (not when the underlying documents
    were processed — that is `DatasetMetrics.processing_duration_seconds`'s concern)."""
    software_version: str | None = None
    provider_versions: dict[str, str] = {}
    runtime_version: str | None = None
    dataset_id: str
    hardware: str | None = None
    configuration: dict[str, str] = {}
    reconciliation_policy_version: int | None = None
    capability_matrix_version: int | None = None
    confidence_policy_version: int | None = None
    triage_policy_version: int | None = None


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: BenchmarkMetadata
    dataset: DatasetMetrics
    runtime: tuple[ProviderRuntimeMetrics, ...]
    quality: tuple[ProviderQualityMetrics, ...]
    observation_types: tuple[ObservationTypeMetrics, ...]
    trust: TrustMetrics
    human_review: HumanReviewMetrics
    operational: OperationalMetrics
    calibration: CalibrationMetrics | None = None
    """`None` for runs persisted before Milestone 11 -- older `benchmark_runs.jsonl` rows still
    load (the field has a default), they simply carry no calibration section."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchmarkRunStore:
    """Append-only JSONL store of `BenchmarkRun`s, one per line — the persistence half of
    "persist benchmark runs, allow comparison between runs, allow trend analysis". Loads its full
    history into memory at construction (small: one row per benchmark run, not per document),
    mirroring `FileProcessingProgressSink`'s read-once-then-mirror design.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        self._runs: list[BenchmarkRun] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._runs.append(BenchmarkRun.model_validate(json.loads(line)))

    def append(self, run: BenchmarkRun) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(run.model_dump_json())
                handle.write("\n")
            self._runs.append(run)

    def runs(self) -> tuple[BenchmarkRun, ...]:
        with self._lock:
            return tuple(self._runs)

    def get(self, run_id: str) -> BenchmarkRun | None:
        return next((r for r in self.runs() if r.metadata.run_id == run_id), None)

    def for_dataset(self, dataset_id: str) -> tuple[BenchmarkRun, ...]:
        """Runs against the same dataset, oldest first — the sequence trend analysis walks."""
        return tuple(
            sorted(
                (r for r in self.runs() if r.metadata.dataset_id == dataset_id),
                key=lambda r: r.metadata.run_at,
            )
        )


class MetricDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    """Dotted path identifying the metric, e.g. `trust.average_canonical_confidence` or
    `quality[docling/1.0].agreement_with_canonical`."""
    before: Any
    after: Any
    before_status: str
    after_status: str


def flatten_metrics(run: BenchmarkRun) -> dict[str, tuple[object, str, str | None, str | None]]:
    """Flattens every `Metric` leaf in a run to `path -> (value, status, unit, note)`, for both
    comparison and export. Group models are pydantic; `model_dump` gives plain dicts/lists we can
    walk generically."""
    flat: dict[str, tuple[object, str, str | None, str | None]] = {}

    def walk(prefix: str, obj: object) -> None:
        if isinstance(obj, dict) and "status" in obj and "value" in obj and set(obj) <= {
            "status", "value", "unit", "note"
        }:
            flat[prefix] = (obj["value"], obj["status"], obj.get("unit"), obj.get("note"))
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                walk(f"{prefix}.{key}" if prefix else str(key), val)
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                label = i
                if isinstance(val, dict) and "provider_id" in val:
                    label = f"{val['provider_id']}/{val.get('provider_version', '?')}"
                elif isinstance(val, dict) and "observation_type" in val:
                    label = val["observation_type"]
                walk(f"{prefix}[{label}]", val)

    dumped = run.model_dump(mode="json")
    for group_name in (
        "dataset",
        "runtime",
        "quality",
        "observation_types",
        "trust",
        "human_review",
        "operational",
    ):
        walk(group_name, dumped[group_name])
    if dumped.get("calibration") is not None:
        walk("calibration", dumped["calibration"])
    return flat


def compare_runs(before: BenchmarkRun, after: BenchmarkRun) -> tuple[MetricDelta, ...]:
    """Every metric present in both runs, paired up — the mechanism behind "allow trend analysis".
    A metric present in only one run (e.g. a provider retired or added between runs) is omitted
    rather than compared against a fabricated baseline."""
    left = flatten_metrics(before)
    right = flatten_metrics(after)
    deltas = []
    for path in sorted(set(left) & set(right)):
        (before_value, before_status, _bu, _bn) = left[path]
        (after_value, after_status, _au, _an) = right[path]
        deltas.append(
            MetricDelta(
                path=path,
                before=before_value,
                after=after_value,
                before_status=before_status,
                after_status=after_status,
            )
        )
    return tuple(deltas)
