"""Benchmark metric groups (Benchmark & Measurement Framework, 2026-07-13).

Six perspectives, kept as six distinct model families rather than one flat table — the framework's
first guiding principle is "never reduce a provider to a single score," and that discipline starts
in the data model, not just the report layout. Every leaf field is a `Metric[T]` (see `metrics.py`)
so its provenance travels with it through aggregation, persistence, and export.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.benchmark.metrics import Metric


class DatasetMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents_processed: Metric[int]
    pages_processed: Metric[int]
    average_pages_per_document: Metric[float]
    document_types: Metric[dict[str, int]]
    processing_duration_seconds: Metric[float]
    average_document_processing_seconds: Metric[float]
    """Mean of `ProcessingProgressEvent.duration_seconds` across every `DOCUMENT_COMPLETED` event
    (Observability milestone, 2026-07-13) — real per-document wall-clock, distinct from
    `processing_duration_seconds`'s whole-run span."""


class ProviderRuntimeMetrics(BaseModel):
    """Runtime/performance figures for one `(provider_id, provider_version)`."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str

    avg_runtime_per_document_seconds: Metric[float]
    avg_runtime_per_page_seconds: Metric[float]
    median_runtime_seconds: Metric[float]
    p95_runtime_seconds: Metric[float]
    peak_runtime_seconds: Metric[float]
    throughput_documents_per_hour: Metric[float]
    throughput_pages_per_hour: Metric[float]
    failure_rate: Metric[float]
    crash_rate: Metric[float]
    retry_count: Metric[int]
    recovery_success_rate: Metric[float]
    gpu_utilization_percent: Metric[float]
    peak_gpu_memory_bytes: Metric[int]
    peak_cpu_memory_bytes: Metric[int]
    cpu_utilization_percent: Metric[float]
    cold_startup_seconds: Metric[float]
    warm_startup_seconds: Metric[float]


class ProviderQualityMetrics(BaseModel):
    """OCR/observation quality figures for one `(provider_id, provider_version)`."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str

    observations_produced: Metric[int]
    observations_accepted: Metric[int]
    observations_rejected: Metric[int]
    agreement_with_canonical: Metric[float]
    disagreement_rate: Metric[float]
    missing_observations: Metric[int]
    unique_observations_contributed: Metric[int]
    average_confidence: Metric[float]
    confidence_distribution: Metric[dict[str, int]]
    observation_completeness: Metric[float]
    review_trigger_rate: Metric[float]
    """Fraction of this provider's contributing slots that triage (`review/triage.py`'s
    `review_reason_for` — the exact function the live Review Center uses) would send to a human
    reviewer. Not "how often this provider is wrong," but a fair per-provider view onto the
    corpus-wide review rate `HumanReviewMetrics.review_percentage` reports in aggregate."""


class ObservationTypeMetrics(BaseModel):
    """One row of the Observation-Type Comparison table — corpus-wide, not per provider, because
    "which provider is best at tables" is a cross-provider question by nature."""

    model_config = ConfigDict(frozen=True)

    observation_type: str
    best_provider: Metric[str]
    agreement: Metric[float]
    confidence: Metric[float]


class TrustMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    average_canonical_confidence: Metric[float]
    confidence_distribution: Metric[dict[str, int]]
    agreement_distribution: Metric[dict[str, int]]
    conflict_count: Metric[int]
    evidence_density: Metric[float]
    average_providers_agreeing: Metric[float]
    average_providers_disagreeing: Metric[float]
    low_confidence_observations: Metric[int]
    high_confidence_observations: Metric[int]
    confidence_by_observation_type: Metric[dict[str, float]]


class HumanReviewMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents_entering_review: Metric[int]
    review_percentage: Metric[float]
    review_items_total: Metric[int]
    average_review_items_per_document: Metric[float]
    average_review_time_seconds: Metric[float]
    human_corrections: Metric[int]
    human_overrides: Metric[int]
    human_agreement_with_canonical: Metric[float]
    review_reduction_vs_baseline: Metric[float]
    """First-class per the framework's mandate: how much triage policy v2 (production) reduces the
    review queue versus the documented v1 baseline that queued every uncorroborated slot
    unconditionally (`review/triage.py`'s own recorded history: "the first validation run put
    86-100% of all canonical facts into review"). Computed by re-running triage over the same
    telemetry under both policies — never a claim about a re-run that didn't happen."""


class OperationalMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    container_restarts: Metric[int]
    provider_startup_failures: Metric[int]
    resource_exhaustion_events: Metric[int]
    timeouts: Metric[int]
    telemetry_completeness: Metric[float]
    pipeline_completion_rate: Metric[float]
    resume_success_rate: Metric[float]
    queue_utilization: Metric[float]


class CalibrationMetrics(BaseModel):
    """Confidence calibration state across the corpus (Milestone 11, Phases 4/7/10) — computed
    from `domain.calibration` + `review.sampling.progress`, counting only
    `ReviewIntent.CALIBRATION` evidence (see `review/sampling/progress.py`'s module docstring for
    why operational corrections are excluded). Optional on `BenchmarkRun` so pre-Milestone-11 runs
    persisted to `benchmark_runs.jsonl` still load."""

    model_config = ConfigDict(frozen=True)

    patterns_total: Metric[int]
    patterns_with_calibration_evidence: Metric[int]
    patterns_statistically_mature: Metric[int]
    calibration_reviews_total: Metric[int]
    average_progress_pct_of_loosest_requirement: Metric[float]
    """Mean, across patterns, of `reviews_completed / required_n(+/-10%)` — the framework's
    headline "how far along is calibration overall" figure."""
    blind_spots_total: Metric[int]
    blind_spot_corpus_share: Metric[float]
    """Fraction of the corpus's slots that belong to a pattern with zero calibration evidence —
    Phase 7's coverage figure, reported here as one number rather than the full per-pattern list
    (see `scripts/calibration_dashboard.py` for the per-pattern detail)."""
