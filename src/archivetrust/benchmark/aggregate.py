"""Benchmark aggregation (Benchmark & Measurement Framework, 2026-07-13).

Every function here is a pure, reproducible read over telemetry ArchiveTrust already records —
`TelemetrySource` (Trust Engine domain events), a `RuntimeTelemetryEvent` sequence (runtime/GPU),
a `ProcessingProgressEvent` sequence (run/document wall-clock), and optionally a
`ReviewInteractionSink` (human review effort). No provider is invoked, no behaviour is changed —
this module only reads what already happened (the framework's own mandate: "measurement, not
optimization").

Wherever the existing Learning Platform analytics (`learning/analytics/*.py`) already compute a
figure, this module calls them rather than recomputing — `provider_health`, `TelemetryIndex`, and
`review.triage.review_reason_for` (the exact function the live Review Center uses) are reused
directly, so a benchmark run's numbers can never silently diverge from what the running system
itself would compute.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime

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
from archivetrust.benchmark.metrics import Metric, derived, estimated, measured, unavailable
from archivetrust.application.progress import ProcessingProgressEvent, ProcessingProgressKind
from archivetrust.domain.calibration.coverage import find_blind_spots
from archivetrust.domain.calibration.progress import CalibrationMaturity
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    HumanCorrectionApplied,
    HumanCorrectionSubmitted,
    ObservationCreated,
    ProviderFailureCategory,
    ProviderInvocationOutcome,
)
from archivetrust.learning.analytics.index import ProviderRef, TelemetryIndex
from archivetrust.learning.analytics.provider_health import provider_health
from archivetrust.learning.review.session import sessions_from_interactions
from archivetrust.learning.review.sink import ReviewInteractionSink
from archivetrust.learning.source import TelemetrySource
from archivetrust.review.sampling.discovery import by_classification_and_type
from archivetrust.review.sampling.log import SamplingLogSink
from archivetrust.review.sampling.progress import compute_calibration_progress_map
from archivetrust.review.triage import TriagePolicy, review_reason_for
from archivetrust.runtime.runtime_telemetry import RuntimeTelemetryEvent, RuntimeTelemetryKind

_CONFIDENCE_BUCKETS = (
    ("High (>=0.90)", 0.90),
    ("Good (0.70-0.90)", 0.70),
    ("Fair (0.50-0.70)", 0.50),
    ("Low (<0.50)", 0.0),
)


def _bucket_confidences(values: list[float]) -> dict[str, int]:
    buckets = {label: 0 for label, _ in _CONFIDENCE_BUCKETS}
    buckets["Not scored"] = 0
    for value in values:
        for label, floor in _CONFIDENCE_BUCKETS:
            if value >= floor:
                buckets[label] += 1
                break
    return buckets


def _slot_document_refs(source: TelemetrySource) -> dict[str, str]:
    """`semantic_slot_id -> document_ref`, the one join `TelemetryIndex` doesn't expose (it tracks
    canonical *content* across documents, never which document a slot belongs to). A minimal,
    single-purpose scan rather than widening the shared index for one caller."""
    mapping: dict[str, str] = {}
    for event in source.all_events():
        if isinstance(event, CanonicalDecisionCreated):
            mapping[event.canonical_observation.semantic_slot_id] = event.document_ref
        elif isinstance(event, HumanCorrectionApplied):
            mapping[event.resulting_canonical_observation.semantic_slot_id] = event.document_ref
    return mapping


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# -- 1. Dataset metrics ------------------------------------------------------------------------


def dataset_metrics(
    source: TelemetrySource,
    *,
    progress_events: tuple[ProcessingProgressEvent, ...] = (),
    document_mime_types: dict[str, str] | None = None,
) -> DatasetMetrics:
    index = TelemetryIndex.build(source)
    documents: set[str] = set()
    max_page_by_document: dict[str, int] = {}
    for event in source.all_events():
        documents.add(event.document_ref)
        if isinstance(event, ObservationCreated):
            page = getattr(event.observation.payload, "page_number", None)
            if isinstance(page, int):
                current = max_page_by_document.get(event.document_ref, 0)
                max_page_by_document[event.document_ref] = max(current, page)

    doc_count = len(documents)
    pages_total = sum(max_page_by_document.values())

    documents_metric: Metric[int] = measured(doc_count, note="distinct document_ref in the stream")
    if pages_total > 0:
        pages_metric: Metric[int] = derived(
            pages_total,
            note="sum of the highest PagePayload.page_number seen per document (assumes "
            "1-indexed, sequential paging)",
        )
        avg_pages_metric: Metric[float] = derived(
            pages_total / doc_count if doc_count else 0.0,
            note="pages_processed / documents_processed",
        )
    else:
        pages_metric = unavailable(
            note="no ObservationCreated event carried a PagePayload with page_number"
        )
        avg_pages_metric = unavailable(note="pages_processed is unavailable")

    if document_mime_types:
        counts: dict[str, int] = defaultdict(int)
        for doc in documents:
            counts[document_mime_types.get(doc, "unknown")] += 1
        types_metric: Metric[dict[str, int]] = estimated(
            dict(counts), note="mime_type used as a proxy — no document-type field exists"
        )
    else:
        types_metric = unavailable(
            note="no document-type field exists on ArchiveObject; supply document_mime_types "
            "for an mime_type-based estimate"
        )

    run_started = next(
        (e for e in progress_events if e.kind == ProcessingProgressKind.RUN_STARTED), None
    )
    run_finished = next(
        (e for e in progress_events if e.kind == ProcessingProgressKind.RUN_FINISHED), None
    )
    if run_started is not None and run_finished is not None:
        start, end = _parse_iso(run_started.recorded_at), _parse_iso(run_finished.recorded_at)
        duration_metric: Metric[float] = (
            measured((end - start).total_seconds(), unit="s")
            if start and end
            else unavailable(note="run start/finish timestamps could not be parsed")
        )
    else:
        duration_metric = unavailable(
            note="no ProcessingRunStarted/ProcessingRunFinished pair in processing.jsonl for this run"
        )

    document_durations = [
        e.duration_seconds
        for e in progress_events
        if e.kind == ProcessingProgressKind.DOCUMENT_COMPLETED and e.duration_seconds is not None
    ]
    avg_document_duration: Metric[float] = (
        measured(
            statistics.mean(document_durations), unit="s",
            note=f"mean of {len(document_durations)} DocumentProcessingCompleted.duration_seconds",
        )
        if document_durations
        else unavailable(note="no DocumentProcessingCompleted event carried a duration_seconds")
    )

    return DatasetMetrics(
        documents_processed=documents_metric,
        pages_processed=pages_metric,
        average_pages_per_document=avg_pages_metric,
        document_types=types_metric,
        processing_duration_seconds=duration_metric,
        average_document_processing_seconds=avg_document_duration,
    )


# -- 2. Runtime metrics (per provider) ---------------------------------------------------------


def _model_matches_provider(model_id: str, provider_id: str) -> bool:
    a, b = model_id.lower(), provider_id.lower()
    return a == b or a in b or b in a


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over already-collected values — no numpy dependency, no
    external library, just `sorted()` and arithmetic (per this milestone's "no new external
    dependencies" constraint)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct * (len(ordered) - 1)
    lower, upper = int(rank), min(int(rank) + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def provider_runtime_metrics(
    source: TelemetrySource,
    *,
    runtime_events: tuple[RuntimeTelemetryEvent, ...] = (),
) -> tuple[ProviderRuntimeMetrics, ...]:
    health = provider_health(source)
    index = TelemetryIndex.build(source)
    attempts_by_ref: dict[ProviderRef, list] = defaultdict(list)
    for attempt in index.provider_attempts:
        attempts_by_ref[ProviderRef(attempt.provider_id, attempt.provider_version)].append(attempt)

    results = []
    for h in health:
        failure_rate: Metric[float] = (
            derived(h.failed_invocations / h.invocation_count, note="failed_invocations / invocation_count")
            if h.invocation_count
            else unavailable(note="provider was never invoked")
        )

        attempts = attempts_by_ref.get(ProviderRef(h.provider_id, h.provider_version), [])
        # Runtime Warm-Up Verification (2026-07-14): `ProviderObservationAttempted.cold_start`
        # marks the one invocation per runtime that paid `VLLMRuntime.warm_up()`'s cost (lazy by
        # design -- `RuntimeManager.get_or_create` never forces a container to start just because
        # a provider was activated, only the first real `infer()` call does, and that call's own
        # duration includes however long warm-up took). Folding that invocation into the same
        # avg/median/p95/throughput figures as every steady-state invocation would let a single
        # one-time cost distort what those figures are meant to measure. Excluded here, not
        # discarded -- `cold_startup_seconds` below still reports it, separately and honestly.
        steady_state = [a for a in attempts if a.cold_start is not True]
        cold_start_count = len(attempts) - len(steady_state)
        durations_s = [a.duration_ms / 1000 for a in steady_state if a.duration_ms is not None]
        per_page_s = [
            a.duration_ms / 1000 / a.pages_processed
            for a in steady_state
            if a.duration_ms is not None and a.pages_processed
        ]
        if durations_s:
            avg_runtime_doc: Metric[float] = measured(
                statistics.mean(durations_s), unit="s",
                note=(
                    f"mean ProviderObservationAttempted.duration_ms over {len(durations_s)} "
                    f"steady-state invocation(s)"
                    + (f" ({cold_start_count} cold-start invocation(s) excluded)" if cold_start_count else "")
                ),
            )
            median_runtime: Metric[float] = measured(statistics.median(durations_s), unit="s")
            p95_runtime: Metric[float] = measured(_percentile(durations_s, 0.95), unit="s")
            peak_runtime: Metric[float] = measured(max(durations_s), unit="s")
            throughput_docs: Metric[float] = derived(
                3600 / statistics.mean(durations_s), unit="documents/hour",
                note="3600 / avg_runtime_per_document_seconds",
            )
        else:
            no_timing = (
                "every invocation for this provider was a cold start (excluded above)"
                if attempts and cold_start_count == len(attempts)
                else "no ProviderObservationAttempted for this provider carried a duration_ms"
            )
            avg_runtime_doc = unavailable(note=no_timing)
            median_runtime = unavailable(note=no_timing)
            p95_runtime = unavailable(note=no_timing)
            peak_runtime = unavailable(note=no_timing)
            throughput_docs = unavailable(note=no_timing)
        if per_page_s:
            avg_runtime_page: Metric[float] = measured(
                statistics.mean(per_page_s), unit="s",
                note=f"mean of duration_ms/pages_processed over {len(per_page_s)} PAGE_IMAGE invocation(s)",
            )
            throughput_pages: Metric[float] = derived(
                3600 / statistics.mean(per_page_s), unit="pages/hour",
                note="3600 / avg_runtime_per_page_seconds",
            )
        else:
            no_page_timing = (
                "no invocation for this provider carried both duration_ms and pages_processed "
                "(DOCUMENT-kind adapters like Docling do not record pages_processed today)"
            )
            avg_runtime_page = unavailable(note=no_page_timing)
            throughput_pages = unavailable(note=no_page_timing)

        retry_counts = [a.retry_count for a in attempts if a.retry_count is not None]
        retry_metric: Metric[int] = (
            measured(
                sum(retry_counts),
                note=f"sum of retry_count over {len(retry_counts)} invocation(s) — always 0 today, "
                "since the pipeline invokes each adapter exactly once",
            )
            if retry_counts
            else unavailable(note="no invocation for this provider carried a retry_count")
        )

        exception_failures = sum(
            1 for a in attempts if a.failure_category == ProviderFailureCategory.EXCEPTION
        )
        crash_rate: Metric[float] = (
            derived(
                exception_failures / len(attempts),
                note="fraction of invocations whose failure_category was EXCEPTION — an "
                "invocation-level exception escaping the adapter contract, not an OS-level "
                "process crash (which remains unmeasured)",
            )
            if attempts
            else unavailable(note="provider was never invoked")
        )

        related = [
            e for e in runtime_events if _model_matches_provider(e.model_id, h.provider_id)
        ]
        warmups = [e.seconds for e in related if e.kind == RuntimeTelemetryKind.WARMUP_TIME and e.seconds is not None]
        cold_startup: Metric[float] = (
            measured(statistics.mean(warmups), unit="s", note=f"mean of {len(warmups)} WarmupTime event(s)")
            if warmups
            else unavailable(note="no WarmupTime runtime event matched this provider's model_id")
        )

        gpu_events = [
            e.reserved_bytes
            for e in related
            if e.kind == RuntimeTelemetryKind.GPU_ALLOCATED and e.reserved_bytes is not None
        ]
        peak_gpu: Metric[int] = (
            measured(max(gpu_events), unit="bytes", note="max GPUAllocated.reserved_bytes for a matched model_id")
            if gpu_events
            else unavailable(note="no GPUAllocated event matched this provider's model_id")
        )

        restarts = sum(1 for e in related if e.kind == RuntimeTelemetryKind.RUNTIME_RESTARTED)
        start_failed = sum(1 for e in related if e.kind == RuntimeTelemetryKind.RUNTIME_START_FAILED)
        recovery: Metric[float] = (
            estimated(
                restarts / (restarts + start_failed),
                note="RuntimeRestarted / (RuntimeRestarted + RuntimeStartFailed) for a matched "
                "model_id — a proxy, not a direct 'recovery attempted -> succeeded' record",
            )
            if (restarts + start_failed) > 0
            else unavailable(note="no restart or start-failure runtime events matched this provider")
        )

        results.append(
            ProviderRuntimeMetrics(
                provider_id=h.provider_id,
                provider_version=h.provider_version,
                avg_runtime_per_document_seconds=avg_runtime_doc,
                avg_runtime_per_page_seconds=avg_runtime_page,
                median_runtime_seconds=median_runtime,
                p95_runtime_seconds=p95_runtime,
                peak_runtime_seconds=peak_runtime,
                throughput_documents_per_hour=throughput_docs,
                throughput_pages_per_hour=throughput_pages,
                failure_rate=failure_rate,
                crash_rate=crash_rate,
                retry_count=retry_metric,
                recovery_success_rate=recovery,
                gpu_utilization_percent=unavailable(
                    note="only VRAM bytes are measured (GPUAllocated.reserved_bytes); no compute "
                    "utilization percentage is recorded"
                ),
                peak_gpu_memory_bytes=peak_gpu,
                peak_cpu_memory_bytes=unavailable(note="no CPU memory measurement exists in telemetry"),
                cpu_utilization_percent=unavailable(note="no CPU utilization measurement exists in telemetry"),
                cold_startup_seconds=cold_startup,
                warm_startup_seconds=unavailable(
                    note="RuntimeReused has no adoption-latency field; only cold-start WarmupTime is recorded"
                ),
            )
        )
    return tuple(results)


# -- 3. OCR quality metrics (per provider) -----------------------------------------------------


def provider_quality_metrics(source: TelemetrySource) -> tuple[ProviderQualityMetrics, ...]:
    index = TelemetryIndex.build(source)
    health = {ProviderRef(h.provider_id, h.provider_version): h for h in provider_health(source)}

    produced_by_ref: dict[ProviderRef, int] = defaultdict(int)
    for attempt in index.provider_attempts:
        if attempt.outcome == ProviderInvocationOutcome.PRODUCED_OBSERVATIONS and attempt.observation_count:
            produced_by_ref[ProviderRef(attempt.provider_id, attempt.provider_version)] += attempt.observation_count

    confidences_by_ref: dict[ProviderRef, list[float]] = defaultdict(list)
    for event in source.all_events():
        if isinstance(event, ObservationCreated):
            obs = event.observation
            if obs.provider_confidence is not None:
                confidences_by_ref[ProviderRef(obs.provider_id, obs.provider_version)].append(
                    obs.provider_confidence.value
                )

    slots_by_ref: dict[ProviderRef, list[ComparisonClassification]] = defaultdict(list)
    unique_by_ref: dict[ProviderRef, int] = defaultdict(int)
    review_triggers_by_ref: dict[ProviderRef, int] = defaultdict(int)
    for canonical in index.latest_canonicals():
        refs = index.contributing_providers(canonical)
        classification = canonical.comparison_confidence.classification
        triggers_review = review_reason_for(canonical) is not None
        for ref in refs:
            slots_by_ref[ref].append(classification)
            if triggers_review:
                review_triggers_by_ref[ref] += 1
        if classification == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE and len(refs) == 1:
            unique_by_ref[next(iter(refs))] += 1

    results = []
    for ref, h in health.items():
        classifications = slots_by_ref.get(ref, [])
        n = len(classifications)
        agreement: Metric[float] = (
            derived(
                sum(1 for c in classifications if c == ComparisonClassification.CORROBORATED) / n,
                note="fraction of this provider's contributing slots classified CORROBORATED",
            )
            if n
            else unavailable(note="provider contributed to no slot")
        )
        disagreement: Metric[float] = (
            derived(
                sum(1 for c in classifications if c == ComparisonClassification.CONTESTED) / n,
                note="fraction of this provider's contributing slots classified CONTESTED",
            )
            if n
            else unavailable(note="provider contributed to no slot")
        )

        confidences = confidences_by_ref.get(ref, [])
        avg_confidence: Metric[float] = (
            measured(statistics.mean(confidences), note=f"mean of {len(confidences)} ProviderConfidence value(s)")
            if confidences
            else unavailable(note="no ObservationCreated event for this provider carried a ProviderConfidence")
        )
        confidence_dist: Metric[dict[str, int]] = (
            measured(_bucket_confidences(confidences))
            if confidences
            else unavailable(note="no ProviderConfidence values to bucket")
        )

        completeness: Metric[float] = (
            derived(
                1 - (h.no_observation_invocations / h.invocation_count),
                note="1 - (no_observation_invocations / invocation_count)",
            )
            if h.invocation_count
            else unavailable(note="provider was never invoked")
        )

        results.append(
            ProviderQualityMetrics(
                provider_id=h.provider_id,
                provider_version=h.provider_version,
                observations_produced=measured(
                    produced_by_ref.get(ref, 0), note="sum of ProviderObservationAttempted.observation_count"
                ),
                observations_accepted=measured(
                    h.contributing_slot_count, note="distinct slots this provider contributed to (provider_health)"
                ),
                observations_rejected=measured(h.rejection_count, note="EvidenceRejected count (provider_health)"),
                agreement_with_canonical=agreement,
                disagreement_rate=disagreement,
                missing_observations=measured(
                    h.no_observation_invocations, note="no_observation_invocations (provider_health)"
                ),
                unique_observations_contributed=measured(
                    unique_by_ref.get(ref, 0),
                    note="slots where this provider was the sole UNCORROBORATED_SINGLE_SOURCE contributor",
                ),
                average_confidence=avg_confidence,
                confidence_distribution=confidence_dist,
                observation_completeness=completeness,
                review_trigger_rate=(
                    derived(
                        review_triggers_by_ref.get(ref, 0) / n,
                        note="fraction of this provider's contributing slots that triage would queue for review",
                    )
                    if n
                    else unavailable(note="provider contributed to no slot")
                ),
            )
        )
    results.sort(key=lambda m: (m.provider_id, m.provider_version))
    return tuple(results)


def observation_type_comparison(source: TelemetrySource) -> tuple[ObservationTypeMetrics, ...]:
    """One row per `ObservationType` actually seen: which provider most often contributed to that
    type's CORROBORATED slots, and the type's overall agreement/confidence. Corpus-wide by nature
    — "best at tables" is a cross-provider comparison, not a per-provider figure."""
    index = TelemetryIndex.build(source)
    by_type: dict[str, list] = defaultdict(list)
    for canonical in index.latest_canonicals():
        by_type[canonical.observation_type.value].append(canonical)

    rows = []
    for obs_type, canonicals in sorted(by_type.items()):
        n = len(canonicals)
        agreement = sum(
            1 for c in canonicals if c.comparison_confidence.classification == ComparisonClassification.CORROBORATED
        ) / n

        contribution_counts: dict[ProviderRef, int] = defaultdict(int)
        for c in canonicals:
            for ref in index.contributing_providers(c):
                contribution_counts[ref] += 1
        best = max(contribution_counts.items(), key=lambda kv: kv[1], default=None)
        best_metric: Metric[str] = (
            derived(f"{best[0].provider_id} ({best[0].provider_version})", note=f"contributed to {best[1]}/{n} slots of this type")
            if best is not None
            else unavailable(note="no provider attribution recorded for this type")
        )

        confidences = [c.canonical_confidence.value for c in canonicals if c.canonical_confidence is not None]
        confidence_metric: Metric[float] = (
            measured(statistics.mean(confidences), note=f"mean Canonical Confidence across {len(confidences)} slot(s)")
            if confidences
            else unavailable(note="no slot of this type has been scored yet")
        )

        rows.append(
            ObservationTypeMetrics(
                observation_type=obs_type,
                best_provider=best_metric,
                agreement=derived(agreement, note="fraction of this type's slots classified CORROBORATED"),
                confidence=confidence_metric,
            )
        )
    return tuple(rows)


# -- 4. Trust metrics ----------------------------------------------------------------------------


def trust_metrics(source: TelemetrySource) -> TrustMetrics:
    index = TelemetryIndex.build(source)
    canonicals = index.latest_canonicals()
    n = len(canonicals)

    confidences = [c.canonical_confidence.value for c in canonicals if c.canonical_confidence is not None]
    avg_confidence: Metric[float] = (
        measured(statistics.mean(confidences), note=f"mean over {len(confidences)} scored slot(s)")
        if confidences
        else unavailable(note="no slot has been scored yet")
    )
    confidence_dist: Metric[dict[str, int]] = (
        measured(_bucket_confidences(confidences)) if n else unavailable(note="no canonical slots yet")
    )

    agreement_counts: dict[str, int] = defaultdict(int)
    for c in canonicals:
        agreement_counts[c.comparison_confidence.classification.value] += 1
    agreement_dist: Metric[dict[str, int]] = (
        measured(dict(agreement_counts)) if n else unavailable(note="no canonical slots yet")
    )
    conflict_count = agreement_counts.get(ComparisonClassification.CONTESTED.value, 0)

    evidence_density: Metric[float] = (
        derived(
            statistics.mean(len(c.contributing_observations) for c in canonicals),
            note="mean count of contributing_observations per canonical slot",
        )
        if n
        else unavailable(note="no canonical slots yet")
    )

    agreeing_sizes = [
        len(index.contributing_providers(c))
        for c in canonicals
        if c.comparison_confidence.classification == ComparisonClassification.CORROBORATED
    ]
    disagreeing_sizes = [
        len(index.contributing_providers(c))
        for c in canonicals
        if c.comparison_confidence.classification == ComparisonClassification.CONTESTED
    ]
    avg_agreeing: Metric[float] = (
        derived(statistics.mean(agreeing_sizes), note="mean distinct providers per CORROBORATED slot")
        if agreeing_sizes
        else unavailable(note="no CORROBORATED slots yet")
    )
    avg_disagreeing: Metric[float] = (
        derived(statistics.mean(disagreeing_sizes), note="mean distinct providers per CONTESTED slot")
        if disagreeing_sizes
        else unavailable(note="no CONTESTED slots yet")
    )

    low = sum(1 for v in confidences if v < 0.5)
    high = sum(1 for v in confidences if v >= 0.9)

    by_type: dict[str, list[float]] = defaultdict(list)
    for c in canonicals:
        if c.canonical_confidence is not None:
            by_type[c.observation_type.value].append(c.canonical_confidence.value)
    confidence_by_type: Metric[dict[str, float]] = (
        measured({t: statistics.mean(v) for t, v in by_type.items()}) if by_type else unavailable(note="no scored slots yet")
    )

    return TrustMetrics(
        average_canonical_confidence=avg_confidence,
        confidence_distribution=confidence_dist,
        agreement_distribution=agreement_dist,
        conflict_count=measured(conflict_count) if n else unavailable(note="no canonical slots yet"),
        evidence_density=evidence_density,
        average_providers_agreeing=avg_agreeing,
        average_providers_disagreeing=avg_disagreeing,
        low_confidence_observations=measured(low) if confidences else unavailable(note="no scored slots yet"),
        high_confidence_observations=measured(high) if confidences else unavailable(note="no scored slots yet"),
        confidence_by_observation_type=confidence_by_type,
    )


# -- 5. Human review metrics ----------------------------------------------------------------------

_BASELINE_POLICY = TriagePolicy(
    version=1, queue_single_source=True, single_source_review_types=None
)
"""Triage Policy v1's documented behaviour, restored explicitly (`review/triage.py`'s own
`single_source_review_types=None` docstring) — queue every uncorroborated slot unconditionally.
The "review reduction" baseline this framework compares production policy against."""


def human_review_metrics(
    source: TelemetrySource,
    *,
    interaction_sink: ReviewInteractionSink | None = None,
    triage_policy: TriagePolicy | None = None,
) -> HumanReviewMetrics:
    index = TelemetryIndex.build(source)
    canonicals = index.latest_canonicals()
    n = len(canonicals)
    policy = triage_policy or TriagePolicy()
    slot_documents = _slot_document_refs(source)

    review_slots = [c for c in canonicals if review_reason_for(c, policy) is not None]
    review_documents = {slot_documents[c.semantic_slot_id] for c in review_slots if c.semantic_slot_id in slot_documents}
    baseline_review_slots = [c for c in canonicals if review_reason_for(c, _BASELINE_POLICY) is not None]

    review_items_total = len(review_slots)
    docs_entering = len(review_documents)

    corrections = index.corrections_submitted
    accepted = sum(1 for c in corrections if c.action == "accept")
    overrides = sum(1 for c in corrections if c.action != "accept")
    durations = [c.review_duration_seconds for c in corrections if c.review_duration_seconds is not None]

    if interaction_sink is not None:
        sessions = sessions_from_interactions(tuple(interaction_sink.all_interactions()))
        session_durations = [s.review_duration for s in sessions if s.review_duration is not None]
        if session_durations:
            durations = durations + session_durations

    return HumanReviewMetrics(
        documents_entering_review=measured(docs_entering, note="distinct documents with >=1 triaged slot"),
        review_percentage=(
            derived(review_items_total / n, note="review_items_total / total canonical slots")
            if n
            else unavailable(note="no canonical slots yet")
        ),
        review_items_total=measured(review_items_total, note=f"triage policy v{policy.version}"),
        average_review_items_per_document=(
            derived(review_items_total / docs_entering, note="review_items_total / documents_entering_review")
            if docs_entering
            else unavailable(note="no documents entered review")
        ),
        average_review_time_seconds=(
            measured(statistics.mean(durations), unit="s", note=f"mean over {len(durations)} recorded review(s)")
            if durations
            else unavailable(note="no HumanCorrectionSubmitted.review_duration_seconds or ReviewSession recorded")
        ),
        human_corrections=measured(len(corrections)),
        human_overrides=measured(overrides, note="corrections whose action was not 'accept'"),
        human_agreement_with_canonical=(
            derived(accepted / len(corrections), note="fraction of corrections that were plain ACCEPT")
            if corrections
            else unavailable(note="no corrections submitted yet")
        ),
        review_reduction_vs_baseline=(
            derived(
                1 - (review_items_total / len(baseline_review_slots)),
                note=f"1 - (v{policy.version} review items / Triage Policy v1 baseline review items); "
                f"baseline would have queued {len(baseline_review_slots)} slot(s)",
            )
            if baseline_review_slots
            else unavailable(note="baseline policy would also queue nothing (no uncorroborated/contested slots)")
        ),
    )


# -- 6. Operational metrics ------------------------------------------------------------------------


def operational_metrics(
    source: TelemetrySource,
    *,
    runtime_events: tuple[RuntimeTelemetryEvent, ...] = (),
    progress_events: tuple[ProcessingProgressEvent, ...] = (),
) -> OperationalMetrics:
    restarts = sum(1 for e in runtime_events if e.kind == RuntimeTelemetryKind.RUNTIME_RESTARTED)
    start_failed = sum(1 for e in runtime_events if e.kind == RuntimeTelemetryKind.RUNTIME_START_FAILED)

    attempted_total = 0
    for event in source.all_events():
        if (
            hasattr(event, "outcome")
            and getattr(event, "outcome", None) == ProviderInvocationOutcome.PRODUCED_OBSERVATIONS
            and getattr(event, "observation_count", None) is not None
        ):
            attempted_total += event.observation_count
    created_total = sum(1 for event in source.all_events() if isinstance(event, ObservationCreated))
    completeness: Metric[float] = (
        derived(
            min(created_total / attempted_total, 1.0),
            note="ObservationCreated count / sum(ProviderObservationAttempted.observation_count) "
            "for PRODUCED_OBSERVATIONS outcomes — an attribution-closure check, not a correctness check",
        )
        if attempted_total
        else unavailable(note="no ProviderObservationAttempted carried an observation_count to check against")
    )

    failed_attempts = [
        event
        for event in source.all_events()
        if getattr(event, "outcome", None) == ProviderInvocationOutcome.FAILED
    ]
    timeout_attempts = [e for e in failed_attempts if getattr(e, "timeout", None) is True]
    unlabeled_failed = sum(1 for e in failed_attempts if getattr(e, "timeout", None) is None)
    if not failed_attempts:
        timeouts_metric: Metric[int] = unavailable(
            note="no ProviderObservationAttempted with outcome=FAILED to check for a timeout flag"
        )
    elif unlabeled_failed == 0:
        timeouts_metric = measured(
            len(timeout_attempts), note="count of ProviderObservationAttempted with timeout=True"
        )
    else:
        timeouts_metric = derived(
            len(timeout_attempts),
            note=f"count of ProviderObservationAttempted with timeout=True; {unlabeled_failed} FAILED "
            "attempt(s) predate this field and could not be checked, so this is a lower bound",
        )

    run_finished_events = [e for e in progress_events if e.kind == ProcessingProgressKind.RUN_FINISHED]
    if run_finished_events:
        last = run_finished_events[-1]
        total = (last.documents_done or 0) + (last.pending_count or 0)
        completion: Metric[float] = (
            derived(last.documents_done / total, note="documents_done / (documents_done + pending_count) at run finish")
            if total
            else unavailable(note="run finished with zero documents in scope")
        )
    else:
        completion = unavailable(note="no ProcessingRunFinished event recorded — run may still be in progress or crashed")

    return OperationalMetrics(
        container_restarts=measured(restarts),
        provider_startup_failures=measured(start_failed),
        resource_exhaustion_events=unavailable(
            note="GPU/resource exhaustion raises an exception (e.g. GPUOversubscriptionError); "
            "it is not recorded as a telemetry event"
        ),
        timeouts=timeouts_metric,
        telemetry_completeness=completeness,
        pipeline_completion_rate=completion,
        resume_success_rate=unavailable(note="no resume/checkpoint feature exists yet (documented readiness gap)"),
        queue_utilization=unavailable(note="no queue-depth telemetry is recorded"),
    )


# -- 8. Calibration metrics (Milestone 11) --------------------------------------------------------


def calibration_metrics(
    source: TelemetrySource,
    sampling_log: SamplingLogSink,
) -> CalibrationMetrics:
    """Confidence calibration state across the corpus (Phases 4/7/10) — built entirely from
    `domain.calibration` and `review.sampling.progress`, which in turn only count
    `ReviewIntent.CALIBRATION` evidence, never operational corrections (see
    `review/sampling/progress.py`'s module docstring for why).
    """
    index = TelemetryIndex.build(source)
    patterns = by_classification_and_type(index)

    if not patterns:
        unavailable_note = "no canonical slots yet"
        return CalibrationMetrics(
            patterns_total=measured(0),
            patterns_with_calibration_evidence=measured(0),
            patterns_statistically_mature=measured(0),
            calibration_reviews_total=measured(0),
            average_progress_pct_of_loosest_requirement=unavailable(note=unavailable_note),
            blind_spots_total=measured(0),
            blind_spot_corpus_share=unavailable(note=unavailable_note),
        )

    progress_map = compute_calibration_progress_map(index, sampling_log)
    total_population = sum(p.population for p in patterns)

    with_evidence = sum(1 for p in progress_map.values() if p.reviews_completed > 0)
    mature = sum(1 for p in progress_map.values() if p.maturity == CalibrationMaturity.STATISTICALLY_MATURE)
    reviews_total = sum(p.reviews_completed for p in progress_map.values())
    pct_values = [
        p.progress_pct_of_loosest_requirement
        for p in progress_map.values()
        if p.progress_pct_of_loosest_requirement is not None
    ]

    blind_spots = find_blind_spots(patterns, progress_map, total_population=total_population)
    blind_share = sum(b.corpus_share for b in blind_spots)

    return CalibrationMetrics(
        patterns_total=measured(len(patterns)),
        patterns_with_calibration_evidence=measured(with_evidence),
        patterns_statistically_mature=measured(mature),
        calibration_reviews_total=measured(reviews_total),
        average_progress_pct_of_loosest_requirement=(
            derived(
                statistics.mean(pct_values),
                note=f"mean of reviews_completed/required_n(+/-10%) over {len(pct_values)} pattern(s)",
            )
            if pct_values
            else unavailable(note="no pattern has a nonzero +/-10% requirement")
        ),
        blind_spots_total=measured(len(blind_spots)),
        blind_spot_corpus_share=derived(
            round(blind_share, 4),
            note=f"{len(blind_spots)} pattern(s) with zero calibration evidence, "
            f"{blind_share:.2%} of corpus slots",
        ),
    )
