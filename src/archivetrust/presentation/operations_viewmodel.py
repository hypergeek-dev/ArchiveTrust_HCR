"""Processing Center ViewModel — the operational home (HUMAN_REVIEW_SPECIFICATION.md §16.2;
ROADMAP_V2.md §8).

Framework-independent. Reads the Trust Engine telemetry stream (a `TelemetrySource`) and the
existing Provider Health analytic, and projects them into the operational picture an operator needs
on landing: what has been processed, what is awaiting review, provider execution, failures, and
recent activity. It computes *view* aggregations (document/observation counts, queue) here in the
presentation layer — it neither changes backend behavior nor adds to the Learning Platform; where a
Learning Platform analytic already exists (Provider Health) it is reused, not reimplemented.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.application.progress import ProcessingProgressKind
from archivetrust.application.current_state import CurrentStateService
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    EvidenceRejected,
    HumanCorrectionApplied,
    ProviderObservationAttempted,
)
from archivetrust.learning.analytics.provider_health import ProviderHealth, provider_health
from archivetrust.learning.source import TelemetrySource
from archivetrust.presentation.operational_stats import TelemetryAggregate
from archivetrust.review.triage import review_reason_for


class ProcessingOverview(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents_processed: int
    observations_captured: int
    canonical_facts: int
    provider_invocations: int
    failures: int
    corrections_applied: int
    documents_awaiting_review: int
    uncertainties_awaiting_review: int


class QueueItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    archive_object_ref: str | None
    canonical_facts: int
    uncertainties: int


class ActivityItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    summary: str


class WorkspaceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    processing_profile: str


class InputSourceStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    kind: str
    enabled: bool
    status: str
    documents_imported: int


class StorageUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    archive_bytes: int
    derived_bytes: int


class ProviderAvailability(BaseModel):
    """Operational terminology, not implementation terminology (Priority 13): whether a configured
    provider is currently able to run at all, distinct from whether it produced anything on any
    particular document (that's `ProviderHealth`)."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    status: str
    """One of "ready", "disabled", "unavailable" -- never a raw boolean pair an operator has to
    interpret."""


class RecoveryStatus(BaseModel):
    """Operator-facing summary of whether the last processing run on this Workspace finished
    cleanly or was interrupted (Production Hardening, 2026-07-14 — Recovery UX): read from the
    durable `processing_progress` stream alone, in operational language, never raw event names or
    internal ids. `None` fields mean "not applicable" (e.g. no run has ever started), never a
    fabricated value.
    """

    model_config = ConfigDict(frozen=True)

    was_interrupted: bool
    """True when the most recent processing run recorded a start but never a
    `ProcessingRunFinished` — the exact shape of the 2026-07-14 incident."""
    documents_completed: int
    documents_remaining: int
    last_completed_document: str | None
    interrupted_document: str | None
    """The Archive Object's original filename that was started but never completed, if any."""
    interrupted_at: str | None
    """ISO-8601 UTC timestamp `interrupted_document` was started, if any."""
    paused_reason: str | None = None
    """Provider Health Guard (Phase 22): set when `was_interrupted` is True *and* the durable
    stream's most recent run recorded a `ProcessingRunPaused` event -- distinguishes "gracefully
    paused, here's why" from an unexplained crash (`was_interrupted=True`, `paused_reason=None`),
    per Constitution Article 18 (silence must be distinguishable from failure)."""


class GpuUsageSummary(BaseModel):
    """Real-time VRAM usage (Long-Running Status, 2026-07-14) -- read from
    `GPUResourceManager`'s own live bookkeeping (`active_reservations()`/`total_memory_bytes()`),
    never a separate probe: two sources of truth for the same GPU would be able to disagree.
    `None` fields mean VRAM could not be measured (no `nvidia-smi`/no GPU), never a fabricated 0.
    """

    model_config = ConfigDict(frozen=True)

    total_bytes: int | None
    reserved_bytes: int
    reservations: tuple[str, ...]
    """`model_id` of every provider currently holding a GPU reservation."""


class PipelineState(str, Enum):
    HEALTHY = "healthy"
    RECOVERING = "recovering"
    WARNING = "warning"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    IDLE = "idle"


class PipelineHealth(BaseModel):
    """The single "is everything okay" signal (Operational Health, 2026-07-14) an operator should
    be able to read at a glance, derived only from facts this ViewModel already has elsewhere
    (recovery status, queue failures, provider failures, remaining work) -- never a new source of
    truth, just their combination named in operational language.
    """

    model_config = ConfigDict(frozen=True)

    state: PipelineState
    reason: str


class ThroughputStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents_measured: int
    """How many `DocumentProcessingCompleted` events (Phase 34) carried a real
    `duration_seconds` -- 0 before any document has completed under the processing-progress
    stream, never a fabricated rate from no data. An interrupted/abandoned attempt never reaches
    `DocumentProcessingCompleted` and so is never counted here, by construction."""
    average_seconds_per_document: float | None
    documents_per_minute: float | None
    estimated_seconds_remaining: float | None
    """`documents_waiting * average_seconds_per_document`, or `None` when there is no measured
    average yet to project from."""


class ProcessingCenterViewModel:
    """Snapshot-style: `overview()`, `queue()`, `provider_execution()`, `recent_activity()` each
    scan the current telemetry stream. The stream is append-only, so re-reading is always safe and
    reflects any corrections applied since the last read.

    `workspace`/`acquisition_manager` are optional so this ViewModel stays constructible wherever a
    Workspace isn't relevant yet (e.g. a bare telemetry-only test) — when given, they back the
    Workspace Selector / Input Sources / Documents Waiting / Storage Usage sections ROADMAP.md
    §5.13 adds to the Processing Center.
    """

    def __init__(
        self,
        source: TelemetrySource,
        *,
        workspace=None,
        acquisition_manager=None,
        queue_failure_count: int = 0,
        enabled_provider_ids: frozenset[str] | None = None,
        configured_provider_ids: frozenset[str] | None = None,
        aggregate: TelemetryAggregate | None = None,
        processing_progress=None,
        gpu_resource_manager=None,
        current_state: CurrentStateService | None = None,
    ) -> None:
        self._source = source
        self._current_state = current_state or CurrentStateService(source)
        self._processing_progress = processing_progress
        """The durable `ProcessingProgressEvent` stream (`application.progress`), if attached --
        backs `recovery_status()`. `None` in contexts with no per-Workspace progress sink (bare
        telemetry-only tests, pre-Workspace state)."""
        self._gpu_resource_manager = gpu_resource_manager
        """`runtime.gpu_resource_manager.GPUResourceManager`, if attached -- backs `gpu_usage()`.
        `None` wherever no GPU-backed runtime is in play (bare telemetry-only tests)."""
        self._aggregate = aggregate if aggregate is not None else TelemetryAggregate()
        """Incremental aggregation over the stream (Production Hardening Review, 2026-07-13) --
        when the caller shares one instance across ViewModel constructions (`AppContext` does,
        per Workspace), each read costs O(events appended since the last read) instead of a full
        rescan of the entire history, the quadratic GUI-thread cost that hung the 2026-07-13
        production run. A privately-constructed aggregate (the default) is aggregated from index
        0 on first read -- identical results, just without cross-refresh reuse."""
        self._workspace = workspace
        self._queue_failure_count = queue_failure_count
        self._acquisition_manager = acquisition_manager
        self._enabled_provider_ids = enabled_provider_ids or frozenset()
        """Providers this Workspace would actually invoke right now (enabled *and* available) --
        `AppContext.enabled_provider_ids()`'s output, passed through so this view can distinguish
        "ran" from "configured but currently unavailable" without duplicating that computation."""
        self._configured_provider_ids = configured_provider_ids or frozenset()
        """Every provider registered for this Workspace, enabled or not, available or not --
        `ProviderRegistry.all()`'s ids. The superset against which "unavailable"/"skipped" is
        computed (Priority 13: absence must be visible, not just what ran)."""

    # -- internal scan -----------------------------------------------------------------------

    def _stats(self) -> TelemetryAggregate:
        """The up-to-date aggregate: consumes whatever the stream appended since the last read
        (O(new events)), then serves every aggregate below from memory."""
        self._aggregate.update(self._source.all_events())
        return self._aggregate

    def _latest_canonicals_by_document(self):
        """Latest Canonical Observation per (document, semantic slot), so supersessions (including
        human corrections) are reflected and never double-counted."""
        return {
            (state.document_ref, slot.semantic_slot_id): slot.current
            for state in self._current_state.documents()
            for slot in state.slots
        }

    @staticmethod
    def _is_uncertain(canonical) -> bool:
        """Delegates to the review triage policy itself (Operational Hardening milestone) — the
        "Review required" tile and the actual review queue must count the same slots, so this view
        can never report a workload the Review Center doesn't have (or hide one it does)."""
        return review_reason_for(canonical) is not None

    # -- public surface ----------------------------------------------------------------------

    def overview(self) -> ProcessingOverview:
        stats = self._stats()
        # Any recorded event marks a document as processed: a document where every provider found
        # nothing produces attempt telemetry but no CanonicalDocumentCreated (Operational Hardening
        # milestone) and is still a completed, countable run — not an invisible one.
        uncertain_by_doc: dict[str, int] = {}
        for (document_ref, _slot), canonical in stats.latest_canonicals.items():
            if self._is_uncertain(canonical):
                uncertain_by_doc[document_ref] = uncertain_by_doc.get(document_ref, 0) + 1
        return ProcessingOverview(
            documents_processed=len(stats.documents),
            observations_captured=stats.observations_captured,
            canonical_facts=len(stats.latest_canonicals),
            provider_invocations=stats.provider_invocations,
            failures=stats.failures,
            corrections_applied=stats.corrections_applied,
            documents_awaiting_review=len(uncertain_by_doc),
            uncertainties_awaiting_review=sum(uncertain_by_doc.values()),
        )

    def _archive_object_ref_by_document(self) -> dict[str, str]:
        return {
            state.document_ref: state.archive_object_ref
            for state in self._current_state.documents()
        }

    def documents_with_canonical_snapshots(self) -> frozenset[str]:
        """Documents with an assembled canonical snapshot, from the shared read-model aggregate."""
        return frozenset(
            state.document_ref
            for state in self._current_state.documents()
            if state.latest_canonical_document is not None
        )

    def queue(self) -> tuple[QueueItem, ...]:
        latest = self._latest_canonicals_by_document()
        archive_refs = self._archive_object_ref_by_document()
        facts: dict[str, int] = {}
        uncertain: dict[str, int] = {}
        for (document_ref, _slot), canonical in latest.items():
            facts[document_ref] = facts.get(document_ref, 0) + 1
            if self._is_uncertain(canonical):
                uncertain[document_ref] = uncertain.get(document_ref, 0) + 1
        items = [
            QueueItem(
                document_ref=document_ref,
                archive_object_ref=archive_refs.get(document_ref),
                canonical_facts=facts[document_ref],
                uncertainties=uncertain.get(document_ref, 0),
            )
            for document_ref in facts
        ]
        # Most uncertainty first — where the operator's attention is needed.
        items.sort(key=lambda i: (-i.uncertainties, i.document_ref))
        return tuple(items)

    def review_queue(self) -> tuple[QueueItem, ...]:
        """Only the documents actually awaiting review (uncertainties > 0) — the Review Center's
        document picker (Operational Completion milestone): "documents requiring review should
        automatically appear," satisfied by a live read of this same, already-real-time queue."""
        return tuple(item for item in self.queue() if item.uncertainties > 0)

    def provider_execution(self) -> tuple[ProviderHealth, ...]:
        return provider_health(self._source)

    def provider_execution_live(self) -> tuple[ProviderHealth, ...]:
        """The cheap per-provider columns (invocations / no-content / failed / rejections) from the
        incremental aggregate — O(1) after `_stats()`, for use *mid-run* where the full
        `provider_execution()` analytic (which rebuilds a `TelemetryIndex` over the entire stream
        per call) would put unbounded work back on the GUI thread (Production Hardening Review,
        2026-07-13). Contribution/correction figures are reported as no-data (`correction_rate`
        `None`, rendered "—"), never fabricated — the full analytic still runs on idle refreshes.
        """
        stats = self._stats()
        results = [
            ProviderHealth(
                provider_id=provider_id,
                provider_version=provider_version,
                invocation_count=counters[0],
                rejection_count=counters[3],
                contributing_slot_count=0,
                corrected_slot_count=0,
                no_observation_invocations=counters[1],
                failed_invocations=counters[2],
            )
            for (provider_id, provider_version), counters in stats.provider_counters.items()
        ]
        results.sort(key=lambda h: (h.provider_id, h.provider_version))
        return tuple(results)

    def provider_availability(self) -> tuple[ProviderAvailability, ...]:
        """Every configured provider's current operational status (Priority 13) -- so a provider
        that never ran because it was disabled or unavailable is visible here, not only absent
        from `provider_execution()`'s per-invocation health rows."""
        return tuple(
            ProviderAvailability(
                provider_id=provider_id,
                status="ready" if provider_id in self._enabled_provider_ids else "unavailable",
            )
            for provider_id in sorted(self._configured_provider_ids)
        )

    def observations_per_document(self) -> float | None:
        """Average `ObservationCreated` count per document processed -- a near-zero value is the
        "this document produced almost nothing" tell an operator should notice (Priority 13)."""
        overview = self.overview()
        if overview.documents_processed == 0:
            return None
        return overview.observations_captured / overview.documents_processed

    def throughput(self) -> ThroughputStats:
        """Processing duration/throughput/ETA computed from the operational processing-progress
        stream's own `DocumentProcessingCompleted.duration_seconds` (Phase 34) -- the authoritative,
        already-recorded wall-clock time `QueueWorker` itself measured via `time.monotonic()` from
        a document's `DOCUMENT_STARTED` to its `DOCUMENT_COMPLETED` (`application/progress.py`).

        Phase 33 found the previous version (spanning a document's earliest-to-latest *domain*
        telemetry event, `TelemetryAggregate.first_seen`/`last_seen`) silently included real-world
        time that was never processing at all: an interrupted run's abandoned first attempt anchors
        `first_seen` while a later, unrelated retry supplies `last_seen` (counting the gap between
        sessions as "processing"), and a human reviewing a document long after it finished pushes
        `last_seen` forward by however long review took to get to (counting review latency as
        processing latency) -- reproduced live, both mechanisms, against a real workspace
        (`docs/PHASE_33_DESKTOP_PIPELINE_ORCHESTRATION_INVESTIGATION_2026-07-16.md`). Neither
        failure mode is possible here: an interrupted attempt never reaches `DOCUMENT_COMPLETED` at
        all (excluded, not miscounted), and human review is a different telemetry stream entirely
        that this method never reads. This is the "operational dashboard metrics must originate
        from operational telemetry" principle this phase establishes -- `TelemetryAggregate.
        first_seen`/`last_seen` remain in place for whatever else may read them (unchanged,
        unremoved); this method simply stops being one of their consumers.

        `None` (via the empty-`durations` branch) wherever no progress stream is attached, or one
        is attached but no document has completed under it yet -- never a fabricated rate from no
        data, matching this method's prior no-data contract exactly.
        """
        durations: list[float] = []
        if self._processing_progress is not None:
            for event in self._processing_progress.events():
                if event.kind is ProcessingProgressKind.DOCUMENT_COMPLETED and event.duration_seconds is not None:
                    durations.append(event.duration_seconds)

        if not durations:
            return ThroughputStats(
                documents_measured=0,
                average_seconds_per_document=None,
                documents_per_minute=None,
                estimated_seconds_remaining=None,
            )
        average = sum(durations) / len(durations)
        remaining = self.documents_waiting()
        return ThroughputStats(
            documents_measured=len(durations),
            average_seconds_per_document=average,
            documents_per_minute=(60.0 / average) if average > 0 else None,
            estimated_seconds_remaining=(remaining * average) if remaining else 0.0,
        )

    def recent_activity(self, limit: int = 12) -> tuple[ActivityItem, ...]:
        stats = self._stats()
        summaries: list[ActivityItem] = []
        for document_ref, event in stats.activity:
            summary = summarize_event(event)
            if summary is not None:
                summaries.append(ActivityItem(document_ref=document_ref, summary=summary))
        return tuple(summaries[-limit:][::-1])  # newest first

    # -- Workspace & Acquisition Management additions (ROADMAP.md §5.13) ---------------------

    def current_workspace_summary(self) -> WorkspaceSummary | None:
        if self._workspace is None:
            return None
        return WorkspaceSummary(
            name=self._workspace.name,
            description=self._workspace.description,
            processing_profile=self._workspace.processing_profile.value,
        )

    def input_sources_status(self) -> tuple[InputSourceStatus, ...]:
        if self._acquisition_manager is None:
            return ()
        return tuple(
            InputSourceStatus(
                source_id=status.source_id,
                kind=status.kind.value,
                enabled=status.enabled,
                status=status.health.status,
                documents_imported=status.health.documents_imported,
            )
            for status in self._acquisition_manager.source_status()
        )

    def documents_waiting(self) -> int:
        """Registered Archive Objects not yet handed to the Trust Engine. `0` when no Acquisition
        Manager is attached."""
        if self._acquisition_manager is None:
            return 0
        return len(self._acquisition_manager.pending)

    def documents_processing(self) -> int:
        """Always `0` today: `WorkspaceProcessingService.process_pending()` runs synchronously, so
        there is currently no observable in-between "processing" state to report — an honestly-noted
        gap (`IMPLEMENTATION_STATUS.md`), not a hidden assumption."""
        return 0

    def documents_completed(self) -> int:
        return self.overview().documents_processed

    def documents_failed(self) -> int:
        """`EvidenceRejected` telemetry (a provider rejecting one page mid-run) plus documents the
        queue worker could not process at all this session (e.g. no provider was both enabled and
        available) -- the latter never reaches telemetry (Article 16), so without it this tile
        would dishonestly read `0` even when every document in a run failed."""
        return self.overview().failures + self._queue_failure_count

    def recovery_status(self) -> RecoveryStatus | None:
        """Reconstructs "what happened, what completed, what's left, what's next" purely from the
        durable processing-progress stream — the Recovery UX this Production Incident (2026-07-14)
        motivated: an operator reopening a Workspace after a crash should see this, not a log file.
        `None` when no progress stream is attached or no run has ever started.
        """
        if self._processing_progress is None:
            return None
        events = self._processing_progress.events()
        if not events:
            return None

        last_run_started_index = max(
            (i for i, e in enumerate(events) if e.kind.value == "ProcessingRunStarted"),
            default=None,
        )
        if last_run_started_index is None:
            return None
        tail = events[last_run_started_index:]
        finish = next(
            (e for e in reversed(tail) if e.kind.value == "ProcessingRunFinished"), None
        )
        was_interrupted = finish is None or getattr(
            finish.run_terminal_state, "value", finish.run_terminal_state
        ) == "interrupted"

        completed_filenames_by_id: dict[str, str] = {}
        started_filenames_by_id: dict[str, str] = {}
        started_at_by_id: dict[str, str] = {}
        last_completed_filename: str | None = None
        for event in events:
            if event.kind.value == "DocumentProcessingStarted" and event.archive_object_id:
                started_filenames_by_id[event.archive_object_id] = event.original_filename or ""
                started_at_by_id[event.archive_object_id] = event.recorded_at
            elif event.kind.value == "DocumentProcessingCompleted" and event.archive_object_id:
                completed_filenames_by_id[event.archive_object_id] = event.original_filename or ""
                last_completed_filename = event.original_filename

        interrupted_document: str | None = None
        interrupted_at: str | None = None
        if was_interrupted:
            never_completed = [
                aid for aid in started_filenames_by_id if aid not in completed_filenames_by_id
            ]
            if never_completed:
                # The queue processes one document at a time, so at most one should ever be
                # in-flight when a run is interrupted; take the most recently started if more
                # than one somehow shows up, rather than guessing.
                latest = max(never_completed, key=lambda aid: started_at_by_id[aid])
                interrupted_document = started_filenames_by_id[latest]
                interrupted_at = started_at_by_id[latest]

        paused_reason: str | None = None
        if was_interrupted:
            paused_event = next(
                (e for e in reversed(tail) if e.kind.value == "ProcessingRunPaused"), None
            )
            if paused_event is not None:
                paused_reason = paused_event.paused_reason

        return RecoveryStatus(
            was_interrupted=was_interrupted,
            documents_completed=len(completed_filenames_by_id),
            documents_remaining=self.documents_waiting(),
            last_completed_document=last_completed_filename,
            interrupted_document=interrupted_document,
            interrupted_at=interrupted_at,
            paused_reason=paused_reason,
        )

    def gpu_usage(self) -> GpuUsageSummary | None:
        """Current VRAM reservation state (Long-Running Status, 2026-07-14). `None` when no GPU
        resource manager is attached; `total_bytes=None` when VRAM itself could not be measured on
        this machine, distinct from "measured and zero"."""
        if self._gpu_resource_manager is None:
            return None
        reservations = self._gpu_resource_manager.active_reservations()
        return GpuUsageSummary(
            total_bytes=self._gpu_resource_manager.total_memory_bytes(),
            reserved_bytes=sum(r.reserved_bytes for r in reservations),
            reservations=tuple(r.model_id for r in reservations),
        )

    def pipeline_health(self, *, is_running: bool = False) -> PipelineHealth:
        """The single glance-and-know signal (Operational Health, 2026-07-14), derived only from
        facts already computed elsewhere on this ViewModel -- never a new measurement. `is_running`
        is passed in because whether a `QueueWorker` is currently active is Qt/session state this
        ViewModel does not itself hold (ROADMAP.md §5.13's existing separation).
        """
        recovery = self.recovery_status()
        if recovery is not None and recovery.was_interrupted and recovery.paused_reason is not None:
            return PipelineHealth(state=PipelineState.PAUSED, reason=recovery.paused_reason)
        if recovery is not None and recovery.was_interrupted:
            return PipelineHealth(
                state=PipelineState.RECOVERING,
                reason=f"Resumed after an interrupted run; {recovery.documents_remaining} documents remaining.",
            )
        if is_running:
            return PipelineHealth(state=PipelineState.HEALTHY, reason="Processing is in progress.")
        if self.documents_failed() > 0:
            return PipelineHealth(
                state=PipelineState.WARNING,
                reason=f"{self.documents_failed()} document(s) failed during processing.",
            )
        if self.documents_waiting() == 0 and self.documents_completed() > 0:
            return PipelineHealth(state=PipelineState.COMPLETED, reason="All documents have been processed.")
        if self.documents_waiting() > 0:
            return PipelineHealth(
                state=PipelineState.IDLE,
                reason=f"{self.documents_waiting()} document(s) waiting to be processed.",
            )
        return PipelineHealth(state=PipelineState.IDLE, reason="No documents in this Workspace yet.")

    def storage_usage(self, layout=None) -> StorageUsage:
        if layout is None:
            return StorageUsage(archive_bytes=0, derived_bytes=0)
        return StorageUsage(
            archive_bytes=_directory_size(layout.archive_dir),
            derived_bytes=_directory_size(layout.derived_dir),
        )


def _directory_size(directory) -> int:
    if not directory.exists():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def summarize_event(event) -> str | None:
    if isinstance(event, ProviderObservationAttempted):
        if event.outcome is None:  # recorded before outcomes were captured
            return f"Provider run: {event.provider_id}"
        if event.failure_reason is not None:
            return f"Provider {event.provider_id} failed: {event.failure_reason}"
        if not event.observation_count:
            return f"Provider {event.provider_id} found no content in this document"
        return f"Provider {event.provider_id}: {event.observation_count} observations"
    if isinstance(event, EvidenceRejected):
        return f"Rejected evidence ({event.provider_id}): {event.rejection_reason}"
    if isinstance(event, CanonicalDecisionCreated):
        return f"Canonical fact: {event.canonical_observation.observation_type.value}"
    if isinstance(event, HumanCorrectionApplied):
        return "Human correction applied"
    if isinstance(event, CanonicalDocumentCreated):
        return "Canonical Document assembled"
    return None
