"""Document lifecycle and processing run projections for operator UI."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.application.progress import ProcessingProgressKind
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    HumanCorrectionApplied,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
    ReviewOutcomeRecorded,
)
from archivetrust.presentation.display_names import document_label, provider_label, short_ref
from archivetrust.presentation.time_format import format_timestamp


class LifecycleState(str, Enum):
    IMPORTED = "Imported"
    WAITING = "Waiting"
    PROCESSING = "Processing"
    NEEDS_ATTENTION = "Needs attention"
    COMPLETE = "Complete"
    FAILED = "Failed"
    INTERRUPTED = "Interrupted"
    NO_OUTPUT = "No output"


class ProviderAttemptSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    version: str
    outcome: str
    observations: int | None
    failure_reason: str | None
    recorded_at: str


class ReviewHistoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: str
    outcome: str
    correction_ref: str | None
    recorded_at: str


class CanonicalHistoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    recorded_at: str


class DocumentRunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    state: str
    started_at: str
    finished_at: str | None
    duration: str
    had_failure: bool
    failure_reason: str | None


class DocumentLifecycleRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    archive_object_ref: str | None
    display_name: str
    state: LifecycleState
    next_action: str
    warnings: tuple[str, ...] = ()


class DocumentLifecycleDetail(DocumentLifecycleRow):
    source_status: str
    original_archived: bool
    canonical_snapshot_available: bool
    output_status: str
    provider_attempts: tuple[ProviderAttemptSummary, ...]
    review_history: tuple[ReviewHistoryItem, ...]
    canonical_history: tuple[CanonicalHistoryItem, ...]
    processing_runs: tuple[DocumentRunSummary, ...]
    technical_reference: str


class ProcessingRunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    state: str
    started_at: str
    finished_at: str | None
    providers: tuple[str, ...]
    documents_total: int
    documents_completed: int
    documents_failed: int
    failure_reason: str | None


def document_lifecycle_rows(
    *,
    processing,
    acquisition_manager,
    telemetry_source,
    processing_progress=None,
) -> tuple[DocumentLifecycleRow, ...]:
    return tuple(
        _row_from_detail(detail)
        for detail in document_lifecycle_details(
            processing=processing,
            acquisition_manager=acquisition_manager,
            telemetry_source=telemetry_source,
            processing_progress=processing_progress,
        )
    )


def document_lifecycle_details(
    *,
    processing,
    acquisition_manager,
    telemetry_source,
    processing_progress=None,
) -> tuple[DocumentLifecycleDetail, ...]:
    docs = _document_index(processing=processing, acquisition_manager=acquisition_manager, telemetry_source=telemetry_source)
    progress_events = tuple(processing_progress.events()) if processing_progress is not None else ()
    telemetry_events = tuple(telemetry_source.all_events()) if telemetry_source is not None else ()
    review_docs = {item.document_ref for item in processing.review_queue()}
    canonical_docs = processing.documents_with_canonical_snapshots()

    details = []
    for document_ref, archive_object_ref in sorted(docs.items(), key=lambda item: item[0]):
        archive_object = _archive_object(acquisition_manager, archive_object_ref)
        display_name = document_label(document_ref, archive_object=archive_object)
        provider_attempts = _provider_attempts(document_ref, telemetry_events)
        review_history = _review_history(document_ref, telemetry_events)
        canonical_history = _canonical_history(document_ref, telemetry_events)
        runs = _document_runs(archive_object_ref or document_ref, progress_events)
        state, warnings = _document_state(
            document_ref=document_ref,
            archive_object_ref=archive_object_ref,
            acquisition_manager=acquisition_manager,
            provider_attempts=provider_attempts,
            review_docs=review_docs,
            canonical_docs=canonical_docs,
            runs=runs,
        )
        details.append(
            DocumentLifecycleDetail(
                document_ref=document_ref,
                archive_object_ref=archive_object_ref,
                display_name=display_name,
                state=state,
                next_action=_next_action(state),
                warnings=warnings,
                source_status="Original archived" if archive_object_ref else "No archive object linked",
                original_archived=archive_object_ref is not None,
                canonical_snapshot_available=document_ref in canonical_docs,
                output_status=(
                    "Ready for JSON export"
                    if document_ref in canonical_docs
                    else "No export available"
                ),
                provider_attempts=provider_attempts,
                review_history=review_history,
                canonical_history=canonical_history,
                processing_runs=runs,
                technical_reference=archive_object_ref or document_ref,
            )
        )
    return tuple(details)


def processing_run_history(processing_progress=None) -> tuple[ProcessingRunSummary, ...]:
    if processing_progress is None:
        return ()
    events = tuple(processing_progress.events())
    by_run: dict[str, list[Any]] = {}
    for event in events:
        by_run.setdefault(event.run_id, []).append(event)

    rows: list[ProcessingRunSummary] = []
    for run_id, run_events in by_run.items():
        start = next((e for e in run_events if e.kind is ProcessingProgressKind.RUN_STARTED), None)
        finish = next((e for e in reversed(run_events) if e.kind is ProcessingProgressKind.RUN_FINISHED), None)
        pause = next((e for e in reversed(run_events) if e.kind is ProcessingProgressKind.RUN_PAUSED), None)
        completed = [e for e in run_events if e.kind is ProcessingProgressKind.DOCUMENT_COMPLETED]
        failed = [e for e in completed if e.had_failure]
        terminal = (
            getattr(finish.run_terminal_state, "value", finish.run_terminal_state)
            if finish is not None
            else None
        )
        state = (
            terminal.replace("_", " ").title()
            if terminal
            else "Complete"
            if finish is not None
            else "Paused"
            if pause is not None
            else "Interrupted"
        )
        rows.append(
            ProcessingRunSummary(
                run_id=run_id,
                state=state,
                started_at=format_timestamp(start.recorded_at) if start is not None else "unknown",
                finished_at=format_timestamp(finish.recorded_at) if finish is not None else None,
                providers=tuple(provider_label(provider_id) for provider_id in (start.provider_ids if start else ())),
                documents_total=start.pending_count or len(completed) if start is not None else len(completed),
                documents_completed=len(completed),
                documents_failed=len(failed),
                failure_reason=pause.paused_reason if pause is not None else _first_failure(failed),
            )
        )
    return tuple(sorted(rows, key=lambda row: row.started_at, reverse=True))


def _row_from_detail(detail: DocumentLifecycleDetail) -> DocumentLifecycleRow:
    return DocumentLifecycleRow(
        document_ref=detail.document_ref,
        archive_object_ref=detail.archive_object_ref,
        display_name=detail.display_name,
        state=detail.state,
        next_action=detail.next_action,
        warnings=detail.warnings,
    )


def _document_index(*, processing, acquisition_manager, telemetry_source) -> dict[str, str | None]:
    docs: dict[str, str | None] = {}
    for archive_object in acquisition_manager.pending:
        docs[archive_object.id] = archive_object.id
    for item in processing.queue():
        docs[item.document_ref] = item.archive_object_ref
    for event in telemetry_source.all_events():
        docs.setdefault(event.document_ref, None)
        if isinstance(event, CanonicalDocumentCreated):
            docs[event.document_ref] = event.canonical_document.archive_object_ref
    return docs


def _archive_object(acquisition_manager, archive_object_ref: str | None):
    if archive_object_ref is None:
        return None
    return acquisition_manager.archive_object_by_ref(archive_object_ref)


def _provider_attempts(document_ref: str, events: tuple[Any, ...]) -> tuple[ProviderAttemptSummary, ...]:
    attempts = []
    for event in events:
        if not isinstance(event, ProviderObservationAttempted) or event.document_ref != document_ref:
            continue
        attempts.append(
            ProviderAttemptSummary(
                provider=provider_label(event.provider_id),
                version=event.provider_version,
                outcome=_provider_outcome(event),
                observations=event.observation_count,
                failure_reason=event.failure_reason,
                recorded_at=format_timestamp(event.recorded_at),
            )
        )
    return tuple(attempts)


def _provider_outcome(event: ProviderObservationAttempted) -> str:
    if event.outcome is ProviderInvocationOutcome.FAILED:
        return "Failed"
    if event.outcome is ProviderInvocationOutcome.NO_OBSERVATIONS:
        return "No content"
    if event.outcome is ProviderInvocationOutcome.PRODUCED_OBSERVATIONS:
        return "Produced observations"
    return "Recorded before outcome tracking"


def _review_history(document_ref: str, events: tuple[Any, ...]) -> tuple[ReviewHistoryItem, ...]:
    return tuple(
        ReviewHistoryItem(
            action=event.action.replace("_", " ").title(),
            outcome=event.outcome.value.title(),
            correction_ref=event.correction_id,
            recorded_at=format_timestamp(event.recorded_at),
        )
        for event in events
        if isinstance(event, ReviewOutcomeRecorded) and event.document_ref == document_ref
    )


def _canonical_history(document_ref: str, events: tuple[Any, ...]) -> tuple[CanonicalHistoryItem, ...]:
    history: list[CanonicalHistoryItem] = []
    for event in events:
        if event.document_ref != document_ref:
            continue
        if isinstance(event, CanonicalDecisionCreated):
            history.append(
                CanonicalHistoryItem(
                    summary=f"Canonical fact: {event.canonical_observation.observation_type.value.replace('_', ' ').title()}",
                    recorded_at=format_timestamp(event.recorded_at),
                )
            )
        elif isinstance(event, HumanCorrectionApplied):
            history.append(CanonicalHistoryItem(summary="Human correction applied", recorded_at=format_timestamp(event.recorded_at)))
        elif isinstance(event, CanonicalDocumentCreated):
            history.append(CanonicalHistoryItem(summary="Canonical snapshot assembled", recorded_at=format_timestamp(event.recorded_at)))
    return tuple(history)


def _document_runs(archive_object_ref: str, events: tuple[Any, ...]) -> tuple[DocumentRunSummary, ...]:
    runs: list[DocumentRunSummary] = []
    by_run: dict[str, list[Any]] = {}
    for event in events:
        if event.archive_object_id == archive_object_ref or event.kind in {
            ProcessingProgressKind.RUN_STARTED,
            ProcessingProgressKind.RUN_FINISHED,
            ProcessingProgressKind.RUN_PAUSED,
        }:
            by_run.setdefault(event.run_id, []).append(event)
    for run_id, run_events in by_run.items():
        started = next(
            (e for e in run_events if e.kind is ProcessingProgressKind.DOCUMENT_STARTED and e.archive_object_id == archive_object_ref),
            None,
        )
        completed = next(
            (e for e in reversed(run_events) if e.kind is ProcessingProgressKind.DOCUMENT_COMPLETED and e.archive_object_id == archive_object_ref),
            None,
        )
        if started is None and completed is None:
            continue
        finished = completed.recorded_at if completed is not None else None
        state = "Complete" if completed is not None and not completed.had_failure else "Failed" if completed is not None else "Interrupted"
        runs.append(
            DocumentRunSummary(
                run_id=run_id,
                state=state,
                started_at=format_timestamp(started.recorded_at) if started is not None else "unknown",
                finished_at=format_timestamp(finished) if finished is not None else None,
                duration=f"{completed.duration_seconds:.1f}s" if completed is not None and completed.duration_seconds is not None else "-",
                had_failure=bool(completed.had_failure) if completed is not None else False,
                failure_reason=completed.failure_reason if completed is not None else None,
            )
        )
    return tuple(runs)


def _document_state(
    *,
    document_ref: str,
    archive_object_ref: str | None,
    acquisition_manager,
    provider_attempts: tuple[ProviderAttemptSummary, ...],
    review_docs: set[str],
    canonical_docs: frozenset[str],
    runs: tuple[DocumentRunSummary, ...],
) -> tuple[LifecycleState, tuple[str, ...]]:
    warnings: list[str] = []
    if any(attempt.outcome == "Failed" for attempt in provider_attempts):
        warnings.append("At least one reading engine failed.")
    if document_ref in review_docs:
        return LifecycleState.NEEDS_ATTENTION, tuple(warnings)
    if any(run.state == "Failed" for run in runs):
        return LifecycleState.FAILED, tuple(warnings)
    if any(run.state == "Interrupted" for run in runs) and not any(run.state == "Complete" for run in runs):
        return LifecycleState.INTERRUPTED, tuple(warnings)
    if document_ref in canonical_docs:
        return LifecycleState.COMPLETE, tuple(warnings)
    if any(run.state == "Complete" for run in runs):
        warnings.append("Processing completed without a canonical snapshot.")
        return LifecycleState.NO_OUTPUT, tuple(warnings)
    if archive_object_ref and any(obj.id == archive_object_ref for obj in acquisition_manager.pending):
        return LifecycleState.WAITING, tuple(warnings)
    return LifecycleState.IMPORTED, tuple(warnings)


def _next_action(state: LifecycleState) -> str:
    return {
        LifecycleState.IMPORTED: "Process queue",
        LifecycleState.WAITING: "Process queue",
        LifecycleState.PROCESSING: "Monitor run",
        LifecycleState.NEEDS_ATTENTION: "Open work queue",
        LifecycleState.COMPLETE: "Inspect document or export JSON",
        LifecycleState.FAILED: "Inspect failure reason",
        LifecycleState.INTERRUPTED: "Resume processing",
        LifecycleState.NO_OUTPUT: "Inspect provider attempts",
    }[state]


def _first_failure(events: list[Any]) -> str | None:
    for event in events:
        if event.failure_reason:
            return event.failure_reason
    return None
