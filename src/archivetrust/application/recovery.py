"""Durable processing retry and restart-reconciliation policy."""

from __future__ import annotations

from dataclasses import dataclass

from archivetrust.application.progress import (
    DocumentProcessingState,
    FailureDisposition,
    ProcessingProgressEvent,
    ProcessingProgressKind,
    ProcessingRunTerminalState,
    utc_now_iso,
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 30.0

    def backoff(self, completed_attempts: int) -> float:
        return min(
            self.maximum_backoff_seconds,
            self.base_backoff_seconds * (2 ** max(0, completed_attempts - 1)),
        )


def classify_processing_failure(error: Exception) -> FailureDisposition:
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return FailureDisposition.RETRYABLE
    return FailureDisposition.TERMINAL


def reconcile_incomplete_runs(progress_sink) -> tuple[str, ...]:
    """Append terminal interruption facts for every historical run left open by a crash."""

    run_ids = incomplete_run_ids(progress_sink)
    events = progress_sink.events()
    by_run: dict[str, list[ProcessingProgressEvent]] = {}
    for event in events:
        by_run.setdefault(event.run_id, []).append(event)
    reconciled: list[str] = []
    for run_id in run_ids:
        run_events = by_run[run_id]
        completed = {
            event.archive_object_id
            for event in run_events
            if event.kind is ProcessingProgressKind.DOCUMENT_COMPLETED
        }
        in_flight = {
            event.archive_object_id: event
            for event in run_events
            if event.kind is ProcessingProgressKind.DOCUMENT_STARTED
            and event.archive_object_id not in completed
        }
        for archive_object_id, started in in_flight.items():
            progress_sink.record(
                ProcessingProgressEvent(
                    kind=ProcessingProgressKind.DOCUMENT_INTERRUPTED,
                    run_id=run_id,
                    recorded_at=utc_now_iso(),
                    archive_object_id=archive_object_id,
                    original_filename=started.original_filename,
                    document_state=DocumentProcessingState.INTERRUPTED,
                    failure_reason="worker stopped before a terminal document event",
                )
            )
        progress_sink.record(
            ProcessingProgressEvent(
                kind=ProcessingProgressKind.RUN_FINISHED,
                run_id=run_id,
                recorded_at=utc_now_iso(),
                documents_done=len(completed),
                run_terminal_state=ProcessingRunTerminalState.INTERRUPTED,
            )
        )
        reconciled.append(run_id)
    return tuple(reconciled)


def incomplete_run_ids(progress_sink) -> tuple[str, ...]:
    by_run: dict[str, list[ProcessingProgressEvent]] = {}
    for event in progress_sink.events():
        by_run.setdefault(event.run_id, []).append(event)
    return tuple(
        run_id
        for run_id, events in by_run.items()
        if any(event.kind is ProcessingProgressKind.RUN_STARTED for event in events)
        and not any(event.kind is ProcessingProgressKind.RUN_FINISHED for event in events)
    )
