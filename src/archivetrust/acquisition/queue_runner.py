"""The queue-processing loop, Qt-free (extracted verbatim from `clients/desktop/queue_worker.py`).

One loop, two callers: the desktop `QueueWorker` (which wraps it in a `QThread` and forwards the
callbacks as Qt signals) and headless operators (campaign scripts, the F3 validation run) — so a
headless run records exactly the durable progress stream, health-guard behavior, and failure
isolation the desktop run does, rather than a script's approximation of it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from archivetrust.acquisition.provider_health_guard import ProviderHealthGuard
from archivetrust.application.progress import (
    DocumentProcessingState,
    FailureDisposition,
    ProcessingProgressEvent,
    ProcessingProgressKind,
    ProcessingRunTerminalState,
    utc_now_iso,
)
from archivetrust.application.recovery import (
    RetryPolicy,
    classify_processing_failure,
    reconcile_incomplete_runs,
)
from archivetrust.domain.shared.ids import new_id


@dataclass
class QueueRunCallbacks:
    """Optional per-step notifications; every field defaults to a no-op so headless callers only
    supply what they observe."""

    document_started: Callable[[str], None] = field(default=lambda _filename: None)
    document_completed: Callable[[str, bool], None] = field(default=lambda _filename, _had_failure: None)
    queue_progress: Callable[[int, int], None] = field(default=lambda _done, _total: None)
    activity: Callable[[str], None] = field(default=lambda _line: None)
    provider_health_alert: Callable[[str], None] = field(default=lambda _reason: None)
    pipeline_events: Callable[[tuple], None] = field(default=lambda _events: None)
    """The document's raw pipeline events, handed up after each successful `process()` call. How
    they are rendered (e.g. `presentation.operations_viewmodel.summarize_event` in the desktop
    client) is the caller's concern — this module must not import upward into presentation."""


@dataclass(frozen=True)
class QueueRunOutcome:
    run_id: str
    documents_done: int
    documents_total: int
    cancelled: bool
    paused_for_health: bool


def run_queue(
    context,
    *,
    paused: threading.Event | None = None,
    cancelled: threading.Event | None = None,
    callbacks: QueueRunCallbacks | None = None,
    health_guard: ProviderHealthGuard | None = None,
    retry_policy: RetryPolicy | None = None,
) -> QueueRunOutcome:
    """Processes the current Workspace's pending queue one Archive Object at a time.

    Behavior is exactly the pre-extraction `QueueWorker.run`: durable RUN_STARTED /
    DOCUMENT_STARTED / DOCUMENT_COMPLETED (with measured `duration_seconds`) / RUN_FINISHED or
    RUN_PAUSED progress events; health-guard check before every document; one bad document never
    stops the queue (it is counted in `context.queue_failure_count` and recorded with its
    failure reason). `paused`, when supplied, follows `QueueWorker`'s convention: *set* means
    "not paused".
    """
    callbacks = callbacks or QueueRunCallbacks()
    retry_policy = retry_policy or RetryPolicy()
    if paused is None:
        paused = threading.Event()
        paused.set()
    if cancelled is None:
        cancelled = threading.Event()

    def record(kind: ProcessingProgressKind, run_id: str, **fields) -> None:
        sink = context.processing_progress
        if sink is None:  # a context built before a Workspace was opened -- nothing to record to
            return
        sink.record(
            ProcessingProgressEvent(kind=kind, run_id=run_id, recorded_at=utc_now_iso(), **fields)
        )

    pending = context.acquisition_manager.pending
    if context.processing_progress is not None:
        reconcile_incomplete_runs(context.processing_progress)
    total = len(pending)
    done = 0
    failures_this_run = 0
    explicitly_cancelled: set[str] = set()
    service = context.processing_service()
    running_ids = context.enabled_provider_ids()
    run_id = new_id("processing_run")
    guard = health_guard or context.provider_health_guard(frozenset(running_ids))
    record(
        ProcessingProgressKind.RUN_STARTED, run_id,
        provider_ids=tuple(sorted(running_ids)), pending_count=total,
    )

    paused_for_health = False
    while pending and not cancelled.is_set():
        paused.wait()
        if cancelled.is_set():
            break

        guard_result = guard.check()
        if not guard_result.healthy:
            record(
                ProcessingProgressKind.RUN_PAUSED, run_id,
                paused_provider_id=guard_result.checks[0].provider_id if guard_result.checks else None,
                paused_reason=guard_result.reason,
                recovery_attempted=any(r.attempted for r in guard_result.recovery_attempts),
                recovery_succeeded=(
                    any(r.succeeded for r in guard_result.recovery_attempts)
                    if guard_result.recovery_attempts else None
                ),
            )
            callbacks.activity(guard_result.reason)
            callbacks.provider_health_alert(guard_result.reason)
            paused.clear()
            paused_for_health = True
            break  # do not process this or any further document in this run

        archive_object = pending[0]
        document_started_at = time.monotonic()
        callbacks.document_started(archive_object.original_filename)
        callbacks.activity(f"Acquiring {archive_object.original_filename}…")
        context.running_provider_ids.update(running_ids)
        record(
            ProcessingProgressKind.DOCUMENT_STARTED, run_id,
            archive_object_id=archive_object.id,
            original_filename=archive_object.original_filename,
            document_state=DocumentProcessingState.PROCESSING,
            attempt_number=1,
            max_attempts=retry_policy.max_attempts,
        )

        had_failure = False
        failure_reason: str | None = None
        failure_disposition: FailureDisposition | None = None
        document_state = DocumentProcessingState.COMPLETED
        attempt = 1
        while True:
            try:
                result = service.process(archive_object)
                pending.pop(0)
                had_failure = False
                failure_reason = None
                failure_disposition = None
                callbacks.pipeline_events(result.events)
                callbacks.activity(f"Completed {archive_object.original_filename}.")
                break
            except Exception as exc:  # noqa: BLE001 -- policy classifies and bounds every failure
                had_failure = True
                failure_reason = str(exc)
                failure_disposition = classify_processing_failure(exc)
                if (
                    failure_disposition is FailureDisposition.RETRYABLE
                    and attempt < retry_policy.max_attempts
                    and not cancelled.is_set()
                ):
                    backoff = retry_policy.backoff(attempt)
                    record(
                        ProcessingProgressKind.DOCUMENT_RETRY_SCHEDULED,
                        run_id,
                        archive_object_id=archive_object.id,
                        original_filename=archive_object.original_filename,
                        document_state=DocumentProcessingState.RETRYABLE_FAILURE,
                        failure_disposition=failure_disposition,
                        attempt_number=attempt,
                        max_attempts=retry_policy.max_attempts,
                        retry_backoff_seconds=backoff,
                        retry_reason=failure_reason,
                    )
                    callbacks.activity(
                        f"Retrying {archive_object.original_filename} after {backoff:.1f}s: {exc}"
                    )
                    if cancelled.wait(backoff):
                        document_state = DocumentProcessingState.CANCELLED
                        break
                    attempt += 1
                    record(
                        ProcessingProgressKind.DOCUMENT_RETRY_STARTED,
                        run_id,
                        archive_object_id=archive_object.id,
                        original_filename=archive_object.original_filename,
                        document_state=DocumentProcessingState.PROCESSING,
                        attempt_number=attempt,
                        max_attempts=retry_policy.max_attempts,
                    )
                    continue
                document_state = (
                    DocumentProcessingState.RETRYABLE_FAILURE
                    if failure_disposition is FailureDisposition.RETRYABLE
                    else DocumentProcessingState.TERMINAL_FAILURE
                )
                if pending and pending[0].id == archive_object.id:
                    pending.pop(0)
                context.queue_failure_count += 1
                failures_this_run += 1
                callbacks.activity(
                    f"Processing failed for {archive_object.original_filename}: {exc}"
                )
                break
            finally:
                context.running_provider_ids.difference_update(running_ids)

        if document_state is DocumentProcessingState.CANCELLED:
            record(
                ProcessingProgressKind.DOCUMENT_CANCELLED,
                run_id,
                archive_object_id=archive_object.id,
                original_filename=archive_object.original_filename,
                document_state=DocumentProcessingState.CANCELLED,
                failure_reason="cancelled during retry backoff",
            )
            explicitly_cancelled.add(archive_object.id)
            break

        done += 1
        record(
            ProcessingProgressKind.DOCUMENT_COMPLETED, run_id,
            archive_object_id=archive_object.id,
            original_filename=archive_object.original_filename,
            had_failure=had_failure, failure_reason=failure_reason, documents_done=done,
            duration_seconds=time.monotonic() - document_started_at,
            document_state=document_state,
            failure_disposition=failure_disposition,
            attempt_number=attempt,
            max_attempts=retry_policy.max_attempts,
        )
        callbacks.document_completed(archive_object.original_filename, had_failure)
        callbacks.queue_progress(done, total)

    if cancelled.is_set():
        for archive_object in pending:
            if archive_object.id in explicitly_cancelled:
                continue
            record(
                ProcessingProgressKind.DOCUMENT_CANCELLED,
                run_id,
                archive_object_id=archive_object.id,
                original_filename=archive_object.original_filename,
                document_state=DocumentProcessingState.CANCELLED,
            )
    terminal = (
        ProcessingRunTerminalState.INTERRUPTED
        if paused_for_health
        else ProcessingRunTerminalState.CANCELLED
        if cancelled.is_set()
        else ProcessingRunTerminalState.COMPLETED_WITH_FAILURES
        if failures_this_run
        else ProcessingRunTerminalState.COMPLETED
    )
    record(
        ProcessingProgressKind.RUN_FINISHED,
        run_id,
        documents_done=done,
        cancelled=cancelled.is_set(),
        run_terminal_state=terminal,
    )
    return QueueRunOutcome(
        run_id=run_id,
        documents_done=done,
        documents_total=total,
        cancelled=cancelled.is_set(),
        paused_for_health=paused_for_health,
    )
