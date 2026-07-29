from __future__ import annotations

import threading
from types import SimpleNamespace

from archivetrust.acquisition.archive_object import ArchiveObject
from archivetrust.acquisition.queue_runner import QueueRunCallbacks, run_queue
from archivetrust.application.progress import (
    DocumentProcessingState,
    InMemoryProcessingProgressSink,
    ProcessingProgressEvent,
    ProcessingProgressKind,
    ProcessingRunTerminalState,
    utc_now_iso,
)
from archivetrust.application.recovery import RetryPolicy, reconcile_incomplete_runs


def _object(identity: str) -> ArchiveObject:
    return ArchiveObject(
        id=identity,
        workspace_id="workspace",
        source_id="source",
        original_filename=f"{identity}.pdf",
        content_hash=identity * 8,
        byte_size=10,
        mime_type="application/pdf",
        storage_path=f"{identity}.pdf",
    )


class _HealthyGuard:
    def check(self):
        return SimpleNamespace(healthy=True)


class _Service:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def process(self, _archive_object):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(events=())


def _context(objects, service):
    progress = InMemoryProcessingProgressSink()
    return SimpleNamespace(
        acquisition_manager=SimpleNamespace(pending=list(objects)),
        processing_progress=progress,
        queue_failure_count=0,
        running_provider_ids=set(),
        enabled_provider_ids=lambda: {"provider"},
        processing_service=lambda: service,
    )


def test_retryable_failure_uses_bounded_durable_retry_then_completes():
    service = _Service((TimeoutError("temporary"), None))
    context = _context((_object("one"),), service)

    outcome = run_queue(
        context,
        health_guard=_HealthyGuard(),
        retry_policy=RetryPolicy(max_attempts=3, base_backoff_seconds=0),
    )

    assert service.calls == 2
    events = context.processing_progress.events()
    assert any(event.kind is ProcessingProgressKind.DOCUMENT_RETRY_SCHEDULED for event in events)
    completed = next(event for event in events if event.kind is ProcessingProgressKind.DOCUMENT_COMPLETED)
    assert completed.document_state is DocumentProcessingState.COMPLETED
    assert completed.had_failure is False
    finish = events[-1]
    assert finish.run_terminal_state is ProcessingRunTerminalState.COMPLETED
    assert outcome.documents_done == 1


def test_terminal_failure_is_not_retried_or_reported_as_success():
    service = _Service((ValueError("invalid document"),))
    context = _context((_object("one"),), service)

    run_queue(
        context,
        health_guard=_HealthyGuard(),
        retry_policy=RetryPolicy(max_attempts=3, base_backoff_seconds=0),
    )

    assert service.calls == 1
    completed = next(
        event
        for event in context.processing_progress.events()
        if event.kind is ProcessingProgressKind.DOCUMENT_COMPLETED
    )
    assert completed.document_state is DocumentProcessingState.TERMINAL_FAILURE
    assert completed.had_failure is True
    assert context.processing_progress.events()[-1].run_terminal_state is (
        ProcessingRunTerminalState.COMPLETED_WITH_FAILURES
    )


def test_cancellation_between_documents_is_explicit_and_run_is_terminal():
    cancelled = threading.Event()
    context = _context((_object("one"), _object("two")), _Service((None, None)))

    run_queue(
        context,
        cancelled=cancelled,
        callbacks=QueueRunCallbacks(
            document_completed=lambda _name, _failed: cancelled.set()
        ),
        health_guard=_HealthyGuard(),
        retry_policy=RetryPolicy(base_backoff_seconds=0),
    )

    events = context.processing_progress.events()
    cancelled_event = next(
        event for event in events if event.kind is ProcessingProgressKind.DOCUMENT_CANCELLED
    )
    assert cancelled_event.archive_object_id == "two"
    assert events[-1].run_terminal_state is ProcessingRunTerminalState.CANCELLED


def test_restart_reconciliation_closes_open_run_and_is_idempotent():
    progress = InMemoryProcessingProgressSink()
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED,
            run_id="abandoned",
            recorded_at=utc_now_iso(),
            pending_count=1,
        )
    )
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_STARTED,
            run_id="abandoned",
            recorded_at=utc_now_iso(),
            archive_object_id="one",
            original_filename="one.pdf",
        )
    )

    assert reconcile_incomplete_runs(progress) == ("abandoned",)
    assert reconcile_incomplete_runs(progress) == ()
    events = progress.events()
    assert events[-2].document_state is DocumentProcessingState.INTERRUPTED
    assert events[-1].run_terminal_state is ProcessingRunTerminalState.INTERRUPTED
