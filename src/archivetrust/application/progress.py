"""Processing Progress Telemetry (Production Hardening Review, 2026-07-13) — a closed event set
describing a processing *run*'s operational lifecycle: run started (with which providers), document
handed to the pipeline, document finished, run finished.

Deliberately **not** part of `domain.telemetry.events`'s `TelemetryEventKind`: that enum is closed
by design to knowledge-evolution events about document facts (Article 16 — "no process-oriented
event may be added"). These events describe a different, process-level concern, exactly as
`acquisition.events` does one stage upstream — same philosophy (immutable, append-only, replayable),
a separate, independently closed stream.

Why this stream must exist (Production Incident, 2026-07-13): the Trust Engine's domain events for
a document are buffered in memory and flushed only when the document *completes*, so a process that
dies mid-document leaves no trace of which document was in flight, which providers the run was
using, or that a run was in progress at all. Reconstructing those facts required Windows Event Log
and Docker inspection — an observability defect this stream closes: every question above is now
answerable from `telemetry/processing.jsonl` alone.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class ProcessingProgressKind(str, Enum):
    RUN_STARTED = "ProcessingRunStarted"
    DOCUMENT_STARTED = "DocumentProcessingStarted"
    DOCUMENT_COMPLETED = "DocumentProcessingCompleted"
    DOCUMENT_RETRY_SCHEDULED = "DocumentRetryScheduled"
    DOCUMENT_RETRY_STARTED = "DocumentRetryStarted"
    DOCUMENT_INTERRUPTED = "DocumentProcessingInterrupted"
    DOCUMENT_CANCELLED = "DocumentProcessingCancelled"
    RUN_PAUSED = "ProcessingRunPaused"
    """Provider Health Guard (Phase 22, 2026-07-15): a required provider was found unavailable
    before the next document began. Distinct from an unexplained crash (a `RUN_STARTED` with no
    matching `RUN_FINISHED` and no `RUN_PAUSED` either) -- this is the explicit, structured record
    of *why* a run stopped, never left to be inferred from an event's absence."""
    RUN_FINISHED = "ProcessingRunFinished"


class DocumentProcessingState(str, Enum):
    ACQUIRED = "acquired"
    QUEUED = "queued"
    PROCESSING = "processing"
    PROVIDER_PARTIAL = "provider_partial"
    COMPLETED = "completed"
    COMPLETED_WITH_REVIEW = "completed_with_review"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    SUPERSEDED = "superseded"
    EXPORTED = "exported"


class ProcessingRunTerminalState(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class FailureDisposition(str, Enum):
    RETRYABLE = "retryable"
    TERMINAL = "terminal"


class ProcessingProgressEvent(BaseModel):
    """One process-level fact about a queue run. `recorded_at` is stamped when the fact occurs
    (not when a document later flushes), so the stream's timestamps bound real wall-clock work —
    the property the domain stream's flush-time `recorded_at` cannot provide mid-document.
    """

    model_config = ConfigDict(frozen=True)

    kind: ProcessingProgressKind
    run_id: str
    recorded_at: str

    archive_object_id: str | None = None
    original_filename: str | None = None
    provider_ids: tuple[str, ...] = ()
    """RUN_STARTED only: the providers this run will actually invoke — the durable record that
    makes a configured-but-silently-absent provider (this incident's PaddleOCR-VL) diagnosable
    from telemetry instead of by its absence from provider attempt counts."""
    pending_count: int | None = None
    documents_done: int | None = None
    had_failure: bool | None = None
    failure_reason: str | None = None
    cancelled: bool | None = None
    duration_seconds: float | None = None
    document_state: DocumentProcessingState | None = None
    run_terminal_state: ProcessingRunTerminalState | None = None
    failure_disposition: FailureDisposition | None = None
    attempt_number: int | None = None
    max_attempts: int | None = None
    retry_backoff_seconds: float | None = None
    retry_reason: str | None = None
    """DOCUMENT_COMPLETED only (Observability milestone, 2026-07-13): total wall-clock time from
    this document's DOCUMENT_STARTED to this event, measured by the same caller with
    `time.monotonic()` (never derived from `recorded_at` string subtraction, which would be one
    ISO-8601 parse away from the same information but strictly less precise than holding the
    monotonic clock reading directly). `None` for every other kind, and for historical
    DOCUMENT_COMPLETED events recorded before this field existed."""
    paused_provider_id: str | None = None
    """RUN_PAUSED only (Provider Health Guard, Phase 22): the primary provider the guard found
    unavailable."""
    paused_reason: str | None = None
    """RUN_PAUSED only: the guard's human-readable summary, naming every unavailable required
    provider -- surfaced verbatim to the operator (`ProcessingCenterViewModel.recovery_status`)."""
    recovery_attempted: bool | None = None
    """RUN_PAUSED only: whether the guard attempted any automatic recovery before pausing."""
    recovery_succeeded: bool | None = None
    """RUN_PAUSED only: `None` when no recovery was attempted; otherwise whether it succeeded.
    (The run still ends up paused even on a successful recovery of *some* providers if others
    remain unavailable -- this field describes recovery outcome, not the final pause decision.)"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryProcessingProgressSink:
    """Process-local, non-durable — tests and throwaway (`persistent_telemetry=False`) contexts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[ProcessingProgressEvent] = []

    def record(self, event: ProcessingProgressEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> tuple[ProcessingProgressEvent, ...]:
        with self._lock:
            return tuple(self._events)


class FileProcessingProgressSink(InMemoryProcessingProgressSink):
    """Durable, append-only JSON-Lines progress stream — the processing-run analogue of
    `FileAcquisitionTelemetrySink`, same write-through-mirror design: the file is the single
    durable record, the in-memory list a cache of it loaded once at construction.
    """

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._events.append(parse_progress_event(json.loads(line)))

    def record(self, event: ProcessingProgressEvent) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json())
                handle.write("\n")
            self._events.append(event)

    def events(self) -> tuple[ProcessingProgressEvent, ...]:
        with self._lock:
            return tuple(self._events)


def parse_progress_event(payload: dict[str, Any]) -> ProcessingProgressEvent:
    return ProcessingProgressEvent.model_validate(payload)
