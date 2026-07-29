"""`QueueWorker` (Operational Completion milestone) — the Process Queue button's background driver.

Wires the **existing, unchanged** `WorkspaceProcessingService` into the UI on a `QThread`, so a real
processing run keeps the interface responsive and Pause/Cancel can take effect between documents.
The actual loop lives in the Qt-free `acquisition/queue_runner.py` (extracted for headless callers
— campaign scripts run the identical code path); this class only supplies the thread, the
pause/cancel events, and the Qt-signal forwarding.

**Honest granularity note:** `run_pipeline` processes every enabled provider for one document
inside a single atomic call — there is no hook to report progress *within* that call. "Provider
currently executing" therefore means "these providers have an invocation in flight for the current
document," not true per-provider real-time progress; Pause/Cancel likewise only take effect between
documents, never mid-document. Both are stated limitations, not a redesign of `run_pipeline`.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from archivetrust.acquisition.provider_health_guard import ProviderHealthGuard
from archivetrust.acquisition.queue_runner import QueueRunCallbacks, run_queue
from archivetrust.composition import AppContext


class QueueWorker(QThread):
    document_started = Signal(str)
    """Emitted with the Archive Object's original filename just before it is handed to the
    processing service."""
    document_completed = Signal(str, bool)
    """(original_filename, had_failure) — `had_failure` is `True` only if processing raised;
    a document that merely produced uncertainties for Review Center is not a failure."""
    queue_progress = Signal(int, int)
    """(documents_done, documents_total_for_this_run)."""
    activity = Signal(str)
    """One human-readable line at a time, derived from real telemetry via `summarize_event` (or a
    small number of fixed, honestly-labeled lifecycle lines: "Acquiring …", "Completed.")."""
    run_finished = Signal()
    """Distinct name from `QThread.finished` (already used internally by Qt) to avoid confusion."""
    provider_health_alert = Signal(str)
    """Same-session live notification (Provider Health Guard, Phase 22) -- the durable
    `ProcessingProgressKind.RUN_PAUSED` event the runner also records is the source of truth
    across restarts; this Signal exists only for immediate in-session feedback, the same role
    `activity` already plays for other, also-durably-recorded facts."""

    def __init__(self, context: AppContext, parent=None, *, health_guard: ProviderHealthGuard | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._paused = threading.Event()
        self._paused.set()  # not paused
        self._cancelled = threading.Event()
        self._health_guard = health_guard

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._paused.set()  # unblock a paused wait so cancellation takes effect immediately

    def _emit_pipeline_summaries(self, events: tuple) -> None:
        from archivetrust.presentation.operations_viewmodel import summarize_event

        for event in events:
            summary = summarize_event(event)
            if summary is not None:
                self.activity.emit(summary)

    def run(self) -> None:  # noqa: D102 -- QThread override
        run_queue(
            self._context,
            paused=self._paused,
            cancelled=self._cancelled,
            health_guard=self._health_guard,
            callbacks=QueueRunCallbacks(
                document_started=self.document_started.emit,
                document_completed=self.document_completed.emit,
                queue_progress=self.queue_progress.emit,
                activity=self.activity.emit,
                provider_health_alert=self.provider_health_alert.emit,
                pipeline_events=self._emit_pipeline_summaries,
            ),
        )
        self.run_finished.emit()
