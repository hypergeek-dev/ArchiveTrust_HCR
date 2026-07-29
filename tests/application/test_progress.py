"""Processing Progress Telemetry (Production Hardening Review, 2026-07-13) — the durable
process-level run/document lifecycle stream that makes a process dying mid-document diagnosable
from ArchiveTrust's own telemetry (the domain stream buffers a document's events until it
completes, so it structurally cannot record what was in flight)."""

from __future__ import annotations

from archivetrust.application.progress import (
    FileProcessingProgressSink,
    InMemoryProcessingProgressSink,
    ProcessingProgressEvent,
    ProcessingProgressKind,
    utc_now_iso,
)


def _event(kind: ProcessingProgressKind, **fields) -> ProcessingProgressEvent:
    return ProcessingProgressEvent(kind=kind, run_id="processing_run_1", recorded_at=utc_now_iso(), **fields)


def test_in_memory_sink_preserves_order() -> None:
    sink = InMemoryProcessingProgressSink()
    sink.record(_event(ProcessingProgressKind.RUN_STARTED, provider_ids=("docling",), pending_count=2))
    sink.record(_event(ProcessingProgressKind.DOCUMENT_STARTED, archive_object_id="archive_object_a"))
    kinds = [e.kind for e in sink.events()]
    assert kinds == [ProcessingProgressKind.RUN_STARTED, ProcessingProgressKind.DOCUMENT_STARTED]


def test_file_sink_survives_reconstruction(tmp_path) -> None:
    path = tmp_path / "telemetry" / "processing.jsonl"
    sink = FileProcessingProgressSink(path)
    sink.record(
        _event(
            ProcessingProgressKind.RUN_STARTED,
            provider_ids=("docling", "paddleocr-vl"), pending_count=1000,
        )
    )
    sink.record(
        _event(
            ProcessingProgressKind.DOCUMENT_STARTED,
            archive_object_id="archive_object_a", original_filename="protokoll.pdf",
        )
    )

    # A process crash mid-document is a reconstruction of exactly this state: run started,
    # document started, no completion. Reopening the file must reproduce it verbatim.
    reopened = FileProcessingProgressSink(path)
    events = reopened.events()
    assert [e.kind for e in events] == [
        ProcessingProgressKind.RUN_STARTED,
        ProcessingProgressKind.DOCUMENT_STARTED,
    ]
    assert events[0].provider_ids == ("docling", "paddleocr-vl")
    assert events[1].archive_object_id == "archive_object_a"
    assert events[1].original_filename == "protokoll.pdf"
