"""Release WS5: the lazy, indexed read architecture of `FileTelemetrySink`.

Guarantees under test: existing telemetry remains readable; corrupt lines fail visibly and
locally (one bad record never invalidates unrelated history); a partial final line is isolated
and never merged into the next append; the `all_events()` view is a snapshot-consistent Sequence
whose slicing supports the incremental read-model contract; the chain sidecar stays valid across
lazy appends.
"""

from __future__ import annotations

import json

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import EvidenceCreated, stamp_recorded_at
from archivetrust.infrastructure.storage.integrity import (
    default_hash_chain_sidecar_path,
    verify_hash_chain_sidecar,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink


def _event(document_ref: str, marker: str) -> EvidenceCreated:
    evidence = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output=marker,
        processing_stage=ProcessingStage.OCR,
    )
    return stamp_recorded_at(
        EvidenceCreated(
            event_id=new_id("event"),
            document_ref=document_ref,
            invocation_id=f"invocation-{marker}",
            evidence=evidence,
        )
    )


def test_all_events_is_a_lazy_snapshot_sequence(tmp_path):
    sink = FileTelemetrySink(tmp_path / "events.jsonl")
    for i in range(5):
        sink.append(_event("doc-1", f"obs-{i}"))

    view = sink.all_events()
    assert len(view) == 5
    assert view[0].evidence.raw_output == "obs-0"
    assert view[-1].evidence.raw_output == "obs-4"
    assert [e.evidence.raw_output for e in view[2:]] == ["obs-2", "obs-3", "obs-4"]

    # Appends after the view was taken do not shift the snapshot.
    sink.append(_event("doc-1", "obs-5"))
    assert len(view) == 5
    assert len(sink.all_events()) == 6


def test_incremental_cursor_slicing_parses_only_new_events(tmp_path):
    """The CoreAggregate contract: `events[cursor:]` over a fresh view yields exactly the events
    appended since the cursor."""
    sink = FileTelemetrySink(tmp_path / "events.jsonl")
    sink.append(_event("doc-1", "obs-0"))
    cursor = len(sink.all_events())
    sink.append(_event("doc-1", "obs-1"))
    sink.append(_event("doc-2", "obs-2"))

    view = sink.all_events()
    assert [e.evidence.raw_output for e in view[cursor:]] == ["obs-1", "obs-2"]


def test_reopened_sink_reads_existing_stream_identically(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = FileTelemetrySink(path)
    events = [_event("doc-1", "a"), _event("doc-2", "b"), _event("doc-1", "c")]
    for event in events:
        writer.append(event)

    reader = FileTelemetrySink(path)
    assert list(reader.all_events()) == events
    assert [e.evidence.raw_output for e in reader.events_for_document("doc-1")] == ["a", "c"]
    assert reader.corrupt_records == ()


def test_corrupt_line_is_isolated_and_reported_not_fatal(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = FileTelemetrySink(path)
    writer.append(_event("doc-1", "before"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "ObservationCreated", "truncated...\n')
    # Reconstruct a second sink so the corruption sits between two valid records.
    tail_writer = FileTelemetrySink(path)
    tail_writer.append(_event("doc-2", "after"))

    reader = FileTelemetrySink(path)
    assert len(reader.corrupt_records) == 1
    assert reader.corrupt_records[0].line_number == 2
    assert [e.evidence.raw_output for e in reader.all_events()] == ["before", "after"]
    assert [e.evidence.raw_output for e in reader.events_for_document("doc-1")] == ["before"]
    assert [e.evidence.raw_output for e in reader.events_for_document("doc-2")] == ["after"]


def test_partial_final_line_is_never_merged_into_next_append(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = FileTelemetrySink(path)
    writer.append(_event("doc-1", "complete"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "ObservationCreated", "document_ref": "doc-1"')  # no newline

    recovered = FileTelemetrySink(path)
    assert len(recovered.corrupt_records) == 1
    recovered.append(_event("doc-1", "post-crash"))

    lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    assert len(lines) == 3  # complete, isolated partial, post-crash — never two merged into one
    json.loads(lines[0])
    json.loads(lines[2])
    reread = FileTelemetrySink(path)
    assert [e.evidence.raw_output for e in reread.events_for_document("doc-1")] == [
        "complete",
        "post-crash",
    ]


def test_chain_sidecar_stays_valid_across_sessions_and_appends(tmp_path):
    path = tmp_path / "events.jsonl"
    first = FileTelemetrySink(path)
    first.append(_event("doc-1", "a"))
    second = FileTelemetrySink(path)
    second.append(_event("doc-1", "b"))

    assert default_hash_chain_sidecar_path(path).exists()
    assert verify_hash_chain_sidecar(path).ok is True


def test_events_for_document_uses_cache_after_first_read(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = FileTelemetrySink(path)
    sink.append(_event("doc-1", "a"))
    first = list(sink.events_for_document("doc-1"))
    second = list(sink.events_for_document("doc-1"))
    assert first == second
    sink.append(_event("doc-1", "b"))
    assert [e.evidence.raw_output for e in sink.events_for_document("doc-1")] == ["a", "b"]
