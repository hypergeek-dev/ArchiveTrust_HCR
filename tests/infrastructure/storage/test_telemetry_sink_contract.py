"""Shared contract test suite for TelemetrySink implementations (ROADMAP.md S11 / Milestone 2
acceptance criteria: "Telemetry sink is pluggable (in-memory + file-based implementations both
pass the same contract test suite)"). Both sink types run through the exact same test bodies,
parametrized by fixture, so neither can silently diverge in append-order or lookup semantics.
"""

from __future__ import annotations

import pytest

from archivetrust.domain.telemetry.events import EvidenceCreated, TelemetryEvent
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.infrastructure.storage.telemetry_sink import (
    FileTelemetrySink,
    InMemoryTelemetrySink,
)


def _event(document_ref: str, suffix: str) -> EvidenceCreated:
    evidence = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output=f"raw-{suffix}",
        processing_stage=ProcessingStage.OCR,
    )
    return EvidenceCreated(
        event_id=f"event_{suffix}",
        document_ref=document_ref,
        invocation_id=f"invocation_{suffix}",
        evidence=evidence,
    )


@pytest.fixture(params=["in_memory", "file"])
def sink(request: pytest.FixtureRequest, tmp_path) -> TelemetrySink:
    if request.param == "in_memory":
        return InMemoryTelemetrySink()
    return FileTelemetrySink(tmp_path / "telemetry.jsonl")


def test_sink_satisfies_the_telemetry_sink_protocol(sink: TelemetrySink):
    assert isinstance(sink, TelemetrySink)


def test_append_then_read_back_round_trips(sink: TelemetrySink):
    event = _event("document-1", "a")
    sink.append(event)
    stored = list(sink.events_for_document("document-1"))
    assert stored == [event]


def test_events_for_document_preserves_append_order(sink: TelemetrySink):
    first = _event("document-1", "a")
    second = _event("document-1", "b")
    third = _event("document-1", "c")
    sink.append(first)
    sink.append(second)
    sink.append(third)
    assert list(sink.events_for_document("document-1")) == [first, second, third]


def test_events_for_document_only_returns_that_documents_events(sink: TelemetrySink):
    doc1_event = _event("document-1", "a")
    doc2_event = _event("document-2", "b")
    sink.append(doc1_event)
    sink.append(doc2_event)
    assert list(sink.events_for_document("document-1")) == [doc1_event]
    assert list(sink.events_for_document("document-2")) == [doc2_event]


def test_events_for_unknown_document_is_empty(sink: TelemetrySink):
    assert list(sink.events_for_document("nonexistent")) == []


def test_all_events_returns_everything_in_append_order(sink: TelemetrySink):
    doc1_event = _event("document-1", "a")
    doc2_event = _event("document-2", "b")
    sink.append(doc1_event)
    sink.append(doc2_event)
    assert list(sink.all_events()) == [doc1_event, doc2_event]


@pytest.mark.parametrize(
    "control_char",
    ["\x85", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", " ", " "],
    ids=["NEL", "VT", "FF", "FS", "GS", "RS", "LINE_SEP", "PARA_SEP"],
)
def test_file_sink_survives_reload_with_line_boundary_characters_in_payload_text(
    tmp_path, control_char: str
) -> None:
    """Real archive/OCR text can legitimately contain characters `str.splitlines()` treats as line
    boundaries even though only "\\n" ever delimits a record (`append()`'s only separator). Loading
    such a file used to tear the JSON line in two at that character, raising a spurious
    `JSONDecodeError` on every subsequent app launch (production incident, 2026-07-15)."""
    path = tmp_path / "telemetry.jsonl"
    first = FileTelemetrySink(path)
    event = _event("document-1", f"payload with a{control_char}boundary char")
    first.append(event)

    reloaded = FileTelemetrySink(path)
    assert list(reloaded.all_events()) == [event]
