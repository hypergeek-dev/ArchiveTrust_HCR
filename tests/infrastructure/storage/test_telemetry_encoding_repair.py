from __future__ import annotations

import json

import pytest

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.telemetry.events import EvidenceCreated, EvidenceRejected
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink, InMemoryTelemetrySink
from archivetrust.infrastructure.storage.blob_store import BlobIntegrityError


def test_file_sink_externalizes_large_evidence_raw_output_and_rehydrates_on_read(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    raw_output = "large raw output " * 20
    evidence = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output=raw_output,
        processing_stage=ProcessingStage.OCR,
    )
    event = EvidenceCreated(
        event_id="event-1",
        document_ref="doc-1",
        invocation_id="invocation-1",
        evidence=evidence,
    )

    sink = FileTelemetrySink(path, raw_output_blob_threshold=10)
    sink.append(event)

    line = path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert raw_output not in line
    assert "__archivetrust_blob_ref__" in payload["evidence"]["raw_output"]
    blob_ref = payload["evidence"]["raw_output"]
    assert blob_ref["media_type"] == "text/plain"
    assert blob_ref["encoding"] == "utf-8"
    assert blob_ref["integrity"].startswith("sha256:")

    reloaded = FileTelemetrySink(path, raw_output_blob_threshold=10)
    restored = tuple(reloaded.all_events())[0]
    assert isinstance(restored, EvidenceCreated)
    assert restored.evidence.raw_output == raw_output


def test_file_sink_deduplicates_identical_raw_output_blobs(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    raw_output = "same blob content " * 20
    sink = FileTelemetrySink(path, raw_output_blob_threshold=10)

    for index in range(2):
        sink.append(
            EvidenceRejected(
                event_id=f"event-{index}",
                document_ref="doc-1",
                provider_id="docling",
                provider_version="1.0",
                invocation_id=f"invocation-{index}",
                processing_stage=ProcessingStage.OCR,
                raw_output=raw_output,
                rejection_reason="bad shape",
            )
        )

    blob_files = [path for path in (tmp_path / "blobs").rglob("*") if path.is_file()]
    assert len(blob_files) == 1
    assert all(event.raw_output == raw_output for event in FileTelemetrySink(path).all_events())


def test_in_memory_sink_keeps_raw_output_inline() -> None:
    sink = InMemoryTelemetrySink()
    event = EvidenceRejected(
        event_id="event-1",
        document_ref="doc-1",
        provider_id="docling",
        provider_version="1.0",
        invocation_id="invocation-1",
        processing_stage=ProcessingStage.OCR,
        raw_output="inline",
        rejection_reason="bad shape",
    )
    sink.append(event)

    assert tuple(sink.all_events()) == (event,)


def test_blob_integrity_is_verified_when_rehydrated(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    sink = FileTelemetrySink(path, raw_output_blob_threshold=1)
    sink.append(
        EvidenceRejected(
            event_id="event-1",
            document_ref="doc-1",
            provider_id="docling",
            provider_version="1.0",
            invocation_id="invocation-1",
            processing_stage=ProcessingStage.OCR,
            raw_output="externalized",
            rejection_reason="bad shape",
        )
    )
    blob_path = next(item for item in (tmp_path / "blobs").rglob("*") if item.is_file())
    blob_path.write_bytes(b"tampered")

    with pytest.raises(BlobIntegrityError, match="digest mismatch"):
        tuple(FileTelemetrySink(path).all_events())
