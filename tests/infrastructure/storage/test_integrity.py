from __future__ import annotations

import json

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.telemetry.events import EvidenceCreated
from archivetrust.infrastructure.storage.integrity import (
    build_hash_chain_manifest,
    default_hash_chain_manifest_path,
    verify_hash_chain_manifest,
    write_hash_chain_manifest,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from scripts.verify_integrity import main as verify_integrity_main


def _event(document_ref: str, suffix: str) -> EvidenceCreated:
    evidence = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output=f"raw-{suffix}",
        processing_stage=ProcessingStage.OCR,
    )
    return EvidenceCreated(
        event_id=f"event-{suffix}",
        document_ref=document_ref,
        invocation_id=f"invocation-{suffix}",
        evidence=evidence,
    )


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_hash_chain_manifest_round_trips_and_verifies_clean_file(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, [{"event": 1}, {"event": 2}])

    manifest_path = write_hash_chain_manifest(path)
    manifest = build_hash_chain_manifest(path)

    assert manifest_path == default_hash_chain_manifest_path(path)
    assert manifest.line_count == 2
    assert verify_hash_chain_manifest(path).ok is True


def test_hash_chain_verifier_detects_bit_flip(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, [{"event": "before"}, {"event": "stable"}])
    write_hash_chain_manifest(path)

    path.write_text(path.read_text(encoding="utf-8").replace("before", "after"), encoding="utf-8")

    result = verify_hash_chain_manifest(path)
    assert result.ok is False
    assert result.reason == "root_hash_mismatch"


def test_hash_chain_verifier_detects_truncation(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, [{"event": 1}, {"event": 2}, {"event": 3}])
    write_hash_chain_manifest(path)

    path.write_text('{"event": 1}\n{"event": 2}\n', encoding="utf-8")

    result = verify_hash_chain_manifest(path)
    assert result.ok is False
    assert result.reason == "line_count_mismatch"


def test_hash_chain_verifier_detects_line_reorder(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    first = {"event": "a"}
    second = {"event": "b"}
    _write_jsonl(path, [first, second])
    write_hash_chain_manifest(path)

    _write_jsonl(path, [second, first])

    result = verify_hash_chain_manifest(path)
    assert result.ok is False
    assert result.reason == "root_hash_mismatch"


def test_file_telemetry_sink_writes_hash_chain_sidecar_on_append(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    sink = FileTelemetrySink(path)
    sink.append(_event("doc-1", "a"))
    sink.append(_event("doc-1", "b"))

    from archivetrust.infrastructure.storage.integrity import (
        default_hash_chain_sidecar_path,
        verify_hash_chain_sidecar,
    )

    assert default_hash_chain_sidecar_path(path).exists()
    assert verify_hash_chain_sidecar(path).ok is True
    assert list(FileTelemetrySink(path).all_events()) == [_event("doc-1", "a"), _event("doc-1", "b")]


def test_verify_integrity_cli_returns_nonzero_for_tampered_stream(tmp_path, capsys) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, [{"event": "clean"}])
    write_hash_chain_manifest(path)

    assert verify_integrity_main([str(path)]) == 0

    path.write_text('{"event": "tampered"}\n', encoding="utf-8")
    assert verify_integrity_main([str(path)]) == 1
    captured = capsys.readouterr()
    assert "root_hash_mismatch" in captured.out
