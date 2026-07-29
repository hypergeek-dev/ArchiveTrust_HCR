from __future__ import annotations

import json

import pytest

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import CanonicalDocumentCreated, stamp_recorded_at
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.replay import ReplayArchive, ReplayArchiveError, export_archive_json, summarize_archive
from archivetrust.replay.__main__ import main as replay_main
from tests.review._helpers import emit_slot, heading


def _append_replayable_document(sink: FileTelemetrySink, document_ref: str = "doc1") -> CanonicalDocument:
    canonical = emit_slot(
        sink,
        document_ref=document_ref,
        canonical_payload=heading("Archive Trust"),
        provider_payloads=(("docling", heading("Archive Trust")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    document = CanonicalDocument.assemble(
        reconciled_graph=ReconciledObservationGraph(
            reconciliation_sequence=canonical.reconciliation_sequence,
            canonical_observations=(canonical,),
        ),
        archive_object_ref=document_ref,
        reassembly_trigger="replay_distribution_test",
    )
    sink.append(
        stamp_recorded_at(
            CanonicalDocumentCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                canonical_document=document,
                reconciliation_policy_version=1,
                capability_matrix_version=2,
                confidence_policy_version=3,
            )
        )
    )
    return document


def _archive_file(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = FileTelemetrySink(path)
    document = _append_replayable_document(sink)
    return path, document


def test_replay_archive_summarizes_direct_jsonl_file(tmp_path) -> None:
    path, _document = _archive_file(tmp_path)

    summary = summarize_archive(path)

    assert summary.event_count == 5
    assert summary.document_count == 1
    assert summary.evidence_count == 1
    assert summary.observation_count == 1
    assert summary.canonical_observation_count == 1
    assert summary.canonical_document_count == 1
    assert summary.integrity.checked is True
    assert summary.integrity.ok is True


def test_replay_archive_discovers_workspace_style_events_path(tmp_path) -> None:
    events_path = tmp_path / "workspace" / "telemetry" / "events.jsonl"
    sink = FileTelemetrySink(events_path)
    _append_replayable_document(sink)

    archive = ReplayArchive.open(tmp_path / "workspace")

    assert archive.event_path == events_path
    assert archive.summary().event_count == 5


def test_replay_archive_reports_missing_integrity_sidecar_as_not_checked(tmp_path) -> None:
    path, _document = _archive_file(tmp_path)
    path.with_name(f"{path.name}.chain.jsonl").unlink()

    summary = summarize_archive(path)

    assert summary.integrity.checked is False
    assert summary.integrity.ok is None
    assert summary.integrity.reason == "manifest_missing"


def test_replay_archive_reports_tampered_sidecar_status(tmp_path) -> None:
    path, _document = _archive_file(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    summary = summarize_archive(path)

    assert summary.integrity.checked is True
    assert summary.integrity.ok is False
    # The append-only sidecar (WS5) reports the first divergent line rather than only the root:
    assert summary.integrity.reason == "checkpoint_mismatch"


def test_replay_archive_exports_a3_workspace_json(tmp_path) -> None:
    path, document = _archive_file(tmp_path)
    output = export_archive_json(
        path,
        tmp_path / "replay-export.json",
        workspace_id="sample",
        workspace_name="Sample replay",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["export_schema"] == "archivetrust.canonical_document.v2"
    assert payload["workspace"]["name"] == "Sample replay"
    assert payload["documents"][0]["canonical_document"]["document_snapshot_id"] == document.document_snapshot_id


def test_cli_summary_emits_json(tmp_path, capsys) -> None:
    path, _document = _archive_file(tmp_path)

    assert replay_main(["summary", str(path)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["event_count"] == 5
    assert payload["integrity"]["ok"] is True


def test_cli_export_writes_json_and_integrity_sidecar(tmp_path, capsys) -> None:
    path, _document = _archive_file(tmp_path)
    output = tmp_path / "export.json"

    assert replay_main(["export", str(path), "--output", str(output), "--integrity-sidecar"]) == 0
    captured = capsys.readouterr()

    assert json.loads(captured.out)["output"] == str(output)
    assert output.exists()
    assert output.with_name("export.json.integrity.json").exists()


def test_replay_archive_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(ReplayArchiveError):
        ReplayArchive.open(tmp_path / "missing")
