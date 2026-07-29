from __future__ import annotations

import json
from dataclasses import dataclass

from archivetrust.application.export.json_export import (
    EXPORT_SCHEMA,
    export_workspace_json,
    exportable_document_count,
    write_workspace_json,
)
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDocumentCreated,
    ReviewOutcome,
    ReviewOutcomeRecorded,
    stamp_recorded_at,
)
from archivetrust.infrastructure.storage.integrity import (
    default_file_digest_manifest_path,
    verify_file_digest_manifest,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from tests.review._helpers import emit_slot, heading


@dataclass(frozen=True)
class WorkspaceStub:
    id: str = "workspace-1"
    name: str = "Municipal archive"


def _append_canonical_document(sink: InMemoryTelemetrySink, document_ref: str) -> CanonicalDocument:
    canonical = emit_slot(
        sink,
        document_ref=document_ref,
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    document = CanonicalDocument.assemble(
        reconciled_graph=ReconciledObservationGraph(
            reconciliation_sequence=canonical.reconciliation_sequence,
            canonical_observations=(canonical,),
        ),
        archive_object_ref=document_ref,
        reassembly_trigger="test_export",
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


def test_workspace_json_export_contains_canonical_snapshot_and_review_outcome() -> None:
    sink = InMemoryTelemetrySink()
    document = _append_canonical_document(sink, "doc1")
    sink.append(
        stamp_recorded_at(
            ReviewOutcomeRecorded(
                event_id=new_id("event"),
                document_ref="doc1",
                outcome_id="outcome-1",
                semantic_slot_id="slot-1",
                canonical_observation_id=document.contained_observations[0],
                action="accept_provider",
                outcome=ReviewOutcome.RESOLVED,
                correction_id="correction-1",
            )
        )
    )

    payload = json.loads(export_workspace_json(workspace=WorkspaceStub(), telemetry_source=sink))

    assert payload["export_schema"] == EXPORT_SCHEMA
    assert payload["workspace"]["name"] == "Municipal archive"
    assert payload["operational_review_data_included"] is True
    assert payload["evaluation_reference_data"] == "excluded_separate_data_plane"
    assert payload["documents"][0]["document_ref"] == "doc1"
    assert payload["documents"][0]["canonical_document"]["document_snapshot_id"] == document.document_snapshot_id
    assert payload["documents"][0]["provenance"]["reconciliation_policy_version"] == 1
    assert payload["documents"][0]["review_outcomes"][0]["outcome"] == "resolved"


def test_write_workspace_json_and_exportable_count(tmp_path) -> None:
    sink = InMemoryTelemetrySink()
    assert exportable_document_count(sink) == 0
    _append_canonical_document(sink, "doc1")

    output = write_workspace_json(tmp_path / "export.json", workspace=WorkspaceStub(), telemetry_source=sink)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output.exists()
    assert exportable_document_count(sink) == 1
    assert len(payload["documents"]) == 1


def test_write_workspace_json_can_emit_integrity_sidecar(tmp_path) -> None:
    sink = InMemoryTelemetrySink()
    _append_canonical_document(sink, "doc1")

    output = write_workspace_json(
        tmp_path / "export.json",
        workspace=WorkspaceStub(),
        telemetry_source=sink,
        integrity_sidecar=True,
    )

    assert default_file_digest_manifest_path(output).exists()
    assert verify_file_digest_manifest(output).ok is True

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["documents"][0]["display_name"] = "tampered"
    output.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_file_digest_manifest(output)
    assert result.ok is False
    assert result.reason in {"digest_mismatch", "size_mismatch"}
