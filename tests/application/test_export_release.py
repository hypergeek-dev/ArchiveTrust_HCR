from __future__ import annotations

import json
from dataclasses import dataclass

from archivetrust.application.export.release import (
    RELEASE_MANIFEST_SCHEMA,
    ExportReleaseManifest,
    write_export_release,
)
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.infrastructure.storage.integrity import verify_file_digest_manifest
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from tests.review._helpers import emit_document_snapshot, emit_slot, heading


@dataclass(frozen=True)
class WorkspaceStub:
    id: str = "workspace-1"
    name: str = "Municipal archive"


def test_release_contains_current_json_interop_versions_integrity_and_config_hash(tmp_path):
    sink = InMemoryTelemetrySink()
    canonical = emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("Interoperable title"),
        provider_payloads=(("docling", heading("Interoperable title")),),
        classification=ComparisonClassification.CONTESTED,
    )
    document = emit_document_snapshot(sink, document_ref="doc1")
    config = tmp_path / "config"
    config.mkdir()
    (config / "policy.json").write_text('{"policy": 1}', encoding="utf-8")

    release = write_export_release(
        tmp_path / "release",
        workspace=WorkspaceStub(),
        telemetry_source=sink,
        configuration_dir=config,
        release_id="release-2026-07-17-001",
    )

    manifest = ExportReleaseManifest.model_validate_json(
        release.release_manifest.read_text(encoding="utf-8")
    )
    assert manifest.manifest_schema == RELEASE_MANIFEST_SCHEMA
    assert manifest.release_id == "release-2026-07-17-001"
    assert manifest.release_version == 1
    assert manifest.configuration_files == ("policy.json",)
    assert len(manifest.configuration_hash) == 64
    assert manifest.integrity_verified is True
    assert manifest.operational_review_data_included is True
    assert manifest.evaluation_references_included is False
    assert manifest.release_status == "contains_unresolved_content"
    assert manifest.unresolved_content_count == 1
    assert manifest.documents[0].document_snapshot_id == document.document_snapshot_id
    assert manifest.documents[0].document_version == document.document_version
    assert {item.role for item in manifest.files} >= {
        "canonical_json",
        "page_xml",
        "alto_xml",
        "mets",
        "interop_manifest",
        "integrity_sidecar",
    }
    assert verify_file_digest_manifest(release.release_manifest).ok

    canonical_json = json.loads(release.canonical_json.read_text(encoding="utf-8"))
    exported = canonical_json["documents"][0]["canonical_observations"][0]
    assert exported["canonical_observation_id"] == canonical.canonical_observation_id
    assert exported["payload"]["text"] == "Interoperable title"
