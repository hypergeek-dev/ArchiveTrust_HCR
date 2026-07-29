from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from archivetrust.application.export.interop import (
    ALTO_NS,
    INTEROP_MANIFEST_SCHEMA,
    METS_NS,
    PAGE_NS,
    InteropPackageManifest,
    validate_interop_package,
    write_interop_package,
)
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import CanonicalDocumentCreated, stamp_recorded_at
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
        canonical_payload=heading("Interoperable title"),
        provider_payloads=(("docling", heading("Interoperable title")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    document = CanonicalDocument.assemble(
        reconciled_graph=ReconciledObservationGraph(
            reconciliation_sequence=canonical.reconciliation_sequence,
            canonical_observations=(canonical,),
        ),
        archive_object_ref=document_ref,
        reassembly_trigger="test_interop_export",
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
                alignment_algorithm_version=4,
            )
        )
    )
    return document


def test_interop_package_writes_page_alto_mets_and_integrity_sidecars(tmp_path) -> None:
    sink = InMemoryTelemetrySink()
    document = _append_canonical_document(sink, "doc1")

    package = write_interop_package(
        tmp_path / "interop",
        workspace=WorkspaceStub(),
        telemetry_source=sink,
    )

    assert package.mets_xml.exists()
    assert package.manifest_json.exists()
    assert package.validation_report.ok is True
    assert len(package.documents) == 1
    exported = package.documents[0]
    assert exported.page_xml.exists()
    assert exported.alto_xml.exists()
    for path in (package.mets_xml, exported.page_xml, exported.alto_xml):
        assert default_file_digest_manifest_path(path).exists()
        assert verify_file_digest_manifest(path).ok is True

    page_root = ET.parse(exported.page_xml).getroot()
    assert page_root.tag == f"{{{PAGE_NS}}}PcGts"
    unicode_nodes = page_root.findall(f".//{{{PAGE_NS}}}Unicode")
    assert [node.text for node in unicode_nodes] == ["Interoperable title"]

    alto_root = ET.parse(exported.alto_xml).getroot()
    assert alto_root.tag == f"{{{ALTO_NS}}}alto"
    strings = alto_root.findall(f".//{{{ALTO_NS}}}String")
    assert [node.attrib["CONTENT"] for node in strings] == ["Interoperable title"]

    mets_root = ET.parse(package.mets_xml).getroot()
    assert mets_root.tag == f"{{{METS_NS}}}mets"
    files = mets_root.findall(f".//{{{METS_NS}}}file")
    assert {file.attrib["MIMETYPE"] for file in files} == {"application/xml"}
    provenance = mets_root.find(".//ArchiveTrustExport/Document")
    assert provenance is not None
    assert provenance.attrib["documentSnapshotId"] == document.document_snapshot_id
    assert provenance.attrib["reconciliationPolicyVersion"] == "1"
    assert provenance.attrib["archiveObjectHash"] == "unavailable"
    export_provenance = mets_root.find(".//ArchiveTrustExport")
    assert export_provenance.attrib["evaluationReferencesIncluded"] == "false"
    assert export_provenance.attrib["operationalReviewDataIncluded"] == "true"

    manifest = InteropPackageManifest.model_validate_json(package.manifest_json.read_text(encoding="utf-8"))
    assert manifest.manifest_schema == INTEROP_MANIFEST_SCHEMA
    assert manifest.validation.ok is True
    assert manifest.validation.external_viewer_validation == "not_run"
    assert manifest.validation.external_parser_validation == "lxml.etree:passed"
    assert {file.role for file in manifest.files} >= {
        "mets",
        "page_xml:doc1",
        "alto_xml:doc1",
        "mets_integrity_sidecar",
        "page_xml:doc1_integrity_sidecar",
        "alto_xml:doc1_integrity_sidecar",
    }
    page = page_root.find(f"{{{PAGE_NS}}}Page")
    assert int(page.attrib["imageWidth"]) > 0
    coords = page_root.find(f".//{{{PAGE_NS}}}Coords")
    assert coords is not None
    assert coords.attrib["points"] == "10,20 200,20 200,48 10,48"
    region = page_root.find(f".//{{{PAGE_NS}}}TextRegion")
    assert "geometryPrecision:pixel_accurate" in region.attrib["custom"]
    block = alto_root.find(f".//{{{ALTO_NS}}}TextBlock")
    assert block.attrib["HPOS"] == "10"
    assert block.attrib["VPOS"] == "20"
    assert block.attrib["WIDTH"] == "190"
    assert block.attrib["HEIGHT"] == "28"
    assert block.attrib["TAGREFS"] == "precision_pixel_accurate"


def test_interop_package_can_skip_integrity_sidecars(tmp_path) -> None:
    sink = InMemoryTelemetrySink()
    _append_canonical_document(sink, "doc1")

    package = write_interop_package(
        tmp_path / "interop",
        workspace=WorkspaceStub(),
        telemetry_source=sink,
        integrity_sidecars=False,
    )

    exported = package.documents[0]
    assert not default_file_digest_manifest_path(package.mets_xml).exists()
    assert not default_file_digest_manifest_path(exported.page_xml).exists()
    assert not default_file_digest_manifest_path(exported.alto_xml).exists()
    assert package.manifest_json.exists()


def test_interop_package_validation_detects_tampered_xml(tmp_path) -> None:
    sink = InMemoryTelemetrySink()
    _append_canonical_document(sink, "doc1")
    package = write_interop_package(
        tmp_path / "interop",
        workspace=WorkspaceStub(),
        telemetry_source=sink,
    )
    package.documents[0].page_xml.write_text("<not-page />", encoding="utf-8")

    report = validate_interop_package(package)

    assert report.ok is False
    assert any(check.name == "doc1-page-xml-parses" and not check.ok for check in report.checks)
    assert any(check.name.startswith("integrity-sidecar:doc1.page.xml") and not check.ok for check in report.checks)
