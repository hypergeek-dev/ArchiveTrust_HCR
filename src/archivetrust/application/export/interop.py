"""A6 archival interoperability export: PAGE-XML, ALTO, and METS package writers.

This is a leaf export surface over recorded canonical telemetry. It does not change canonical
facts, invoke providers, or infer layout that was not recorded. The first profile exports
text-bearing canonical observations as simple text regions/blocks and records provenance in METS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict

from archivetrust.application.current_state import CurrentStateService
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.evidence.models import BoundingBox, Precision
from archivetrust.domain.current_state import CurrentDocumentState
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.infrastructure.storage.integrity import (
    build_file_digest_manifest,
    default_file_digest_manifest_path,
    verify_file_digest_manifest,
    write_file_digest_manifest,
)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
METS_NS = "http://www.loc.gov/METS/"
XLINK_NS = "http://www.w3.org/1999/xlink"
INTEROP_MANIFEST_SCHEMA = "archivetrust.interop_package.v1"
INTEROP_MANIFEST_NAME = "archivetrust_interop_manifest.json"

for prefix, uri in (
    ("pc", PAGE_NS),
    ("alto", ALTO_NS),
    ("mets", METS_NS),
    ("xlink", XLINK_NS),
):
    ET.register_namespace(prefix, uri)


@dataclass(frozen=True)
class InteropExportedDocument:
    document_ref: str
    page_xml: Path
    alto_xml: Path


@dataclass(frozen=True)
class InteropExportPackage:
    output_dir: Path
    mets_xml: Path
    manifest_json: Path
    documents: tuple[InteropExportedDocument, ...]
    validation_report: "InteropValidationReport"


class InteropValidationCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool
    detail: str


class InteropValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    checks: tuple[InteropValidationCheck, ...]
    external_viewer_validation: str = "not_run"
    external_parser_validation: str = "not_run"


@dataclass(frozen=True)
class InteropGeometry:
    page: int
    box: BoundingBox


class InteropPackageManifestFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    path: str
    size_bytes: int
    sha256: str


class InteropPackageManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_schema: str = INTEROP_MANIFEST_SCHEMA
    package_kind: str = "PAGE_ALTO_METS"
    files: tuple[InteropPackageManifestFile, ...]
    validation: InteropValidationReport


def write_interop_package(
    output_dir: str | Path,
    *,
    workspace,
    telemetry_source,
    acquisition_manager=None,
    integrity_sidecars: bool = True,
) -> InteropExportPackage:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    snapshots = {
        state.document_ref: state
        for state in CurrentStateService(telemetry_source).documents()
        if state.export_eligible
    }
    exported: list[InteropExportedDocument] = []

    for document_ref, state in sorted(snapshots.items()):
        document = state.latest_canonical_document
        if document is None:  # filtered by export_eligible; retained for type narrowing
            continue
        current_by_id = {
            observation.canonical_observation_id: observation
            for observation in state.current_canonical_observations
        }
        document_observations = tuple(
            current_by_id[obs_id]
            for obs_id in state.effective_contained_observations
            if obs_id in current_by_id
        )
        geometries = {
            slot.current.canonical_observation_id: geometry
            for slot in state.slots
            if (geometry := _best_geometry(slot.evidence)) is not None
        }
        page_path = root / f"{_safe_name(document_ref)}.page.xml"
        alto_path = root / f"{_safe_name(document_ref)}.alto.xml"
        _write_xml(
            page_path,
            page_xml_for_document(document, document_observations, geometries=geometries),
        )
        _write_xml(
            alto_path,
            alto_xml_for_document(document, document_observations, geometries=geometries),
        )
        if integrity_sidecars:
            write_file_digest_manifest(page_path)
            write_file_digest_manifest(alto_path)
        exported.append(
            InteropExportedDocument(
                document_ref=document_ref,
                page_xml=page_path,
                alto_xml=alto_path,
            )
        )

    archive_hashes = {}
    if acquisition_manager is not None:
        for state in snapshots.values():
            document = state.latest_canonical_document
            if document is None:
                continue
            archive_object = acquisition_manager.archive_object_by_ref(document.archive_object_ref)
            archive_hashes[state.document_ref] = getattr(archive_object, "content_hash", None)
    mets_path = root / "mets.xml"
    _write_xml(
        mets_path,
        mets_xml_for_package(
            workspace=workspace,
            snapshots=snapshots,
            exported=tuple(exported),
            archive_hashes=archive_hashes,
        ),
    )
    if integrity_sidecars:
        write_file_digest_manifest(mets_path)

    manifest_path = root / INTEROP_MANIFEST_NAME
    package = InteropExportPackage(
        output_dir=root,
        mets_xml=mets_path,
        manifest_json=manifest_path,
        documents=tuple(exported),
        validation_report=InteropValidationReport(ok=False, checks=()),
    )
    validation = validate_interop_package(package, require_integrity_sidecars=integrity_sidecars)
    manifest = interop_package_manifest(package, validation)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if integrity_sidecars:
        write_file_digest_manifest(manifest_path)
    return InteropExportPackage(
        output_dir=root,
        mets_xml=mets_path,
        manifest_json=manifest_path,
        documents=tuple(exported),
        validation_report=validation,
    )


def validate_interop_package(
    package: InteropExportPackage,
    *,
    require_integrity_sidecars: bool = True,
) -> InteropValidationReport:
    checks: list[InteropValidationCheck] = []
    checks.append(_xml_root_check("mets-xml-parses", package.mets_xml, _tag(METS_NS, "mets")))
    for document in package.documents:
        checks.append(_xml_root_check(f"{document.document_ref}-page-xml-parses", document.page_xml, _tag(PAGE_NS, "PcGts")))
        checks.append(_xml_root_check(f"{document.document_ref}-alto-xml-parses", document.alto_xml, _tag(ALTO_NS, "alto")))

    referenced = _mets_referenced_files(package.mets_xml)
    expected = {doc.page_xml.name for doc in package.documents} | {doc.alto_xml.name for doc in package.documents}
    checks.append(
        InteropValidationCheck(
            name="mets-references-exported-files",
            ok=expected <= referenced,
            detail=f"expected={sorted(expected)}, referenced={sorted(referenced)}",
        )
    )
    for filename in sorted(referenced):
        path = package.output_dir / filename
        checks.append(
            InteropValidationCheck(
                name=f"referenced-file-exists:{filename}",
                ok=path.exists(),
                detail=str(path),
            )
        )

    if require_integrity_sidecars:
        for path in _package_payload_files(package):
            result = verify_file_digest_manifest(path)
            checks.append(
                InteropValidationCheck(
                    name=f"integrity-sidecar:{path.name}",
                    ok=result.ok,
                    detail=result.reason or "ok",
                )
            )
    external_parser = _independent_parser_checks(package)
    checks.extend(external_parser[0])
    return InteropValidationReport(
        ok=all(check.ok for check in checks),
        checks=tuple(checks),
        external_parser_validation=external_parser[1],
    )


def interop_package_manifest(
    package: InteropExportPackage,
    validation: InteropValidationReport | None = None,
) -> InteropPackageManifest:
    validation = validation or validate_interop_package(package)
    files = []
    for role, path in _package_manifest_paths(package):
        digest = build_file_digest_manifest(path)
        files.append(
            InteropPackageManifestFile(
                role=role,
                path=path.name,
                size_bytes=digest.size_bytes,
                sha256=digest.digest,
            )
        )
        sidecar = default_file_digest_manifest_path(path)
        if sidecar.exists():
            sidecar_digest = build_file_digest_manifest(sidecar)
            files.append(
                InteropPackageManifestFile(
                    role=f"{role}_integrity_sidecar",
                    path=sidecar.name,
                    size_bytes=sidecar_digest.size_bytes,
                    sha256=sidecar_digest.digest,
                )
            )
    return InteropPackageManifest(files=tuple(files), validation=validation)


def page_xml_for_document(
    document: CanonicalDocument,
    observations: tuple[CanonicalObservation, ...],
    *,
    geometries: dict[str, InteropGeometry] | None = None,
) -> ET.Element:
    geometries = geometries or {}
    pcgts = ET.Element(_tag(PAGE_NS, "PcGts"))
    metadata = ET.SubElement(pcgts, _tag(PAGE_NS, "Metadata"))
    ET.SubElement(metadata, _tag(PAGE_NS, "Creator")).text = "ArchiveTrust"
    page_geometries = tuple(geometries.values())
    width = math.ceil(max((geometry.box.x1 for geometry in page_geometries), default=0))
    height = math.ceil(max((geometry.box.y1 for geometry in page_geometries), default=0))
    page_number = min((geometry.page for geometry in page_geometries), default=1)
    page = ET.SubElement(
        pcgts,
        _tag(PAGE_NS, "Page"),
        imageFilename=document.archive_object_ref,
        imageWidth=str(width),
        imageHeight=str(height),
        custom=f"sourcePage:{page_number}; dimensionBasis:recordedEvidenceExtent",
    )
    for index, canonical in enumerate(_text_observations(observations), start=1):
        region = ET.SubElement(
            page,
            _tag(PAGE_NS, "TextRegion"),
            id=f"r{index}",
            custom=_custom(canonical, geometries.get(canonical.canonical_observation_id)),
        )
        geometry = geometries.get(canonical.canonical_observation_id)
        if geometry is not None:
            ET.SubElement(region, _tag(PAGE_NS, "Coords"), points=_page_points(geometry.box))
        line = ET.SubElement(region, _tag(PAGE_NS, "TextLine"), id=f"r{index}_l1")
        if geometry is not None:
            ET.SubElement(line, _tag(PAGE_NS, "Coords"), points=_page_points(geometry.box))
        equiv = ET.SubElement(line, _tag(PAGE_NS, "TextEquiv"))
        ET.SubElement(equiv, _tag(PAGE_NS, "Unicode")).text = _text_of(canonical)
    return pcgts


def alto_xml_for_document(
    document: CanonicalDocument,
    observations: tuple[CanonicalObservation, ...],
    *,
    geometries: dict[str, InteropGeometry] | None = None,
) -> ET.Element:
    geometries = geometries or {}
    alto = ET.Element(_tag(ALTO_NS, "alto"))
    description = ET.SubElement(alto, _tag(ALTO_NS, "Description"))
    ET.SubElement(description, _tag(ALTO_NS, "MeasurementUnit")).text = "pixel"
    tags = ET.SubElement(alto, _tag(ALTO_NS, "Tags"))
    for precision in sorted({geometry.box.precision.value for geometry in geometries.values()}):
        ET.SubElement(
            tags,
            _tag(ALTO_NS, "OtherTag"),
            ID=f"precision_{precision}",
            LABEL=f"ArchiveTrust geometry precision: {precision}",
        )
    layout = ET.SubElement(alto, _tag(ALTO_NS, "Layout"))
    page_geometries = tuple(geometries.values())
    width = math.ceil(max((geometry.box.x1 for geometry in page_geometries), default=0))
    height = math.ceil(max((geometry.box.y1 for geometry in page_geometries), default=0))
    page_number = min((geometry.page for geometry in page_geometries), default=1)
    page = ET.SubElement(
        layout,
        _tag(ALTO_NS, "Page"),
        ID=_xml_id(document.document_snapshot_id),
        PHYSICAL_IMG_NR=str(page_number),
        WIDTH=str(width),
        HEIGHT=str(height),
    )
    print_space = ET.SubElement(page, _tag(ALTO_NS, "PrintSpace"), ID=f"{_xml_id(document.document_snapshot_id)}_ps")
    for index, canonical in enumerate(_text_observations(observations), start=1):
        geometry = geometries.get(canonical.canonical_observation_id)
        attributes = {"ID": f"b{index}"}
        if geometry is not None:
            attributes["TAGREFS"] = f"precision_{geometry.box.precision.value}"
            attributes.update(_alto_box(geometry.box))
        block = ET.SubElement(
            print_space,
            _tag(ALTO_NS, "TextBlock"),
            attributes,
        )
        line_attributes = {"ID": f"b{index}_l1"}
        if geometry is not None:
            line_attributes.update(_alto_box(geometry.box))
        line = ET.SubElement(block, _tag(ALTO_NS, "TextLine"), line_attributes)
        ET.SubElement(
            line,
            _tag(ALTO_NS, "String"),
            ID=f"b{index}_s1",
            CONTENT=_text_of(canonical),
        )
    return alto


def mets_xml_for_package(
    *,
    workspace,
    snapshots: dict[str, CurrentDocumentState],
    exported: tuple[InteropExportedDocument, ...],
    archive_hashes: dict[str, str | None] | None = None,
) -> ET.Element:
    archive_hashes = archive_hashes or {}
    mets = ET.Element(
        _tag(METS_NS, "mets"),
        OBJID=getattr(workspace, "id", "workspace"),
        LABEL=getattr(workspace, "name", "ArchiveTrust export"),
    )
    ET.SubElement(
        mets,
        _tag(METS_NS, "metsHdr"),
        CREATEDATE=max(
            (state.latest_snapshot_recorded_at or "" for state in snapshots.values()),
            default="",
        ),
    )
    dmd = ET.SubElement(mets, _tag(METS_NS, "dmdSec"), ID="archiveTrustProvenance")
    md_wrap = ET.SubElement(dmd, _tag(METS_NS, "mdWrap"), MDTYPE="OTHER", OTHERMDTYPE="ArchiveTrust")
    xml_data = ET.SubElement(md_wrap, _tag(METS_NS, "xmlData"))
    provenance = ET.SubElement(
        xml_data,
        "ArchiveTrustExport",
        operationalReviewDataIncluded="true",
        evaluationReferencesIncluded="false",
    )
    for document_ref, state in sorted(snapshots.items()):
        document = state.latest_canonical_document
        if document is None:
            continue
        document_element = ET.SubElement(
            provenance,
            "Document",
            documentRef=document_ref,
            documentSnapshotId=document.document_snapshot_id,
            archiveObjectRef=document.archive_object_ref,
            archiveObjectHash=archive_hashes.get(document_ref) or "unavailable",
            reconciliationPolicyVersion=str(
                state.policy_versions.reconciliation_version or ""
            ),
            capabilityMatrixVersion=str(
                state.policy_versions.capability_matrix_version or ""
            ),
            confidencePolicyVersion=str(state.policy_versions.confidence_version or ""),
            alignmentAlgorithmVersion=str(state.policy_versions.alignment_version or ""),
            integrityStatus=state.integrity_status,
            unresolvedContentCount=str(state.unresolved_count),
            operationalReviewOutcomeCount=str(len(state.review_outcomes)),
        )
        for slot in state.slots:
            ET.SubElement(
                document_element,
                "CanonicalObservation",
                canonicalObservationId=slot.current.canonical_observation_id,
                semanticSlotId=slot.semantic_slot_id,
                classification=slot.classification.value,
                corrected=str(slot.corrected).lower(),
                value=_text_of(slot.current),
            )

    file_sec = ET.SubElement(mets, _tag(METS_NS, "fileSec"))
    page_group = ET.SubElement(file_sec, _tag(METS_NS, "fileGrp"), USE="PAGE_XML")
    alto_group = ET.SubElement(file_sec, _tag(METS_NS, "fileGrp"), USE="ALTO")
    for doc in exported:
        _mets_file(page_group, file_id=f"page-{_safe_name(doc.document_ref)}", path=doc.page_xml)
        _mets_file(alto_group, file_id=f"alto-{_safe_name(doc.document_ref)}", path=doc.alto_xml)
    return mets


def _mets_file(parent: ET.Element, *, file_id: str, path: Path) -> None:
    file_el = ET.SubElement(parent, _tag(METS_NS, "file"), ID=file_id, MIMETYPE="application/xml")
    ET.SubElement(
        file_el,
        _tag(METS_NS, "FLocat"),
        {"LOCTYPE": "URL", _tag(XLINK_NS, "href"): path.name},
    )


def _xml_root_check(name: str, path: Path, expected_root: str) -> InteropValidationCheck:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return InteropValidationCheck(name=name, ok=False, detail=f"parse_error:{exc}")
    return InteropValidationCheck(
        name=name,
        ok=root.tag == expected_root,
        detail=f"root={root.tag}",
    )


def _mets_referenced_files(mets_path: Path) -> set[str]:
    try:
        root = ET.parse(mets_path).getroot()
    except ET.ParseError:
        return set()
    return {
        node.attrib[_tag(XLINK_NS, "href")]
        for node in root.findall(f".//{{{METS_NS}}}FLocat")
        if _tag(XLINK_NS, "href") in node.attrib
    }


def _package_payload_files(package: InteropExportPackage) -> tuple[Path, ...]:
    return (package.mets_xml,) + tuple(
        path for doc in package.documents for path in (doc.page_xml, doc.alto_xml)
    )


def _package_manifest_paths(package: InteropExportPackage) -> tuple[tuple[str, Path], ...]:
    pairs: list[tuple[str, Path]] = [("mets", package.mets_xml)]
    for document in package.documents:
        safe_ref = _safe_name(document.document_ref)
        pairs.append((f"page_xml:{safe_ref}", document.page_xml))
        pairs.append((f"alto_xml:{safe_ref}", document.alto_xml))
    return tuple(pairs)


def _text_observations(observations: tuple[CanonicalObservation, ...]) -> tuple[CanonicalObservation, ...]:
    return tuple(obs for obs in observations if _text_of(obs))


def _text_of(canonical: CanonicalObservation) -> str:
    text = getattr(canonical.payload, "text", "")
    return text if isinstance(text, str) else ""


def _custom(canonical: CanonicalObservation, geometry: InteropGeometry | None = None) -> str:
    base = (
        f"canonical_observation_id:{canonical.canonical_observation_id}; "
        f"semantic_slot_id:{canonical.semantic_slot_id}; "
        f"observation_type:{canonical.observation_type.value}"
    )
    if geometry is None:
        return f"{base}; geometryPrecision:unavailable"
    return f"{base}; sourcePage:{geometry.page}; geometryPrecision:{geometry.box.precision.value}"


def _best_geometry(evidence) -> InteropGeometry | None:
    candidates = [item for item in evidence if item.bounding_box is not None and item.page is not None]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda item: (
            item.bounding_box.precision is not Precision.PIXEL_ACCURATE,
            item.page,
            item.evidence_id,
        ),
    )
    return InteropGeometry(page=selected.page, box=selected.bounding_box)


def _page_points(box: BoundingBox) -> str:
    return " ".join(
        f"{round(x)},{round(y)}"
        for x, y in (
            (box.x0, box.y0),
            (box.x1, box.y0),
            (box.x1, box.y1),
            (box.x0, box.y1),
        )
    )


def _alto_box(box: BoundingBox) -> dict[str, str]:
    return {
        "HPOS": str(round(box.x0)),
        "VPOS": str(round(box.y0)),
        "WIDTH": str(max(0, round(box.x1 - box.x0))),
        "HEIGHT": str(max(0, round(box.y1 - box.y0))),
    }


def _independent_parser_checks(
    package: InteropExportPackage,
) -> tuple[list[InteropValidationCheck], str]:
    try:
        from lxml import etree
    except ImportError:
        return [], "unavailable:lxml_not_installed"
    checks: list[InteropValidationCheck] = []
    for path in _package_payload_files(package):
        try:
            etree.parse(str(path))
            ok, detail = True, "lxml.etree parsed document"
        except etree.XMLSyntaxError as exc:
            ok, detail = False, f"lxml_parse_error:{exc}"
        checks.append(
            InteropValidationCheck(
                name=f"independent-parser:lxml:{path.name}", ok=ok, detail=detail
            )
        )
    return checks, "lxml.etree:passed" if all(check.ok for check in checks) else "lxml.etree:failed"


def _write_xml(path: Path, root: ET.Element) -> None:
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe or "document"


def _xml_id(value: str) -> str:
    safe = _safe_name(value)
    if safe[0].isdigit():
        return f"id_{safe}"
    return safe
