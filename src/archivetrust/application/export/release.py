"""One supported, versioned archival release containing JSON plus PAGE/ALTO/METS."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.application.export.interop import write_interop_package
from archivetrust.application.export.json_export import workspace_export, write_workspace_json
from archivetrust.domain.shared.ids import new_id
from archivetrust.infrastructure.storage.integrity import (
    build_file_digest_manifest,
    default_file_digest_manifest_path,
    verify_file_digest_manifest,
    write_file_digest_manifest,
)

RELEASE_MANIFEST_SCHEMA = "archivetrust.export_release.v1"
RELEASE_MANIFEST_NAME = "release_manifest.json"


class ReleaseDocumentVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    logical_document_id: str
    document_snapshot_id: str
    document_version: int
    unresolved_content_count: int
    review_outcome_count: int


class ReleaseFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    path: str
    size_bytes: int
    sha256: str


class ExportReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_schema: str = RELEASE_MANIFEST_SCHEMA
    release_id: str
    release_version: int
    created_at: str
    workspace_id: str
    workspace_name: str
    generator_version: str
    configuration_hash: str
    configuration_files: tuple[str, ...]
    documents: tuple[ReleaseDocumentVersion, ...]
    unresolved_content_count: int
    review_completion_summary: dict[str, int]
    operational_review_data_included: bool = True
    evaluation_references_included: bool = False
    integrity_verified: bool
    release_status: str
    files: tuple[ReleaseFile, ...]


class ExportRelease(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    root: Path
    canonical_json: Path
    interop_manifest: Path
    release_manifest: Path
    manifest: ExportReleaseManifest


def write_export_release(
    output_dir: str | Path,
    *,
    workspace,
    telemetry_source,
    acquisition_manager=None,
    configuration_dir: str | Path | None = None,
    release_id: str | None = None,
    current_state_service=None,
) -> ExportRelease:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    canonical_path = root / "canonical_documents.json"
    write_workspace_json(
        canonical_path,
        workspace=workspace,
        telemetry_source=telemetry_source,
        acquisition_manager=acquisition_manager,
        integrity_sidecar=True,
        current_state_service=current_state_service,
    )
    interop = write_interop_package(
        root / "interop",
        workspace=workspace,
        telemetry_source=telemetry_source,
        acquisition_manager=acquisition_manager,
        integrity_sidecars=True,
    )
    exported = workspace_export(
        workspace=workspace,
        telemetry_source=telemetry_source,
        acquisition_manager=acquisition_manager,
        current_state_service=current_state_service,
    )
    config_hash, config_files = _configuration_hash(configuration_dir)
    documents = tuple(
        ReleaseDocumentVersion(
            document_ref=document.document_ref,
            logical_document_id=document.canonical_document.logical_document_id,
            document_snapshot_id=document.canonical_document.document_snapshot_id,
            document_version=document.canonical_document.document_version,
            unresolved_content_count=len(document.contested_slots),
            review_outcome_count=len(document.review_outcomes),
        )
        for document in exported.documents
    )
    unresolved = sum(document.unresolved_content_count for document in documents)
    review_summary: dict[str, int] = {}
    for document in exported.documents:
        for outcome in document.review_outcomes:
            review_summary[outcome.outcome] = review_summary.get(outcome.outcome, 0) + 1

    paths = [canonical_path, interop.mets_xml, interop.manifest_json]
    paths.extend(path for document in interop.documents for path in (document.page_xml, document.alto_xml))
    paths.extend(
        sidecar
        for path in tuple(paths)
        if (sidecar := default_file_digest_manifest_path(path)).exists()
    )
    files = tuple(_release_file(root, path) for path in sorted(paths, key=lambda item: str(item)))
    integrity_ok = interop.validation_report.ok and all(
        verify_file_digest_manifest(path).ok
        for path in (canonical_path, interop.mets_xml)
        + tuple(path for document in interop.documents for path in (document.page_xml, document.alto_xml))
    )
    manifest = ExportReleaseManifest(
        release_id=release_id or new_id("export_release"),
        release_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        workspace_id=workspace.id,
        workspace_name=workspace.name,
        generator_version=_generator_version(),
        configuration_hash=config_hash,
        configuration_files=config_files,
        documents=documents,
        unresolved_content_count=unresolved,
        review_completion_summary=review_summary,
        integrity_verified=integrity_ok,
        release_status=(
            "integrity_failed"
            if not integrity_ok
            else "contains_unresolved_content"
            if unresolved
            else "complete"
        ),
        files=files,
    )
    manifest_path = root / RELEASE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_file_digest_manifest(manifest_path)
    return ExportRelease(
        root=root,
        canonical_json=canonical_path,
        interop_manifest=interop.manifest_json,
        release_manifest=manifest_path,
        manifest=manifest,
    )


def _release_file(root: Path, path: Path) -> ReleaseFile:
    digest = build_file_digest_manifest(path)
    return ReleaseFile(
        role=_role(path),
        path=path.relative_to(root).as_posix(),
        size_bytes=digest.size_bytes,
        sha256=digest.digest,
    )


def _role(path: Path) -> str:
    name = path.name
    if name == "canonical_documents.json":
        return "canonical_json"
    if name == "mets.xml":
        return "mets"
    if name.endswith(".page.xml"):
        return "page_xml"
    if name.endswith(".alto.xml"):
        return "alto_xml"
    if name.endswith(".integrity.json"):
        return "integrity_sidecar"
    if name == "archivetrust_interop_manifest.json":
        return "interop_manifest"
    return "release_payload"


def _configuration_hash(configuration_dir: str | Path | None) -> tuple[str, tuple[str, ...]]:
    digest = hashlib.sha256()
    if configuration_dir is None:
        return digest.hexdigest(), ()
    root = Path(configuration_dir)
    files = tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=str)) if root.exists() else ()
    relative: list[str] = []
    for path in files:
        name = path.relative_to(root).as_posix()
        relative.append(name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), tuple(relative)


def _generator_version() -> str:
    try:
        return version("archivetrust")
    except PackageNotFoundError:
        return "0.1.0+source"
