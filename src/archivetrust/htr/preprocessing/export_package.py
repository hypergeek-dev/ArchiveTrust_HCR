"""Export-package preparation for external, manual Transkribus processing.

The ArchiveTrust half of a workflow whose middle is deliberately outside this application:

    select source pages -> normalize -> write export package + manifest
        -> [ researcher manually uploads to Transkribus, runs Swedish Lion I, exports the result ]
        -> import the PAGE/ALTO result -> record the researcher-confirmed association

**ArchiveTrust never performs the upload.** This module writes a directory of files to local disk and
stops. It opens no socket, holds no credential, and has no code path to Transkribus -- the same
enforcement-by-design that `providers/transkribus/README.md` documents for the import side ("A
researcher opens Transkribus (a separate, external, browser-based or desktop application -- not
reachable from or launched by ArchiveTrust)"). The manifest's own `external_processing_boundary` field
states this in the artifact itself, so a package handed to a colleague carries the fact with it.

## Why the manifest matters more than it looks

Transkribus does not preserve, echo, or accept ArchiveTrust's content hashes. When a result comes back
there is therefore **no cryptographic link** from the imported PAGE XML to the normalized image that
produced it -- only the researcher's own knowledge of which file they uploaded. The manifest is what
makes that knowledge checkable rather than remembered: it records, per page, the original hash, the
normalized hash, and the exact output filename that was handed over. `associate_imported_result` then
records the correspondence as explicitly `RESEARCHER_CONFIRMED`, never as proof (see
`providers/transkribus/external_import.py::CorrespondenceBasis`).

A package is written only for pages that **successfully** normalized. A page whose normalization
failed raises before any file is written for it, so an un-normalized original cannot reach a package
directory -- the guarantee is structural, not a check a caller might skip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import new_id
from archivetrust.htr.preprocessing.models import (
    RGB_NORMALIZATION_STAGE,
    NormalizationError,
    NormalizationFailure,
    NormalizationFailureCategory,
    NormalizedPageArtifact,
)
from archivetrust.htr.preprocessing.normalization_service import NormalizationService

MANIFEST_FILENAME = "manifest.json"

EXTERNAL_PROCESSING_BOUNDARY = (
    "Transkribus is an external service. ArchiveTrust performs no upload, holds no credential, and "
    "makes no network call: a researcher must manually upload the images in this package to "
    "Transkribus, run Swedish Lion I there, and manually export the result back to local disk."
)
"""Recorded verbatim into every manifest. Reuses the established "external service, manual only"
language from `providers/transkribus/README.md` rather than paraphrasing it into a second wording that
could drift into implying an automated upload."""


@dataclass(frozen=True)
class PageImageSelection:
    """One selected source page: the bytes to normalize plus the corpus identity to record.

    Bytes rather than a path, so a caller may select pages from a blob store, an acquisition folder,
    or a test fixture without this module growing a second input mechanism.
    """

    page_id: str
    image_bytes: bytes
    archive_object_ref: str
    page_number: int
    original_filename: str | None = None
    """The source file's own name, when there was one -- recorded for the researcher's benefit so
    they can tell which archival file a package entry came from. `None` when the bytes did not come
    from a named file; never invented."""
    output_stem: str | None = None
    """Overrides the package filename stem, which defaults to
    `<archive_object_ref>_p<page_number>`. An `archive_object_ref` is `document_<sha256>`, so the
    default stem is 79 characters; a 60-page package of those sits close enough to Windows'
    `MAX_PATH` to have already broken `git worktree add` on this repository. A caller may supply a
    short, still-unique stem instead. The manifest continues to record the real
    `archive_object_ref` in its own field, so shortening the *filename* loses no identity -- the
    entry still states which archival document it came from, and `output_filename` still states
    exactly what the researcher uploads."""


class ExportPackageEntry(BaseModel):
    """One page in an export package's manifest."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    archive_object_ref: str
    page_number: int
    original_filename: str | None = None
    original_content_hash: str
    normalized_content_hash: str
    output_filename: str
    """The file's name *inside the package directory* -- what the researcher actually uploads, and
    therefore the only handle they will have when confirming correspondence later."""
    width: int
    height: int
    source_color_mode: str
    """Carried into the manifest because it is the single most useful fact for a researcher auditing
    why normalization was necessary for this page at all."""


class ExportPackageManifest(BaseModel):
    """The manifest written alongside a package's images.

    Frozen and content-complete: everything needed to identify what was handed to an external service
    and what it was derived from, without consulting the telemetry log. That redundancy is deliberate
    -- the package leaves the workspace, the telemetry log does not.
    """

    model_config = ConfigDict(frozen=True)

    package_id: str
    stage: str = RGB_NORMALIZATION_STAGE
    normalization_version: str
    configuration_hash: str
    created_at: str
    entries: tuple[ExportPackageEntry, ...]
    external_processing_boundary: str = EXTERNAL_PROCESSING_BOUNDARY
    correspondence_caveat: str = (
        "Transkribus does not preserve or return these content hashes. Any association between an "
        "imported Transkribus result and a normalized page below is a researcher-confirmed "
        "correspondence, not cryptographic proof."
    )

    @model_validator(mode="after")
    def _validate(self) -> "ExportPackageManifest":
        if not self.entries:
            raise ValueError(
                "ExportPackageManifest.entries must be non-empty -- an empty package would claim a "
                "handover that has nothing in it"
            )
        filenames = [entry.output_filename for entry in self.entries]
        if len(set(filenames)) != len(filenames):
            raise ValueError(
                "ExportPackageManifest output filenames must be unique -- a collision would make "
                "two pages indistinguishable to the researcher doing the upload"
            )
        return self


@dataclass(frozen=True)
class ExportPackage:
    """A written package: where it is, what is in it, and the artifacts it was built from."""

    package_id: str
    directory: Path
    manifest: ExportPackageManifest
    manifest_path: Path
    normalized_artifacts: tuple[NormalizedPageArtifact, ...]


def build_export_package(
    selections: tuple[PageImageSelection, ...],
    *,
    destination: Path | str,
    service: NormalizationService,
    correlation_id: str | None = None,
    caused_by: str | None = None,
    package_id: str | None = None,
    created_at: str | None = None,
) -> ExportPackage:
    """Normalizes every selected page and writes a package directory plus its manifest.

    Every page is normalized through the real `NormalizationService`, so every page in a package has
    real telemetry behind it. A `NormalizationError` from any page propagates: the package is not
    written, because a partially-normalized package whose manifest omitted the failures would be
    exactly the "silently submit the original" outcome this stage exists to prevent.
    """
    if not selections:
        raise ValueError("build_export_package requires at least one selected page")

    package_id = package_id or new_id("export_package")
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    directory = Path(destination) / package_id

    artifacts: list[NormalizedPageArtifact] = []
    entries: list[ExportPackageEntry] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NormalizationError(
            NormalizationFailure.create(
                page_id=selections[0].page_id,
                category=NormalizationFailureCategory.ARTIFACT_NOT_PERSISTED,
                reason=f"cannot create package directory {directory}: {type(exc).__name__}: {exc}",
                configuration_hash=service.config.configuration_hash,
                occurred_at=created_at,
            )
        ) from exc

    for selection in selections:
        # Raises (after recording an ImageNormalizationFailed event) if this page cannot be
        # normalized. Nothing is written for it, and no later page is written either.
        recorded = service.normalize_and_record(
            selection.image_bytes,
            page_id=selection.page_id,
            correlation_id=correlation_id,
            caused_by=caused_by,
        )
        artifact = recorded.normalized_artifact
        stem = (
            selection.output_stem
            or f"{selection.archive_object_ref}_p{selection.page_number:04d}"
        )
        output_filename = f"{stem}.{service.config.output_format.lower()}"
        target = directory / output_filename
        try:
            # Read back through the content-addressed store rather than reusing an in-memory copy:
            # this verifies the stored artifact hashes to what the manifest is about to claim, so a
            # package can never disagree with the blob store it came from.
            target.write_bytes(service.read_normalized_bytes(artifact))
        except OSError as exc:
            failure = NormalizationFailure.create(
                page_id=selection.page_id,
                category=NormalizationFailureCategory.ARTIFACT_NOT_PERSISTED,
                reason=f"cannot write {target}: {type(exc).__name__}: {exc}",
                configuration_hash=service.config.configuration_hash,
                occurred_at=created_at,
                source_content_hash=artifact.source_content_hash,
            )
            raise NormalizationError(failure) from exc

        artifacts.append(artifact)
        entries.append(
            ExportPackageEntry(
                page_id=selection.page_id,
                archive_object_ref=selection.archive_object_ref,
                page_number=selection.page_number,
                original_filename=selection.original_filename,
                original_content_hash=artifact.source_content_hash,
                normalized_content_hash=artifact.normalized_content_hash,
                output_filename=output_filename,
                width=artifact.output_width,
                height=artifact.output_height,
                source_color_mode=artifact.source_color_mode,
            )
        )

    manifest = ExportPackageManifest(
        package_id=package_id,
        normalization_version=artifacts[0].normalization_version,
        configuration_hash=service.config.configuration_hash,
        created_at=created_at,
        entries=tuple(entries),
    )
    manifest_path = directory / MANIFEST_FILENAME
    try:
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise NormalizationError(
            NormalizationFailure.create(
                page_id=selections[0].page_id,
                category=NormalizationFailureCategory.MANIFEST_NOT_WRITTEN,
                reason=f"cannot write {manifest_path}: {type(exc).__name__}: {exc}",
                configuration_hash=service.config.configuration_hash,
                occurred_at=created_at,
            )
        ) from exc

    return ExportPackage(
        package_id=package_id,
        directory=directory,
        manifest=manifest,
        manifest_path=manifest_path,
        normalized_artifacts=tuple(artifacts),
    )


def read_manifest(directory: Path | str) -> ExportPackageManifest:
    """Reads a written package's manifest back, validated. Used by the import side to look up which
    normalized hash a given output filename corresponded to."""
    path = Path(directory) / MANIFEST_FILENAME
    return ExportPackageManifest.model_validate_json(path.read_text(encoding="utf-8"))
