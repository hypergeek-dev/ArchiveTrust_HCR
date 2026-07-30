"""Line-targeted export preparation: hand Transkribus the *same lines* the local recognizers got.

**Additive to `export_package.py`, never a replacement.** The two packages answer different
questions and both are produced:

| Package | What it tests | Who segments |
|---|---|---|
| `export_package.py` (whole page) | Lion I's real end-to-end workflow | Transkribus, internally |
| this module (line-targeted) | Lion I's *recognition* against the same lines SATRN/Florence-2 read | ArchiveTrust's `htr/segmentation/` |

`providers/transkribus/adapter.py::get_capabilities()` declares **both** `line_level_supported=True`
and `page_level_supported=True`, so both workflows are genuinely within the method's declared
capability. The earlier baseline exercised only the page-level path because that fixture's text had
no correspondence to the shared ground truth -- a data limitation of that run, not a capability
limit of the method.

## Why PAGE XML, and not a directory of line-crop images

Two options were considered for "give Transkribus our lines":

* **(a) Upload each `InputCrop` PNG as its own image.** Byte-identical to what SATRN and Florence-2
  read -- but it is not a workflow anyone runs. Each line would become a separate one-line "page" in
  a Transkribus document, Lion I would still run its own layout analysis over each strip, and a
  researcher would be hand-managing hundreds of single-line uploads per volume.
* **(b) Upload the page image plus a PAGE XML carrying our line geometry with empty `TextEquiv`,
  and run recognition only.** This *is* Transkribus's "skip layout analysis" workflow, it is one
  upload per page rather than one per line, and the line regions Lion I recognizes are exactly the
  ones this repository detected.

**(b) is what this module produces.** (a) was rejected as unrealistic, not as unimplementable.

## The honest limit of (b) -- state this wherever a three-way comparison is claimed

Choosing (b) buys workflow realism at a real cost: Transkribus extracts its own pixels from the page
using our `Coords`. The line *regions* are identical; the line *bytes* are not guaranteed to be.
Transkribus applies its own polygon extraction, and may deskew, pad, or rescale a line before
recognition. So this export gives all three methods **the same detected line geometry**, not the
byte-identical `InputCrop` that `InputCropHashMismatchError` guards between SATRN and Florence-2.
A three-way comparison built on it is controlled *for segmentation*, and must not be described as
byte-identical-input controlled.

## What the manifest is for

Each exported line records the `InputCrop.crop_id` and `InputCrop.hash` it corresponds to. If a
researcher does manually run Lion I on this package and import the result, that record is what lets
ArchiveTrust associate a returned `TextLine` with the exact `InputCrop` SATRN and Florence-2 read.
That association remains **researcher-confirmed correspondence, not cryptographic proof** -- the
same standing caveat `export_package.py` and `providers/transkribus/external_import.py` already
carry, for the same reason: Transkribus neither preserves nor returns ArchiveTrust's hashes.

**No upload happens here, and none is possible.** This module writes local files and stops. Nothing
in it opens a socket or holds a credential.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.shared.ids import new_id
from archivetrust.htr.corpus.models import InputCrop, Region, TextLine
from archivetrust.htr.preprocessing.export_package import EXTERNAL_PROCESSING_BOUNDARY
from archivetrust.htr.preprocessing.models import NormalizedPageArtifact
from archivetrust.providers.transkribus.page_xml_writer import (
    PageXmlLine,
    PageXmlRegion,
    build_layout_only_page_xml,
)

LINE_TARGETED_MANIFEST_FILENAME = "line_targeted_manifest.json"

BYTE_IDENTITY_CAVEAT = (
    "This export gives Transkribus the same detected line GEOMETRY that SATRN and Florence-2 "
    "received as byte-identical InputCrops -- not the same bytes. Transkribus extracts its own "
    "pixels from the page image using these Coords and may deskew, pad or rescale a line before "
    "recognition. A comparison built on this export is controlled for segmentation; it must not be "
    "described as byte-identical-input controlled."
)

RECOGNITION_ONLY_INSTRUCTIONS = (
    "Upload the page image and its accompanying .xml (PAGE XML) together to Transkribus as one "
    "document, then run Swedish Lion I with layout analysis DISABLED / 'text recognition only'. "
    "The XML already contains every text line's Coords and Baseline with empty TextEquiv; letting "
    "Transkribus re-run layout analysis would discard the segmentation this package exists to "
    "share, and would silently turn a segmentation-controlled comparison back into an end-to-end "
    "one."
)

MANUAL_STEP_REQUIRED = (
    "Producing Transkribus results from this package requires a human to manually upload it, run "
    "Swedish Lion I, and manually export the result back to local disk. ArchiveTrust performs no "
    "upload and has no code path to Transkribus; preparing this package does not obtain, schedule, "
    "or imply any Transkribus result."
)


class LineTargetedLineEntry(BaseModel):
    """One exported line: where it is on the page, and which `InputCrop` it corresponds to."""

    model_config = ConfigDict(frozen=True)

    line_id: str
    """The `TextLine.text_line_id`, written verbatim as the PAGE XML `TextLine/@id`, so a returned
    file's line ids map straight back without a lookup table."""
    region_id: str
    reading_order_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    input_crop_id: str
    input_crop_hash: str
    """The exact `InputCrop.hash` SATRN and Florence-2 read for this line. Recorded so a later
    import can be associated with it -- researcher-confirmed, never proof."""
    crop_width: int | None = None
    crop_height: int | None = None


class LineTargetedPageEntry(BaseModel):
    """One page in a line-targeted package."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    archive_object_ref: str
    page_number: int
    original_content_hash: str
    normalized_content_hash: str
    image_filename: str
    page_xml_filename: str
    image_width: int
    image_height: int
    region_count: int
    line_count: int
    lines: tuple[LineTargetedLineEntry, ...]

    @model_validator(mode="after")
    def _validate(self) -> "LineTargetedPageEntry":
        if not self.lines:
            raise ValueError(
                f"LineTargetedPageEntry for page {self.page_id!r} has no lines -- a line-targeted "
                "export of a page with no detected lines would hand Transkribus an empty layout"
            )
        return self


class LineTargetedManifest(BaseModel):
    """The manifest written alongside a line-targeted package's images and PAGE XML files.

    Content-complete for the same reason `ExportPackageManifest` is: the package leaves the
    workspace and the telemetry log does not.
    """

    model_config = ConfigDict(frozen=True)

    package_id: str
    kind: str = "line_targeted"
    created_at: str
    segmentation_adapter_name: str
    segmentation_confound_statement: str
    """The Florence-2-family detector confound, carried into the manifest so it travels with a
    package handed to a colleague -- it is the crops in this package that it is about."""
    normalization_version: str
    configuration_hash: str
    entries: tuple[LineTargetedPageEntry, ...]
    external_processing_boundary: str = EXTERNAL_PROCESSING_BOUNDARY
    manual_step_required: str = MANUAL_STEP_REQUIRED
    byte_identity_caveat: str = BYTE_IDENTITY_CAVEAT
    recognition_only_instructions: str = RECOGNITION_ONLY_INSTRUCTIONS
    correspondence_caveat: str = (
        "Transkribus does not preserve or return the InputCrop hashes recorded here. Associating "
        "an imported Transkribus line with the InputCrop named beside it is a researcher-confirmed "
        "correspondence, not cryptographic proof."
    )

    @model_validator(mode="after")
    def _validate(self) -> "LineTargetedManifest":
        if not self.entries:
            raise ValueError("LineTargetedManifest.entries must be non-empty")
        filenames = [entry.page_xml_filename for entry in self.entries] + [
            entry.image_filename for entry in self.entries
        ]
        if len(set(filenames)) != len(filenames):
            raise ValueError(
                "LineTargetedManifest filenames must be unique -- a collision would make two pages "
                "indistinguishable to the researcher doing the upload"
            )
        return self

    def crop_hash_for_line(self, line_id: str) -> str | None:
        """The `InputCrop.hash` an imported line id corresponds to, or `None` if this package never
        exported that line. An honest `None` rather than a guess: an unrecognized line id in a
        returned file means Transkribus produced a line this package did not ask for (it re-ran
        layout analysis), which is exactly the case a caller must be able to detect."""
        for entry in self.entries:
            for line in entry.lines:
                if line.line_id == line_id:
                    return line.input_crop_hash
        return None


@dataclass(frozen=True)
class LineTargetedSelection:
    """One page to export: its already-normalized artifact and its already-detected segmentation.

    Takes the *outputs* of the normalization and segmentation stages rather than re-running either.
    The whole-page package (`build_export_package`) has already normalized these pages and recorded
    the telemetry for it; re-normalizing here would emit a second, duplicate set of normalization
    events for the same bytes and claim work that did not happen twice.
    """

    page_id: str
    archive_object_ref: str
    page_number: int
    normalized_artifact: NormalizedPageArtifact
    normalized_image_bytes: bytes
    regions: tuple[Region, ...]
    text_lines: tuple[TextLine, ...]
    input_crops: tuple[InputCrop, ...]
    output_stem: str | None = None
    """Overrides the package filename stem for this page's `.png`/`.xml` pair, defaulting to
    `<archive_object_ref>_p<page_number>`. Same rationale as
    `export_package.py::PageImageSelection.output_stem`: the default stem is 79 characters and a
    60-page package of them approaches Windows' `MAX_PATH`. The manifest entry continues to record
    the real `archive_object_ref`, `page_id` and both content hashes, so the shorter filename costs
    no identity."""


@dataclass(frozen=True)
class LineTargetedPackage:
    """A written line-targeted package."""

    package_id: str
    directory: Path
    manifest: LineTargetedManifest
    manifest_path: Path
    page_xml_paths: tuple[Path, ...]
    image_paths: tuple[Path, ...]


def build_line_targeted_export(
    selections: tuple[LineTargetedSelection, ...],
    *,
    destination: Path | str,
    segmentation_adapter_name: str,
    segmentation_confound_statement: str,
    normalization_version: str,
    configuration_hash: str,
    package_id: str | None = None,
    created_at: str | None = None,
) -> LineTargetedPackage:
    """Writes one page image + one layout-only PAGE XML per selected page, plus the manifest.

    Raises `ValueError` when a selection carries a line with no matching `InputCrop`: the entire
    point of this package is the crop correspondence, so a line that cannot state which `InputCrop`
    it is would silently weaken exactly the guarantee the manifest exists to record.
    """
    if not selections:
        raise ValueError("build_line_targeted_export requires at least one selected page")

    package_id = package_id or new_id("line_targeted_package")
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    directory = Path(destination) / package_id
    directory.mkdir(parents=True, exist_ok=True)

    entries: list[LineTargetedPageEntry] = []
    page_xml_paths: list[Path] = []
    image_paths: list[Path] = []

    for selection in selections:
        crop_by_line = {crop.text_line_id: crop for crop in selection.input_crops}
        lines_by_region: dict[str, list[TextLine]] = {}
        for line in selection.text_lines:
            if line.text_line_id not in crop_by_line:
                raise ValueError(
                    f"text line {line.text_line_id!r} on page {selection.page_id!r} has no "
                    "InputCrop -- a line-targeted export cannot record the crop correspondence "
                    "that is its entire purpose"
                )
            lines_by_region.setdefault(line.region_id, []).append(line)

        stem = (
            selection.output_stem
            or f"{selection.archive_object_ref}_p{selection.page_number:04d}"
        )
        image_filename = f"{stem}.png"
        page_xml_filename = f"{stem}.xml"

        image_path = directory / image_filename
        image_path.write_bytes(selection.normalized_image_bytes)
        image_paths.append(image_path)

        xml_regions = tuple(
            PageXmlRegion(
                region_id=region.region_id,
                x0=region.bounding_box.x0,
                y0=region.bounding_box.y0,
                x1=region.bounding_box.x1,
                y1=region.bounding_box.y1,
                reading_order_index=region.order_index or 0,
                region_type="paragraph",
                lines=tuple(
                    PageXmlLine(
                        line_id=line.text_line_id,
                        x0=line.bounding_box.x0,
                        y0=line.bounding_box.y0,
                        x1=line.bounding_box.x1,
                        y1=line.bounding_box.y1,
                        reading_order_index=line.reading_order_index,
                        input_crop_id=crop_by_line[line.text_line_id].crop_id,
                        input_crop_hash=crop_by_line[line.text_line_id].hash,
                    )
                    for line in sorted(
                        lines_by_region.get(region.region_id, ()),
                        key=lambda line: line.reading_order_index,
                    )
                ),
            )
            for region in sorted(selection.regions, key=lambda r: r.order_index or 0)
            if lines_by_region.get(region.region_id)
        )
        if not xml_regions:
            raise ValueError(
                f"page {selection.page_id!r} has no region carrying any detected line -- refusing "
                "to write a layout-only PAGE XML with nothing in it"
            )

        page_xml = build_layout_only_page_xml(
            image_filename=image_filename,
            image_width=selection.normalized_artifact.output_width,
            image_height=selection.normalized_artifact.output_height,
            regions=xml_regions,
            created_at=created_at,
            segmentation_adapter_name=segmentation_adapter_name,
            extra_metadata=(
                ("archiveTrustPageId", selection.page_id),
                ("normalizedContentHash", selection.normalized_artifact.normalized_content_hash),
                ("byteIdentityCaveat", BYTE_IDENTITY_CAVEAT),
                ("segmentationConfound", segmentation_confound_statement),
            ),
        )
        page_xml_path = directory / page_xml_filename
        page_xml_path.write_text(page_xml, encoding="utf-8")
        page_xml_paths.append(page_xml_path)

        entries.append(
            LineTargetedPageEntry(
                page_id=selection.page_id,
                archive_object_ref=selection.archive_object_ref,
                page_number=selection.page_number,
                original_content_hash=selection.normalized_artifact.source_content_hash,
                normalized_content_hash=selection.normalized_artifact.normalized_content_hash,
                image_filename=image_filename,
                page_xml_filename=page_xml_filename,
                image_width=selection.normalized_artifact.output_width,
                image_height=selection.normalized_artifact.output_height,
                region_count=len(xml_regions),
                line_count=len(selection.text_lines),
                lines=tuple(
                    LineTargetedLineEntry(
                        line_id=line.text_line_id,
                        region_id=line.region_id,
                        reading_order_index=line.reading_order_index,
                        x0=line.bounding_box.x0,
                        y0=line.bounding_box.y0,
                        x1=line.bounding_box.x1,
                        y1=line.bounding_box.y1,
                        input_crop_id=crop_by_line[line.text_line_id].crop_id,
                        input_crop_hash=crop_by_line[line.text_line_id].hash,
                        crop_width=crop_by_line[line.text_line_id].width,
                        crop_height=crop_by_line[line.text_line_id].height,
                    )
                    for line in selection.text_lines
                ),
            )
        )

    manifest = LineTargetedManifest(
        package_id=package_id,
        created_at=created_at,
        segmentation_adapter_name=segmentation_adapter_name,
        segmentation_confound_statement=segmentation_confound_statement,
        normalization_version=normalization_version,
        configuration_hash=configuration_hash,
        entries=tuple(entries),
    )
    manifest_path = directory / LINE_TARGETED_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return LineTargetedPackage(
        package_id=package_id,
        directory=directory,
        manifest=manifest,
        manifest_path=manifest_path,
        page_xml_paths=tuple(page_xml_paths),
        image_paths=tuple(image_paths),
    )


def read_line_targeted_manifest(directory: Path | str) -> LineTargetedManifest:
    """Reads a written line-targeted manifest back, validated -- the lookup path an import would
    use to find which `InputCrop` a returned line id corresponds to."""
    path = Path(directory) / LINE_TARGETED_MANIFEST_FILENAME
    return LineTargetedManifest.model_validate_json(path.read_text(encoding="utf-8"))
