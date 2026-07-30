#!/usr/bin/env python
"""Produces the real RGB-normalization demonstration under docs/experiments/rgb-normalization-demo/.

A **separate, small demonstration**, deliberately not touching
`docs/experiments/baseline-comparison/` in any way -- that baseline ran before this stage existed and
its Transkribus run had no page image at all, so no normalization event may be added to it
(`tests/htr/preprocessing/test_baseline_history_not_fabricated.py` enforces this).

What this exercises, for real:

* the real `NormalizationService` over real image bytes,
* a real `FileTelemetrySink` on disk, with its hash-chain sidecar,
* real `ImageNormalizationStarted`/`DerivedImageArtifactCreated`/`ImageNormalizationCompleted`
  events, and a real `ImageNormalizationFailed` for a genuinely undecodable input,
* a real export package + manifest,
* a real researcher-confirmed import association,
* a destroy-and-replay reconstruction, verified against the in-process artifacts.

Run:

    QT_QPA_PLATFORM=offscreen PYTHONPATH=src .venv/Scripts/python.exe \
        scripts/run_rgb_normalization_demo.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from archivetrust.application.htr_journal import HtrJournal  # noqa: E402
from archivetrust.htr.experiment.baseline_template import (  # noqa: E402
    build_baseline_experiment,
    swedish_lion_1_page_level_definition,
)
from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.htr.preprocessing.export_package import (  # noqa: E402
    PageImageSelection,
    build_export_package,
)
from archivetrust.htr.preprocessing.models import NormalizationError  # noqa: E402
from archivetrust.htr.preprocessing.normalization_service import (  # noqa: E402
    NormalizationService,
)
from archivetrust.infrastructure.storage.blob_store import (  # noqa: E402
    ContentAddressedBlobStore,
)
from archivetrust.infrastructure.storage.telemetry_sink import (  # noqa: E402
    FileTelemetrySink,
)
from archivetrust.providers.transkribus.external_import import (  # noqa: E402
    ExternalImport,
    ExternalImportFormat,
    associate_imported_result_with_normalized_page,
)
from tests.htr.preprocessing import _images  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "rgb-normalization-demo"
REAL_IMAGE = REPO_ROOT / "tests" / "fixtures" / "htr" / "trolldomskommissionen_sample_line.jpg"
ARCHIVE_OBJECT_REF = "archive_object_rgb_normalization_demo"
AT = "2026-07-30T00:00:00+00:00"


def main() -> int:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    telemetry_path = OUTPUT_DIR / "normalization_events.jsonl"
    store = DurableHtrResearchStore(
        FileTelemetrySink(telemetry_path), actor_id="rgb-normalization-demo"
    )
    service = NormalizationService(
        blob_store=ContentAddressedBlobStore(OUTPUT_DIR / "blobs"), store=store
    )

    # -- The selected pages ---------------------------------------------------------------------
    #
    # Page 1 is the one *real* historical-document image this repository contains: a genuine line
    # crop from Riksarkivet's trolldomskommissionen_lines dataset (tests/fixtures/htr/README.md).
    # It is a line crop, not a page, and is used here as a **stand-in page** -- said plainly rather
    # than described as a page. Pages 2-4 are synthesized in-process by tests/htr/preprocessing/
    # _images.py to cover colour modes the real JPEG cannot (CMYK, 16-bit, palette+transparency).
    selections = (
        PageImageSelection(
            page_id="page_demo_real_line_as_page",
            image_bytes=REAL_IMAGE.read_bytes(),
            archive_object_ref=ARCHIVE_OBJECT_REF,
            page_number=1,
            original_filename=REAL_IMAGE.name,
        ),
        PageImageSelection(
            page_id="page_demo_cmyk",
            image_bytes=_images.cmyk(),
            archive_object_ref=ARCHIVE_OBJECT_REF,
            page_number=2,
            original_filename="synthetic_cmyk.tif",
        ),
        PageImageSelection(
            page_id="page_demo_16bit_gray",
            image_bytes=_images.grayscale_16bit(),
            archive_object_ref=ARCHIVE_OBJECT_REF,
            page_number=3,
            original_filename="synthetic_i16.png",
        ),
        PageImageSelection(
            page_id="page_demo_palette_transparency",
            image_bytes=_images.palette_with_transparency(),
            archive_object_ref=ARCHIVE_OBJECT_REF,
            page_number=4,
            original_filename="synthetic_palette_transparency.png",
        ),
    )

    package = build_export_package(
        selections,
        destination=OUTPUT_DIR / "export-packages",
        service=service,
        correlation_id="rgb_normalization_demo_run_1",
        package_id="export_package_rgb_normalization_demo",
    )

    # -- A real recorded failure ----------------------------------------------------------------
    # Not a simulated one: genuinely undecodable bytes, producing a genuine ImageNormalizationFailed.
    failure_recorded = None
    try:
        service.normalize_and_record(
            _images.undecodable(),
            page_id="page_demo_undecodable",
            correlation_id="rgb_normalization_demo_run_1",
        )
    except NormalizationError as exc:
        failure_recorded = exc.failure

    # -- The experiment version this configuration belongs to -----------------------------------
    built = build_baseline_experiment(
        swedish_lion_1_page_level_definition(
            dataset_description=(
                "Four page-level inputs for the RGB-normalization demonstration: one real "
                "Riksarkivet trolldomskommissionen line crop used as a stand-in page, plus three "
                "synthesized images covering CMYK, 16-bit grayscale, and palette-with-transparency. "
                "This is a stage demonstration, not a research corpus."
            ),
            dataset_version_note="DatasetVersion.version=1; no membership change has occurred.",
            scope_caveats=(
                "Page 1 is a line crop used as a stand-in page -- this repository contains no real "
                "full-page archival scan.",
                "Pages 2-4 are synthesized colour-mode fixtures, not archival material.",
                "No Transkribus upload was performed and no real Transkribus result was imported; "
                "the import association below uses a hand-authored ExternalImport to demonstrate "
                "the mechanism, and is labeled as such.",
            ),
        ),
        research_project_id="research_project_rgb_normalization_demo",
        dataset_version_id="dataset_version_rgb_normalization_demo",
        created_at=AT,
    )

    # -- The researcher-confirmed association ---------------------------------------------------
    first_artifact = package.normalized_artifacts[0]
    association = associate_imported_result_with_normalized_page(
        ExternalImport.create(
            method_run_id="method_run_rgb_normalization_demo",
            source_file_path=str(
                REPO_ROOT / "tests" / "fixtures" / "transkribus" / "sample_page.xml"
            ),
            export_format=ExternalImportFormat.PAGE_XML,
            imported_by="rgb-normalization-demo",
            imported_at=AT,
        ),
        normalized_artifact_id=first_artifact.normalized_artifact_id,
        normalized_content_hash=first_artifact.normalized_content_hash,
        confirmed_by="rgb-normalization-demo",
        associated_at=AT,
        export_package_id=package.package_id,
    )

    # -- Destroy and replay ---------------------------------------------------------------------
    in_process = {
        a.normalized_artifact_id: a for a in store.normalized_page_artifacts()
    }
    del service, store
    replayed = HtrJournal().replay(FileTelemetrySink(telemetry_path).all_events())
    reconstructed = {
        a.normalized_artifact_id: a for a in replayed.normalized_page_artifacts()
    }
    assert reconstructed == in_process, "replay did not reproduce the in-process artifacts"

    summary = {
        "demonstration": "RGB normalization stage, end to end",
        "not_the_committed_baseline": (
            "This directory is separate from docs/experiments/baseline-comparison/, which was not "
            "read, written, or modified by this run. That baseline's Transkribus MethodRun had no "
            "page image (input_crop_id=None) and no normalization event was added to it."
        ),
        "normalization_version": first_artifact.normalization_version,
        "configuration_hash": first_artifact.configuration_hash,
        "experiment_version_id": built.experiment_version.experiment_version_id,
        "experiment_version_requires_normalization": (
            built.definition.image_color_normalization.required
        ),
        "telemetry_file": telemetry_path.name,
        "telemetry_event_counts": _event_counts(telemetry_path),
        "export_package": {
            "package_id": package.package_id,
            "manifest": str(package.manifest_path.relative_to(OUTPUT_DIR)),
            "entries": [
                {
                    "page_id": entry.page_id,
                    "original_filename": entry.original_filename,
                    "source_color_mode": entry.source_color_mode,
                    "original_content_hash": entry.original_content_hash,
                    "normalized_content_hash": entry.normalized_content_hash,
                    "output_filename": entry.output_filename,
                    "dimensions": f"{entry.width}x{entry.height}",
                }
                for entry in package.manifest.entries
            ],
        },
        "artifacts": [
            {
                "page_id": a.page_id,
                "source_color_mode": a.source_color_mode,
                "source_bit_depth": a.source_bit_depth,
                "source_channel_count": a.source_channel_count,
                "source_format": a.source_format,
                "source_alpha_present": a.source_alpha_present,
                "source_icc_profile_present": a.source_icc_profile_present,
                "source_exif_orientation": a.source_exif_orientation,
                "output_color_mode": a.output_color_mode,
                "output_bit_depth": a.output_bit_depth,
                "output_dimensions": f"{a.output_width}x{a.output_height}",
                "alpha_compositing_applied": a.alpha_compositing_applied,
                "compositing_background": a.compositing_background,
                "exif_orientation_applied": a.exif_orientation_applied,
                "source_content_hash": a.source_content_hash,
                "normalized_content_hash": a.normalized_content_hash,
                "warnings": list(a.warnings),
            }
            for a in sorted(package.normalized_artifacts, key=lambda x: x.page_id)
        ],
        "recorded_failure": (
            {
                "page_id": failure_recorded.page_id,
                "category": failure_recorded.category.value,
                "reason": failure_recorded.reason,
            }
            if failure_recorded
            else None
        ),
        "import_association": {
            "target_kind": association.target_kind.value,
            "target_id": association.target_id,
            "correspondence_basis": association.correspondence_basis.value,
            "caveat": (
                "Researcher-confirmed correspondence, NOT cryptographic proof: Transkribus does "
                "not preserve or return ArchiveTrust's content hashes."
            ),
            "confirmed_by": association.associated_by,
            "note": (
                "The ExternalImport here points at the hand-authored PAGE XML test fixture. No real "
                "Transkribus upload or export occurred in this demonstration."
            ),
        },
        "replay_verified": True,
    }

    (OUTPUT_DIR / "demonstration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _event_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            kind = json.loads(line)["kind"]
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
