#!/usr/bin/env python
"""Stage 2 + Checkpoint 3 smoke test -- Swedish Historical HTR Technical Reliability Screening.

Runs the whole new stage for real, on a handful of real corpus pages, and stops:

    real segmentation (Florence-2 <OD>)  ->  real byte-identical line crops
        -> real SATRN inference on those crops
        -> real Florence-2 inference on the *same* crops (hash equality enforced)
        -> real RGB normalization + whole-page Transkribus export package (no upload)
        -> real line-targeted Transkribus export package (no upload)

!! CONFOUND -- restated here because this script is what produces the numbers a reader will quote.
The line detector is a **Florence-2-family model** (`nazounoryuu/florence_base__mixed__page__line_od`).
Its crops are fed byte-identically to both SATRN and Florence-2, so one screened method's own model
family controls the input the other is judged on. Accepted and explicitly flagged at Checkpoint 1;
it is not neutralized by anything here and must be restated wherever these results are.

**No accuracy metric is computed, and none can be.** This corpus has no ground truth. CER, WER and
accuracy are deliberately absent -- the recorded signals are execution success/failure, timing,
GPU memory, output presence/length and empty/degenerate-output detection.

**HARD BOUNDARY.** This is the smoke test only. It does not run, and must not be extended to run,
the full sample -- that requires a separate human review of these results first.

Run:

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_segmentation_smoke_test.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from archivetrust.application.htr_journal import HtrJournal  # noqa: E402
from archivetrust.domain.shared.ids import new_id  # noqa: E402
from archivetrust.htr.corpus.models import InputCrop  # noqa: E402
from archivetrust.htr.experiment.baseline_execution import (  # noqa: E402
    InputCropHashMismatchError,
)
from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.htr.preprocessing.export_package import (  # noqa: E402
    PageImageSelection,
    build_export_package,
)
from archivetrust.htr.preprocessing.line_targeted_export import (  # noqa: E402
    LineTargetedSelection,
    build_line_targeted_export,
)
from archivetrust.htr.preprocessing.models import RgbNormalizationConfig  # noqa: E402
from archivetrust.htr.preprocessing.normalization_service import (  # noqa: E402
    NormalizationService,
)
from archivetrust.htr.segmentation import (  # noqa: E402
    CONFOUND_STATEMENT,
    Florence2LineDetectorAdapter,
    LineDetectionFailedError,
    PageImage,
    SegmentationService,
)
from archivetrust.infrastructure.storage.blob_store import (  # noqa: E402
    ContentAddressedBlobStore,
)
from archivetrust.infrastructure.storage.telemetry_sink import (  # noqa: E402
    FileTelemetrySink,
)
from archivetrust.providers.florence2_htr.adapter import Florence2Adapter  # noqa: E402
from archivetrust.providers.florence2_htr.adapter import (  # noqa: E402
    normalize_transcription as florence2_normalize,
)
from archivetrust.providers.htr_adapter import RecognitionInput  # noqa: E402
from archivetrust.providers.satrn.adapter import SatrnAdapter  # noqa: E402
from archivetrust.providers.satrn.adapter import (  # noqa: E402
    normalize_transcription as satrn_normalize,
)

OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "smoke-test"
INVENTORY = (
    REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "dataset-inventory.csv"
)
CORRELATION_ID = "segmentation_smoke_test_run_1"

MAX_CROPS_PER_PAGE_FOR_RECOGNITION = 3
"""Recognition is run on the first N crops of each page, not all of them.

SATRN pays a full subprocess + model load on *every* call (`providers/satrn/facade.py`), so
recognizing every detected line on every page would take hours and would measure the same thing
this smoke test already measures with a bounded sample: that the path runs, on real crops, and what
it costs. The count is stated in the output so no reader mistakes it for full coverage."""


def _selected_pages() -> tuple[dict, ...]:
    """The smoke-test pages, chosen from Stage 1's real inventory.

    A deliberate mix rather than the smallest or the first: at least one `Ordinary` page and at
    least one `Structurally difficult` double-page spread, because 93.6% of the corpus is a spread
    and a smoke test that dodged that property would exercise the wrong pipeline. The single
    `Technically difficult` page is included too -- it is the whole stratum, so this is the only
    chance to see it run at all.
    """
    with INVENTORY.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["automatic_category"], []).append(row)

    for group in by_category.values():
        group.sort(key=lambda r: int(r["pixel_count"]))

    selected: list[dict] = []
    # Two spreads: the smallest, and one from the middle of the size distribution, so the sample
    # is not accidentally all-small.
    spreads = by_category.get("Structurally difficult", [])
    if spreads:
        selected.append(spreads[0])
        selected.append(spreads[len(spreads) // 2])
    # Two Ordinary pages.
    ordinary = by_category.get("Ordinary", [])
    selected.extend(ordinary[:2])
    # The one and only Technically difficult page.
    selected.extend(by_category.get("Technically difficult", [])[:1])
    return tuple(selected)


def _degenerate(text: str) -> bool:
    """Character/n-gram repetition loop -- a known VLM degradation mode, detected rather than
    assumed absent. Not an accuracy judgement: it needs no reference text."""
    stripped = text.strip()
    if len(stripped) < 12:
        return False
    if len(set(stripped)) <= 2:
        return True
    for size in (2, 3, 4):
        if len(stripped) >= size * 6:
            unit = stripped[:size]
            if unit * (len(stripped) // size) == stripped[: size * (len(stripped) // size)]:
                return True
    return False


def _recognition_signals(text: str | None, raw_response: dict) -> dict:
    """The reference-free technical-reliability signals. **No CER/WER/accuracy** -- there is no
    ground truth for this corpus and none is planned."""
    return {
        "produced_output": text is not None,
        "failure_category": None if text is not None else raw_response.get("category"),
        "failure_message": None if text is not None else raw_response.get("message"),
        "output_length": len(text) if text is not None else None,
        "empty_or_whitespace_only": (text is not None and not text.strip()),
        "degenerate_repetition": (text is not None and _degenerate(text)),
    }


def main() -> int:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    started_wall = datetime.now(timezone.utc).isoformat()
    telemetry_path = OUTPUT_DIR / "smoke_test_events.jsonl"
    store = DurableHtrResearchStore(
        FileTelemetrySink(telemetry_path), actor_id="segmentation-smoke-test"
    )

    detector = Florence2LineDetectorAdapter()
    segmentation = SegmentationService(
        adapter=detector, store=store, crop_directory=OUTPUT_DIR / "crops"
    )
    normalization = NormalizationService(
        blob_store=ContentAddressedBlobStore(OUTPUT_DIR / "blobs"), store=store
    )
    satrn = SatrnAdapter()
    florence2 = Florence2Adapter()

    print("=" * 100)
    print("STAGE 2 + CHECKPOINT 3 SMOKE TEST -- Swedish Historical HTR Technical Reliability Screening")
    print("=" * 100)
    print()
    print("CONFOUND:", CONFOUND_STATEMENT)
    print()
    print("NO ACCURACY METRIC IS COMPUTED. This corpus has no ground truth; CER/WER/accuracy are")
    print("deliberately absent. Recorded signals are execution, timing, memory and output shape.")
    print()

    for name, adapter in (("satrn", satrn), ("florence2", florence2)):
        validation = adapter.validate_environment()
        print(f"  environment[{name}]: valid={validation.valid} :: {'; '.join(validation.messages)}")
    print()

    pages = _selected_pages()
    print(f"Selected {len(pages)} real pages from Stage 1's inventory:")
    for row in pages:
        print(
            f"  {row['page_id'][:34]}  {row['automatic_category']:<24} "
            f"{row['width']}x{row['height']}  ar={row['aspect_ratio']}  "
            f"{int(row['file_size_bytes'])/1e6:.1f} MB  {row['archival_volume']}/{row['page_number']}"
        )
    print()

    page_reports: list[dict] = []
    line_targeted_selections: list[LineTargetedSelection] = []
    whole_page_selections: list[PageImageSelection] = []
    crop_hash_checks: list[dict] = []

    with store.correlated_to(CORRELATION_ID):
        for index, row in enumerate(pages, start=1):
            page_path = REPO_ROOT / row["relative_path"]
            page_id = row["page_id"]
            archive_object_ref = row["document_id"]
            print("-" * 100)
            print(f"[{index}/{len(pages)}] {page_id}")
            print(f"        {row['automatic_category']} | {page_path.name[:80]}")

            image_bytes = page_path.read_bytes()
            page_image = PageImage(
                page_id=page_id, image_bytes=image_bytes, source_path=str(page_path)
            )

            # -- 1. Real segmentation ------------------------------------------------------------
            segment_started = time.monotonic()
            try:
                segmented = segmentation.segment_and_record(
                    page_image,
                    archive_object_ref=archive_object_ref,
                    page_number=int(row["page_number"]),
                    correlation_id=CORRELATION_ID,
                )
            except LineDetectionFailedError as exc:
                print(f"        SEGMENTATION FAILED [{exc.category}]: {exc}")
                page_reports.append(
                    {
                        "page_id": page_id,
                        "category": row["automatic_category"],
                        "segmentation_failed": True,
                        "failure_category": exc.category,
                        "failure_message": str(exc),
                    }
                )
                continue
            segment_seconds = time.monotonic() - segment_started

            evidence = segmented.region_evidence
            print(
                f"        spread={evidence.spread_detected} aspect={evidence.aspect_ratio:.4f} "
                f"gutter_x={evidence.gutter_x} "
                f"(naive mid {evidence.gutter_evidence.get('naive_midpoint_x')}, "
                f"offset {evidence.gutter_evidence.get('offset_from_midpoint_px')}, "
                f"contrast {evidence.gutter_evidence.get('gutter_contrast')})"
            )
            for region, outcome in zip(segmented.regions, segmented.line_outcomes, strict=False):
                print(
                    f"          region {region.region_type:<16} lines={len(outcome.lines):<4} "
                    f"boxes={outcome.boxes_returned:<4} discarded_wrong_side="
                    f"{outcome.boxes_discarded_wrong_side:<3} {outcome.inference_seconds:.2f}s "
                    f"truncated={outcome.output_truncated} "
                    f"{'SUSPECT(edge-crowded)' if outcome.detection_suspect else ''}"
                )
                if len(outcome.lines) == 0:
                    print(
                        f"            !! this region yielded ZERO lines -- a densely written leaf "
                        f"cannot; treat as a segmentation failure, not an empty page"
                    )
            print(
                f"        TOTAL lines={len(segmented.text_lines)} crops={len(segmented.input_crops)} "
                f"detect={segmented.total_inference_seconds:.2f}s stage={segment_seconds:.2f}s "
                f"peak_gpu={segmented.peak_gpu_memory_mb and round(segmented.peak_gpu_memory_mb)} MB"
            )

            # -- 2. Real RGB normalization (expect: already compliant) ---------------------------
            recorded = normalization.normalize_and_record(
                image_bytes, page_id=page_id, correlation_id=CORRELATION_ID
            )
            artifact = recorded.normalized_artifact
            already_compliant = (
                artifact.source_color_mode == "RGB"
                and artifact.source_bit_depth == 8
                and artifact.source_channel_count == 3
                and not artifact.source_icc_profile_present
                and not artifact.source_alpha_present
                and artifact.source_exif_orientation is None
            )
            bytes_unchanged = (
                artifact.source_content_hash.split("_", 2)[-1]
                == artifact.normalized_content_hash.split("_", 2)[-1]
            )
            print(
                f"        normalization: source={artifact.source_color_mode}/"
                f"{artifact.source_bit_depth}bit/{artifact.source_channel_count}ch "
                f"icc={artifact.source_icc_profile_present} alpha={artifact.source_alpha_present} "
                f"exif={artifact.source_exif_orientation} -> already_compliant={already_compliant}, "
                f"re-encode byte-identical={bytes_unchanged}"
            )

            whole_page_selections.append(
                PageImageSelection(
                    page_id=page_id,
                    image_bytes=image_bytes,
                    archive_object_ref=archive_object_ref,
                    page_number=int(row["page_number"]),
                    original_filename=page_path.name,
                )
            )
            line_targeted_selections.append(
                LineTargetedSelection(
                    page_id=page_id,
                    archive_object_ref=archive_object_ref,
                    page_number=int(row["page_number"]),
                    normalized_artifact=artifact,
                    normalized_image_bytes=normalization.read_normalized_bytes(artifact),
                    regions=segmented.regions,
                    text_lines=segmented.text_lines,
                    input_crops=segmented.input_crops,
                )
            )

            # -- 3. Real recognition on the byte-identical crops ---------------------------------
            crops = segmented.input_crops[:MAX_CROPS_PER_PAGE_FOR_RECOGNITION]
            print(
                f"        recognition on first {len(crops)} of {len(segmented.input_crops)} crops "
                f"(SATRN restarts a subprocess per call -- see MAX_CROPS_PER_PAGE_FOR_RECOGNITION)"
            )
            crop_reports: list[dict] = []
            for crop in crops:
                crop_path = Path(crop.storage_path)
                on_disk = crop_path.read_bytes()

                # The controlled-comparison guard: both methods must read the *same bytes*.
                # Recomputed from the file each adapter is about to open, not trusted from the
                # record -- htr-domain-design.md §7's "enforced by an assertion in the runner".
                satrn_input_hash = InputCrop.compute_hash(on_disk)
                florence2_input_hash = InputCrop.compute_hash(crop_path.read_bytes())
                if not (crop.hash == satrn_input_hash == florence2_input_hash):
                    raise InputCropHashMismatchError(
                        f"crop {crop.crop_id} is not byte-identical across methods: "
                        f"recorded={crop.hash!r} satrn={satrn_input_hash!r} "
                        f"florence2={florence2_input_hash!r}"
                    )
                crop_hash_checks.append(
                    {
                        "crop_id": crop.crop_id,
                        "recorded_hash": crop.hash,
                        "satrn_input_hash": satrn_input_hash,
                        "florence2_input_hash": florence2_input_hash,
                        "equal": True,
                    }
                )

                recognition_input = RecognitionInput(input_crop_id=str(crop_path))

                satrn_started = time.monotonic()
                satrn_result = satrn.recognize(recognition_input)
                satrn_wall = time.monotonic() - satrn_started

                florence2_started = time.monotonic()
                florence2_result = florence2.recognize(recognition_input)
                florence2_wall = time.monotonic() - florence2_started

                satrn_text = satrn_result.text
                florence2_text = florence2_result.text
                print(f"          crop {crop.crop_id[:26]} {crop.width}x{crop.height} {crop.hash[:20]}...")
                print(
                    f"            SATRN     [{satrn_wall:6.2f}s wall] "
                    f"{'OK ' if satrn_text is not None else 'FAIL'} "
                    f"conf={satrn_result.confidence} :: "
                    f"{(satrn_text if satrn_text is not None else satrn_result.raw_response.get('message'))!r}"
                )
                print(
                    f"            Florence2 [{florence2_wall:6.2f}s wall] "
                    f"{'OK ' if florence2_text is not None else 'FAIL'} "
                    f"conf={florence2_result.confidence and round(florence2_result.confidence, 4)} :: "
                    f"{(florence2_text if florence2_text is not None else florence2_result.raw_response.get('message'))!r}"
                )

                crop_reports.append(
                    {
                        "crop_id": crop.crop_id,
                        "crop_hash": crop.hash,
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "hash_equal_across_methods": True,
                        "satrn": {
                            "text": satrn_text,
                            "normalized_text": satrn_normalize(satrn_text) if satrn_text else None,
                            "confidence": satrn_result.confidence,
                            "wall_seconds": round(satrn_wall, 3),
                            "adapter_reported_ms": satrn_result.execution_time_ms,
                            "device": satrn_result.raw_response.get("device_used"),
                            "peak_gpu_memory_mb": satrn_result.raw_response.get("peak_gpu_memory_mb"),
                            **_recognition_signals(satrn_text, satrn_result.raw_response),
                        },
                        "florence2": {
                            "text": florence2_text,
                            "normalized_text": (
                                florence2_normalize(florence2_text) if florence2_text else None
                            ),
                            "confidence_proxy": florence2_result.confidence,
                            "wall_seconds": round(florence2_wall, 3),
                            "adapter_reported_ms": florence2_result.execution_time_ms,
                            "device": florence2_result.raw_response.get("device_used"),
                            "peak_gpu_memory_mb": florence2_result.raw_response.get(
                                "peak_gpu_memory_mb"
                            ),
                            **_recognition_signals(florence2_text, florence2_result.raw_response),
                        },
                    }
                )

            page_reports.append(
                {
                    "page_id": page_id,
                    "archive_object_ref": archive_object_ref,
                    "category_automatic": row["automatic_category"],
                    "source_file": row["relative_path"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "aspect_ratio": float(row["aspect_ratio"]),
                    "segmentation_failed": False,
                    "spread_detected": evidence.spread_detected,
                    "gutter_evidence": evidence.gutter_evidence,
                    "regions": [
                        {
                            "region_id": region.region_id,
                            "region_type": region.region_type,
                            "window_x": [region.bounding_box.x0, region.bounding_box.x1],
                            "lines_detected": len(outcome.lines),
                            "boxes_returned": outcome.boxes_returned,
                            "boxes_discarded_wrong_side": outcome.boxes_discarded_wrong_side,
                            "inference_seconds": round(outcome.inference_seconds, 3),
                            "output_truncated": outcome.output_truncated,
                            "boxes_in_edge_band": outcome.boxes_in_edge_band,
                            "detection_suspect": outcome.detection_suspect,
                            "zero_lines": len(outcome.lines) == 0,
                            "device_used": outcome.device_used,
                            "peak_gpu_memory_mb": (
                                round(outcome.peak_gpu_memory_mb)
                                if outcome.peak_gpu_memory_mb
                                else None
                            ),
                        }
                        for region, outcome in zip(
                            segmented.regions, segmented.line_outcomes, strict=False
                        )
                    ],
                    "total_lines_detected": len(segmented.text_lines),
                    "total_crops_written": len(segmented.input_crops),
                    "segmentation_detect_seconds": round(segmented.total_inference_seconds, 3),
                    "segmentation_stage_seconds": round(segment_seconds, 3),
                    "segmentation_peak_gpu_memory_mb": (
                        round(segmented.peak_gpu_memory_mb) if segmented.peak_gpu_memory_mb else None
                    ),
                    "segmentation_run_id": segmented.segmentation_run_id,
                    "normalization": {
                        "source_color_mode": artifact.source_color_mode,
                        "source_bit_depth": artifact.source_bit_depth,
                        "source_channel_count": artifact.source_channel_count,
                        "source_icc_profile_present": artifact.source_icc_profile_present,
                        "source_alpha_present": artifact.source_alpha_present,
                        "source_exif_orientation": artifact.source_exif_orientation,
                        "verified_already_compliant": already_compliant,
                        "reencode_byte_identical": bytes_unchanged,
                        "alpha_compositing_applied": artifact.alpha_compositing_applied,
                        "exif_orientation_applied": artifact.exif_orientation_applied,
                        "source_content_hash": artifact.source_content_hash,
                        "normalized_content_hash": artifact.normalized_content_hash,
                        "warnings": list(artifact.warnings),
                    },
                    "recognition_crops": crop_reports,
                }
            )

    # -- 4. Real Transkribus preparation, both packages. NO UPLOAD. --------------------------------
    print("-" * 100)
    whole_page_package = build_export_package(
        tuple(whole_page_selections),
        destination=OUTPUT_DIR / "export-packages",
        service=normalization,
        correlation_id=CORRELATION_ID,
        package_id="export_package_smoke_test_whole_page",
    )
    print(
        f"whole-page export package: {whole_page_package.package_id} "
        f"({len(whole_page_package.manifest.entries)} pages) -> "
        f"{whole_page_package.directory.relative_to(OUTPUT_DIR)}"
    )

    line_targeted_package = build_line_targeted_export(
        tuple(line_targeted_selections),
        destination=OUTPUT_DIR / "export-packages",
        segmentation_adapter_name=detector.name,
        segmentation_confound_statement=CONFOUND_STATEMENT,
        normalization_version=line_targeted_selections[0].normalized_artifact.normalization_version,
        configuration_hash=RgbNormalizationConfig().configuration_hash,
        package_id="export_package_smoke_test_line_targeted",
    )
    total_exported_lines = sum(e.line_count for e in line_targeted_package.manifest.entries)
    print(
        f"line-targeted export package: {line_targeted_package.package_id} "
        f"({len(line_targeted_package.manifest.entries)} pages, {total_exported_lines} lines, "
        f"each carrying its InputCrop id+hash) -> "
        f"{line_targeted_package.directory.relative_to(OUTPUT_DIR)}"
    )
    print("NO UPLOAD PERFORMED. Both packages are local files only; obtaining Lion I results from")
    print("either requires a human to upload and export manually. No automatic path exists.")
    print()

    # -- 5. Durability: destroy and replay ---------------------------------------------------------
    in_process_regions = {region.region_id: region for region in store.regions()}
    in_process_lines = {line.text_line_id: line for line in store.text_lines()}
    del store, segmentation, normalization
    replayed = HtrJournal().replay(FileTelemetrySink(telemetry_path).all_events())
    replay_ok = (
        {region.region_id: region for region in replayed.regions()} == in_process_regions
        and {line.text_line_id: line for line in replayed.text_lines()} == in_process_lines
    )
    print(f"replay reconstruction: {replay_ok} ({len(in_process_regions)} regions, {len(in_process_lines)} lines)")

    event_counts = Counter()
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event_counts[json.loads(line)["kind"]] += 1
    print()
    print("durable telemetry event counts:")
    for kind, count in sorted(event_counts.items()):
        print(f"  {kind:<44} {count}")

    # -- 6. Summary --------------------------------------------------------------------------------
    succeeded = [p for p in page_reports if not p["segmentation_failed"]]
    all_crops = [c for p in succeeded for c in p["recognition_crops"]]
    all_regions = [r for p in succeeded for r in p["regions"]]

    def _output_constancy(method: str) -> dict:
        """Distinct outputs over distinct inputs -- a **reference-free** degradation signal that no
        per-string check can see.

        A method can execute successfully on every crop, return non-empty, non-repetitive text, and
        still be broken, if it returns nearly the *same* text regardless of what it was shown. That
        is exactly what the Checkpoint 3 smoke test found for one of the two recognizers. Since
        every crop here is a different line of a different document, a low distinct ratio is
        evidence the method is not reading its input. It is **not** an accuracy measure: it needs no
        ground truth and says nothing about whether any output is correct.
        """
        texts = [c[method]["text"] for c in all_crops if c[method]["text"] is not None]
        distinct = len(set(texts))
        most_repeated = Counter(texts).most_common(1)
        return {
            "outputs": len(texts),
            "distinct_outputs": distinct,
            "distinct_ratio": round(distinct / len(texts), 4) if texts else None,
            "most_repeated_output": most_repeated[0][0] if most_repeated else None,
            "most_repeated_count": most_repeated[0][1] if most_repeated else None,
        }
    summary = {
        "stage": "Stage 2 + Checkpoint 3 -- smoke test only. STOPPED HERE BY DESIGN.",
        "hard_boundary": (
            "The full ~60-page sample was NOT run and must not be run until a human has reviewed "
            "these smoke-test results."
        ),
        "confound": CONFOUND_STATEMENT,
        "no_accuracy_metric": (
            "No CER, WER or accuracy is computed or reported. This corpus has no ground truth and "
            "none is planned; only reference-free technical-reliability signals are recorded."
        ),
        "started_at": started_wall,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "correlation_id": CORRELATION_ID,
        "segmentation_configuration": detector.configuration.model_dump(mode="json"),
        "recognition_crop_cap_per_page": MAX_CROPS_PER_PAGE_FOR_RECOGNITION,
        "pages_attempted": len(page_reports),
        "pages_segmented_successfully": len(succeeded),
        "total_lines_detected": sum(p["total_lines_detected"] for p in succeeded),
        "total_crops_written": sum(p["total_crops_written"] for p in succeeded),
        "crop_hash_equality": {
            "crops_checked": len(crop_hash_checks),
            "all_equal_across_methods": all(check["equal"] for check in crop_hash_checks),
            "checks": crop_hash_checks,
        },
        "recognition_totals": {
            "crops_recognized": len(all_crops),
            "satrn_succeeded": sum(1 for c in all_crops if c["satrn"]["produced_output"]),
            "satrn_failed": sum(1 for c in all_crops if not c["satrn"]["produced_output"]),
            "satrn_empty_output": sum(1 for c in all_crops if c["satrn"]["empty_or_whitespace_only"]),
            "satrn_degenerate": sum(1 for c in all_crops if c["satrn"]["degenerate_repetition"]),
            "florence2_succeeded": sum(1 for c in all_crops if c["florence2"]["produced_output"]),
            "florence2_failed": sum(1 for c in all_crops if not c["florence2"]["produced_output"]),
            "florence2_empty_output": sum(
                1 for c in all_crops if c["florence2"]["empty_or_whitespace_only"]
            ),
            "florence2_degenerate": sum(
                1 for c in all_crops if c["florence2"]["degenerate_repetition"]
            ),
            "satrn_output_constancy": _output_constancy("satrn"),
            "florence2_output_constancy": _output_constancy("florence2"),
        },
        "segmentation_reliability": {
            "regions_total": len(all_regions),
            "regions_with_zero_lines": sum(1 for r in all_regions if r["zero_lines"]),
            "regions_flagged_edge_crowded_suspect": sum(
                1 for r in all_regions if r["detection_suspect"]
            ),
            "regions_output_truncated": sum(1 for r in all_regions if r["output_truncated"]),
        },
        "transkribus_preparation": {
            "upload_performed": False,
            "upload_possible": False,
            "note": (
                "Two packages are prepared, both local files only. The whole-page package exercises "
                "Lion I's real end-to-end workflow (Transkribus does its own segmentation). The "
                "line-targeted package hands Transkribus the same detected lines SATRN and "
                "Florence-2 received, as layout-only PAGE XML with empty TextEquiv, for a "
                "recognition-only run. The latter makes a three-way segmentation-controlled "
                "comparison POSSIBLE; it does not obtain one -- a human must still upload, run Lion "
                "I, and export manually."
            ),
            "byte_identity_caveat": line_targeted_package.manifest.byte_identity_caveat,
            "whole_page_package": {
                "package_id": whole_page_package.package_id,
                "directory": str(whole_page_package.directory.relative_to(OUTPUT_DIR)),
                "pages": len(whole_page_package.manifest.entries),
                "entries": [
                    {
                        "page_id": e.page_id,
                        "output_filename": e.output_filename,
                        "source_color_mode": e.source_color_mode,
                        "original_content_hash": e.original_content_hash,
                        "normalized_content_hash": e.normalized_content_hash,
                        "dimensions": f"{e.width}x{e.height}",
                    }
                    for e in whole_page_package.manifest.entries
                ],
            },
            "line_targeted_package": {
                "package_id": line_targeted_package.package_id,
                "directory": str(line_targeted_package.directory.relative_to(OUTPUT_DIR)),
                "pages": len(line_targeted_package.manifest.entries),
                "total_lines_exported": total_exported_lines,
                "entries": [
                    {
                        "page_id": e.page_id,
                        "image_filename": e.image_filename,
                        "page_xml_filename": e.page_xml_filename,
                        "region_count": e.region_count,
                        "line_count": e.line_count,
                    }
                    for e in line_targeted_package.manifest.entries
                ],
            },
        },
        "durable_telemetry": {
            "file": telemetry_path.name,
            "event_counts": dict(sorted(event_counts.items())),
            "total_events": sum(event_counts.values()),
            "replay_reconstruction_verified": replay_ok,
        },
        "pages": page_reports,
    }

    (OUTPUT_DIR / "smoke_test_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 100)
    print(
        f"SUMMARY  pages {summary['pages_segmented_successfully']}/{summary['pages_attempted']} segmented | "
        f"lines {summary['total_lines_detected']} | crops {summary['total_crops_written']} | "
        f"crop-hash equality {summary['crop_hash_equality']['all_equal_across_methods']}"
    )
    totals = summary["recognition_totals"]
    print(
        f"         recognition on {totals['crops_recognized']} crops: "
        f"SATRN {totals['satrn_succeeded']} ok / {totals['satrn_failed']} fail | "
        f"Florence-2 {totals['florence2_succeeded']} ok / {totals['florence2_failed']} fail"
    )
    for method in ("satrn", "florence2"):
        constancy = totals[f"{method}_output_constancy"]
        print(
            f"         {method:<10} output constancy: {constancy['distinct_outputs']}"
            f"/{constancy['outputs']} distinct (ratio {constancy['distinct_ratio']}); "
            f"most repeated {constancy['most_repeated_output']!r} x{constancy['most_repeated_count']}"
        )
    seg = summary["segmentation_reliability"]
    print(
        f"         segmentation: {seg['regions_with_zero_lines']}/{seg['regions_total']} regions "
        f"yielded zero lines, {seg['regions_flagged_edge_crowded_suspect']} flagged edge-crowded, "
        f"{seg['regions_output_truncated']} truncated"
    )
    print("         NO CER/WER/accuracy computed -- no ground truth exists for this corpus.")
    print("STOPPING AT THE SMOKE TEST. The full sample was not run and requires human review first.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
