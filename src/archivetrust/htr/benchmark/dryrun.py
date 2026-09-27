"""Mechanical dry run on page images that have NO ground truth: pages + an imported line segmentation
-> deterministic line crops -> a hashed dry-run manifest -> smoke inference through the real backends.

This exercises every stage the benchmark will use except GT alignment and scoring. It is labelled
`NO_GROUND_TRUTH / NOT_AN_ACCURACY_BENCHMARK` everywhere. It has no GT fields and no scoring entry
point. Its predictions carry `benchmark_id = "dryrun:<id>"`, so `score` refuses them (they match no
frozen manifest).

Segmentation is *imported*, never computed here. The harness does not segment (see
BENCHMARK_PROTOCOL.md §4). The import format is JSONL, one row per line:

    {"page_image": "<path relative to pages_dir>", "line_key": "l0001", "bbox": [x0, y0, x1, y1],
     "reading_order": 0, "segmentation_source": "<tool@revision + run id>",
     "document": "<optional, default: parent folder>", "page": "<optional, default: file stem>",
     "page_width": <optional declared width>, "page_height": <optional declared height>}

`polygon` ([[x, y], ...]) may be given instead of `bbox`. Crops use `bbox_v1`: the same rounding,
clamping and PNG encoding as a real benchmark build.
"""

from __future__ import annotations

import json
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.benchmark.contract import (
    CropRecord,
    line_image_relpath,
    make_line_id,
    safe_id_component,
    sha256_bytes,
    sha256_file,
)
from archivetrust.htr.benchmark.findings import Finding, findings_to_jsonl
from archivetrust.htr.benchmark.imaging import crop_line, open_page_rgb, polygon_bbox, probe_image_bytes
from archivetrust.htr.benchmark.inference import (
    BackendResult,
    PredictionRecord,
    RecognizerBackend,
    RunError,
    prediction_filename,
)
from archivetrust.htr.benchmark.layout import assert_outside
from archivetrust.htr.benchmark.normalization import canonicalize_prediction
from archivetrust.htr.benchmark.provenance import provenance, utc_now

DRYRUN_SCHEMA = "benchmark-dryrun/1"
LABEL = "NO_GROUND_TRUTH / NOT_AN_ACCURACY_BENCHMARK"


class DryRunError(RuntimeError):
    pass


class DryRunLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dryrun_id: str
    document_id: str
    page_id: str
    line_key: str
    line_id: str
    line_order: int
    line_image_path: str
    image_sha256: str
    image_width: int
    image_height: int
    crop: CropRecord
    segmentation_source: str
    ground_truth: Literal["NO_GROUND_TRUTH"] = "NO_GROUND_TRUTH"


def _read_segmentation(path: Path) -> list[dict]:
    rows = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        missing = {"page_image", "line_key", "segmentation_source"} - row.keys()
        if missing or ("bbox" not in row and "polygon" not in row):
            raise DryRunError(f"{path}:{number}: segmentation row needs page_image, line_key, segmentation_source and bbox or polygon")
        rows.append(row)
    return rows


def build_dryrun(pages_dir: Path, segmentation_path: Path, out_dir: Path, *, dryrun_id: str,
                 max_pages: int | None = None) -> dict:
    """Writes `lines/`, `manifest.jsonl`, `findings.jsonl` and `DRYRUN.json` into a new `out_dir`.
    Pages that are unreadable, carry a non-identity EXIF orientation, or contradict a declared page
    size are skipped with a finding. Orientation is never applied silently."""
    pages_dir, out_dir = Path(pages_dir), Path(out_dir)
    if out_dir.exists():
        raise DryRunError(f"{out_dir} exists; dry runs are never overwritten")
    assert_outside(out_dir, pages_dir)
    rows = _read_segmentation(Path(segmentation_path))
    by_page: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_page[row["page_image"]].append(row)
    page_paths = sorted(by_page)[:max_pages] if max_pages else sorted(by_page)

    findings: list[Finding] = []
    lines: list[DryRunLine] = []
    out_dir.mkdir(parents=True)
    for page_rel in page_paths:
        page_path = pages_dir / page_rel
        if not page_path.is_file():
            findings.append(Finding(severity="blocker", code="page.image_missing", message="segmented page image not found", path=page_rel))
            continue
        data = page_path.read_bytes()
        probe = probe_image_bytes(data)
        if not probe.ok:
            findings.append(Finding(severity="blocker", code="image.unreadable", message=probe.error or "unreadable", path=page_rel))
            continue
        if probe.exif_orientation not in (None, 1):
            findings.append(Finding(severity="needs_review", code="image.exif_orientation",
                                    message=f"EXIF orientation {probe.exif_orientation}; page skipped, not rotated", path=page_rel))
            continue
        page_rows = sorted(by_page[page_rel], key=lambda r: (r.get("reading_order", 0), str(r["line_key"])))
        declared = {(r.get("page_width"), r.get("page_height")) for r in page_rows} - {(None, None)}
        if declared and declared != {(probe.width, probe.height)}:
            findings.append(Finding(severity="blocker", code="page.size_mismatch",
                                    message=f"segmentation declares {sorted(declared)}, raster is {probe.width}x{probe.height}", path=page_rel))
            continue
        page = open_page_rgb(data)
        rel = Path(page_rel)
        document_id = safe_id_component(page_rows[0].get("document") or (rel.parent.name or "root"))
        page_id = safe_id_component(page_rows[0].get("page") or rel.stem)
        for order, row in enumerate(page_rows):
            if "polygon" in row and row["polygon"]:
                points = [tuple(p) for p in row["polygon"]]
            else:
                x0, y0, x1, y1 = row["bbox"]
                points = [(x0, y0), (x1, y1)]
            bbox = polygon_bbox(points, width=probe.width, height=probe.height)
            line_key = safe_id_component(str(row["line_key"]))
            if bbox is None:
                findings.append(Finding(severity="warning", code="layout.empty_bbox", message="line box is empty after clamping; skipped",
                                        path=page_rel, line_id=line_key))
                continue
            crop = crop_line(page, bbox)
            image_rel = line_image_relpath(document_id, page_id, line_key, ".png")
            (out_dir / image_rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / image_rel).write_bytes(crop)
            lines.append(DryRunLine(
                dryrun_id=dryrun_id, document_id=document_id, page_id=page_id, line_key=line_key,
                line_id=make_line_id(document_id, page_id, line_key), line_order=order, line_image_path=image_rel,
                image_sha256=sha256_bytes(crop), image_width=bbox[2] - bbox[0], image_height=bbox[3] - bbox[1],
                crop=CropRecord(policy="bbox_v1", source_image_relative_path=rel.as_posix(), source_image_sha256=probe.sha256, bbox=bbox),
                segmentation_source=str(row["segmentation_source"]),
            ))

    duplicates = [k for k, n in Counter(line.line_id for line in lines).items() if n > 1]
    if duplicates:
        raise DryRunError(f"duplicate line ids after sanitizing: {duplicates[:5]}")
    manifest = "".join(json.dumps(line.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for line in lines)
    (out_dir / "manifest.jsonl").write_text(manifest, encoding="utf-8", newline="\n")
    (out_dir / "findings.jsonl").write_text(findings_to_jsonl(findings), encoding="utf-8", newline="\n")
    record = {
        "schema": DRYRUN_SCHEMA,
        "label": LABEL,
        "dryrun_id": dryrun_id,
        "pages_dir": str(pages_dir),
        "segmentation_file": str(segmentation_path),
        "segmentation_sha256": sha256_file(Path(segmentation_path)),
        "segmentation_sources": sorted({line.segmentation_source for line in lines}),
        "crop_policy": "bbox_v1",
        "pages_requested": len(page_paths),
        "pages_cropped": len({(line.document_id, line.page_id) for line in lines}),
        "lines": len(lines),
        "findings": dict(Counter(f.code for f in findings)),
        "manifest_sha256": sha256_bytes(manifest.encode("utf-8")),
        "created_at_utc": utc_now(),
        "provenance": provenance(official=False),
    }
    (out_dir / "DRYRUN.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def read_dryrun(dryrun_dir: Path) -> tuple[dict, list[DryRunLine]]:
    dryrun_dir = Path(dryrun_dir)
    record = json.loads((dryrun_dir / "DRYRUN.json").read_text(encoding="utf-8"))
    manifest = dryrun_dir / "manifest.jsonl"
    if sha256_file(manifest) != record["manifest_sha256"]:
        raise DryRunError("dry-run manifest does not match DRYRUN.json")
    lines = [DryRunLine.model_validate(json.loads(raw)) for raw in manifest.read_text(encoding="utf-8").splitlines() if raw.strip()]
    return record, lines


def run_dryrun(dryrun_dir: Path, backend: RecognizerBackend, *, limit: int | None = None) -> dict:
    """Smoke inference over (the first `limit`) dry-run lines. Writes immutable
    `predictions/<model>.jsonl` + `.run.json`. The output says nothing about accuracy."""
    dryrun_dir = Path(dryrun_dir)
    record, lines = read_dryrun(dryrun_dir)
    lines = lines[:limit] if limit else lines
    for line in lines:
        if sha256_file(dryrun_dir / line.line_image_path) != line.image_sha256:
            raise DryRunError(f"{line.line_image_path} does not match its manifest hash")
    predictions_dir = dryrun_dir / "predictions"
    target = predictions_dir / prediction_filename(backend.model, backend.profile)
    work_dir = dryrun_dir / "work" / target.stem
    if target.exists() or work_dir.exists():
        raise RunError(f"{target.name} or its work dir exists; dry-run predictions are never overwritten")
    work_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    results: dict[str, BackendResult] = {}
    for result in backend.recognize([(line.line_id, dryrun_dir / line.line_image_path) for line in lines], work_dir):
        results[result.line_id] = result
    finished_at = utc_now()
    rows = []
    for line in lines:
        result = results.get(line.line_id) or BackendResult(line.line_id, "missing", error_category="not_returned")
        rows.append(PredictionRecord(
            benchmark_id=f"dryrun:{record['dryrun_id']}", manifest_sha256=record["manifest_sha256"], dataset_id=record["dryrun_id"],
            document_id=line.document_id, page_id=line.page_id, line_id=line.line_id, image_sha256=line.image_sha256,
            model_id=backend.model.model_id, decoding_profile=backend.profile.profile_id, status=result.status,
            prediction_raw=result.text, prediction=canonicalize_prediction(result.text) if result.text else "",
            confidence=result.confidence, duration_seconds=result.duration_seconds, duration_kind=result.duration_kind,
            error_category=result.error_category, error_message=result.error_message,
        ))
    target.write_text("".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                      encoding="utf-8", newline="\n")
    os.chmod(target, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    run_record = {
        "label": LABEL,
        "dryrun_id": record["dryrun_id"],
        "manifest_sha256": record["manifest_sha256"],
        "model": backend.model.record(backend.profile.profile_id),
        "backend_environment": backend.environment(),
        "predictions_file": target.name,
        "predictions_sha256": sha256_file(target),
        "lines": len(rows),
        "limited_to": limit,
        "status_counts": dict(Counter(r.status for r in rows)),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "provenance": provenance(official=False),
    }
    (predictions_dir / f"{target.stem}.run.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                                           encoding="utf-8")
    return run_record
