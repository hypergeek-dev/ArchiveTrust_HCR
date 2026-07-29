"""Phase 16 -- High-IoU Divergence Investigation (2026-07-15).

Measurement-only, per this phase's own charter: no clustering threshold, comparison logic,
confidence logic, or Review Center behavior is touched. Read-only over already-persisted
artifacts, including -- for the first time in this investigation series -- the original rendered
page images already produced by the pipeline run (no OCR rerun, no replay).

Reproduces:
- The high-IoU population (top ~13.5% of the docling/tesseract pair by IoU, per the corrected
  Phase 15 feature dataset -- see the erratum at the top of
  `docs/PHASE_15_PROVIDER_SEGMENTATION_RELATIONSHIP_STUDY_2026-07-15.md`).
- Layout statistics for that population (§2 of the Phase 16 report).
- Annotated crop images (both providers' bounding boxes overlaid on the source page) for manual
  visual audit (§1) -- written to `benchmarks/phase16_crops/`.

Run: `python scripts/phase16_high_iou_divergence_investigation.py`
Writes: `benchmarks/phase16_high_iou_divergence_investigation.json`, `benchmarks/phase16_crops/*.png`
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "benchmarks" / "calibration_corpus_1.jsonl"
FEATURE_CSV = REPO_ROOT / "benchmarks" / "phase15_feature_dataset.csv"
TELEMETRY_PATH = (
    REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
    / "telemetry" / "events.jsonl"
)
ACQUISITION_PATH = (
    REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11"
    / "telemetry" / "acquisition.jsonl"
)
DERIVED_DIR = REPO_ROOT / "archivetrust_data" / "workspaces" / "46e94f15d73347c2be79f1527e646c11" / "derived"
CROPS_DIR = REPO_ROOT / "benchmarks" / "phase16_crops"
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "phase16_high_iou_divergence_investigation.json"

# The population this phase's visual audit was performed against: the top 53 docling/tesseract
# pairs by IoU (top ~13.5% of the 392-pair population, IoU 0.42-0.77) -- a rank-based definition,
# not a threshold, so its size is exactly reproducible regardless of how IoU is distributed near
# any particular cut point. This closely (not identically -- ~85% overlap) reproduces the original
# KMeans-selected population Phase 15 first flagged; see the "note on scope" in the Phase 16
# report for why a clean rank-based definition replaced that clustering-derived one.
POPULATION_SIZE = 53


def resolve_population() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_CSV)
    return df.nlargest(POPULATION_SIZE, "iou").copy()


def _archive_object_to_hash() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with ACQUISITION_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if '"ArchiveObjectRegistered"' not in line:
                continue
            obj = json.loads(line)
            ao = obj.get("archive_object")
            if ao:
                mapping[ao["id"]] = ao["content_hash"]
    return mapping


def _fetch_evidence(evidence_ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with TELEMETRY_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if '"kind":"EvidenceCreated"' not in line:
                continue
            if not any(eid in line for eid in evidence_ids):
                continue
            obj = json.loads(line)
            evidence = obj.get("evidence")
            if not evidence:
                continue
            eid = evidence.get("evidence_id")
            if eid in evidence_ids:
                found[eid] = evidence
            if len(found) == len(evidence_ids):
                break
    return found


def layout_analysis(population: pd.DataFrame, corpus_records: list[dict], evidence: dict) -> dict:
    y_centers_norm = []
    pages = []
    for r in corpus_records:
        ev = [evidence[eid] for eid in r["evidence_ids"] if eid in evidence]
        if len(ev) != 2:
            continue
        meta = ev[0].get("supporting_metadata", {})
        page_height = meta.get("page_height")
        if not page_height:
            continue
        y_center = sum((e["bounding_box"]["y0"] + e["bounding_box"]["y1"]) / 2 for e in ev) / 2
        y_centers_norm.append(y_center / page_height)
        pages.append((r["pages"] or [None])[0])

    return {
        "n": len(population),
        "mean_width_docling": round(float(population["width_docling"].mean()), 1),
        "mean_width_tesseract": round(float(population["width_tesseract"].mean()), 1),
        "mean_height_docling": round(float(population["height_docling"].mean()), 1),
        "mean_height_tesseract": round(float(population["height_tesseract"].mean()), 1),
        "docling_bigger_area_count": int((population["area_docling"] > population["area_tesseract"]).sum()),
        "docling_longer_text_count": int(population["docling_is_longer"].sum()),
        "mean_length_ratio": round(float(population["length_ratio"].mean()), 2),
        "median_length_ratio": round(float(population["length_ratio"].median()), 2),
        "y_center_norm_mean": round(sum(y_centers_norm) / len(y_centers_norm), 3) if y_centers_norm else None,
        "y_center_norm_in_top_20pct_of_page": sum(1 for y in y_centers_norm if y < 0.2),
        "pct_page_1": round(100 * sum(1 for p in pages if p == 1) / len(pages), 1) if pages else None,
        "median_page_no": sorted(pages)[len(pages) // 2] if pages else None,
    }


def build_crops(population: pd.DataFrame, corpus_records_by_slot: dict[str, dict], evidence: dict, archive_map: dict[str, str]) -> int:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for slot_id in population["semantic_slot_id"]:
        r = corpus_records_by_slot.get(slot_id)
        if r is None:
            continue
        content_hash = archive_map.get(r["document_ref"])
        if not content_hash:
            continue
        dirs = glob.glob(os.path.join(DERIVED_DIR, content_hash, "r_pypdfium2_*"))
        if not dirs:
            continue
        page = (r["pages"] or [None])[0]
        img_path = os.path.join(dirs[0], f"page-{page}.png")
        if not os.path.exists(img_path):
            continue
        ev = [evidence[eid] for eid in r["evidence_ids"] if eid in evidence]
        by_provider = {e["provider"]: e for e in ev}
        if set(by_provider) != {"docling", "tesseract_layoutparser"}:
            continue
        d_bb = by_provider["docling"]["bounding_box"]
        t_bb = by_provider["tesseract_layoutparser"]["bounding_box"]
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle([d_bb["x0"], d_bb["y0"], d_bb["x1"], d_bb["y1"]], outline=(255, 0, 0), width=3)
        draw.rectangle([t_bb["x0"], t_bb["y0"], t_bb["x1"], t_bb["y1"]], outline=(0, 90, 255), width=3)
        x0 = max(0, min(d_bb["x0"], t_bb["x0"]) - 60)
        y0 = max(0, min(d_bb["y0"], t_bb["y0"]) - 60)
        x1 = min(im.width, max(d_bb["x1"], t_bb["x1"]) + 60)
        y1 = min(im.height, max(d_bb["y1"], t_bb["y1"]) + 60)
        crop = im.crop((x0, y0, x1, y1))
        crop.save(CROPS_DIR / f"{slot_id}.png")
        written += 1
    return written


def main() -> None:
    population = resolve_population()
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_by_slot = {r["semantic_slot_id"]: r for r in corpus if r["semantic_slot_id"] in set(population["semantic_slot_id"])}
    corpus_records = list(corpus_by_slot.values())

    evidence_ids = {eid for r in corpus_records for eid in r["evidence_ids"]}
    evidence = _fetch_evidence(evidence_ids)
    archive_map = _archive_object_to_hash()

    results = {
        "population_definition": f"top {POPULATION_SIZE} docling/tesseract pairs by IoU (rank-based)",
        "population_size": len(population),
        "layout_analysis": layout_analysis(population, corpus_records, evidence),
    }

    n_crops = build_crops(population, corpus_by_slot, evidence, archive_map)
    results["crops_written"] = n_crops

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {n_crops} annotated crops to {CROPS_DIR}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
