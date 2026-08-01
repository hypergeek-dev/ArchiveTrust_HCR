#!/usr/bin/env python
"""Inventories the Dutch corpus (`dataset-dutch-rgb/`) -- real per-page content hashes and image
properties, computed from the actual extracted files. No classification heuristics are borrowed from
the Swedish corpus's inventory (aspect-ratio/bytes-per-pixel thresholds there were derived for that
specific corpus and are not assumed to transfer); this script reports only what can be measured
directly, and leaves difficulty/layout classification to human review
(docs/experiments/lion-loghi-comparison/dataset-comparability.md).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/inventory_dutch_dataset.py

Writes:
  docs/experiments/lion-loghi-comparison/dataset-dutch-manifest.json
  docs/experiments/lion-loghi-comparison/dataset-dutch-inventory.csv

Source: `dataset-dutch-rgb/republic7/{train,val}/*.jpg` + `{train,val}/page/*.xml` (extracted from
`D:\\Downloads\\republic7.zip` -- see dataset-provenance-dutch.md for what is and is not known about
this archive's origin and licensing).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402

DATASET_ROOT = REPO_ROOT / "dataset-dutch-rgb" / "republic7"
OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "lion-loghi-comparison"
MANIFEST_PATH = OUTPUT_DIR / "dataset-dutch-manifest.json"
INVENTORY_CSV_PATH = OUTPUT_DIR / "dataset-dutch-inventory.csv"

CSV_FIELDS = (
    "split",
    "page_id",
    "relative_path",
    "content_hash",
    "file_size_bytes",
    "image_format",
    "width",
    "height",
    "color_mode",
    "has_ground_truth_page_xml",
    "ground_truth_relative_path",
)


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"page_image_{digest.hexdigest()}"


def _inventory_split(split: str) -> list[dict]:
    split_dir = DATASET_ROOT / split
    page_dir = split_dir / "page"
    rows: list[dict] = []
    for image_path in sorted(split_dir.glob("*.jpg")):
        page_id = image_path.stem
        xml_path = page_dir / f"{page_id}.xml"
        with Image.open(image_path) as img:
            width, height = img.size
            color_mode = img.mode
            image_format = img.format

        rows.append(
            {
                "split": split,
                "page_id": page_id,
                "relative_path": str(image_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "content_hash": _content_hash(image_path),
                "file_size_bytes": image_path.stat().st_size,
                "image_format": image_format,
                "width": width,
                "height": height,
                "color_mode": color_mode,
                "has_ground_truth_page_xml": xml_path.exists(),
                "ground_truth_relative_path": (
                    str(xml_path.relative_to(REPO_ROOT)).replace("\\", "/") if xml_path.exists() else ""
                ),
            }
        )
    return rows


def main() -> int:
    if not DATASET_ROOT.exists():
        raise SystemExit(
            f"Missing {DATASET_ROOT} -- extract D:\\Downloads\\republic7.zip into dataset-dutch-rgb/ first"
        )

    all_rows: list[dict] = []
    for split in ("train", "val"):
        if (DATASET_ROOT / split).exists():
            all_rows.extend(_inventory_split(split))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INVENTORY_CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/inventory_dutch_dataset.py",
        "dataset": {
            "dataset_name": "dutch-republic-corpus-rgb",
            "root_relative_path": "dataset-dutch-rgb/republic7",
            "description": (
                "Dutch historical handwriting pages (Nationaal Archief scans, predominantly the "
                "Staten-Generaal fonds, 1.01.02) with Transkribus-annotated PAGE XML ground truth. "
                "Extracted from a locally-provided archive (republic7.zip) -- see "
                "dataset-provenance-dutch.md for provenance and licensing status."
            ),
        },
        "counts": {
            "total_pages": len(all_rows),
            "train": sum(1 for r in all_rows if r["split"] == "train"),
            "val": sum(1 for r in all_rows if r["split"] == "val"),
            "pages_with_ground_truth_page_xml": sum(1 for r in all_rows if r["has_ground_truth_page_xml"]),
            "distinct_content_hashes": len({r["content_hash"] for r in all_rows}),
        },
        "note": (
            "No difficulty/layout classification is computed here -- the Swedish corpus's inventory "
            "thresholds (aspect ratio, bytes-per-pixel) were derived for that specific corpus and are "
            "not assumed to transfer. See dataset-comparability.md for what is Known vs. Unknown."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {INVENTORY_CSV_PATH.relative_to(REPO_ROOT)} ({len(all_rows)} rows)")
    print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
