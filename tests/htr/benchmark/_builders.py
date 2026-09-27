"""Shared builders for benchmark tests: small, valid `BenchmarkLine`s with overridable fields."""

from __future__ import annotations

from archivetrust.htr.benchmark.contract import BenchmarkLine, CropRecord, make_line_id, sha256_text
from archivetrust.htr.benchmark.normalization import canonicalize_gt


def make_line(*, document_id="doc1", page_id="p001", line_key="l001", line_order=0, gt="Anno 1723 d. 4 Maj",
              image_sha256="a" * 64, dataset_id="ds1", **overrides) -> BenchmarkLine:
    canonical, applied = canonicalize_gt(gt)
    fields = dict(
        dataset_id=dataset_id, source_id="src1", document_id=document_id, page_id=page_id, line_key=line_key,
        line_id=make_line_id(document_id, page_id, line_key), line_order=line_order,
        source_relative_path=f"{document_id}/{page_id}_{line_key}.png", original_filename=f"{page_id}_{line_key}.png",
        gt_source_relative_path=f"{document_id}/{page_id}_{line_key}.gt.txt", adapter_id="test", adapter_version="0",
        line_image_path=f"lines/{document_id}/{page_id}/{line_key}.png", image_sha256=image_sha256,
        image_width=400, image_height=48, crop=CropRecord(policy="supplied_line_image"),
        segmentation_source="supplied_line_image", gt_source=gt, gt_canonical=canonical,
        gt_source_sha256=sha256_text(gt), gt_canonical_sha256=sha256_text(canonical), normalization_applied=applied,
    )
    fields.update(overrides)
    return BenchmarkLine(**fields)
