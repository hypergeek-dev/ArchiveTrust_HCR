"""Real line detection: downloads the real checkpoint and runs a real forward pass on a real page.

Marked `real_model` and skipped -- never failed -- when its prerequisites are absent, matching
`tests/htr/experiment/test_real_inference.py`'s discipline. Two prerequisites:

* `torch`/`transformers`/`timm`/`einops` importable (the `transformers` extra), and
* `dataset-rgb/` present at the repository root. That corpus is real archival material and is
  deliberately never committed to git, so this test cannot assume it exists.

Deliberately **one** page and a bounded assertion set. This is a "the real thing genuinely runs and
returns plausible geometry" check, not the screening: the screening is
`scripts/run_segmentation_smoke_test.py`, which records its results durably.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from archivetrust.htr.segmentation import (
    Florence2LineDetectorAdapter,
    PageImage,
    is_double_page_spread,
)
from archivetrust.providers.florence2_htr.facade import florence2_dependencies_available

pytestmark = pytest.mark.real_model

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "dataset-rgb"
INVENTORY = (
    REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "dataset-inventory.csv"
)

_deps_ok, _deps_message = florence2_dependencies_available()

requires_real_detection = pytest.mark.skipif(
    not (_deps_ok and DATASET_ROOT.is_dir() and INVENTORY.is_file()),
    reason=(
        f"florence2_dependencies_available={_deps_ok} ({_deps_message}); "
        f"dataset-rgb present={DATASET_ROOT.is_dir()}; inventory present={INVENTORY.is_file()}"
    ),
)


def _first_spread_page() -> Path | None:
    with INVENTORY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["automatic_category"] == "Structurally difficult":
                return REPO_ROOT / row["relative_path"]
    return None


@requires_real_detection
def test_real_detector_finds_lines_on_a_real_spread():
    page_path = _first_spread_page()
    if page_path is None or not page_path.is_file():
        pytest.skip(f"no readable spread page found under {DATASET_ROOT}")

    page = PageImage(
        page_id="page_real_detection_test",
        image_bytes=page_path.read_bytes(),
        source_path=str(page_path),
    )
    adapter = Florence2LineDetectorAdapter()

    regions = adapter.detect_regions(page)

    decoded = page.open()
    assert is_double_page_spread(decoded.width, decoded.height)
    # A spread is split into exactly two page-side regions -- the stage's central decision, on a
    # real page rather than a synthetic one.
    assert len(regions) == 2
    assert [r.region_type for r in regions] == ["page_side_left", "page_side_right"]

    evidence = adapter.last_region_evidence(page.page_id)
    assert evidence.spread_detected is True
    # The gutter is somewhere near, but not necessarily at, the naive midpoint.
    assert evidence.gutter_evidence["search_band_px"][0] < evidence.gutter_x
    assert evidence.gutter_x < evidence.gutter_evidence["search_band_px"][1]

    lines = adapter.detect_lines(regions[0], page_image=page)

    # A page of a 17th-century court record has lines; zero would mean the detector or the task
    # token is wrong, not that the page is blank.
    assert len(lines) > 0
    outcome = adapter.last_line_outcome(regions[0].region_id)
    assert outcome.inference_seconds > 0.0
    assert outcome.device_used in ("cuda", "cpu")

    for line in lines:
        box = line.bounding_box
        # Every box lies inside the page raster and has real extent.
        assert 0 <= box.x0 < box.x1 <= decoded.width
        assert 0 <= box.y0 < box.y1 <= decoded.height
        # Every box's centre is on the left leaf -- the ownership rule, on real geometry.
        assert (box.x0 + box.x1) / 2 < evidence.gutter_x
    assert [line.reading_order_index for line in lines] == list(range(len(lines)))
