"""Segmentation metric tests (docs/htr-migration-plan.md Stage 9).

Reference geometry comes from `tests/fixtures/transkribus/sample_page.xml`'s real PAGE XML
`Coords` (parsed via the real `providers/transkribus/parsing.py::parse_export_file`, the same
parser the Transkribus adapter itself uses) -- two real `TextRegion`s, three real `TextLine`s.
`BoundingBox` (`domain/evidence/models.py`) has no polygon field, so each polygon's axis-aligned
bounding box (min/max of its points) is used, per this module's own axis-aligned-only design (see
`segmentation.py`'s module docstring).

There is only one real segmentation (the ground truth) in this fixture -- no second, independent
"hypothesis" segmentation of the same page exists in the repo. Every hypothesis geometry below is
therefore a small, clearly-labeled synthetic perturbation *derived from* the real reference boxes
(shifted, merged, split, duplicated, reordered, dropped) -- not real data on its own, but not an
arbitrary unrelated fixture either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.domain.evidence.models import BoundingBox, Precision
from archivetrust.htr.corpus.models import Region, TextLine
from archivetrust.htr.evaluation import segmentation
from archivetrust.providers.transkribus.parsing import parse_export_file

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "transkribus" / "sample_page.xml"


def _bbox_from_polygon(points: tuple[tuple[float, float], ...]) -> BoundingBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys), precision=Precision.PIXEL_ACCURATE)


def _load_real_reference():
    """Parses the real fixture and returns (regions, lines) as real `htr.corpus.models` objects,
    geometry taken verbatim from the file's real Coords. `reading_order_index`/`order_index` are
    re-numbered page-wide (0, 1, 2 for lines; 0, 1 for regions) since `ParsedLine`/`ParsedRegion`
    (the Transkribus parser's transient parse-time shape) only carries a *region-relative* reading
    order, not `htr.corpus.models.TextLine`'s page-wide one -- a real ingestion step (out of this
    phase's scope) would do this renumbering when turning a `ParsedDocument` into corpus entities.
    """
    parsed, _raw = parse_export_file(str(FIXTURE), export_format="page_xml")
    regions = []
    lines = []
    line_order = 0
    for region_order, parsed_region in enumerate(parsed.regions):
        region = Region.create(
            page_id="page_1",
            bounding_box=_bbox_from_polygon(parsed_region.polygon),
            region_type=parsed_region.region_type,
            order_index=region_order,
        )
        regions.append(region)
        for parsed_line in parsed_region.lines:
            lines.append(
                TextLine.create(
                    region_id=region.region_id,
                    bounding_box=_bbox_from_polygon(parsed_line.polygon),
                    reading_order_index=line_order,
                )
            )
            line_order += 1
    return regions, lines


def test_real_fixture_parses_into_two_regions_and_three_lines():
    regions, lines = _load_real_reference()
    assert len(regions) == 2
    assert len(lines) == 3


def test_iou_of_identical_boxes_is_one():
    regions, lines = _load_real_reference()
    box = lines[0].bounding_box
    assert segmentation.iou(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    regions, lines = _load_real_reference()
    assert segmentation.iou(lines[0].bounding_box, lines[2].bounding_box) == 0.0
    # sanity: these two real lines really are disjoint (region r1 vs region r2, far apart in y)
    assert lines[0].bounding_box.y1 < lines[2].bounding_box.y0


# --- Perfect-match hypothesis (identical geometry) -------------------------------------------


def test_perfect_match_hypothesis_scores_full_precision_recall_and_iou():
    regions, lines = _load_real_reference()
    result = segmentation.precision_recall(lines, lines)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mean_matched_iou == pytest.approx(1.0)
    assert segmentation.missed_line_rate(lines, lines) == 0.0
    assert segmentation.duplicate_line_rate(lines, lines) == 0.0
    assert segmentation.merged_line_rate(lines, lines) == 0.0
    assert segmentation.split_line_rate(lines, lines) == 0.0


def test_perfect_match_hypothesis_has_full_crop_coverage_and_zero_contamination():
    _regions, lines = _load_real_reference()
    assert segmentation.crop_coverage(lines, lines) == pytest.approx(1.0)
    assert segmentation.crop_contamination(lines, lines) == pytest.approx(0.0)


def test_perfect_match_hypothesis_has_perfect_reading_and_region_order_accuracy():
    regions, lines = _load_real_reference()
    assert segmentation.reading_order_accuracy(lines, lines) == pytest.approx(1.0)
    assert segmentation.region_order_accuracy(regions, regions) == pytest.approx(1.0)


# --- Missed line (a hypothesis segmentation stage drops the marginalia line) -----------------


def test_missing_hypothesis_line_shows_up_as_missed_and_lowers_recall():
    _regions, lines = _load_real_reference()
    hypothesis = lines[:2]  # drops the real "NB dombook" marginalia line
    assert segmentation.missed_line_rate(lines, hypothesis) == pytest.approx(1 / 3)
    result = segmentation.precision_recall(lines, hypothesis)
    assert result.recall == pytest.approx(2 / 3)
    assert result.precision == 1.0  # every predicted line was a real, correct line


# --- Shifted-but-overlapping hypothesis (still matches, imperfect IoU) ------------------------


def test_shifted_hypothesis_line_still_matches_with_reduced_iou():
    _regions, lines = _load_real_reference()
    real_box = lines[0].bounding_box  # x0=130 y0=150 x1=2000 y1=220 (real Coords)
    shifted = TextLine.create(
        region_id=lines[0].region_id,
        bounding_box=BoundingBox(
            x0=real_box.x0 + 10, y0=real_box.y0, x1=real_box.x1 + 10, y1=real_box.y1, precision=Precision.PIXEL_ACCURATE
        ),
        reading_order_index=0,
    )
    score = segmentation.iou(real_box, shifted.bounding_box)
    assert 0.0 < score < 1.0
    result = segmentation.precision_recall([lines[0]], [shifted])
    assert result.matched == 1
    assert result.mean_matched_iou == pytest.approx(score)


# --- Duplicate detection: two hypothesis lines both claim the same real line ------------------


def test_duplicate_hypothesis_lines_for_the_same_real_line_raise_duplicate_line_rate():
    _regions, lines = _load_real_reference()
    real_line = lines[0]
    duplicate_a = TextLine.create(
        region_id=real_line.region_id, bounding_box=real_line.bounding_box, reading_order_index=0
    )
    duplicate_b = TextLine.create(
        region_id=real_line.region_id, bounding_box=real_line.bounding_box, reading_order_index=1
    )
    rate = segmentation.duplicate_line_rate([real_line], [duplicate_a, duplicate_b])
    assert rate == pytest.approx(0.5)  # one of the two hypothesis lines is the "extra" duplicate


# --- Merged line: one hypothesis box spans two real lines --------------------------------------


def test_merged_hypothesis_line_spanning_two_real_lines_raises_merged_line_rate():
    _regions, lines = _load_real_reference()
    line_1, line_2 = lines[0], lines[1]  # both real lines in region r1, vertically adjacent
    merged_box = BoundingBox(
        x0=min(line_1.bounding_box.x0, line_2.bounding_box.x0),
        y0=line_1.bounding_box.y0,
        x1=max(line_1.bounding_box.x1, line_2.bounding_box.x1),
        y1=line_2.bounding_box.y1,
        precision=Precision.PIXEL_ACCURATE,
    )
    merged_hypothesis_line = TextLine.create(
        region_id=line_1.region_id, bounding_box=merged_box, reading_order_index=0
    )
    rate = segmentation.merged_line_rate([line_1, line_2], [merged_hypothesis_line], iou_threshold=0.3)
    assert rate == 1.0  # the one hypothesis line overlaps both real lines above the threshold


# --- Split line: two hypothesis boxes each cover half of one real line -------------------------


def test_split_real_line_into_two_hypothesis_lines_raises_split_line_rate():
    _regions, lines = _load_real_reference()
    real_line = lines[0]
    mid_x = (real_line.bounding_box.x0 + real_line.bounding_box.x1) / 2
    left_half = TextLine.create(
        region_id=real_line.region_id,
        bounding_box=BoundingBox(
            x0=real_line.bounding_box.x0, y0=real_line.bounding_box.y0, x1=mid_x, y1=real_line.bounding_box.y1,
            precision=Precision.PIXEL_ACCURATE,
        ),
        reading_order_index=0,
    )
    right_half = TextLine.create(
        region_id=real_line.region_id,
        bounding_box=BoundingBox(
            x0=mid_x, y0=real_line.bounding_box.y0, x1=real_line.bounding_box.x1, y1=real_line.bounding_box.y1,
            precision=Precision.PIXEL_ACCURATE,
        ),
        reading_order_index=1,
    )
    rate = segmentation.split_line_rate([real_line], [left_half, right_half], iou_threshold=0.2)
    assert rate == 1.0


# --- Reading order accuracy: reversed hypothesis order is fully discordant ---------------------


def test_reversed_reading_order_hypothesis_is_fully_discordant():
    _regions, lines = _load_real_reference()
    reversed_hypothesis = [
        TextLine.create(
            region_id=line.region_id,
            bounding_box=line.bounding_box,
            reading_order_index=len(lines) - 1 - i,
        )
        for i, line in enumerate(lines)
    ]
    accuracy = segmentation.reading_order_accuracy(lines, reversed_hypothesis)
    assert accuracy == pytest.approx(0.0)


def test_reading_order_accuracy_is_none_with_fewer_than_two_matches():
    _regions, lines = _load_real_reference()
    assert segmentation.reading_order_accuracy(lines[:1], lines[:1]) is None


# --- crop_coverage / crop_contamination on a partial-overlap hypothesis ------------------------


def test_partial_overlap_hypothesis_line_yields_partial_coverage_and_nonzero_contamination():
    _regions, lines = _load_real_reference()
    real_line = lines[0]
    box = real_line.bounding_box
    half_width = (box.x1 - box.x0) / 2
    # A crop shifted right by half its own width: overlaps the real line's right half only, and
    # extends past it on the right (contamination).
    shifted_crop = TextLine.create(
        region_id=real_line.region_id,
        bounding_box=BoundingBox(
            x0=box.x0 + half_width, y0=box.y0, x1=box.x1 + half_width, y1=box.y1, precision=Precision.PIXEL_ACCURATE
        ),
        reading_order_index=0,
    )
    coverage = segmentation.crop_coverage([real_line], [shifted_crop])
    contamination = segmentation.crop_contamination([real_line], [shifted_crop])
    assert coverage == pytest.approx(0.5, abs=0.01)
    assert contamination == pytest.approx(0.5, abs=0.01)


# --- segmentation_metric_results builds a full MetricResult set --------------------------------


def test_segmentation_metric_results_never_fabricates_undefined_metrics():
    regions, lines = _load_real_reference()
    results = segmentation.segmentation_metric_results(
        reference_regions=regions,
        hypothesis_regions=regions,
        reference_lines=lines,
        hypothesis_lines=lines,
        method_run_id="method_run_1",
    )
    assert len(results) > 0
    for result in results:
        assert result.value is not None


def test_segmentation_metric_results_empty_hypothesis_still_reports_recall_and_missed_rate():
    regions, lines = _load_real_reference()
    results = segmentation.segmentation_metric_results(
        reference_regions=regions,
        hypothesis_regions=[],
        reference_lines=lines,
        hypothesis_lines=[],
        method_run_id="method_run_1",
    )
    values_by_definition = {r.metric_definition_id: r.value for r in results}
    from archivetrust.htr.evaluation import definitions

    assert values_by_definition[definitions.LINE_RECALL.metric_definition_id] == 0.0
    assert values_by_definition[definitions.MISSED_LINE_RATE.metric_definition_id] == 1.0
    # Zero hypothesis lines against a non-empty reference: precision is defined as 0.0 (an
    # evaluable population existed and nothing predicted was correct), not fabricated -- distinct
    # from the both-empty case below, which is genuinely undefined.
    assert values_by_definition[definitions.LINE_PRECISION.metric_definition_id] == 0.0
    # duplicate/merged rate are computed over hypothesis lines, of which there are none -- both
    # legitimately None (nothing to divide by that means anything) and therefore absent.
    assert definitions.DUPLICATE_LINE_RATE.metric_definition_id not in values_by_definition
    assert definitions.MERGED_LINE_RATE.metric_definition_id not in values_by_definition


def test_precision_is_undefined_not_fabricated_when_both_reference_and_hypothesis_are_empty():
    result = segmentation.precision_recall([], [])
    assert result.precision is None
    assert result.recall is None
    assert result.mean_matched_iou is None
