"""Segmentation metrics (docs/htr-migration-plan.md Stage 9's "segmentation metrics") -- only
meaningful where ground-truth geometry exists, per the brief. Operates on `Region`/`TextLine`
(`htr/corpus/models.py`), whose `bounding_box` field is `domain.evidence.models.BoundingBox`
(`x0, y0, x1, y1, precision`) -- **axis-aligned only**: `BoundingBox` carries no polygon, so IoU
here is plain rectangle IoU, not a richer polygon overlap (the brief explicitly allows this: "axis-
aligned bounding box IoU is fine unless the geometry model already has richer polygon support" --
it does not).

`InputCrop` (`htr/corpus/models.py`) carries no geometry field of its own: per its own docstring,
"Hash-addressed image bytes cropped for one `TextLine`" -- a crop's spatial extent *is* its source
`TextLine.bounding_box`, by construction. `crop_coverage`/`crop_contamination` below therefore take
`TextLine` sequences directly (the hypothesis lines a segmentation stage produced, standing in for
the crops cut from them), not a separate crop-geometry type that does not exist in the domain
model.

Real-geometry test fixture: `tests/fixtures/transkribus/sample_page.xml`'s two `TextRegion`s / three
`TextLine`s, each with real PAGE XML `Coords` -- converted to axis-aligned bounding boxes (min/max
of the polygon's points) in `tests/htr/evaluation/test_segmentation.py`, exactly as this module's
own bbox-only geometry model requires.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.evidence.models import BoundingBox
from archivetrust.htr.evaluation import definitions
from archivetrust.htr.experiment.models import MetricResult


class _HasBoundingBox(Protocol):
    bounding_box: BoundingBox


def area(box: BoundingBox) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Axis-aligned bounding-box intersection-over-union, in `[0, 1]`. `0.0` for two zero-area or
    disjoint boxes (never a division by zero)."""
    intersection = intersection_area(a, b)
    union = area(a) + area(b) - intersection
    return intersection / union if union > 0 else 0.0


class GeometryMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    reference_index: int
    hypothesis_index: int
    iou: float


class GeometryMatchResult(BaseModel):
    """Greedy one-to-one matching result: every reference item matched to at most one hypothesis
    item and vice versa, each pair's IoU at or above `iou_threshold`, chosen to maximize IoU
    (highest-IoU candidate pairs assigned first -- a standard greedy approximation of optimal
    bipartite matching, adequate at line/region counts per page)."""

    model_config = ConfigDict(frozen=True)

    matches: tuple[GeometryMatch, ...]
    unmatched_reference: tuple[int, ...]
    unmatched_hypothesis: tuple[int, ...]


def match_geometry(
    reference: Sequence[_HasBoundingBox],
    hypothesis: Sequence[_HasBoundingBox],
    *,
    iou_threshold: float = 0.5,
) -> GeometryMatchResult:
    candidates: list[tuple[float, int, int]] = []
    for ri, ref_item in enumerate(reference):
        for hi, hyp_item in enumerate(hypothesis):
            score = iou(ref_item.bounding_box, hyp_item.bounding_box)
            if score >= iou_threshold:
                candidates.append((score, ri, hi))
    candidates.sort(key=lambda triple: triple[0], reverse=True)

    matched_ref: set[int] = set()
    matched_hyp: set[int] = set()
    matches: list[GeometryMatch] = []
    for score, ri, hi in candidates:
        if ri in matched_ref or hi in matched_hyp:
            continue
        matched_ref.add(ri)
        matched_hyp.add(hi)
        matches.append(GeometryMatch(reference_index=ri, hypothesis_index=hi, iou=score))

    unmatched_reference = tuple(i for i in range(len(reference)) if i not in matched_ref)
    unmatched_hypothesis = tuple(i for i in range(len(hypothesis)) if i not in matched_hyp)
    return GeometryMatchResult(
        matches=tuple(matches),
        unmatched_reference=unmatched_reference,
        unmatched_hypothesis=unmatched_hypothesis,
    )


class PrecisionRecallResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    reference_total: int
    hypothesis_total: int
    matched: int
    precision: float | None
    """`None` only when `hypothesis_total == 0` and `reference_total == 0` (nothing to evaluate,
    never a fabricated 1.0)."""
    recall: float | None
    """`None` only when `reference_total == 0` (no ground truth to recall against)."""
    f1: float | None
    mean_matched_iou: float | None
    """Mean IoU over matched pairs only -- `None` when there are no matches."""


def precision_recall(
    reference: Sequence[_HasBoundingBox],
    hypothesis: Sequence[_HasBoundingBox],
    *,
    iou_threshold: float = 0.5,
) -> PrecisionRecallResult:
    """Generic over `Region` or `TextLine` -- the caller decides granularity by which sequence
    type it passes (this is `region_precision_recall`/`line_precision_recall`'s shared
    implementation, per `definitions.py`'s separate metric-name/description pair for each)."""
    result = match_geometry(reference, hypothesis, iou_threshold=iou_threshold)
    matched = len(result.matches)
    ref_total, hyp_total = len(reference), len(hypothesis)

    precision = matched / hyp_total if hyp_total else (None if not ref_total else 0.0)
    recall = matched / ref_total if ref_total else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    mean_matched_iou = (
        sum(m.iou for m in result.matches) / matched if matched else None
    )
    return PrecisionRecallResult(
        reference_total=ref_total,
        hypothesis_total=hyp_total,
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=mean_matched_iou,
    )


def missed_line_rate(
    reference_lines: Sequence[_HasBoundingBox],
    hypothesis_lines: Sequence[_HasBoundingBox],
    *,
    iou_threshold: float = 0.5,
) -> float | None:
    """Reference lines with no matching hypothesis line / all reference lines. `None` when there
    are no reference lines (undefined, not a fabricated 0.0)."""
    if not reference_lines:
        return None
    result = match_geometry(reference_lines, hypothesis_lines, iou_threshold=iou_threshold)
    return len(result.unmatched_reference) / len(reference_lines)


def duplicate_line_rate(
    reference_lines: Sequence[_HasBoundingBox],
    hypothesis_lines: Sequence[_HasBoundingBox],
    *,
    iou_threshold: float = 0.5,
) -> float | None:
    """Over-prediction: hypothesis lines whose *best*-IoU reference line was already claimed by
    another hypothesis line in the one-to-one match (i.e. the method predicted more than one line
    for the same true line). `None` when there are no hypothesis lines."""
    if not hypothesis_lines:
        return None
    best_reference_for: dict[int, int] = {}
    for hi, hyp_item in enumerate(hypothesis_lines):
        best_ref_index = None
        best_score = 0.0
        for ri, ref_item in enumerate(reference_lines):
            score = iou(ref_item.bounding_box, hyp_item.bounding_box)
            if score >= iou_threshold and score > best_score:
                best_score = score
                best_ref_index = ri
        if best_ref_index is not None:
            best_reference_for[hi] = best_ref_index

    claimed: set[int] = set()
    duplicates = 0
    for hi in sorted(best_reference_for):
        ref_index = best_reference_for[hi]
        if ref_index in claimed:
            duplicates += 1
        else:
            claimed.add(ref_index)
    return duplicates / len(hypothesis_lines)


def merged_line_rate(
    reference_lines: Sequence[_HasBoundingBox],
    hypothesis_lines: Sequence[_HasBoundingBox],
    *,
    iou_threshold: float = 0.5,
) -> float | None:
    """Under-segmentation: hypothesis lines overlapping (IoU >= threshold) more than one distinct
    reference line / all hypothesis lines. `None` when there are no hypothesis lines."""
    if not hypothesis_lines:
        return None
    merged = 0
    for hyp_item in hypothesis_lines:
        overlapping = sum(
            1
            for ref_item in reference_lines
            if iou(ref_item.bounding_box, hyp_item.bounding_box) >= iou_threshold
        )
        if overlapping > 1:
            merged += 1
    return merged / len(hypothesis_lines)


def split_line_rate(
    reference_lines: Sequence[_HasBoundingBox],
    hypothesis_lines: Sequence[_HasBoundingBox],
    *,
    iou_threshold: float = 0.5,
) -> float | None:
    """Over-segmentation: reference lines overlapping (IoU >= threshold) more than one distinct
    hypothesis line / all reference lines -- the mirror of `merged_line_rate`. `None` when there
    are no reference lines."""
    if not reference_lines:
        return None
    split = 0
    for ref_item in reference_lines:
        overlapping = sum(
            1
            for hyp_item in hypothesis_lines
            if iou(ref_item.bounding_box, hyp_item.bounding_box) >= iou_threshold
        )
        if overlapping > 1:
            split += 1
    return split / len(reference_lines)


def _order_concordance(
    reference: Sequence[_HasBoundingBox],
    hypothesis: Sequence[_HasBoundingBox],
    *,
    reference_order: Sequence[int],
    hypothesis_order: Sequence[int],
    iou_threshold: float,
) -> float | None:
    """Shared implementation for `reading_order_accuracy`/`region_order_accuracy`: pairwise order
    concordance over every pair of matched (reference, hypothesis) items -- for each unordered pair
    of matches, do reference and hypothesis agree on which came first? `None` when fewer than two
    matched pairs exist (order is undefined with 0 or 1 points)."""
    result = match_geometry(reference, hypothesis, iou_threshold=iou_threshold)
    if len(result.matches) < 2:
        return None
    concordant = 0
    total = 0
    matches = result.matches
    for a in range(len(matches)):
        for b in range(a + 1, len(matches)):
            ref_a, ref_b = reference_order[matches[a].reference_index], reference_order[matches[b].reference_index]
            hyp_a, hyp_b = hypothesis_order[matches[a].hypothesis_index], hypothesis_order[matches[b].hypothesis_index]
            if ref_a == ref_b:
                continue  # undefined reference ordering for this pair -- skip, don't penalize
            total += 1
            ref_says_a_first = ref_a < ref_b
            hyp_says_a_first = hyp_a < hyp_b
            if ref_says_a_first == hyp_says_a_first:
                concordant += 1
    return concordant / total if total else None


def reading_order_accuracy(
    reference_lines: Sequence, hypothesis_lines: Sequence, *, iou_threshold: float = 0.5
) -> float | None:
    """Pairwise reading-order concordance over matched `TextLine`s, using
    `TextLine.reading_order_index`."""
    return _order_concordance(
        reference_lines,
        hypothesis_lines,
        reference_order=[line.reading_order_index for line in reference_lines],
        hypothesis_order=[line.reading_order_index for line in hypothesis_lines],
        iou_threshold=iou_threshold,
    )


def region_order_accuracy(
    reference_regions: Sequence, hypothesis_regions: Sequence, *, iou_threshold: float = 0.5
) -> float | None:
    """Pairwise reading-order concordance over matched `Region`s, using `Region.order_index`.
    Regions with `order_index=None` are treated as ordered last among themselves (index `-1`
    replaced by a large sentinel would bias the comparison; instead any region lacking an
    order_index is excluded from the pairwise comparison entirely -- see the `None` filtering
    below)."""
    indexable_reference = [(i, r) for i, r in enumerate(reference_regions) if r.order_index is not None]
    if len(indexable_reference) < 2:
        return None
    # Only compare among regions where both sides declare an order -- undeclared order is not
    # scored as wrong, it is simply not evaluable.
    ref_items = [r for _, r in indexable_reference]
    hyp_items = [r for r in hypothesis_regions if r.order_index is not None]
    return _order_concordance(
        ref_items,
        hyp_items,
        reference_order=[r.order_index for r in ref_items],
        hypothesis_order=[r.order_index for r in hyp_items],
        iou_threshold=iou_threshold,
    )


def crop_coverage(reference_lines: Sequence, hypothesis_lines: Sequence) -> float | None:
    """How much of the true text area is covered by *some* predicted line/crop. Sums, per
    reference line, that line's single best-overlapping hypothesis line's intersection area (not
    a true multi-crop union -- see module docstring's `definitions.CROP_COVERAGE` description for
    why this is a lower bound when hypothesis lines overlap the same reference line). `None` when
    there are no reference lines (nothing to cover)."""
    total_reference_area = sum(area(line.bounding_box) for line in reference_lines)
    if total_reference_area == 0:
        return None
    covered = 0.0
    for ref_line in reference_lines:
        best = max(
            (intersection_area(ref_line.bounding_box, hyp_line.bounding_box) for hyp_line in hypothesis_lines),
            default=0.0,
        )
        covered += best
    return covered / total_reference_area


def crop_contamination(reference_lines: Sequence, hypothesis_lines: Sequence) -> float | None:
    """How much of the produced crops/lines falls outside any true line -- the inverse concern
    from `crop_coverage` (over-cropping neighboring content, e.g. an adjacent line's descenders).
    `None` when there are no hypothesis lines (nothing produced, nothing to contaminate)."""
    total_hypothesis_area = sum(area(line.bounding_box) for line in hypothesis_lines)
    if total_hypothesis_area == 0:
        return None
    contaminated = 0.0
    for hyp_line in hypothesis_lines:
        hyp_area = area(hyp_line.bounding_box)
        best_overlap = max(
            (intersection_area(hyp_line.bounding_box, ref_line.bounding_box) for ref_line in reference_lines),
            default=0.0,
        )
        contaminated += hyp_area - best_overlap
    return contaminated / total_hypothesis_area


def segmentation_metric_results(
    *,
    reference_regions: Sequence,
    hypothesis_regions: Sequence,
    reference_lines: Sequence,
    hypothesis_lines: Sequence,
    method_run_id: str,
    iou_threshold: float = 0.5,
) -> tuple[MetricResult, ...]:
    """Builds every segmentation `MetricResult` this module computes, for one method run's
    predicted regions/lines against one document's ground-truth regions/lines. Metrics whose
    computation returned `None` (undefined for this input, e.g. no reference lines at all) produce
    no `MetricResult` -- never a fabricated placeholder value."""
    results: list[MetricResult] = []

    def add(definition, value: float | None) -> None:
        if value is not None:
            results.append(
                MetricResult.create(
                    metric_definition_id=definition.metric_definition_id,
                    method_run_id=method_run_id,
                    value=value,
                )
            )

    region_pr = precision_recall(reference_regions, hypothesis_regions, iou_threshold=iou_threshold)
    add(definitions.REGION_PRECISION, region_pr.precision)
    add(definitions.REGION_RECALL, region_pr.recall)

    line_pr = precision_recall(reference_lines, hypothesis_lines, iou_threshold=iou_threshold)
    add(definitions.LINE_PRECISION, line_pr.precision)
    add(definitions.LINE_RECALL, line_pr.recall)
    add(definitions.MEAN_MATCHED_IOU, line_pr.mean_matched_iou)

    add(definitions.MISSED_LINE_RATE, missed_line_rate(reference_lines, hypothesis_lines, iou_threshold=iou_threshold))
    add(
        definitions.DUPLICATE_LINE_RATE,
        duplicate_line_rate(reference_lines, hypothesis_lines, iou_threshold=iou_threshold),
    )
    add(definitions.MERGED_LINE_RATE, merged_line_rate(reference_lines, hypothesis_lines, iou_threshold=iou_threshold))
    add(definitions.SPLIT_LINE_RATE, split_line_rate(reference_lines, hypothesis_lines, iou_threshold=iou_threshold))

    add(
        definitions.READING_ORDER_ACCURACY,
        reading_order_accuracy(reference_lines, hypothesis_lines, iou_threshold=iou_threshold),
    )
    add(
        definitions.REGION_ORDER_ACCURACY,
        region_order_accuracy(reference_regions, hypothesis_regions, iou_threshold=iou_threshold),
    )

    add(definitions.CROP_COVERAGE, crop_coverage(reference_lines, hypothesis_lines))
    add(definitions.CROP_CONTAMINATION, crop_contamination(reference_lines, hypothesis_lines))

    return tuple(results)
