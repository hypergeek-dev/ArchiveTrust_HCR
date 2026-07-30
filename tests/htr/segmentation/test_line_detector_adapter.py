"""`Florence2LineDetectorAdapter` against a fake detector facade -- the stage's logic, not its model.

What these prove: that the adapter satisfies §7's Protocol, that a spread is genuinely split before
detection (observable in the window sizes the detector was shown), that window-local boxes are
translated into page coordinates, that the overlap margin cannot double-count a line, that reading
order is region-major, that crops are content-addressed, and that a detector failure is raised as a
named error rather than degraded into "zero lines".
"""

from __future__ import annotations

import pytest

from archivetrust.domain.evidence.models import Precision
from archivetrust.htr.corpus.models import InputCrop
from archivetrust.htr.segmentation import (
    CONFOUND_STATEMENT,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    TASK_PROMPT,
    Florence2LineDetectorAdapter,
    LineDetectionFailedError,
    SegmentationAdapter,
    SegmentationConfiguration,
)

from tests.htr.segmentation import _fakes


def _adapter(facade) -> Florence2LineDetectorAdapter:
    return Florence2LineDetectorAdapter(facade=facade)


def test_adapter_satisfies_the_domain_design_protocol():
    """docs/htr-domain-design.md §7's `SegmentationAdapter`, verified by isinstance -- the same
    contract check `SatrnAdapter`/`Florence2Adapter` carry against `HtrMethodAdapter`."""
    assert isinstance(_adapter(_fakes.FakeLineDetectorFacade(())), SegmentationAdapter)


def test_configuration_pins_the_checkpoint_and_carries_the_confound():
    configuration = SegmentationConfiguration()
    assert configuration.model_id == DEFAULT_MODEL_ID == "nazounoryuu/florence_base__mixed__page__line_od"
    # A resolved commit sha, never a floating "main".
    assert configuration.model_revision == DEFAULT_MODEL_REVISION
    assert len(configuration.model_revision) == 40
    assert configuration.task_prompt == TASK_PROMPT == "<OD>"
    assert configuration.spread_handling == "split_at_gutter_before_detection"
    # The confound is a field, not merely documentation -- so it reaches telemetry and reports.
    assert configuration.confound_statement == CONFOUND_STATEMENT
    assert "Florence-2-family" in configuration.confound_statement
    assert "SATRN" in configuration.confound_statement


def test_single_page_yields_one_full_width_region():
    facade = _fakes.FakeLineDetectorFacade(((),))
    adapter = _adapter(facade)
    page = _fakes.page_image("page_single", _fakes.single_page_bytes(600, 800))

    regions = adapter.detect_regions(page)

    assert len(regions) == 1
    assert regions[0].region_type == "page_full"
    assert (regions[0].bounding_box.x0, regions[0].bounding_box.x1) == (0.0, 600.0)
    evidence = adapter.last_region_evidence("page_single")
    assert evidence is not None
    assert evidence.spread_detected is False
    assert evidence.gutter_x is None


def test_spread_yields_two_page_side_regions_in_reading_order():
    facade = _fakes.FakeLineDetectorFacade(((), ()))
    adapter = _adapter(facade)
    page = _fakes.page_image("page_spread", _fakes.spread_bytes(1600, 1000, gutter_x=810))

    regions = adapter.detect_regions(page)

    assert [region.region_type for region in regions] == ["page_side_left", "page_side_right"]
    assert [region.order_index for region in regions] == [0, 1]
    evidence = adapter.last_region_evidence("page_spread")
    assert evidence.spread_detected is True
    assert abs(evidence.gutter_x - 810) <= 12
    # The gutter evidence is recorded for a reviewer, including how far it sits from the naive guess.
    assert evidence.gutter_evidence["naive_midpoint_x"] == 800
    assert evidence.gutter_evidence["gutter_contrast"] > 0.5


def test_region_geometry_is_recorded_as_a_coarse_estimate():
    """A page-side window comes from a projection profile and an aspect-ratio rule; calling it
    pixel-accurate would be a false precision claim."""
    adapter = _adapter(_fakes.FakeLineDetectorFacade(((), ())))
    regions = adapter.detect_regions(_fakes.page_image("p", _fakes.spread_bytes()))
    assert all(region.bounding_box.precision is Precision.COARSE_ESTIMATE for region in regions)


def test_spread_is_actually_split_before_detection_not_after():
    """The observable consequence of the split decision: the detector is shown two ~half-width
    windows, never the whole spread. This is the test that would fail if the stage silently
    switched to whole-spread detection with post-hoc grouping."""
    facade = _fakes.FakeLineDetectorFacade(((), ()))
    adapter = _adapter(facade)
    page = _fakes.page_image("page_spread", _fakes.spread_bytes(1600, 1000, gutter_x=810))

    regions = adapter.detect_regions(page)
    for region in regions:
        adapter.detect_lines(region, page_image=page)

    assert len(facade.calls) == 2
    shown_widths = [call["image_size"][0] for call in facade.calls]
    assert all(width < 1600 for width in shown_widths), shown_widths
    # Each window is roughly half the spread plus the overlap margin, not the whole thing.
    assert all(700 < width < 1000 for width in shown_widths), shown_widths


def test_detector_is_prompted_with_the_od_task_token():
    facade = _fakes.FakeLineDetectorFacade(((),))
    adapter = _adapter(facade)
    page = _fakes.page_image("p", _fakes.single_page_bytes())
    (region,) = adapter.detect_regions(page)
    adapter.detect_lines(region, page_image=page)
    assert facade.calls[0]["task_prompt"] == "<OD>"
    assert facade.calls[0]["model_id"] == DEFAULT_MODEL_ID
    assert facade.calls[0]["model_revision"] == DEFAULT_MODEL_REVISION


def test_window_local_boxes_are_translated_into_page_coordinates():
    """The right-hand window starts at an x offset; a box at x=10 inside it is not at x=10 on the
    page. Getting this wrong would silently crop the wrong part of every right-hand page."""
    left_boxes = ((50.0, 100.0, 400.0, 160.0),)
    right_boxes = ((10.0, 100.0, 500.0, 160.0),)
    facade = _fakes.FakeLineDetectorFacade((left_boxes, right_boxes))
    adapter = _adapter(facade)
    page = _fakes.page_image("page_spread", _fakes.spread_bytes(1600, 1000, gutter_x=810))

    left_region, right_region = adapter.detect_regions(page)
    (left_line,) = adapter.detect_lines(left_region, page_image=page)
    (right_line,) = adapter.detect_lines(right_region, page_image=page)

    # Left window starts at 0, so its coordinates are unchanged.
    assert left_line.bounding_box.x0 == 50.0
    # Right window starts at (gutter - margin), so the box is shifted by exactly that offset.
    offset = right_region.bounding_box.x0
    assert right_line.bounding_box.x0 == 10.0 + offset
    assert right_line.bounding_box.x1 == 500.0 + offset


def test_a_box_in_the_overlap_belonging_to_the_other_side_is_discarded_and_counted():
    """The overlap margin lets both windows see the gutter area. Centre-based ownership is what
    stops the same line being detected twice -- and the discard is counted, never silent."""
    # Gutter lands near 810; the left window extends to 810+32. A box centred past 810 is the right
    # page's, seen through the left window's margin.
    left_boxes = (
        (100.0, 100.0, 700.0, 160.0),  # genuinely left
        (815.0, 300.0, 838.0, 360.0),  # centre ~826 -> the right page's, must be discarded
    )
    facade = _fakes.FakeLineDetectorFacade((left_boxes, ()))
    adapter = _adapter(facade)
    page = _fakes.page_image("page_spread", _fakes.spread_bytes(1600, 1000, gutter_x=810))

    left_region, _ = adapter.detect_regions(page)
    lines = adapter.detect_lines(left_region, page_image=page)

    assert len(lines) == 1
    outcome = adapter.last_line_outcome(left_region.region_id)
    assert outcome.boxes_returned == 2
    assert outcome.boxes_discarded_wrong_side == 1


def test_detection_crowded_into_one_edge_band_is_flagged_suspect():
    """The exact shape of the Checkpoint 3 smoke-test failure: a window whose every detection sits
    against one boundary is the detector locking onto adjacent content, not reading its page. It
    must be visible, because centre-based ownership will then discard them all and leave a bare
    "0 lines" that looks like an empty page."""
    # Six boxes, all inside the leftmost 10% of a 600-wide page window (x < 60).
    boxes = tuple((5.0, 50.0 + i * 60, 45.0, 100.0 + i * 60) for i in range(6))
    adapter = _adapter(_fakes.FakeLineDetectorFacade((boxes,)))
    page = _fakes.page_image("p", _fakes.single_page_bytes(600, 800))
    (region,) = adapter.detect_regions(page)
    adapter.detect_lines(region, page_image=page)

    outcome = adapter.last_line_outcome(region.region_id)
    assert outcome.boxes_in_edge_band == 6
    assert outcome.detection_suspect is True


def test_normally_distributed_detections_are_not_flagged_suspect():
    boxes = tuple((60.0, 50.0 + i * 60, 540.0, 100.0 + i * 60) for i in range(6))
    adapter = _adapter(_fakes.FakeLineDetectorFacade((boxes,)))
    page = _fakes.page_image("p", _fakes.single_page_bytes(600, 800))
    (region,) = adapter.detect_regions(page)
    adapter.detect_lines(region, page_image=page)

    outcome = adapter.last_line_outcome(region.region_id)
    assert outcome.detection_suspect is False


def test_a_handful_of_edge_boxes_is_not_enough_to_be_suspect():
    """Three short marginal notes down one edge are a real page feature, not a detector failure."""
    boxes = tuple((5.0, 50.0 + i * 60, 45.0, 100.0 + i * 60) for i in range(3))
    adapter = _adapter(_fakes.FakeLineDetectorFacade((boxes,)))
    page = _fakes.page_image("p", _fakes.single_page_bytes(600, 800))
    (region,) = adapter.detect_regions(page)
    adapter.detect_lines(region, page_image=page)
    assert adapter.last_line_outcome(region.region_id).detection_suspect is False


def test_line_geometry_is_recorded_as_a_coarse_estimate():
    """Florence-2 emits `<loc_N>` geometry quantized to 1000 bins per axis -- never pixel-accurate."""
    facade = _fakes.FakeLineDetectorFacade((((10.0, 20.0, 300.0, 80.0),),))
    adapter = _adapter(facade)
    page = _fakes.page_image("p", _fakes.single_page_bytes())
    (region,) = adapter.detect_regions(page)
    (line,) = adapter.detect_lines(region, page_image=page)
    assert line.bounding_box.precision is Precision.COARSE_ESTIMATE


def test_reading_order_within_a_region_is_top_to_bottom():
    boxes = (
        (60.0, 400.0, 500.0, 460.0),
        (60.0, 100.0, 500.0, 160.0),
        (60.0, 250.0, 500.0, 310.0),
    )
    facade = _fakes.FakeLineDetectorFacade((boxes,))
    adapter = _adapter(facade)
    page = _fakes.page_image("p", _fakes.single_page_bytes())
    (region,) = adapter.detect_regions(page)
    lines = adapter.detect_lines(region, page_image=page)

    assert [line.reading_order_index for line in lines] == [0, 1, 2]
    assert [line.bounding_box.y0 for line in lines] == [100.0, 250.0, 400.0]


def test_global_reading_order_is_region_major_not_y_interleaved():
    """For a spread, ordering by y across both sides would alternate between two physically separate
    leaves. Left page's lines must all precede the right page's."""
    left_boxes = ((60.0, 500.0, 700.0, 560.0), (60.0, 100.0, 700.0, 160.0))
    right_boxes = ((10.0, 300.0, 600.0, 360.0), (10.0, 50.0, 600.0, 110.0))
    facade = _fakes.FakeLineDetectorFacade((left_boxes, right_boxes))
    adapter = _adapter(facade)
    page = _fakes.page_image("page_spread", _fakes.spread_bytes(1600, 1000, gutter_x=810))

    left_region, right_region = adapter.detect_regions(page)
    lines = adapter.detect_lines(left_region, page_image=page) + adapter.detect_lines(
        right_region, page_image=page
    )
    ordered = adapter.order_lines(lines)

    assert [line.region_id for line in ordered] == [
        left_region.region_id,
        left_region.region_id,
        right_region.region_id,
        right_region.region_id,
    ]


def test_order_lines_does_not_renumber_the_entities():
    """`reading_order_index` is within-region by definition; re-sequencing the tuple must not
    silently rewrite it into a global index."""
    left_boxes = ((60.0, 100.0, 700.0, 160.0),)
    right_boxes = ((10.0, 50.0, 600.0, 110.0),)
    adapter = _adapter(_fakes.FakeLineDetectorFacade((left_boxes, right_boxes)))
    page = _fakes.page_image("page_spread", _fakes.spread_bytes(1600, 1000, gutter_x=810))
    left_region, right_region = adapter.detect_regions(page)
    lines = adapter.detect_lines(left_region, page_image=page) + adapter.detect_lines(
        right_region, page_image=page
    )
    ordered = adapter.order_lines(lines)
    assert [line.reading_order_index for line in ordered] == [0, 0]


def test_crops_are_content_addressed_and_cut_from_the_full_resolution_page(tmp_path):
    boxes = ((60.0, 100.0, 500.0, 160.0), (60.0, 250.0, 500.0, 310.0))
    adapter = _adapter(_fakes.FakeLineDetectorFacade((boxes,)))
    page = _fakes.page_image("p", _fakes.single_page_bytes(600, 800))
    (region,) = adapter.detect_regions(page)
    lines = adapter.detect_lines(region, page_image=page)

    crops = adapter.crop_lines(lines, page_image=page, destination=tmp_path)

    assert len(crops) == 2
    for crop, line in zip(crops, lines, strict=True):
        assert crop.hash.startswith("crop_")
        assert crop.text_line_id == line.text_line_id
        written = (tmp_path / f"{line.text_line_id}.png").read_bytes()
        # The recorded hash is the hash of the bytes actually on disk -- InputCrop.compute_hash,
        # not a second scheme.
        assert crop.hash == InputCrop.compute_hash(written)
        assert crop.byte_size == len(written)
    # Crop dimensions come from the original page raster, not the detector's 768x768 input.
    assert crops[0].width == 440
    assert crops[0].height == 60


def test_crop_boxes_are_clamped_to_the_raster():
    """A quantized box can land fractionally outside the page; Pillow would pad such a crop with
    black, changing the bytes the recognizers read for a non-document reason."""
    boxes = ((-20.0, -10.0, 700.0, 160.0),)
    adapter = _adapter(_fakes.FakeLineDetectorFacade((boxes,)))
    page = _fakes.page_image("p", _fakes.single_page_bytes(600, 800))
    (region,) = adapter.detect_regions(page)
    (line,) = adapter.detect_lines(region, page_image=page)
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        (crop,) = adapter.crop_lines((line,), page_image=page, destination=directory)
    assert crop.width == 600
    assert crop.height == 160


def test_zero_area_crop_is_refused_rather_than_written(tmp_path):
    """A sliver box that rounds to zero width. Its centre is still on the page, so it is genuinely
    owned by this region -- it is the crop, not the ownership test, that must reject it."""
    boxes = ((10.2, 10.0, 10.4, 60.0),)
    adapter = _adapter(_fakes.FakeLineDetectorFacade((boxes,)))
    page = _fakes.page_image("p", _fakes.single_page_bytes(600, 800))
    (region,) = adapter.detect_regions(page)
    (line,) = adapter.detect_lines(region, page_image=page)
    with pytest.raises(LineDetectionFailedError):
        adapter.crop_lines((line,), page_image=page, destination=tmp_path)


def test_detector_failure_raises_a_named_error_rather_than_returning_no_lines():
    """A detector that *failed* and a page that genuinely has no text are different facts
    (Constitution Article 18)."""
    facade = _fakes.FailingLineDetectorFacade(category="cuda_oom", message="CUDA out of memory")
    adapter = _adapter(facade)
    page = _fakes.page_image("p", _fakes.single_page_bytes())
    (region,) = adapter.detect_regions(page)

    with pytest.raises(LineDetectionFailedError) as excinfo:
        adapter.detect_lines(region, page_image=page)
    assert excinfo.value.category == "cuda_oom"
    assert "CUDA out of memory" in str(excinfo.value)


def test_a_page_with_genuinely_no_lines_returns_empty_rather_than_raising():
    adapter = _adapter(_fakes.FakeLineDetectorFacade(((),)))
    page = _fakes.page_image("p", _fakes.single_page_bytes())
    (region,) = adapter.detect_regions(page)
    assert adapter.detect_lines(region, page_image=page) == ()


def test_adapter_name_pins_the_checkpoint_revision():
    """`segmentation_adapter_name` reaches SegmentationRunCompleted; a bare "florence2" there would
    make a recorded run unable to say which detector produced it."""
    adapter = _adapter(_fakes.FakeLineDetectorFacade(()))
    assert adapter.name == f"florence2_line_od@{DEFAULT_MODEL_REVISION}"
