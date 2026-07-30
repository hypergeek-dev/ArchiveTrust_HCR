"""The double-page-spread decision, tested on its own terms.

93.6% of this corpus is a double-page spread, so these are the tests that matter most about whether
the stage handles the corpus it was built for -- see `htr/segmentation/spread.py`'s module docstring
for the decision and the risk it accepts.
"""

from __future__ import annotations

from PIL import Image

import io

from archivetrust.htr.segmentation.spread import (
    GUTTER_OVERLAP_FRACTION,
    SPREAD_ASPECT_RATIO_THRESHOLD,
    column_profile,
    estimate_gutter,
    is_double_page_spread,
    page_side_windows,
    whole_page_window,
)

from tests.htr.segmentation import _fakes


def _open(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def test_spread_threshold_matches_stage_1_inventory_policy():
    """The threshold is Stage 1's own 1.15 (dataset-manifest.json's
    `classification_policy.spread_aspect_ratio_threshold`), reused rather than re-derived, so
    segmentation and the inventory cannot disagree about what a spread is."""
    assert SPREAD_ASPECT_RATIO_THRESHOLD == 1.15


def test_landscape_spread_is_detected_and_portrait_page_is_not():
    assert is_double_page_spread(3816, 2736) is True  # a real corpus spread's dimensions
    assert is_double_page_spread(2688, 3176) is False  # a real corpus Ordinary page's dimensions


def test_aspect_ratio_exactly_at_threshold_is_a_spread():
    assert is_double_page_spread(1150, 1000) is True
    assert is_double_page_spread(1149, 1000) is False


def test_zero_height_is_not_a_spread_rather_than_a_division_error():
    assert is_double_page_spread(1000, 0) is False


def test_column_profile_is_one_mean_per_column():
    image = _open(_fakes.spread_bytes())
    profile = column_profile(image, profile_width=200)
    assert len(profile) == 200
    assert all(0.0 <= value <= 255.0 for value in profile)


def test_gutter_estimate_finds_the_real_gutter_not_the_naive_midpoint():
    """The synthetic spread's gutter is at x=810, deliberately offset from the midpoint (800), so
    this cannot pass by accidentally agreeing with `width // 2`."""
    image = _open(_fakes.spread_bytes(width=1600, height=1000, gutter_x=810))
    estimate = estimate_gutter(image)
    assert abs(estimate.gutter_x - 810) <= 12
    assert estimate.gutter_x != 1600 // 2


def test_gutter_contrast_is_high_for_a_real_gutter():
    image = _open(_fakes.spread_bytes())
    assert estimate_gutter(image).gutter_contrast > 0.5


def test_gutter_contrast_is_low_when_there_is_no_gutter():
    """A page with no dark central band still yields *an* estimate -- the darkest central column
    always exists -- but its contrast is near zero. This is why contrast is reported rather than
    thresholded away inside the estimator: the number is the evidence."""
    image = Image.new("RGB", (1600, 1000), (238, 232, 214))
    assert estimate_gutter(image).gutter_contrast < 0.05


def test_overlap_margin_is_zero_on_measured_evidence():
    """A 2% margin was measured to cost an entire densely-written leaf on a real corpus page: it
    pulled the opposite page's line-endings into the window, the detector locked onto them, and the
    leaf yielded zero lines. See `spread.py::GUTTER_OVERLAP_FRACTION`'s table."""
    assert GUTTER_OVERLAP_FRACTION == 0.0


def test_page_side_windows_split_at_the_gutter_and_own_disjoint_halves():
    left, right = page_side_windows(1000, 500)
    margin = int(round(1000 * GUTTER_OVERLAP_FRACTION))

    assert left.window == (0, 500 + margin)
    assert right.window == (500 - margin, 1000)

    # Ownership never overlaps, so the same line cannot be claimed by both sides. This holds
    # independently of the margin, and is what keeps a reintroduced margin correct.
    assert left.owns == (0, 500)
    assert right.owns == (500, 1000)
    assert left.contains_centre(499) and not right.contains_centre(499)
    assert right.contains_centre(500) and not left.contains_centre(500)


def test_no_window_reaches_into_the_opposite_page():
    """The asymmetry the smoke test measured: a window stopping inside its own page loses a few
    pixels; a window reaching into the opposite page can lose the whole leaf."""
    left, right = page_side_windows(5159, 2580)
    assert left.window[1] <= 2580
    assert right.window[0] >= 2580


def test_page_side_windows_are_clamped_to_the_raster():
    """A gutter estimate near an edge must not produce a negative or out-of-raster window."""
    left, right = page_side_windows(1000, 5)
    assert left.window[0] == 0
    assert right.window[1] == 1000


def test_whole_page_window_owns_everything_and_uses_the_same_type():
    (window,) = whole_page_window(1234)
    assert window.window == (0, 1234)
    assert window.owns == (0, 1234)
    assert window.label == "page_full"
    assert window.contains_centre(0) and window.contains_centre(1233)
