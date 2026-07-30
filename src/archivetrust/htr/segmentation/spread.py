"""Double-page-spread detection and gutter location.

**Why this module exists at all.** Stage 1's inventory measured that **717 of 766 pages (93.6%)**
of this corpus are double-page spreads: an open bound volume photographed flat, with a central
gutter, two facing text blocks, and *two independent reading orders*
(`docs/experiments/technical-reliability-screening/design-audit.md` §4.4.2). Ignoring that property
would mean 94% of the corpus is segmented as though it were one page with one reading order.

## The decision: split before line detection

Of the two options, this stage **splits the spread into two logical page-sides before running line
detection**, rather than detecting across the whole spread and grouping afterwards. Each side
becomes one `Region` and receives its own detector invocation and its own
`reading_order_index` sequence.

Reasons, in order of weight:

1. **The checkpoint is a single-page detector.** `nazounoryuu/florence_base__mixed__page__line_od`
   is fine-tuned for line detection over a *page*. A spread is out-of-distribution input for it.
2. **Horizontal resolution.** Florence-2's processor resizes any input to a fixed 768x768 square.
   A 3816x2736 spread is squeezed 5.0x horizontally; each 1908x2736 half is squeezed 2.5x. Halving
   the horizontal squeeze is a direct, measurable improvement in the detail the detector sees.
   (Vertical scale is unaffected by splitting, so this argument is horizontal-only -- stated
   precisely rather than as a general "higher resolution" claim.)
3. **Two reading orders become structural, not reconstructed.** Splitting makes each side's reading
   order the detector's own emission order within that side. Whole-spread detection produces one
   flat list that must be re-split by a coordinate rule afterwards -- the same gutter estimate is
   needed either way, but in that design an error in it silently corrupts reading order instead of
   visibly clipping a crop.
4. **A cut at the gutter cannot destroy a real line.** The two sides of an open bound volume are
   physically separate leaves; no line of text runs from one to the other. So, unlike splitting a
   single page, splitting at a correctly located gutter cannot cut a line in half.

**The measured risk this accepts.** Reason 4 holds only while the gutter estimate lands *inside*
the gutter. A wrong estimate clips line ends. Three mitigations, all implemented:

* the split is attempted **only** when the page's aspect ratio is at or above
  `SPREAD_ASPECT_RATIO_THRESHOLD` -- Stage 1's own 1.15, reused verbatim rather than re-derived, so
  this stage and the inventory agree by construction on what a spread is. This gate is load-bearing
  and was verified necessary: run on a genuine single page (aspect ratio 0.846), the darkest-column
  search below returns a confident-looking but meaningless "gutter" at 56% of width, and splitting
  there would cut every full-width line in half.
* the detector input window for each side carries a symmetric **overlap margin**
  (`GUTTER_OVERLAP_FRACTION` of page width) past the gutter estimate, so a modestly misplaced
  estimate still shows the detector whole lines rather than clipped ones;
* a line detected inside that overlap is assigned to a side by its **centre**, and discarded by the
  other side, so the margin cannot produce the same line twice. Discards are counted and reported
  (`LineDetectionOutcome.boxes_discarded_wrong_side`), never silent.

## The estimator

A vertical projection profile: the page is reduced to one row of column means and the darkest
column within the central band is taken as the gutter. A bound volume's gutter is shadowed, so it
is the darkest central column; this is the standard, dependency-free formulation, and it matters
that it *is* dependency-free -- the design audit §4.4.1 records that this repository has no OpenCV,
scikit-image, kraken or docTR dependency, and this stage deliberately does not add one for a
20-line projection profile.

The reduction is done by `Image.resize((width, 1), BOX)`, which is exactly a per-column mean, in one
C-level call rather than a Python loop over millions of pixels.

**The estimate's confidence is reported, never assumed.** `gutter_contrast` is how much darker the
chosen column is than the page's median column. A spread whose gutter is not appreciably darker
than its page body produces a low value, and that number travels into the telemetry and the report
instead of being thresholded away here -- this stage's job is to measure the split, not to decide
that a weak one is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PillowImage

SPREAD_ASPECT_RATIO_THRESHOLD = 1.15
"""Width/height at or above which a page is treated as a double-page spread.

Taken verbatim from Stage 1's `classification_policy.spread_aspect_ratio_threshold` in
`docs/experiments/technical-reliability-screening/dataset-manifest.json`, so segmentation and the
inventory cannot drift into disagreeing about which pages are spreads."""

GUTTER_SEARCH_BAND = (0.35, 0.65)
"""Fraction-of-width window the gutter is searched in. Wide enough for a volume photographed
off-centre, narrow enough that a dark margin at the page edge cannot win."""

GUTTER_OVERLAP_FRACTION = 0.0
"""Symmetric margin, as a fraction of page width, that each side's detector window extends past the
gutter estimate. **Set to zero on measured evidence** -- see below.

A 0.02 margin was the original design (absorb a modest gutter-estimate error; suppress the
resulting duplicates by centre-based ownership). The Checkpoint 3 smoke test showed that reasoning
was wrong, and expensively so, on a real corpus page
(`Kommission_ang__trolldomsvasendet_i_S_ta_Katarina_forsamling__1675_`, 5159x4073, gutter 2580,
margin 103 px):

| right-side detector window | boxes returned | boxes beyond the window's leftmost 10% | max box x |
|---|---|---|---|
| gutter - 103 px (0.02 margin) | 27 | **0** | 122 |
| gutter (no margin) | 27 | 27 | 2521 |
| gutter + 103 px (inset) | 28 | 28 | 2418 |

With the margin, all 27 detections landed in the 103 px strip of *left*-page line-endings the margin
had pulled in; the detector locked onto them and returned nothing at all for the right leaf. Every
one of those boxes was then correctly discarded by centre-based ownership -- and the right leaf,
which is densely written, yielded **zero** lines. The margin did not absorb an error; it fed the
detector a foreign-page distractor and cost an entire leaf.

The asymmetry is the lesson: a window that stops slightly *inside* its own page loses a few pixels
of line start, while a window that reaches slightly *into the opposite page* can lose the whole
leaf. Zero is therefore the safe side of an estimate that cannot be exact. `PageSideWindow` keeps
its `window`/`owns` distinction rather than collapsing to one field, so restoring a margin later is
a one-constant change -- and so the centre-ownership rule remains in place as a safety net that is
now a no-op in the normal case, and still correct if a margin is ever reintroduced."""

_PROFILE_WIDTH = 800
"""Column count the projection profile is computed at. The gutter is a wide shadow band, not a
one-pixel feature, so a ~800-column profile locates it to well within its own width while keeping
the reduction cheap on a 20 MP raster."""


@dataclass(frozen=True)
class GutterEstimate:
    """Where the gutter was found, and how much to believe it."""

    gutter_x: int
    gutter_darkness: float
    """Mean intensity (0-255) of the chosen column."""
    median_column_darkness: float
    """Mean intensity of the page's median column, the baseline it is compared against."""
    search_band_px: tuple[int, int]

    @property
    def gutter_contrast(self) -> float:
        """How much darker the gutter is than a typical column, normalized to the page's own
        brightness. Higher is a more confident split. Reported, never thresholded here."""
        if self.median_column_darkness <= 0:
            return 0.0
        return (self.median_column_darkness - self.gutter_darkness) / self.median_column_darkness


def is_double_page_spread(width: int, height: int) -> bool:
    """Stage 1's own rule, applied to this page's real dimensions."""
    if height <= 0:
        return False
    return (width / height) >= SPREAD_ASPECT_RATIO_THRESHOLD


def column_profile(image: "PillowImage", *, profile_width: int = _PROFILE_WIDTH) -> tuple[float, ...]:
    """Vertical projection profile: one mean intensity per column.

    `resize((profile_width, 1), BOX)` averages every pixel of each column into a single sample --
    a genuine mean, not a subsample, computed in Pillow's C resampling path.
    """
    from PIL import Image  # noqa: PLC0415

    grayscale = image.convert("L")
    reduced = grayscale.resize((profile_width, 1), Image.BOX)
    return tuple(float(value) for value in reduced.getdata())


_DARK_RUN_TOLERANCE = 0.10
"""How much brighter than the darkest column a column may be and still count as part of the gutter
band, as a fraction of the profile's own dark-to-median range."""


def estimate_gutter(image: "PillowImage", *, profile_width: int = _PROFILE_WIDTH) -> GutterEstimate:
    """Locates the gutter as the **centre of the darkest contiguous band** inside the search window.

    Taking the single darkest column would be wrong in a way that matters: a gutter is a shadow
    *band* tens of pixels wide, and its columns are frequently equal-valued or near-equal. `min()`
    over them returns whichever tied column comes first -- the band's left *edge*, not its centre --
    which biases every split consistently towards the left page and clips the right page's line
    starts. Locating the darkest run and taking its midpoint removes that bias.

    Never raises: an unusable image is a caller-level concern, and a flat profile simply yields a
    low `gutter_contrast`.
    """
    profile = column_profile(image, profile_width=profile_width)
    columns = len(profile)
    low = int(columns * GUTTER_SEARCH_BAND[0])
    high = max(low + 1, int(columns * GUTTER_SEARCH_BAND[1]))
    band = profile[low:high]

    ordered = sorted(profile)
    median = ordered[len(ordered) // 2]

    darkest_value = min(band)
    # Tolerance is relative to this page's own dark-to-median range, so it adapts to a faint scan
    # rather than assuming an absolute intensity that only holds for well-exposed images.
    tolerance = max(1.0, (median - darkest_value) * _DARK_RUN_TOLERANCE)
    threshold = darkest_value + tolerance

    best_start = best_end = min(range(len(band)), key=lambda index: band[index])
    run_start: int | None = None
    for index, value in enumerate(band):
        if value <= threshold:
            if run_start is None:
                run_start = index
            if (index - run_start) > (best_end - best_start):
                best_start, best_end = run_start, index
        else:
            run_start = None

    darkest_column = low + (best_start + best_end) / 2.0

    scale = image.width / columns
    return GutterEstimate(
        gutter_x=int(round(darkest_column * scale)),
        gutter_darkness=darkest_value,
        median_column_darkness=median,
        search_band_px=(int(round(low * scale)), int(round(high * scale))),
    )


@dataclass(frozen=True)
class PageSideWindow:
    """One side of a spread: the window shown to the detector, and the boundary that owns a line.

    `window` is wider than `owns` by the overlap margin. A detected line belongs to this side only
    when its centre-x falls inside `owns` -- see module docstring.
    """

    label: str
    order_index: int
    window: tuple[int, int]
    owns: tuple[int, int]

    def contains_centre(self, centre_x: float) -> bool:
        return self.owns[0] <= centre_x < self.owns[1]


def page_side_windows(width: int, gutter_x: int) -> tuple[PageSideWindow, ...]:
    """The two detector windows for a spread, with their overlap margins applied."""
    # No `max(1, ...)` floor: at GUTTER_OVERLAP_FRACTION == 0 the margin must be genuinely zero, or
    # each window still reaches one pixel into the opposite page -- the exact direction the smoke
    # test measured as catastrophic.
    margin = int(round(width * GUTTER_OVERLAP_FRACTION))
    left_edge = max(0, gutter_x - margin)
    right_edge = min(width, gutter_x + margin)
    return (
        PageSideWindow(label="page_side_left", order_index=0, window=(0, right_edge), owns=(0, gutter_x)),
        PageSideWindow(
            label="page_side_right", order_index=1, window=(left_edge, width), owns=(gutter_x, width)
        ),
    )


def whole_page_window(width: int) -> tuple[PageSideWindow, ...]:
    """The single window used for a page that is not a spread. Same type as the spread case so the
    adapter has one code path, not two."""
    return (
        PageSideWindow(label="page_full", order_index=0, window=(0, width), owns=(0, width)),
    )
