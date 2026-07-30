"""Segmentation-stage types: the `SegmentationAdapter` Protocol and its transport objects.

`docs/htr-domain-design.md` §7 specifies this package's interface directly, in code, not merely in
prose. That sketch is followed here -- same four methods, same order, same names, same return
types::

    class SegmentationAdapter(Protocol):
        def detect_regions(self, page_image: PageImage) -> tuple[Region, ...]: ...
        def detect_lines(self, region: Region) -> tuple[TextLine, ...]: ...
        def order_lines(self, lines: tuple[TextLine, ...]) -> tuple[TextLine, ...]: ...
        def crop_lines(self, lines: tuple[TextLine, ...]) -> tuple[InputCrop, ...]: ...

**One deliberate, documented deviation: `page_image` is threaded explicitly into `detect_lines`
and `crop_lines`.** §7's sketch passes only a `Region`/`TextLine`, but `Region` and `TextLine`
(`htr/corpus/models.py`) carry *geometry and ids only* -- deliberately, under the "reference by id,
never embed" rule (Constitution Article 7) that keeps image bytes out of the domain entities. So a
`Region` alone cannot yield the pixels a detector must actually look at. The two available ways to
close that gap are:

1. hold the page's pixels in adapter instance state between `detect_regions` and `detect_lines`, or
2. pass them explicitly.

(1) makes the adapter order-dependent and silently wrong when a caller interleaves pages, and hides
a required input inside mutable state. (2) is chosen: the extra parameter is keyword-only, so the
signature still reads as §7's, and the data dependency is visible at every call site. This is
recorded here rather than left as an unexplained difference between the doc and the code.

**`PageImage` is defined here, not in `htr/corpus/models.py`.** It carries raw bytes, so it is not a
domain entity in the corpus package's sense (nothing there embeds pixels); it is this stage's input
transport, the exact counterpart of `htr/preprocessing/export_package.py::PageImageSelection` for
the normalization stage.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from archivetrust.htr.corpus.models import InputCrop, Region, TextLine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PillowImage

SEGMENTATION_STAGE = "htr.segmentation"
"""Stage name, mirroring `htr/preprocessing/models.py::RGB_NORMALIZATION_STAGE`'s role."""

EDGE_BAND_FRACTION = 0.10
"""Fraction of a detector window's width, at each side, that counts as its "edge band" for
`LineDetectionOutcome.detection_suspect`."""

EDGE_BAND_SUSPECT_RATIO = 0.8
"""Fraction of a window's boxes that must fall in one edge band before its detection is called
suspect."""

EDGE_BAND_SUSPECT_MIN_BOXES = 5
"""Below this many boxes, edge crowding is not evidence of anything -- a window that legitimately
contains three short marginal notes would otherwise be flagged."""


class SegmentationError(RuntimeError):
    """Base for every failure this stage raises rather than silently absorbs."""


class LineDetectionFailedError(SegmentationError):
    """The line detector returned a failure outcome for a page or region.

    A named, catchable error rather than "zero lines detected": a detector that *failed* and a page
    that genuinely *has* no text lines are different facts, and Constitution Article 18 requires
    failure to be a recorded fact rather than an absence indistinguishable from an empty result.
    """

    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class PageImage:
    """One page's pixels plus the corpus identity they belong to -- this stage's input.

    Bytes rather than a path (matching `PageImageSelection`), so a caller may segment a page held in
    a blob store, an acquisition folder, or a test fixture without this package growing a second
    input mechanism. `source_path` is recorded when there was a file, for provenance only, and is
    never used to re-read the pixels.
    """

    page_id: str
    image_bytes: bytes
    source_path: str | None = None

    def open(self) -> "PillowImage":
        """Decodes to an RGB Pillow image. Decoding is forced here (`.load()`) so a truncated file
        fails at a named boundary rather than deep inside a detector call."""
        from PIL import Image  # noqa: PLC0415 -- Pillow is lazy-imported, as elsewhere in htr/

        image = Image.open(io.BytesIO(self.image_bytes))
        image.load()
        return image.convert("RGB")


@dataclass(frozen=True)
class DetectedBox:
    """One raw detector output box in *page* pixel coordinates, before any domain entity exists.

    Kept separate from `TextLine` because a detection is not yet a `TextLine`: it has no
    `region_id`, no reading order, and may be discarded (e.g. a box whose centre falls on the other
    page of a spread). Promoting every raw box straight to a domain entity would make discards
    invisible.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    label: str | None = None

    @property
    def centre_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def centre_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class RegionDetectionOutcome:
    """What `detect_regions` produced, plus the evidence for *why* it produced that shape.

    A caller needs the regions; a reviewer needs to know whether a spread was detected, where the
    gutter estimate landed, and how confident that estimate was. Returning only `tuple[Region, ...]`
    (as §7's sketch does) would discard that, so the adapter exposes this alongside -- §7's method
    still returns exactly `tuple[Region, ...]`, and this record is reachable via
    `last_region_evidence()`.
    """

    regions: tuple[Region, ...]
    spread_detected: bool
    aspect_ratio: float
    gutter_x: int | None
    gutter_evidence: dict[str, Any]


@dataclass(frozen=True)
class LineDetectionOutcome:
    """One real detector invocation over one region: its lines and its measured cost."""

    lines: tuple[TextLine, ...]
    boxes_returned: int
    boxes_discarded_wrong_side: int
    raw_decoded: str
    inference_seconds: float
    device_used: str
    peak_gpu_memory_mb: float | None
    output_truncated: bool
    """True when the detector's generation hit its token budget instead of emitting a stop token --
    a real, silent line-loss mode for a page with many lines, so it is measured, not assumed absent."""
    boxes_in_edge_band: int
    """How many returned boxes have their centre in the leftmost or rightmost
    `EDGE_BAND_FRACTION` of the detector's own window."""
    detection_suspect: bool
    """True when this window's detections are crowded into one edge band -- the signature of the
    detector locking onto content at a window boundary instead of the page it was given.

    Introduced because the Checkpoint 3 smoke test hit exactly that: a right-hand page-side window
    returned 27 boxes, every one of them in the leftmost 4% of the window (they were the *left*
    page's line-endings, pulled in by what was then a 2% overlap margin), and the densely-written
    right leaf yielded zero lines. Centre-based ownership then discarded all 27, so the page's own
    record showed "0 lines on this side" with no indication anything had gone wrong.

    The margin that caused it is now zero (`spread.py::GUTTER_OVERLAP_FRACTION`), but the *failure
    mode* is a property of the detector, not of the margin, so it is measured rather than assumed
    fixed. `docs/.../design-audit.md` §9.1 names silent segmentation failure the highest risk on the
    project; this is the signal that keeps this instance of it from being silent."""


@runtime_checkable
class SegmentationAdapter(Protocol):
    """`docs/htr-domain-design.md` §7's interface. See this module's docstring for the one
    documented deviation (`page_image` threaded explicitly)."""

    def detect_regions(self, page_image: PageImage) -> tuple[Region, ...]: ...

    def detect_lines(self, region: Region, *, page_image: PageImage) -> tuple[TextLine, ...]: ...

    def order_lines(self, lines: tuple[TextLine, ...]) -> tuple[TextLine, ...]: ...

    def crop_lines(
        self, lines: tuple[TextLine, ...], *, page_image: PageImage, destination: Any
    ) -> tuple[InputCrop, ...]: ...
