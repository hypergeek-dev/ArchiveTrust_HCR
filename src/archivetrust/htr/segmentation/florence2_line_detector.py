"""`Florence2LineDetectorAdapter` -- the first real `SegmentationAdapter` implementation.

`docs/experiments/technical-reliability-screening/design-audit.md` §4.2 states the gap this closes
plainly: *"`src/archivetrust/htr/segmentation/` does not exist. Not 'exists but is a stub' -- the
package was never created."* The domain entities (`Region`/`TextLine`/`InputCrop`), the telemetry
vocabulary (`REGION_DETECTED`/`TEXT_LINE_DETECTED`/`INPUT_CROP_CREATED`/
`SEGMENTATION_RUN_COMPLETED`) and the segmentation metrics all already existed; only the thing that
looks at an image and finds lines was missing. This is that thing.

## !! CONFOUND -- READ THIS BEFORE READING ANY RESULT PRODUCED WITH THIS ADAPTER !!

**The line detector used here is from Florence-2's own model family.** The crops this adapter
produces are fed byte-identically to *both* SATRN and Florence-2 in the technical reliability
screening. One of the two methods being screened therefore has its own model family in control of
the input the other is judged on.

This is an **accepted, explicitly-flagged confound**, decided by the human supervisor at
Checkpoint 1, not an oversight. It is stated here, in
`docs/experiments/technical-reliability-screening/design-audit.md` §4.4.3 and §9.2, in this
package's `README.md`, in `SegmentationConfiguration.confound_statement` below, and in every
telemetry record and report this stage produces. It must be restated in any output derived from
them. It is **not** neutralized by anything in this file; the only thing that would neutralize it
is a method-neutral detector, which was considered and not chosen.

A concrete consequence to keep in view: if detection is poor, *all* methods score poorly and the
screening measures the detector rather than the recognizers (design audit §9.1, its highest-listed
risk). Segmentation quality here has no ground truth and is assessable only by human plausibility
review.

## Provenance of the checkpoint

`nazounoryuu/florence_base__mixed__page__line_od`, the companion line-detection checkpoint to the
OCR checkpoint `providers/florence2_htr/adapter.py` already runs. Both were located through the
`hoanghapham/vlm-htr` thesis project's linked Gradio Space, whose README names both as the exact
models it loads -- see `providers/florence2_htr/README.md` for that investigation, which is not
repeated here. The real vlm-htr pipeline is two-stage (`<OD>` detection then `<OCR>` recognition);
`providers/florence2_htr/adapter.py` deliberately implements only the recognition half, recording
this checkpoint as *"a real, confirmed-to-exist extension point for a future `SegmentationAdapter`
implementation"*. This module is that implementation, in the package the domain design assigns it
to -- so the boundary that adapter refused to blur stays unblurred: recognition is an
`HtrMethodAdapter`, detection is a `SegmentationAdapter`, and neither imports the other.

Repository revision, checkpoint revision and processor revision are all pinned to resolved commit
shas below, never to a floating `main`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.evidence.models import BoundingBox, Precision
from archivetrust.htr.corpus.models import InputCrop, Region, TextLine
from archivetrust.htr.segmentation.facade import (
    Florence2LineDetectorFacade,
    real_florence2_line_detector_facade,
)
from archivetrust.htr.segmentation.models import (
    EDGE_BAND_FRACTION,
    EDGE_BAND_SUSPECT_MIN_BOXES,
    EDGE_BAND_SUSPECT_RATIO,
    DetectedBox,
    LineDetectionFailedError,
    LineDetectionOutcome,
    PageImage,
    RegionDetectionOutcome,
)
from archivetrust.htr.segmentation.spread import (
    GutterEstimate,
    PageSideWindow,
    estimate_gutter,
    is_double_page_spread,
    page_side_windows,
    whole_page_window,
)

ADAPTER_ID = "florence2_line_od"
ADAPTER_NAME = "Florence-2 line detector (vlm-htr <OD>)"
VENDOR = "Uppsala University / Riksarkivet (hoanghapham/vlm-htr thesis project)"

DEFAULT_MODEL_ID = "nazounoryuu/florence_base__mixed__page__line_od"
"""The fine-tuned line-detection checkpoint. Confirmed public and non-gated (`gated=False`,
`private=False` via `HfApi().model_info`), so no user token is required."""
DEFAULT_MODEL_REVISION = "48a18ecb7e6815c3a7d33dff53133ea87a33053f"
"""The exact commit this stage downloaded and ran real detection against, resolved via
`huggingface_hub.HfApi().model_info(...).sha` -- not a floating `main`."""

DEFAULT_PROCESSOR_MODEL_ID = "microsoft/Florence-2-base-ft"
DEFAULT_PROCESSOR_REVISION = "e0b8f375661041228a6431c950adac1a5c539b98"
"""The same pinned base-model processor revision `providers/florence2_htr/adapter.py` uses
(`refs/pr/6`, resolved to a sha). Deliberately the identical pin: the detection and recognition
checkpoints share a base model, and letting the two stages drift onto different processor revisions
would make their tokenization/preprocessing silently non-comparable."""

VLM_HTR_REPO_REVISION = "ced3b30222770911dcb900c3a1f83a247100d1a3"
"""`github.com/hoanghapham/vlm-htr` main HEAD, cited for provenance; no code is vendored from it."""

ADAPTER_VERSION = "1.0.0"

TASK_PROMPT = "<OD>"
"""Florence-2's object-detection task token. The vlm-htr project fine-tunes line detection onto this
token (`scripts/train/finetune_florence_od.py`), which is why detection is prompted with `<OD>` and
not with a caption-style or grounding-style task."""

DEFAULT_MAX_NEW_TOKENS = 1024
"""Generation budget for one detector window. Each detected line costs ~5 tokens (`line` + four
`<loc_N>`), so 1024 tokens is roughly 200 lines -- comfortably above a page of 17th-century court
record, and a window that hits the budget is reported as `output_truncated`, never silently short."""

CONFOUND_STATEMENT = (
    "The line crops produced by this stage are detected by a Florence-2-family model "
    "(nazounoryuu/florence_base__mixed__page__line_od) and are then fed byte-identically to both "
    "SATRN and Florence-2. One of the two screened methods therefore has its own model family in "
    "control of the input the other is judged on. This is an accepted, explicitly-flagged confound "
    "decided at Checkpoint 1, not an oversight, and must be restated in any output derived from "
    "these crops. It is not neutralized by anything in this stage."
)
"""Carried in `SegmentationConfiguration`, so the confound travels with the configuration into
telemetry and any report, rather than living only in a docstring someone may not read."""


class SegmentationConfiguration(BaseModel):
    """The versioned, recorded configuration one segmentation run was executed under.

    `docs/htr-domain-design.md` §1 lists `SegmentationConfiguration` as an entity referenced by
    `ExperimentVersion`; this is it, owned by the segmentation package per §5's ownership rule.
    Frozen, and carrying `confound_statement` as a *field* rather than as documentation.
    """

    model_config = ConfigDict(frozen=True)

    adapter_id: str = ADAPTER_ID
    adapter_version: str = ADAPTER_VERSION
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    processor_model_id: str = DEFAULT_PROCESSOR_MODEL_ID
    processor_revision: str = DEFAULT_PROCESSOR_REVISION
    task_prompt: str = TASK_PROMPT
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    device_request: str = "auto"
    spread_handling: str = "split_at_gutter_before_detection"
    confound_statement: str = CONFOUND_STATEMENT


class Florence2LineDetectorAdapter:
    """Implements `htr.segmentation.models.SegmentationAdapter` (verified by an `isinstance`
    contract test, as `SatrnAdapter`/`Florence2Adapter` are against `HtrMethodAdapter`)."""

    def __init__(
        self,
        *,
        configuration: SegmentationConfiguration | None = None,
        facade: Florence2LineDetectorFacade | None = None,
    ) -> None:
        self._configuration = configuration or SegmentationConfiguration()
        self._facade = facade if facade is not None else real_florence2_line_detector_facade()
        self._region_evidence: dict[str, RegionDetectionOutcome] = {}
        self._region_windows: dict[str, PageSideWindow] = {}
        self._line_outcomes: dict[str, LineDetectionOutcome] = {}

    @property
    def configuration(self) -> SegmentationConfiguration:
        return self._configuration

    @property
    def name(self) -> str:
        """The `segmentation_adapter_name` recorded on `SegmentationRunCompleted`, pinned to the
        checkpoint revision so a recorded run names the exact detector that produced it."""
        return f"{ADAPTER_ID}@{self._configuration.model_revision}"

    # -- docs/htr-domain-design.md §7's four methods ----------------------------------------------

    def detect_regions(self, page_image: PageImage) -> tuple[Region, ...]:
        """One `Region` per logical page-side: two for a double-page spread, one otherwise.

        This is where the spread decision is applied -- see `spread.py`'s module docstring for the
        decision, its four reasons, and the risk it accepts. The `Region.bounding_box` recorded is
        the **detector input window** (including its overlap margin), because that is the area a
        detector genuinely saw; the narrower ownership boundary that decides which side a line
        belongs to is applied in `detect_lines`.
        """
        image = page_image.open()
        width, height = image.size
        aspect_ratio = width / height if height else 0.0
        spread = is_double_page_spread(width, height)

        gutter: GutterEstimate | None = None
        if spread:
            gutter = estimate_gutter(image)
            windows = page_side_windows(width, gutter.gutter_x)
        else:
            windows = whole_page_window(width)

        regions = tuple(
            Region.create(
                page_id=page_image.page_id,
                bounding_box=BoundingBox(
                    x0=float(window.window[0]),
                    y0=0.0,
                    x1=float(window.window[1]),
                    y1=float(height),
                    # A page-side window is derived from a projection profile and an aspect-ratio
                    # rule, not measured to the pixel -- the same honesty Precision exists to
                    # enforce for detector boxes.
                    precision=Precision.COARSE_ESTIMATE,
                ),
                region_type=window.label,
                order_index=window.order_index,
            )
            for window in windows
        )

        outcome = RegionDetectionOutcome(
            regions=regions,
            spread_detected=spread,
            aspect_ratio=aspect_ratio,
            gutter_x=gutter.gutter_x if gutter else None,
            gutter_evidence=(
                {
                    "gutter_x": gutter.gutter_x,
                    "gutter_darkness": round(gutter.gutter_darkness, 2),
                    "median_column_darkness": round(gutter.median_column_darkness, 2),
                    "gutter_contrast": round(gutter.gutter_contrast, 4),
                    "search_band_px": list(gutter.search_band_px),
                    "naive_midpoint_x": width // 2,
                    "offset_from_midpoint_px": gutter.gutter_x - width // 2,
                }
                if gutter
                else {"reason": "aspect ratio below spread threshold; page not split"}
            ),
        )
        self._region_evidence[page_image.page_id] = outcome
        for region, window in zip(regions, windows, strict=True):
            self._region_windows[region.region_id] = window
        return regions

    def last_region_evidence(self, page_id: str) -> RegionDetectionOutcome | None:
        """The spread/gutter evidence behind `detect_regions`' result for this page. Kept reachable
        rather than returned, so §7's declared signature stays exactly as specified."""
        return self._region_evidence.get(page_id)

    def last_line_outcome(self, region_id: str) -> LineDetectionOutcome | None:
        """The measured cost and discard count of `detect_lines` for this region."""
        return self._line_outcomes.get(region_id)

    def detect_lines(self, region: Region, *, page_image: PageImage) -> tuple[TextLine, ...]:
        """Runs the real detector over one region's window and returns that region's `TextLine`s.

        Boxes come back in *window* coordinates and are translated into *page* coordinates
        immediately, so every geometry this stage stores is in one frame of reference -- the page's.
        A box whose centre lies on the other side of the gutter is discarded and counted (the
        overlap margin's duplicate-suppression, see `spread.py`).

        Raises `LineDetectionFailedError` on a detector failure. A page that genuinely has no lines
        returns an empty tuple; these are different outcomes and are never collapsed.
        """
        window = self._region_windows.get(region.region_id)
        if window is None:
            # Reconstructed from the region's own recorded geometry rather than refusing: a caller
            # replaying stored regions into a fresh adapter is legitimate.
            window = PageSideWindow(
                label=region.region_type or "page_full",
                order_index=region.order_index or 0,
                window=(int(region.bounding_box.x0), int(region.bounding_box.x1)),
                owns=(int(region.bounding_box.x0), int(region.bounding_box.x1)),
            )

        image = page_image.open()
        crop = image.crop((window.window[0], 0, window.window[1], image.height))

        result = self._facade.detect_lines(
            image=crop,
            device_request=self._configuration.device_request,
            model_id=self._configuration.model_id,
            model_revision=self._configuration.model_revision,
            processor_model_id=self._configuration.processor_model_id,
            processor_revision=self._configuration.processor_revision,
            task_prompt=self._configuration.task_prompt,
            max_new_tokens=self._configuration.max_new_tokens,
        )
        if not result.get("ok"):
            raise LineDetectionFailedError(
                f"line detection failed for region {region.region_id} "
                f"({region.region_type}): {result.get('message')}",
                category=str(result.get("category", "unknown_error")),
            )

        offset_x = float(window.window[0])
        boxes = tuple(
            DetectedBox(x0=box[0] + offset_x, y0=box[1], x1=box[2] + offset_x, y1=box[3], label=label)
            for box, label in zip(
                result.get("boxes", ()),
                tuple(result.get("labels", ())) + ("",) * len(result.get("boxes", ())),
                strict=False,
            )
        )

        owned = tuple(box for box in boxes if window.contains_centre(box.centre_x))
        discarded = len(boxes) - len(owned)

        # Reading order within a side: top to bottom, then left to right for boxes that start on the
        # same line. The detector's own emission order is already close to this, but relying on it
        # would make reading order an undocumented property of beam search rather than a stated rule.
        ordered = sorted(owned, key=lambda box: (round(box.centre_y), box.x0))

        lines = tuple(
            TextLine.create(
                region_id=region.region_id,
                bounding_box=BoundingBox(
                    x0=box.x0,
                    y0=box.y0,
                    x1=box.x1,
                    y1=box.y1,
                    # Quantized <loc_N> geometry -- see facade.py's module docstring.
                    precision=Precision.COARSE_ESTIMATE,
                ),
                reading_order_index=index,
            )
            for index, box in enumerate(ordered)
        )

        # Edge-band crowding: the signature of the detector locking onto content at a window
        # boundary rather than the page it was given. See LineDetectionOutcome.detection_suspect.
        window_width = max(1.0, float(window.window[1] - window.window[0]))
        band = window_width * EDGE_BAND_FRACTION
        left_edge = offset_x + band
        right_edge = offset_x + window_width - band
        in_left_band = sum(1 for box in boxes if box.centre_x < left_edge)
        in_right_band = sum(1 for box in boxes if box.centre_x > right_edge)
        crowded = max(in_left_band, in_right_band)
        suspect = (
            len(boxes) >= EDGE_BAND_SUSPECT_MIN_BOXES
            and crowded >= len(boxes) * EDGE_BAND_SUSPECT_RATIO
        )

        self._line_outcomes[region.region_id] = LineDetectionOutcome(
            lines=lines,
            boxes_returned=len(boxes),
            boxes_discarded_wrong_side=discarded,
            boxes_in_edge_band=crowded,
            detection_suspect=suspect,
            raw_decoded=str(result.get("raw_decoded", "")),
            inference_seconds=float(result.get("elapsed_seconds", 0.0)),
            device_used=str(result.get("device_used", "unknown")),
            peak_gpu_memory_mb=result.get("peak_gpu_memory_mb"),
            output_truncated=bool(result.get("output_truncated", False)),
        )
        return lines

    def order_lines(self, lines: tuple[TextLine, ...]) -> tuple[TextLine, ...]:
        """Global reading order across a page's regions.

        Region order first (left page-side before right -- `Region.order_index`, assigned in
        `detect_regions`), then each side's own `reading_order_index`. For a spread this is the only
        correct rule: interleaving the two sides by y-coordinate would produce a reading order that
        alternates between two physically separate leaves.

        `TextLine.reading_order_index` is *within* its region and is left untouched, so this
        function returns a re-sequenced tuple, not renumbered entities.
        """
        region_order = {
            region_id: (window.order_index, region_id)
            for region_id, window in self._region_windows.items()
        }
        return tuple(
            sorted(
                lines,
                key=lambda line: (
                    region_order.get(line.region_id, (0, line.region_id)),
                    line.reading_order_index,
                ),
            )
        )

    def crop_lines(
        self, lines: tuple[TextLine, ...], *, page_image: PageImage, destination: Path | str
    ) -> tuple[InputCrop, ...]:
        """Cuts each line's box from the **original, full-resolution** page and content-addresses it.

        Crops are taken from the page as decoded, never from the downscaled tensor the detector saw:
        the detector's 768x768 input decides *where* a line is, not what the recognizers read. So
        detection resolution bounds box precision only, not recognition input quality.

        `InputCrop.create` computes the hash (`InputCrop.compute_hash`, sha256, `crop_` prefix); no
        second hashing scheme is introduced here, per event-model doc §2.
        """
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        image = page_image.open()
        page_width, page_height = image.size

        crops: list[InputCrop] = []
        for line in lines:
            box = line.bounding_box
            # Clamped to the raster: a quantized box can land a fraction of a pixel outside the page,
            # and Pillow would silently pad such a crop with black, which would change the bytes the
            # recognizers read for reasons that have nothing to do with the document.
            left = max(0, int(round(box.x0)))
            top = max(0, int(round(box.y0)))
            right = min(page_width, int(round(box.x1)))
            bottom = min(page_height, int(round(box.y1)))
            if right <= left or bottom <= top:
                raise LineDetectionFailedError(
                    f"text line {line.text_line_id} has an empty extent after clamping to the page "
                    f"raster ({left},{top},{right},{bottom}) -- refusing to write a zero-area crop",
                    category="malformed_output",
                )
            cut = image.crop((left, top, right, bottom))

            buffer_path = directory / f"{line.text_line_id}.png"
            # PNG, lossless, with the same pinned encoder parameters the RGB-normalization stage
            # uses (`optimize=False`, `compress_level=6`), so a Pillow default change cannot silently
            # alter crop hashes across runs.
            import io  # noqa: PLC0415

            buffer = io.BytesIO()
            cut.save(buffer, format="PNG", optimize=False, compress_level=6)
            image_bytes = buffer.getvalue()
            buffer_path.write_bytes(image_bytes)

            crops.append(
                InputCrop.create(
                    image_bytes=image_bytes,
                    text_line_id=line.text_line_id,
                    storage_path=str(buffer_path),
                    width=cut.width,
                    height=cut.height,
                )
            )
        return tuple(crops)
