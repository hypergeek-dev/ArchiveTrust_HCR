"""`htr/segmentation/` -- segmentation as an independent pipeline stage
(`docs/htr-domain-design.md` §7).

Created for the first time by Stage 2 of the Swedish Historical HTR Technical Reliability
Screening. Until then this package was referenced by `htr/corpus/models.py`'s `Region`/`TextLine`
docstrings, by `domain/telemetry/events.py`'s `SegmentationRunCompleted`, by `providers/base.py`,
by `providers/transkribus/parsing_models.py` and by `providers/florence2_htr/adapter.py` -- every
one of them a reference to a package that did not exist
(`docs/experiments/technical-reliability-screening/design-audit.md` §4.2).

**Confound, restated here so it is unmissable at the package boundary:** the only real adapter in
this package detects lines with a **Florence-2-family model**, and its crops are fed
byte-identically to both SATRN and Florence-2. One screened method's family controls the other's
input. Accepted and explicitly flagged at Checkpoint 1; see
`florence2_line_detector.py`'s module docstring and `CONFOUND_STATEMENT`.
"""

from archivetrust.htr.segmentation.facade import (
    Florence2LineDetectorFacade,
    LineDetectionWorkerResult,
    real_florence2_line_detector_facade,
)
from archivetrust.htr.segmentation.florence2_line_detector import (
    ADAPTER_ID,
    ADAPTER_VERSION,
    CONFOUND_STATEMENT,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    TASK_PROMPT,
    Florence2LineDetectorAdapter,
    SegmentationConfiguration,
)
from archivetrust.htr.segmentation.models import (
    SEGMENTATION_STAGE,
    DetectedBox,
    LineDetectionFailedError,
    LineDetectionOutcome,
    PageImage,
    RegionDetectionOutcome,
    SegmentationAdapter,
    SegmentationError,
)
from archivetrust.htr.segmentation.service import SegmentationService, SegmentedPage
from archivetrust.htr.segmentation.spread import (
    GUTTER_OVERLAP_FRACTION,
    SPREAD_ASPECT_RATIO_THRESHOLD,
    GutterEstimate,
    PageSideWindow,
    estimate_gutter,
    is_double_page_spread,
    page_side_windows,
    whole_page_window,
)

__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "CONFOUND_STATEMENT",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "GUTTER_OVERLAP_FRACTION",
    "SEGMENTATION_STAGE",
    "SPREAD_ASPECT_RATIO_THRESHOLD",
    "TASK_PROMPT",
    "DetectedBox",
    "Florence2LineDetectorAdapter",
    "Florence2LineDetectorFacade",
    "GutterEstimate",
    "LineDetectionFailedError",
    "LineDetectionOutcome",
    "LineDetectionWorkerResult",
    "PageImage",
    "PageSideWindow",
    "RegionDetectionOutcome",
    "SegmentationAdapter",
    "SegmentationConfiguration",
    "SegmentationError",
    "SegmentationService",
    "SegmentedPage",
    "estimate_gutter",
    "is_double_page_spread",
    "page_side_windows",
    "real_florence2_line_detector_facade",
    "whole_page_window",
]
