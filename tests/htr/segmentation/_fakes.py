"""Synthetic pages and a fake line detector for the segmentation tests.

Synthetic on purpose, and for the same reason `tests/htr/persistence/_fixtures.py` is: these tests
prove the stage's *logic* -- spread splitting, side ownership, coordinate translation, reading
order, crop hashing, event emission -- none of which needs a real 20 MP archival scan or a real
GPU forward pass to be proven. Real detection over real corpus pages is exercised separately, by
the `real_model`-marked tests in `test_real_line_detection.py`.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from archivetrust.htr.segmentation.facade import LineDetectionWorkerResult
from archivetrust.htr.segmentation.models import PageImage


def single_page_bytes(width: int = 600, height: int = 800) -> bytes:
    """A portrait single page: light paper, a few dark horizontal bars standing in for text."""
    image = Image.new("RGB", (width, height), (238, 232, 214))
    draw = ImageDraw.Draw(image)
    for index in range(6):
        top = 80 + index * 100
        draw.rectangle((60, top, width - 60, top + 40), fill=(40, 35, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def spread_bytes(width: int = 1600, height: int = 1000, gutter_x: int = 810) -> bytes:
    """A landscape double-page spread with a genuinely dark central gutter band.

    `gutter_x` is deliberately *not* the exact midpoint (800), so a test asserting the estimator
    found the gutter cannot pass by accidentally agreeing with `width // 2`.
    """
    image = Image.new("RGB", (width, height), (238, 232, 214))
    draw = ImageDraw.Draw(image)
    for index in range(5):
        top = 100 + index * 150
        draw.rectangle((80, top, gutter_x - 80, top + 50), fill=(40, 35, 30))
        draw.rectangle((gutter_x + 80, top, width - 80, top + 50), fill=(40, 35, 30))
    # The gutter shadow: a dark vertical band, which is what estimate_gutter looks for.
    draw.rectangle((gutter_x - 14, 0, gutter_x + 14, height), fill=(18, 16, 14))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def page_image(page_id: str, image_bytes: bytes) -> PageImage:
    return PageImage(page_id=page_id, image_bytes=image_bytes)


class FakeLineDetectorFacade:
    """A `Florence2LineDetectorFacade` that returns pre-programmed boxes.

    Records every call so a test can assert on *what the detector was actually shown* -- the size of
    each window is the observable difference between splitting a spread and not splitting it.
    """

    def __init__(self, boxes_per_call: tuple[tuple[tuple[float, float, float, float], ...], ...]):
        self._boxes_per_call = boxes_per_call
        self.calls: list[dict] = []

    def detect_lines(self, **kwargs) -> LineDetectionWorkerResult:
        index = len(self.calls)
        self.calls.append({**kwargs, "image_size": kwargs["image"].size})
        boxes = self._boxes_per_call[index] if index < len(self._boxes_per_call) else ()
        return LineDetectionWorkerResult(
            ok=True,
            boxes=boxes,
            labels=tuple("line" for _ in boxes),
            raw_decoded="<fake>",
            output_truncated=False,
            generated_token_count=len(boxes) * 5,
            elapsed_seconds=0.01,
            model_load_seconds=0.0,
            device_used="cpu",
            peak_gpu_memory_mb=None,
            gpu_name=None,
            software_environment={"torch": "fake", "transformers": "fake"},
            model_revision="fake_revision",
        )


class FailingLineDetectorFacade:
    """A facade that reports a real failure outcome, for the failure-is-recorded tests."""

    def __init__(self, category: str = "cuda_oom", message: str = "out of memory") -> None:
        self._category = category
        self._message = message
        self.calls = 0

    def detect_lines(self, **kwargs) -> LineDetectionWorkerResult:
        self.calls += 1
        return LineDetectionWorkerResult(ok=False, category=self._category, message=self._message)
