"""Fixtures for the reliability-run harness tests.

Synthetic *inputs*, real *machinery*. Every test in `test_reliability_run.py` runs the real
`SegmentationService`, the real `DurableHtrResearchStore`, the real `FileTelemetrySink` with its
real hash-chain sidecar, and the real replay path -- only the two things that need a GPU (the line
detector's forward pass and the recognizers) are stood in for, and even those are stood in for by
objects that return the same `RecognitionResult` envelope the real adapters return.

That split is deliberate. What these tests must prove is that completion is decided from the durable
log, that an interrupted call is never marked complete, and that the two recognizers resume
independently -- none of which is a statement about model weights. The claim that this works against
real subprocess and GPU behaviour is proven separately and literally by
`scripts/reliability_fixture_demo.py`, which runs real SATRN and real Florence-2.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.screening.run_configuration import (
    FLORENCE2_METHOD_ID,
    RECOGNITION_METHOD_IDS,
    RELIABILITY_RUN_SCHEMA_VERSION,
    SAMPLING_VERSION,
    SATRN_METHOD_ID,
    PageSelection,
    ResolvedConfiguration,
)
from archivetrust.htr.screening.runner import ReliabilityRunner
from archivetrust.htr.segmentation import Florence2LineDetectorAdapter, SegmentationService
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.providers.htr_adapter import RecognitionResult
from tests.htr.segmentation._fakes import FakeLineDetectorFacade


def write_page(path, *, width: int = 600, height: int = 800, seed: int = 0) -> bytes:
    """A synthetic single page with `seed`-dependent pixels, so two pages differ by content."""
    image = Image.new("RGB", (width, height), (238, 232, 214))
    draw = ImageDraw.Draw(image)
    for index in range(6):
        top = 80 + index * 100
        draw.rectangle((60 + seed, top, width - 60, top + 40), fill=(40, 35, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def build_configuration(
    repo_root,
    *,
    page_count: int = 2,
    crops_per_page: int = 2,
    model_revisions: dict[str, str] | None = None,
    reliability_thresholds_hash: str = "reliability_thresholds_test",
    sampling_version: str = SAMPLING_VERSION,
) -> ResolvedConfiguration:
    """A real `ResolvedConfiguration` over synthetic page files written under `repo_root`."""
    import hashlib

    pages = []
    for index in range(page_count):
        relative = f"dataset-rgb/synthetic/page_{index}.png"
        payload = write_page(repo_root / relative, seed=index * 7)
        pages.append(
            PageSelection(
                page_id=f"page_{hashlib.sha256(payload).hexdigest()}",
                document_id=f"document_synthetic_{index}",
                page_number=index + 1,
                relative_path=relative,
                content_hash=f"page_image_{hashlib.sha256(payload).hexdigest()}",
                automatic_category="Ordinary",
                width=600,
                height=800,
            )
        )
    return ResolvedConfiguration(
        schema_version=RELIABILITY_RUN_SCHEMA_VERSION,
        profile="fixture",
        dataset_id="dataset_test",
        dataset_version_id="dataset_version_test",
        dataset_root_relative_path="dataset-rgb",
        corpus_digest_sha256="0" * 64,
        corpus_page_count=page_count,
        sampling_version=sampling_version,
        sampling_seed=20260730,
        pages=tuple(pages),
        crops_per_page=crops_per_page,
        crop_selection_rule="evenly-spaced-over-detected-lines/v1",
        method_ids=RECOGNITION_METHOD_IDS,
        model_revisions=model_revisions or {SATRN_METHOD_ID: "satrn-r1", FLORENCE2_METHOD_ID: "flo-r1"},
        segmentation_adapter_name="fake-detector@r1",
        segmentation_configuration_hash="segmentation_config_test",
        preprocessing_version="1.0.0",
        preprocessing_configuration_hash="rgb_normalization_config_test",
        reliability_heuristics_version="1.0.0",
        reliability_thresholds_hash=reliability_thresholds_hash,
    )


class FakeRecognizer:
    """Returns a deterministic `RecognitionResult` per crop, and records every call.

    `fail_on` makes a call return a real *failure* envelope (`text=None` with a category), which is
    a recorded outcome. `raise_on` makes a call raise, which simulates a genuine interruption
    mid-call -- the case where nothing may be appended.
    """

    def __init__(
        self,
        method_id: str,
        *,
        fail_on: set[int] | None = None,
        raise_on: set[int] | None = None,
        exception: type[BaseException] = KeyboardInterrupt,
    ) -> None:
        self.method_id = method_id
        self.calls: list[str] = []
        self._fail_on = fail_on or set()
        self._raise_on = raise_on or set()
        self._exception = exception

    def recognize(self, recognition_input) -> RecognitionResult:
        index = len(self.calls)
        self.calls.append(recognition_input.input_crop_id)
        if index in self._raise_on:
            raise self._exception(f"{self.method_id} interrupted on call {index}")
        if index in self._fail_on:
            return RecognitionResult(
                text=None,
                raw_response={"category": "worker_error", "message": "synthetic failure"},
                model_revision=f"{self.method_id}-rev",
            )
        return RecognitionResult(
            text=f"{self.method_id} transcript {index} of {recognition_input.input_crop_id[-12:]}",
            confidence=0.5,
            raw_response={"device_used": "cpu", "raw_decoded": f"<raw {index}>"},
            execution_time_ms=1.0,
            model_revision=f"{self.method_id}-rev",
        )


def detector_facade(pages: int, lines_per_page: int = 4) -> FakeLineDetectorFacade:
    """Boxes for `pages` single-page detections, `lines_per_page` each."""
    boxes = tuple(
        tuple(
            (60.0, 80.0 + index * 100, 540.0, 120.0 + index * 100) for index in range(lines_per_page)
        )
        for _ in range(pages)
    )
    return FakeLineDetectorFacade(boxes)


def make_runner(
    *,
    repo_root,
    paths,
    configuration,
    adapters,
    facade,
    observer=None,
):
    """Wires a runner over a real store and a real segmentation service."""
    paths.directory.mkdir(parents=True, exist_ok=True)
    sink = FileTelemetrySink(paths.events, blob_dir=paths.blobs)
    store = DurableHtrResearchStore.open(sink, actor_id="reliability-test-harness")
    detector = Florence2LineDetectorAdapter(facade=facade)
    segmentation = SegmentationService(
        adapter=detector, store=store, crop_directory=paths.crops
    )
    runner = ReliabilityRunner(
        repo_root=repo_root,
        paths=paths,
        configuration=configuration,
        store=store,
        segmentation_service=segmentation,
        adapters=adapters,
        observer=observer,
    )
    return runner, store, sink
