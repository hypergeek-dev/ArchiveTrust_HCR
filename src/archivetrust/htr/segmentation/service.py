"""Orchestration for the segmentation stage: run the adapter, record every entity it produced.

The counterpart of `htr/preprocessing/normalization_service.py` for this stage, and it exists for
the same reason: the adapter is the pure-ish transform (pixels in, entities out), the durable store
owns event emission, and this module decides the *order* things happen in -- which is where the
guarantees live.

1. register the `Page` (if the caller has not already);
2. `detect_regions` -> register each `Region` (`RegionDetected`), caused by the page registration;
3. `detect_lines` per region -> register each `TextLine` (`TextLineDetected`), caused by its region;
4. `order_lines` across all regions -> the page's global reading order;
5. `crop_lines` -> register each `InputCrop` (`InputCropCreated`), caused by its text line;
6. `record_segmentation_run` -> `SegmentationRunCompleted`, naming all three id sets.

**`SEGMENTATION_RUN_COMPLETED` has never been emitted before.** The design audit §4.2 records the
event kind, the domain entities and the metrics as all pre-existing with *zero producers anywhere in
`src/`* -- the geometry in this system has only ever arrived pre-made, parsed from a hand-authored
PAGE XML fixture. This module is that kind's first real producer.

**Correlation and causation** follow `docs/architecture/htr-event-model.md` §4 exactly:
`correlation_id` is one value for the whole logical unit of work (the caller's run id, scoped via
`store.correlated_to(...)`), and `causation_id` is the `event_id` of the event that directly caused
each one -- so `page -> region -> text line -> input crop -> segmentation run` is a walkable causal
DAG rather than a set of events sharing a timestamp.

**Failure is recorded, then raised.** A `LineDetectionFailedError` from any region propagates after
the regions detected so far are already durably recorded, so a crashed or failed page leaves behind
what it genuinely completed, never a projection entry with no event behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from archivetrust.htr.corpus.models import InputCrop, Page, Region, TextLine
from archivetrust.htr.segmentation.florence2_line_detector import (
    Florence2LineDetectorAdapter,
    SegmentationConfiguration,
)
from archivetrust.htr.segmentation.models import (
    LineDetectionOutcome,
    PageImage,
    RegionDetectionOutcome,
)


@dataclass(frozen=True)
class SegmentedPage:
    """Everything one `segment_and_record` call produced -- the entities, the measured cost, and the
    event ids announcing them, so a caller (or a test) can assert on both the results and the causal
    edges between them. Same shape as `RecordedNormalization`."""

    page: Page
    regions: tuple[Region, ...]
    text_lines: tuple[TextLine, ...]
    """In global reading order (`order_lines`), not region-grouped order."""
    input_crops: tuple[InputCrop, ...]
    region_evidence: RegionDetectionOutcome
    line_outcomes: tuple[LineDetectionOutcome, ...]
    page_event_id: str
    region_event_ids: tuple[str, ...]
    segmentation_run_event_id: str
    segmentation_run_id: str
    configuration: SegmentationConfiguration

    @property
    def total_inference_seconds(self) -> float:
        return sum(outcome.inference_seconds for outcome in self.line_outcomes)

    @property
    def peak_gpu_memory_mb(self) -> float | None:
        observed = [o.peak_gpu_memory_mb for o in self.line_outcomes if o.peak_gpu_memory_mb is not None]
        return max(observed) if observed else None

    @property
    def any_output_truncated(self) -> bool:
        return any(outcome.output_truncated for outcome in self.line_outcomes)


class SegmentationService:
    """Segments pages, persisting crop bytes to disk and provenance to a `DurableHtrResearchStore`.

    `store` is typed loosely (any object exposing the `register_*`/`record_segmentation_run`
    methods) for the same reason `NormalizationService` types its store loosely: a test must be able
    to pass an in-memory-sink store without this module caring, and this module must never be what
    decides how durable the log is.
    """

    def __init__(
        self,
        *,
        adapter: Florence2LineDetectorAdapter,
        store,
        crop_directory: Path | str,
    ) -> None:
        self._adapter = adapter
        self._store = store
        self._crop_directory = Path(crop_directory)

    @property
    def adapter(self) -> Florence2LineDetectorAdapter:
        return self._adapter

    def segment_and_record(
        self,
        page_image: PageImage,
        *,
        archive_object_ref: str,
        page_number: int,
        correlation_id: str | None = None,
        caused_by: str | None = None,
        crop_directory_name: str | None = None,
    ) -> SegmentedPage:
        """Segments one page and records every entity and event it produced.

        `crop_directory_name` overrides the per-page crop subdirectory name, which defaults to the
        `page_id`. A `page_id` is `page_<sha256>` -- 69 characters -- and on Windows a run over a
        60-page sample writes several hundred crops beneath it, which pushed the smoke test's paths
        close enough to `MAX_PATH` that `git worktree add` failed on this repository. A caller may
        therefore pass a short, still-unique per-page name (e.g. a sequential index). It affects the
        *filesystem location only*: the `InputCrop.storage_path` recorded in telemetry is the real
        path written, and the `page_id` remains the identity everywhere it is an identity.
        """
        decoded = page_image.open()
        page = Page(
            page_id=page_image.page_id,
            archive_object_ref=archive_object_ref,
            page_number=page_number,
            width=decoded.width,
            height=decoded.height,
        )
        page_event_id = self._store.register_page(
            page, caused_by=caused_by, correlation_id=correlation_id
        )

        regions = self._adapter.detect_regions(page_image)
        region_event_ids: list[str] = []
        for region in regions:
            region_event_ids.append(
                self._store.register_region(
                    region, caused_by=page_event_id, correlation_id=correlation_id
                )
            )

        all_lines: list[TextLine] = []
        line_event_ids: dict[str, str] = {}
        line_outcomes: list[LineDetectionOutcome] = []
        for region, region_event_id in zip(regions, region_event_ids, strict=True):
            lines = self._adapter.detect_lines(region, page_image=page_image)
            outcome = self._adapter.last_line_outcome(region.region_id)
            if outcome is not None:
                line_outcomes.append(outcome)
            for line in lines:
                line_event_ids[line.text_line_id] = self._store.register_text_line(
                    line, caused_by=region_event_id, correlation_id=correlation_id
                )
            all_lines.extend(lines)

        ordered_lines = self._adapter.order_lines(tuple(all_lines))

        crops = self._adapter.crop_lines(
            ordered_lines,
            page_image=page_image,
            destination=self._crop_directory / (crop_directory_name or page_image.page_id),
        )
        for crop in crops:
            self._store.register_input_crop(
                crop,
                caused_by=line_event_ids.get(crop.text_line_id),
                correlation_id=correlation_id,
            )

        from archivetrust.domain.shared.ids import new_id  # noqa: PLC0415

        segmentation_run_id = new_id("segmentation_run")
        run_event_id = self._store.record_segmentation_run(
            page_id=page.page_id,
            segmentation_adapter_name=self._adapter.name,
            region_ids=tuple(region.region_id for region in regions),
            text_line_ids=tuple(line.text_line_id for line in ordered_lines),
            input_crop_ids=tuple(crop.crop_id for crop in crops),
            caused_by=page_event_id,
            correlation_id=correlation_id,
            segmentation_run_id=segmentation_run_id,
        )

        region_evidence = self._adapter.last_region_evidence(page_image.page_id)
        assert region_evidence is not None  # detect_regions always records it

        return SegmentedPage(
            page=page,
            regions=regions,
            text_lines=ordered_lines,
            input_crops=crops,
            region_evidence=region_evidence,
            line_outcomes=tuple(line_outcomes),
            page_event_id=page_event_id,
            region_event_ids=tuple(region_event_ids),
            segmentation_run_event_id=run_event_id,
            segmentation_run_id=segmentation_run_id,
            configuration=self._adapter.configuration,
        )
