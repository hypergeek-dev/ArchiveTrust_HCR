"""`SegmentationService`: durable recording, and the causal chain the event model requires.

`SEGMENTATION_RUN_COMPLETED` was defined by the prior transformation with **zero producers anywhere
in `src/`** (design audit §4.2). These tests are the first that can assert it was actually emitted,
by a real producer, with the correlation/causation wiring
`docs/architecture/htr-event-model.md` §4 specifies.
"""

from __future__ import annotations

import json

import pytest

from archivetrust.application.htr_journal import HtrJournal
from archivetrust.domain.telemetry.events import TelemetryEventKind
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.segmentation import Florence2LineDetectorAdapter, SegmentationService
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink

from tests.htr.segmentation import _fakes

ARCHIVE_OBJECT_REF = "archive_object_segmentation_test"


def _service(tmp_path, facade):
    store = DurableHtrResearchStore(
        FileTelemetrySink(tmp_path / "events.jsonl"), actor_id="segmentation-test"
    )
    service = SegmentationService(
        adapter=Florence2LineDetectorAdapter(facade=facade),
        store=store,
        crop_directory=tmp_path / "crops",
    )
    return store, service


def _event_kinds(path):
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            kind = json.loads(line)["kind"]
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def test_segmenting_a_spread_records_every_entity_and_the_run_summary(tmp_path):
    left_boxes = ((60.0, 100.0, 700.0, 160.0), (60.0, 300.0, 700.0, 360.0))
    right_boxes = ((10.0, 120.0, 600.0, 180.0),)
    facade = _fakes.FakeLineDetectorFacade((left_boxes, right_boxes))
    store, service = _service(tmp_path, facade)

    result = service.segment_and_record(
        _fakes.page_image("page_spread_1", _fakes.spread_bytes(1600, 1000, gutter_x=810)),
        archive_object_ref=ARCHIVE_OBJECT_REF,
        page_number=1,
        correlation_id="segmentation_run_correlation_1",
    )

    assert len(result.regions) == 2
    assert len(result.text_lines) == 3
    assert len(result.input_crops) == 3
    assert result.region_evidence.spread_detected is True

    counts = _event_kinds(tmp_path / "events.jsonl")
    assert counts[TelemetryEventKind.PAGE_REGISTERED.value] == 1
    assert counts[TelemetryEventKind.REGION_DETECTED.value] == 2
    assert counts[TelemetryEventKind.TEXT_LINE_DETECTED.value] == 3
    assert counts[TelemetryEventKind.INPUT_CROP_CREATED.value] == 3
    # The first real emission of this kind anywhere in this system.
    assert counts[TelemetryEventKind.SEGMENTATION_RUN_COMPLETED.value] == 1


def test_segmentation_run_event_names_the_adapter_and_every_produced_id(tmp_path):
    facade = _fakes.FakeLineDetectorFacade((((60.0, 100.0, 500.0, 160.0),),))
    store, service = _service(tmp_path, facade)

    result = service.segment_and_record(
        _fakes.page_image("page_single_1", _fakes.single_page_bytes()),
        archive_object_ref=ARCHIVE_OBJECT_REF,
        page_number=1,
    )

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text("utf-8").splitlines() if line.strip()]
    (run_event,) = [e for e in events if e["kind"] == TelemetryEventKind.SEGMENTATION_RUN_COMPLETED.value]

    assert run_event["segmentation_adapter_name"] == service.adapter.name
    assert run_event["page_id"] == "page_single_1"
    assert tuple(run_event["region_ids"]) == tuple(r.region_id for r in result.regions)
    assert tuple(run_event["text_line_ids"]) == tuple(line.text_line_id for line in result.text_lines)
    assert tuple(run_event["input_crop_ids"]) == tuple(crop.crop_id for crop in result.input_crops)
    # Reference-by-id only: the summary never embeds the entities themselves.
    assert "region" not in run_event and "text_line" not in run_event


def test_correlation_and_causation_form_a_walkable_chain(tmp_path):
    """page -> region -> text line -> input crop, and the run summary caused by the page --
    an explicit DAG, not an inference from shared timestamps (event-model doc §4)."""
    facade = _fakes.FakeLineDetectorFacade((((60.0, 100.0, 500.0, 160.0),),))
    store, service = _service(tmp_path, facade)

    result = service.segment_and_record(
        _fakes.page_image("page_chain", _fakes.single_page_bytes()),
        archive_object_ref=ARCHIVE_OBJECT_REF,
        page_number=1,
        correlation_id="unit_of_work_1",
    )

    events = {
        e["event_id"]: e
        for e in (
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        )
    }
    assert all(e["correlation_id"] == "unit_of_work_1" for e in events.values())

    region_event = next(e for e in events.values() if e["kind"] == TelemetryEventKind.REGION_DETECTED.value)
    line_event = next(e for e in events.values() if e["kind"] == TelemetryEventKind.TEXT_LINE_DETECTED.value)
    crop_event = next(e for e in events.values() if e["kind"] == TelemetryEventKind.INPUT_CROP_CREATED.value)
    run_event = next(
        e for e in events.values() if e["kind"] == TelemetryEventKind.SEGMENTATION_RUN_COMPLETED.value
    )

    assert region_event["causation_id"] == result.page_event_id
    assert line_event["causation_id"] == region_event["event_id"]
    assert crop_event["causation_id"] == line_event["event_id"]
    assert run_event["causation_id"] == result.page_event_id


def test_a_failed_page_still_leaves_what_it_completed_durably_recorded(tmp_path):
    """Failure is recorded, then raised: the page and its regions are already durable when the
    detector fails, so a crashed run never loses what it genuinely finished."""
    from archivetrust.htr.segmentation import LineDetectionFailedError

    store, service = _service(tmp_path, _fakes.FailingLineDetectorFacade())

    with pytest.raises(LineDetectionFailedError):
        service.segment_and_record(
            _fakes.page_image("page_failing", _fakes.single_page_bytes()),
            archive_object_ref=ARCHIVE_OBJECT_REF,
            page_number=1,
        )

    counts = _event_kinds(tmp_path / "events.jsonl")
    assert counts[TelemetryEventKind.PAGE_REGISTERED.value] == 1
    assert counts[TelemetryEventKind.REGION_DETECTED.value] == 1
    # No run summary, because no run completed -- absence here is the honest record.
    assert TelemetryEventKind.SEGMENTATION_RUN_COMPLETED.value not in counts
    assert TelemetryEventKind.INPUT_CROP_CREATED.value not in counts


def test_entities_replay_from_the_event_log_alone(tmp_path):
    """Destroy the in-process store, rebuild from the log -- the durability claim, actually tested."""
    left_boxes = ((60.0, 100.0, 700.0, 160.0),)
    right_boxes = ((10.0, 120.0, 600.0, 180.0),)
    store, service = _service(tmp_path, _fakes.FakeLineDetectorFacade((left_boxes, right_boxes)))

    result = service.segment_and_record(
        _fakes.page_image("page_replay", _fakes.spread_bytes(1600, 1000, gutter_x=810)),
        archive_object_ref=ARCHIVE_OBJECT_REF,
        page_number=7,
    )
    in_process_regions = {r.region_id: r for r in store.regions()}
    in_process_lines = {line.text_line_id: line for line in store.text_lines()}
    in_process_crops = {
        crop.crop_id: crop
        for line in result.text_lines
        for crop in store.crops_for_line(line.text_line_id)
    }
    del store, service

    replayed = HtrJournal().replay(FileTelemetrySink(tmp_path / "events.jsonl").all_events())

    assert {r.region_id: r for r in replayed.regions()} == in_process_regions
    assert {line.text_line_id: line for line in replayed.text_lines()} == in_process_lines
    assert {
        crop.crop_id: crop
        for line in result.text_lines
        for crop in replayed.crops_for_line(line.text_line_id)
    } == in_process_crops
    assert len(in_process_regions) == 2
    assert len(in_process_crops) == len(result.input_crops)


def test_crop_files_are_written_under_the_page_id(tmp_path):
    store, service = _service(tmp_path, _fakes.FakeLineDetectorFacade((((60.0, 100.0, 500.0, 160.0),),)))
    result = service.segment_and_record(
        _fakes.page_image("page_files", _fakes.single_page_bytes()),
        archive_object_ref=ARCHIVE_OBJECT_REF,
        page_number=1,
    )
    (crop,) = result.input_crops
    written = tmp_path / "crops" / "page_files" / f"{result.text_lines[0].text_line_id}.png"
    assert written.exists()
    assert crop.storage_path == str(written)
