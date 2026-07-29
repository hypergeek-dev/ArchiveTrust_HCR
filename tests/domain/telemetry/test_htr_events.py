from __future__ import annotations

from archivetrust.domain.telemetry.events import (
    EVENT_TYPE_BY_KIND,
    AdjudicationRecorded,
    CanonicalResultCreated,
    MethodRunCompleted,
    ReviewSubmissionRecorded,
    SegmentationRunCompleted,
    TelemetryEventKind,
    parse_event,
)


def test_segmentation_run_completed_round_trips_via_parse_event():
    event = SegmentationRunCompleted(
        event_id="event_1",
        document_ref="doc-1",
        segmentation_run_id="segmentation_run_1",
        page_id="page_1",
        segmentation_adapter_name="fake-segmenter",
        region_ids=("region_1",),
        text_line_ids=("text_line_1",),
        input_crop_ids=("input_crop_1",),
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event


def test_method_run_completed_registered_for_its_kind():
    assert EVENT_TYPE_BY_KIND[TelemetryEventKind.METHOD_RUN_COMPLETED] is MethodRunCompleted
    event = MethodRunCompleted(
        event_id="event_1",
        document_ref="doc-1",
        method_run_id="method_run_1",
        experiment_run_id="experiment_run_1",
        method_id="method_satrn",
        evidence_id="evidence_1",
        outcome="succeeded",
    )
    assert event.kind == TelemetryEventKind.METHOD_RUN_COMPLETED


def test_review_submission_and_adjudication_events_are_registered():
    assert EVENT_TYPE_BY_KIND[TelemetryEventKind.REVIEW_SUBMISSION_RECORDED] is ReviewSubmissionRecorded
    assert EVENT_TYPE_BY_KIND[TelemetryEventKind.ADJUDICATION_RECORDED] is AdjudicationRecorded


def test_canonical_result_created_round_trips():
    event = CanonicalResultCreated(
        event_id="event_1",
        document_ref="doc-1",
        canonical_result_id="canonical_result_1",
        page_id="page_1",
        strategy="best_single_method",
        strategy_version=1,
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event
