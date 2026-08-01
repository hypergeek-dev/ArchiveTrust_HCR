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


def test_the_nine_active_method_transition_kinds_are_registered():
    """Active-method transition / Loghi integration -- every new kind resolves to its own class, no
    kind left producerless-and-unregistered."""
    from archivetrust.domain.telemetry.events import (
        CrossDomainComparisonCreated,
        DomainRelationshipRecorded,
        LoghiEnvironmentValidated,
        LoghiPageXmlGenerated,
        LoghiPipelineStarted,
        LoghiStageCompleted,
        LoghiStageFailed,
        LoghiStageStarted,
        MethodResearchStatusChanged,
    )

    expected = {
        TelemetryEventKind.METHOD_RESEARCH_STATUS_CHANGED: MethodResearchStatusChanged,
        TelemetryEventKind.LOGHI_ENVIRONMENT_VALIDATED: LoghiEnvironmentValidated,
        TelemetryEventKind.LOGHI_PIPELINE_STARTED: LoghiPipelineStarted,
        TelemetryEventKind.LOGHI_STAGE_STARTED: LoghiStageStarted,
        TelemetryEventKind.LOGHI_STAGE_COMPLETED: LoghiStageCompleted,
        TelemetryEventKind.LOGHI_STAGE_FAILED: LoghiStageFailed,
        TelemetryEventKind.LOGHI_PAGE_XML_GENERATED: LoghiPageXmlGenerated,
        TelemetryEventKind.DOMAIN_RELATIONSHIP_RECORDED: DomainRelationshipRecorded,
        TelemetryEventKind.CROSS_DOMAIN_COMPARISON_CREATED: CrossDomainComparisonCreated,
    }
    for kind, cls in expected.items():
        assert EVENT_TYPE_BY_KIND[kind] is cls


def test_every_telemetry_event_kind_has_exactly_one_registered_class():
    """The closed-vocabulary discipline this module's docstring insists on: every enum member
    resolves to exactly one event class, no kind orphaned."""
    assert set(EVENT_TYPE_BY_KIND.keys()) == set(TelemetryEventKind)


def test_loghi_stage_failed_round_trips_via_parse_event():
    from archivetrust.domain.telemetry.events import LoghiStageFailed

    event = LoghiStageFailed(
        event_id="event_1",
        document_ref="htr:research",
        method_run_id="method_run_1",
        stage_result={"stage_name": "laypa", "ok": False, "errors": ["boom"]},
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event


def test_training_session_failed_round_trips_via_parse_event():
    from archivetrust.domain.telemetry.events import TrainingSessionFailed

    event = TrainingSessionFailed(
        event_id="event_1",
        document_ref="htr:research",
        run_id="run_1",
        session_id="session_1",
        error_message="container exited 137 (OOM killed)",
        final_epoch=2,
        final_global_step=0,
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event


def test_run_warning_recorded_round_trips_via_parse_event():
    from archivetrust.domain.telemetry.events import RunWarningRecorded

    event = RunWarningRecorded(
        event_id="event_1",
        document_ref="htr:research",
        run_id="run_1",
        reason="low_disk_space",
        message="Only 2.0GB free at the run's output location.",
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event


def test_the_three_training_session_kinds_are_registered():
    from archivetrust.domain.telemetry.events import (
        TrainingSessionCheckpointed,
        TrainingSessionCompleted,
        TrainingSessionStarted,
    )

    expected = {
        TelemetryEventKind.TRAINING_SESSION_STARTED: TrainingSessionStarted,
        TelemetryEventKind.TRAINING_SESSION_CHECKPOINTED: TrainingSessionCheckpointed,
        TelemetryEventKind.TRAINING_SESSION_COMPLETED: TrainingSessionCompleted,
    }
    for kind, cls in expected.items():
        assert EVENT_TYPE_BY_KIND[kind] is cls


def test_training_session_started_round_trips_via_parse_event():
    from archivetrust.domain.telemetry.events import TrainingSessionStarted

    event = TrainingSessionStarted(
        event_id="event_1",
        document_ref="htr:research",
        run_id="run_1",
        session_id="session_1",
        configuration_hash="hash_1",
        initial_epoch=0,
        initial_global_step=0,
        source_checkpoint="generic-2023-02-15",
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event


def test_training_session_checkpointed_round_trips_via_parse_event():
    from archivetrust.domain.telemetry.events import TrainingSessionCheckpointed

    event = TrainingSessionCheckpointed(
        event_id="event_1",
        document_ref="htr:research",
        run_id="run_1",
        session_id="session_1",
        epoch=1,
        global_step=100,
        checkpoint_dir="/training/epoch_1/checkpoint",
        checkpoint_kind="best_val",
        train_cer=0.5,
        val_cer=0.4,
        duration_seconds=900.0,
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event


def test_training_session_completed_round_trips_via_parse_event():
    from archivetrust.domain.telemetry.events import TrainingSessionCompleted

    event = TrainingSessionCompleted(
        event_id="event_1",
        document_ref="htr:research",
        run_id="run_1",
        session_id="session_1",
        stop_reason="time_budget_reached",
        final_epoch=3,
        final_global_step=300,
        session_training_seconds=1800.0,
        cumulative_training_seconds=1800.0,
        latest_checkpoint_dir="/training/epoch_3/checkpoint",
        best_checkpoint_dir="/training/epoch_2/checkpoint",
    )
    restored = parse_event(event.model_dump(mode="json"))
    assert restored == event
