"""`DurableHtrResearchStore`'s new methods for the active-method transition / Loghi integration:
research-phase registration, comparison-group registration, and the Loghi stage/environment/
domain-relationship telemetry recorders. Each follows the existing emit-then-project (or
emit-only-marker) discipline already established by `record_normalization_started/completed/failed`.
"""

from __future__ import annotations

from archivetrust.htr.experiment.models import DomainRelationship, ExperimentComparisonGroup
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.research_status import CURRENT_RESEARCH_PHASE
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink


def _store(tmp_path) -> DurableHtrResearchStore:
    return DurableHtrResearchStore(FileTelemetrySink(tmp_path / "events.jsonl"))


def test_register_research_phase_appends_telemetry_then_projects(tmp_path) -> None:
    store = _store(tmp_path)
    event_id = store.register_research_phase(CURRENT_RESEARCH_PHASE)
    assert event_id
    assert store.research_phases() == (CURRENT_RESEARCH_PHASE,)

    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert [e.kind.value for e in events] == ["MethodResearchStatusChanged"]
    assert events[0].phase == CURRENT_RESEARCH_PHASE


def test_register_research_phase_refuses_duplicate_registration(tmp_path) -> None:
    from archivetrust.htr.research_store import DuplicateRegistrationError

    store = _store(tmp_path)
    store.register_research_phase(CURRENT_RESEARCH_PHASE)
    try:
        store.register_research_phase(CURRENT_RESEARCH_PHASE)
        raise AssertionError("expected DuplicateRegistrationError")
    except DuplicateRegistrationError:
        pass


def test_register_comparison_group(tmp_path) -> None:
    store = _store(tmp_path)
    group = ExperimentComparisonGroup.create(
        name="Lion vs Loghi", experiment_ids=("experiment_1", "experiment_2"),
        created_at="2026-08-01T00:00:00Z",
    )
    store.register_comparison_group(group)
    assert store.comparison_groups() == (group,)


def test_record_loghi_environment_validated_is_a_marker_event(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_loghi_environment_validated(
        method_id="loghi", valid=False, environment_report={"host_os": "Windows"}
    )
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert events[0].kind.value == "LoghiEnvironmentValidated"
    assert events[0].valid is False
    assert events[0].environment_report == {"host_os": "Windows"}


def test_loghi_stage_events_carry_method_run_id_and_stage_result(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_loghi_pipeline_started(
        method_run_id="method_run_1", input_image_ref="/tmp/p.jpg", component_versions={"x": "y"}
    )
    store.record_loghi_stage_started(method_run_id="method_run_1", stage_name="laypa", started_at="t0")
    store.record_loghi_stage_completed(
        method_run_id="method_run_1", stage_result={"stage_name": "laypa", "ok": True}
    )
    store.record_loghi_page_xml_generated(
        method_run_id="method_run_1", source_xml_hash="abc123", page_schema_version="2019-07-15"
    )

    kinds = [e.kind.value for e in FileTelemetrySink(tmp_path / "events.jsonl").all_events()]
    assert kinds == [
        "LoghiPipelineStarted",
        "LoghiStageStarted",
        "LoghiStageCompleted",
        "LoghiPageXmlGenerated",
    ]


def test_loghi_stage_failed_preserves_the_failure_never_swallows_it(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_loghi_stage_failed(
        method_run_id="method_run_1", stage_result={"stage_name": "laypa", "ok": False, "errors": ["boom"]}
    )
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert events[0].kind.value == "LoghiStageFailed"
    assert events[0].stage_result["errors"] == ["boom"]


def test_record_domain_relationship(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_domain_relationship(
        experiment_version_id="experiment_version_1",
        corpus_language="nl",
        method_primary_language_domain="nl",
        domain_relationship=DomainRelationship.IN_DOMAIN,
    )
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert events[0].kind.value == "DomainRelationshipRecorded"
    assert events[0].domain_relationship is DomainRelationship.IN_DOMAIN


def test_no_swedish_lion_execution_event_is_ever_fabricated_by_loghi_recorders(tmp_path) -> None:
    """None of the new Loghi-specific recorder methods may emit anything that looks like a Lion
    execution event -- these methods only ever describe Loghi."""
    store = _store(tmp_path)
    store.record_loghi_environment_validated(method_id="loghi", valid=True, environment_report={})
    store.record_loghi_pipeline_started(method_run_id="m1", input_image_ref="/x", component_versions={})
    for event in FileTelemetrySink(tmp_path / "events.jsonl").all_events():
        assert "swedish_lion" not in str(event.model_dump())


def test_training_session_lifecycle_events_carry_run_and_session_ids(tmp_path) -> None:
    """`record_training_session_started/checkpointed/completed` -- Work Package 14's 3 new kinds,
    following the same emit-only-marker discipline `record_loghi_environment_validated` already uses."""
    store = _store(tmp_path)
    store.record_training_session_started(
        run_id="run_1", session_id="session_1", configuration_hash="hash_1",
        initial_epoch=0, initial_global_step=0, source_checkpoint="generic-2023-02-15",
    )
    store.record_training_session_checkpointed(
        run_id="run_1", session_id="session_1", epoch=1, global_step=100,
        checkpoint_dir="/training/.../epoch_1/checkpoint", checkpoint_kind="latest",
        duration_seconds=900.0, train_cer=0.5, val_cer=0.4,
    )
    store.record_training_session_completed(
        run_id="run_1", session_id="session_1", stop_reason="time_budget_reached",
        final_epoch=1, final_global_step=100, session_training_seconds=900.0,
        cumulative_training_seconds=900.0, latest_checkpoint_dir="/training/.../epoch_1/checkpoint",
    )
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    kinds = [e.kind.value for e in events]
    assert kinds == ["TrainingSessionStarted", "TrainingSessionCheckpointed", "TrainingSessionCompleted"]
    assert all(e.run_id == "run_1" and e.session_id == "session_1" for e in events)


def test_training_session_checkpointed_records_the_real_checkpoint_kind(tmp_path) -> None:
    """Distinguishes `"latest"` / `"best_val"` / `"end_of_session"` -- never collapsed into one
    undifferentiated "checkpoint saved" event, matching `checkpoint_index.py`'s own 3 categories."""
    store = _store(tmp_path)
    for kind in ("latest", "best_val", "end_of_session"):
        store.record_training_session_checkpointed(
            run_id="run_1", session_id="session_1", epoch=1, global_step=100,
            checkpoint_dir=f"/ckpt/{kind}", checkpoint_kind=kind, duration_seconds=900.0,
        )
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert [e.checkpoint_kind for e in events] == ["latest", "best_val", "end_of_session"]


def test_training_session_completed_preserves_an_honest_stop_reason(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_training_session_completed(
        run_id="run_1", session_id="session_1", stop_reason="epoch_would_not_fit",
        final_epoch=3, final_global_step=300, session_training_seconds=1800.0,
        cumulative_training_seconds=1800.0,
    )
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert events[0].stop_reason == "epoch_would_not_fit"


def test_training_session_failed_is_distinct_from_a_clean_completion(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_training_session_failed(
        run_id="run_1", session_id="session_1", error_message="container exited 137 (OOM killed)",
        final_epoch=2, final_global_step=0,
    )
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert events[0].kind.value == "TrainingSessionFailed"
    assert events[0].error_message == "container exited 137 (OOM killed)"


def test_run_warning_recorded_carries_the_real_health_finding(tmp_path) -> None:
    store = _store(tmp_path)
    store.record_run_warning(run_id="run_1", reason="low_disk_space", message="Only 2.0GB free.")
    events = list(FileTelemetrySink(tmp_path / "events.jsonl").all_events())
    assert events[0].kind.value == "RunWarningRecorded"
    assert events[0].reason == "low_disk_space"
    assert events[0].message == "Only 2.0GB free."
