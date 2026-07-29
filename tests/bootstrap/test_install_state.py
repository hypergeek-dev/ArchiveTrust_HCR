from __future__ import annotations

import json

import pytest

from archivetrust.bootstrap.install_state import (
    STATE_SCHEMA_VERSION,
    InstallState,
    InstallStateCorruptionError,
    InstallStateError,
    InstallStateStore,
    Phase,
    PhaseStatus,
)


def test_new_state_has_every_phase_not_started() -> None:
    state = InstallState(deployment_root="d:/tmp/x")
    for phase in Phase:
        record = state.phase_record(phase)
        assert record.status is PhaseStatus.NOT_STARTED
        assert record.attempts == 0


def test_with_phase_result_increments_attempts_every_call() -> None:
    state = InstallState(deployment_root="d:/tmp/x")
    state = state.with_phase_result(Phase.PREFLIGHT, PhaseStatus.RUNNING)
    state = state.with_phase_result(Phase.PREFLIGHT, PhaseStatus.COMPLETED, implemented=True)
    record = state.phase_record(Phase.PREFLIGHT)
    assert record.attempts == 2
    assert record.status is PhaseStatus.COMPLETED
    assert record.implemented is True
    assert record.started_at is not None


def test_failed_result_records_error_classification_and_is_sticky() -> None:
    state = InstallState(deployment_root="d:/tmp/x")
    state = state.with_phase_result(Phase.DOCKER_DESKTOP_DETECTED, PhaseStatus.FAILED, error_classification="missing_prerequisite")
    record = state.phase_record(Phase.DOCKER_DESKTOP_DETECTED)
    assert record.last_error_classification == "missing_prerequisite"
    # a later non-failed result must not erase the last known failure classification
    state = state.with_phase_result(Phase.DOCKER_DESKTOP_DETECTED, PhaseStatus.RUNNING)
    assert state.phase_record(Phase.DOCKER_DESKTOP_DETECTED).last_error_classification == "missing_prerequisite"


def test_overall_blocked_and_reboot_required() -> None:
    state = InstallState(deployment_root="d:/tmp/x")
    assert state.overall_blocked() is False
    assert state.reboot_required() is False
    blocked = state.with_phase_result(Phase.PREREQUISITES, PhaseStatus.BLOCKED)
    assert blocked.overall_blocked() is True
    rebooting = state.with_phase_result(Phase.DOCKER_RUNTIME_AVAILABLE, PhaseStatus.REBOOT_REQUIRED)
    assert rebooting.reboot_required() is True


def test_store_atomic_round_trip(tmp_path) -> None:
    store = InstallStateStore.for_deployment(tmp_path)
    assert not store.exists()
    state = InstallState(deployment_root=str(tmp_path))
    state = state.with_phase_result(Phase.PREFLIGHT, PhaseStatus.COMPLETED, implemented=True)
    store.save(state)
    assert store.exists()

    reloaded = store.load()
    assert reloaded.phase_record(Phase.PREFLIGHT).status is PhaseStatus.COMPLETED
    assert reloaded.schema_version == STATE_SCHEMA_VERSION


def test_load_without_existing_state_and_without_deployment_root_raises(tmp_path) -> None:
    store = InstallStateStore.for_deployment(tmp_path)
    with pytest.raises(InstallStateError):
        store.load()


def test_load_without_existing_state_creates_fresh_state_when_root_given(tmp_path) -> None:
    store = InstallStateStore.for_deployment(tmp_path)
    state = store.load(deployment_root=str(tmp_path))
    assert state.deployment_root == str(tmp_path)
    assert state.phase_record(Phase.PREFLIGHT).status is PhaseStatus.NOT_STARTED


def test_corrupted_json_is_detected_not_silently_replaced(tmp_path) -> None:
    store = InstallStateStore.for_deployment(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(InstallStateCorruptionError):
        store.load(deployment_root=str(tmp_path))


def test_unsupported_schema_version_is_detected(tmp_path) -> None:
    store = InstallStateStore.for_deployment(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_id": "x", "schema_version": 999, "deployment_root": str(tmp_path), "phases": {}}
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InstallStateCorruptionError):
        store.load(deployment_root=str(tmp_path))


def test_resume_reads_back_prior_progress_across_store_instances(tmp_path) -> None:
    store_a = InstallStateStore.for_deployment(tmp_path)
    state = InstallState(deployment_root=str(tmp_path))
    state = state.with_phase_result(Phase.PREFLIGHT, PhaseStatus.COMPLETED, implemented=True)
    state = state.with_phase_result(Phase.PYTHON_ENVIRONMENT, PhaseStatus.RUNNING)
    store_a.save(state)

    store_b = InstallStateStore.for_deployment(tmp_path)
    resumed = store_b.load()
    assert resumed.is_complete(Phase.PREFLIGHT)
    assert resumed.phase_record(Phase.PYTHON_ENVIRONMENT).status is PhaseStatus.RUNNING


def test_repeated_save_is_idempotent_in_shape(tmp_path) -> None:
    store = InstallStateStore.for_deployment(tmp_path)
    state = InstallState(deployment_root=str(tmp_path)).with_phase_result(Phase.PREFLIGHT, PhaseStatus.COMPLETED, implemented=True)
    store.save(state)
    first = store.load()
    store.save(state)
    second = store.load()
    assert first.phases == second.phases


def test_no_secret_or_document_fields_exist_on_phase_record() -> None:
    from archivetrust.bootstrap.install_state import PhaseRecord

    fields = set(PhaseRecord.model_fields.keys())
    forbidden = {"password", "secret", "token", "hmac_key", "document_content", "credentials"}
    assert not (fields & forbidden)
