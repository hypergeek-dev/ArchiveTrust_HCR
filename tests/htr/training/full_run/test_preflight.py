from __future__ import annotations

import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from archivetrust.htr.training.full_run.monitoring_config import derive_monitoring_config
from archivetrust.htr.training.full_run.pilot_analysis import analyze_pilot_run
from archivetrust.htr.training.full_run.preflight import EXPECTED_PILOT_VAL_LINE_COUNT, run_preflight
from archivetrust.htr.training.training_session import EpochResult
from tests.htr.training.full_run.conftest import build_pilot_fixture


class _FakeSmokeTestRunner:
    def __init__(self, *, ok: bool = True, produces_checkpoint: bool = True):
        self._ok = ok
        self._produces_checkpoint = produces_checkpoint
        self.calls = 0

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        from pathlib import Path

        self.calls += 1
        if not self._ok:
            return EpochResult(ok=False, duration_seconds=1.0, error_message="scripted smoke-test failure")
        if not self._produces_checkpoint:
            return EpochResult(ok=True, duration_seconds=1.0, checkpoint_dir=None)
        ckpt_dir = Path(output_dir) / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
            zf.writestr("config.json", "{}")
        return EpochResult(ok=True, duration_seconds=1.0, checkpoint_dir=str(ckpt_dir))


def _make_checkpoint_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path / "model.keras", "w") as zf:
        zf.writestr("config.json", "{}")
    return path


def _make_manifest(path, line_ids):
    table = pa.table({"line_id": pa.array(line_ids, type=pa.string())})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture()
def base_setup(tmp_path, pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)

    base_model_dir = _make_checkpoint_dir(tmp_path / "base_model")
    parent_checkpoint_dir = base_model_dir
    val_manifest = _make_manifest(tmp_path / "val_manifest.parquet", [f"v{i}" for i in range(EXPECTED_PILOT_VAL_LINE_COUNT)])
    test_manifest = _make_manifest(tmp_path / "test_manifest.parquet", ["t1", "t2"])
    output_dir = tmp_path / "output"
    training_root = tmp_path / "training_root"

    resume_proof_path = tmp_path / "resume-proof.json"
    resume_proof_path.write_text(
        json.dumps({"proof_passed": True, "epoch_continued_not_restarted": True, "global_step_continued_not_restarted": True}),
        encoding="utf-8",
    )

    keras_bytes = (base_model_dir / "model.keras").read_bytes()
    import hashlib

    base_checkpoint_pinned_hash = hashlib.sha256(keras_bytes).hexdigest()

    return {
        "base_model_dir": base_model_dir, "base_checkpoint_pinned_hash": base_checkpoint_pinned_hash,
        "parent_checkpoint_dir": parent_checkpoint_dir,
        "pilot_val_manifest_path": val_manifest, "pilot_test_manifest_path": test_manifest,
        "output_dir": output_dir, "training_root": training_root, "monitoring_config": config,
        "dataset_hash": "realhash123", "recorded_dataset_hash": "realhash123", "random_seed": 42,
        "pilot_resume_proof_path": resume_proof_path,
        "check_docker_daemon": False, "check_container_image": False,
    }


def test_all_checks_pass_with_a_clean_setup_and_successful_smoke_test(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner(ok=True))
    assert report.all_critical_passed, [c for c in report.checks if not c.passed]
    assert report.failed_checks == ()


def test_summary_reports_passed_over_total(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner(ok=True))
    passed, total = report.summary.split("/")
    assert passed == total
    assert int(total) == len(report.checks)


class _MutatingSmokeTestRunner:
    """Simulates the real, once-observed Loghi container bug: a smoke test that overwrites files
    inside `existing_model_dir` in place (as if resaving/converting the checkpoint it loaded)."""

    def __init__(self):
        self.seen_existing_model_dir: str | None = None

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        from pathlib import Path

        self.seen_existing_model_dir = existing_model_dir
        # "Corrupt" the checkpoint the smoke test was actually given -- this must be the staged
        # copy, never the real pristine source.
        (Path(existing_model_dir) / "model.keras").write_bytes(b"MUTATED")
        ckpt_dir = Path(output_dir) / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
            zf.writestr("config.json", "{}")
        return EpochResult(ok=True, duration_seconds=1.0, checkpoint_dir=str(ckpt_dir))


def test_smoke_test_never_mutates_the_real_pristine_base_checkpoint(base_setup):
    """Regression test for a real incident: the smoke test must stage a writable copy of the base
    checkpoint and hand *that* to the runner -- never the real pinned `base_model_dir` itself, which
    a container's own checkpoint-loading/resave behavior could otherwise corrupt in place."""
    import hashlib

    original_bytes = base_setup["base_model_dir"].joinpath("model.keras").read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    runner = _MutatingSmokeTestRunner()
    run_preflight(**base_setup, require_gpu=False, smoke_test_runner=runner)

    # The runner must never have been handed the real base_model_dir path.
    assert runner.seen_existing_model_dir is not None
    from pathlib import Path

    assert Path(runner.seen_existing_model_dir).resolve() != base_setup["base_model_dir"].resolve()

    # And the real pristine source file must be byte-identical to before the smoke test ran.
    current_bytes = base_setup["base_model_dir"].joinpath("model.keras").read_bytes()
    assert hashlib.sha256(current_bytes).hexdigest() == original_hash


def test_base_model_check_fails_when_directory_has_no_keras_file(base_setup, tmp_path):
    empty_dir = tmp_path / "empty_base_model"
    empty_dir.mkdir()
    base_setup["base_model_dir"] = empty_dir
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "base_model_loadable")
    assert check.passed is False
    assert not report.all_critical_passed


def test_base_checkpoint_hash_mismatch_fails_the_check(base_setup):
    base_setup["base_checkpoint_pinned_hash"] = "0" * 64
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "base_checkpoint_hash_matches_pin")
    assert check.passed is False


def test_parent_checkpoint_pilot_rejection_fails_the_check(base_setup):
    pilot_like_dir = base_setup["parent_checkpoint_dir"].parent / "training" / "loghi-swedish-v1" / "epoch_22"
    pilot_like_dir.mkdir(parents=True)
    with zipfile.ZipFile(pilot_like_dir / "model.keras", "w") as zf:
        zf.writestr("config.json", "{}")
    base_setup["parent_checkpoint_dir"] = pilot_like_dir
    base_setup["known_pilot_run_dirs"] = (pilot_like_dir.parent.parent,)
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "parent_not_pilot")
    assert check.passed is False


def test_val_manifest_empty_fails_the_check(base_setup, tmp_path):
    empty_val = _make_manifest(tmp_path / "empty_val.parquet", [])
    base_setup["pilot_val_manifest_path"] = empty_val
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "val_manifest_non_empty")
    assert check.passed is False


def test_val_test_overlap_fails_the_check(base_setup, tmp_path):
    overlapping_test = _make_manifest(tmp_path / "overlap_test.parquet", ["v0", "t9"])
    base_setup["pilot_test_manifest_path"] = overlapping_test
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "val_test_no_overlap")
    assert check.passed is False


def test_no_val_test_overlap_passes(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "val_test_no_overlap")
    assert check.passed is True


def test_expected_line_counts_plausible_fails_on_wrong_count(base_setup, tmp_path):
    wrong_count_val = _make_manifest(tmp_path / "wrong_val.parquet", ["v1", "v2"])
    base_setup["pilot_val_manifest_path"] = wrong_count_val
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "expected_line_counts_plausible")
    assert check.passed is False


def test_disk_space_check_fails_below_the_threshold(base_setup):
    report = run_preflight(
        **base_setup, require_gpu=False, min_free_disk_gb=1e12,  # absurdly high, guaranteed to fail
        smoke_test_runner=_FakeSmokeTestRunner(),
    )
    check = next(c for c in report.checks if c.name == "output_dir_writable")
    assert check.passed is False


def test_dataset_hash_missing_fails_the_check(base_setup):
    base_setup["dataset_hash"] = None
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "dataset_hash_recorded")
    assert check.passed is False


def test_dataset_hash_unstable_fails_the_check(base_setup):
    base_setup["recorded_dataset_hash"] = "differenthash"
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "dataset_hash_stable")
    assert check.passed is False


def test_random_seed_missing_fails_the_check(base_setup):
    base_setup["random_seed"] = None
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "random_seed_explicit")
    assert check.passed is False


def test_no_conflicting_active_run_passes_on_empty_training_root(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "no_conflicting_active_run")
    assert check.passed is True


def test_conflicting_active_run_fails_the_check(base_setup):
    from archivetrust.htr.training.full_run.run_state import create_initial_run_state, mark_running, save_run_state

    training_root = base_setup["training_root"]
    other_run_state_dir = training_root / "other-run" / "run-state"
    state = mark_running(create_initial_run_state(run_id="other", configuration_hash="h"))
    save_run_state(other_run_state_dir, state)
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "no_conflicting_active_run")
    assert check.passed is False


def test_dashboard_can_discover_check_passes(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "dashboard_can_discover_run")
    assert check.passed is True


def test_optimizer_scheduler_serialization_reuses_pilot_resume_proof(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "optimizer_scheduler_state_serialization")
    assert check.passed is True


def test_optimizer_scheduler_serialization_fails_when_proof_did_not_pass(base_setup):
    base_setup["pilot_resume_proof_path"].write_text(json.dumps({"proof_passed": False}), encoding="utf-8")
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "optimizer_scheduler_state_serialization")
    assert check.passed is False


def test_optimizer_scheduler_serialization_omitted_when_no_proof_path_given(base_setup):
    base_setup["pilot_resume_proof_path"] = None
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    names = [c.name for c in report.checks]
    assert "optimizer_scheduler_state_serialization" not in names


def test_monitoring_config_invalid_fails_the_check(base_setup):
    bad_config = base_setup["monitoring_config"].model_copy(update={"shard_line_count": 0})
    base_setup["monitoring_config"] = bad_config
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "monitoring_config_valid")
    assert check.passed is False


def test_smoke_test_failure_fails_both_smoke_and_round_trip_checks(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner(ok=False))
    smoke = next(c for c in report.checks if c.name == "smoke_test_forward_backward")
    round_trip = next(c for c in report.checks if c.name == "checkpoint_save_reload_round_trip")
    assert smoke.passed is False
    assert round_trip.passed is False


def test_smoke_test_never_runs_twice_for_one_preflight_call(base_setup):
    runner = _FakeSmokeTestRunner(ok=True)
    run_preflight(**base_setup, require_gpu=False, smoke_test_runner=runner)
    assert runner.calls == 1


def test_checkpoint_round_trip_fails_when_smoke_test_produces_no_checkpoint(base_setup):
    report = run_preflight(
        **base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner(ok=True, produces_checkpoint=False),
    )
    round_trip = next(c for c in report.checks if c.name == "checkpoint_save_reload_round_trip")
    assert round_trip.passed is False


def test_no_smoke_test_runner_fails_gracefully_not_a_crash(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=None)
    smoke = next(c for c in report.checks if c.name == "smoke_test_forward_backward")
    assert smoke.passed is False
    assert "No smoke-test runner" in smoke.message


def test_run_smoke_test_false_skips_the_smoke_checks_entirely(base_setup):
    report = run_preflight(**base_setup, require_gpu=False, run_smoke_test=False, smoke_test_runner=None)
    names = [c.name for c in report.checks]
    assert "smoke_test_forward_backward" not in names
    assert "checkpoint_save_reload_round_trip" not in names


def test_a_check_that_raises_is_recorded_as_failed_not_crashing_the_whole_preflight(base_setup, monkeypatch):
    import archivetrust.htr.training.full_run.preflight as preflight_mod

    def _broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(preflight_mod, "_check_disk_space", _broken)
    report = run_preflight(**base_setup, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "output_dir_writable")
    assert check.passed is False
    assert "boom" in check.message
