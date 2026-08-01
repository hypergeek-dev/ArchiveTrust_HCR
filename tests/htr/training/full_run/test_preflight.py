from __future__ import annotations

import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from archivetrust.htr.training.full_run.monitoring_config import derive_monitoring_config
from archivetrust.htr.training.full_run.pilot_analysis import analyze_pilot_run
from archivetrust.htr.training.full_run.preflight import run_preflight
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
    train_manifest = _make_manifest(tmp_path / "shards" / "shard_0.parquet", ["a", "b", "c"])
    val_manifest = _make_manifest(tmp_path / "val_manifest.parquet", ["v1", "v2"])
    output_dir = tmp_path / "output"

    return {
        "base_model_dir": base_model_dir, "parent_checkpoint_dir": parent_checkpoint_dir,
        "train_manifest_paths": (train_manifest,), "val_manifest_path": val_manifest,
        "output_dir": output_dir, "monitoring_config": config,
    }


def test_all_checks_pass_with_a_clean_setup_and_successful_smoke_test(base_setup):
    report = run_preflight(
        **base_setup, dataset_hash="realhash123", require_gpu=False,
        smoke_test_runner=_FakeSmokeTestRunner(ok=True),
    )
    assert report.all_critical_passed
    assert report.failed_checks == ()


def test_base_model_check_fails_when_directory_has_no_keras_file(base_setup, tmp_path):
    empty_dir = tmp_path / "empty_base_model"
    empty_dir.mkdir()
    base_setup["base_model_dir"] = empty_dir
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "base_model_loadable")
    assert check.passed is False
    assert not report.all_critical_passed


def test_parent_checkpoint_pilot_rejection_fails_the_check(base_setup):
    pilot_like_dir = base_setup["parent_checkpoint_dir"].parent / "training" / "loghi-swedish-v1" / "epoch_22"
    pilot_like_dir.mkdir(parents=True)
    with zipfile.ZipFile(pilot_like_dir / "model.keras", "w") as zf:
        zf.writestr("config.json", "{}")
    base_setup["parent_checkpoint_dir"] = pilot_like_dir
    base_setup["known_pilot_run_dirs"] = (base_setup["parent_checkpoint_dir"].parent.parent,)
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "parent_not_pilot")
    assert check.passed is False


def test_val_manifest_empty_fails_the_check(base_setup, tmp_path):
    empty_val = _make_manifest(tmp_path / "empty_val.parquet", [])
    base_setup["val_manifest_path"] = empty_val
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "val_manifest_non_empty")
    assert check.passed is False


def test_train_val_overlap_fails_the_check(base_setup, tmp_path):
    overlapping_train = _make_manifest(tmp_path / "overlap_shard.parquet", ["v1", "x"])
    base_setup["train_manifest_paths"] = (overlapping_train,)
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "train_val_no_overlap")
    assert check.passed is False
    assert "v1" in check.message


def test_no_train_val_overlap_passes(base_setup):
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "train_val_no_overlap")
    assert check.passed is True


def test_disk_space_check_fails_below_the_threshold(base_setup):
    report = run_preflight(
        **base_setup, dataset_hash="h", require_gpu=False, min_free_disk_gb=1e12,  # absurdly high, guaranteed to fail
        smoke_test_runner=_FakeSmokeTestRunner(),
    )
    check = next(c for c in report.checks if c.name == "output_dir_writable")
    assert check.passed is False


def test_dataset_hash_missing_fails_the_check(base_setup):
    report = run_preflight(**base_setup, dataset_hash=None, require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "dataset_hash_recorded")
    assert check.passed is False


def test_monitoring_config_invalid_fails_the_check(base_setup):
    bad_config = base_setup["monitoring_config"].model_copy(update={"shard_line_count": 0})
    base_setup["monitoring_config"] = bad_config
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "monitoring_config_valid")
    assert check.passed is False


def test_smoke_test_failure_fails_both_smoke_and_round_trip_checks(base_setup):
    report = run_preflight(
        **base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner(ok=False),
    )
    smoke = next(c for c in report.checks if c.name == "smoke_test_forward_backward")
    round_trip = next(c for c in report.checks if c.name == "checkpoint_save_reload_round_trip")
    assert smoke.passed is False
    assert round_trip.passed is False


def test_smoke_test_never_runs_twice_for_one_preflight_call(base_setup):
    runner = _FakeSmokeTestRunner(ok=True)
    run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=runner)
    assert runner.calls == 1


def test_checkpoint_round_trip_fails_when_smoke_test_produces_no_checkpoint(base_setup):
    report = run_preflight(
        **base_setup, dataset_hash="h", require_gpu=False,
        smoke_test_runner=_FakeSmokeTestRunner(ok=True, produces_checkpoint=False),
    )
    round_trip = next(c for c in report.checks if c.name == "checkpoint_save_reload_round_trip")
    assert round_trip.passed is False


def test_no_smoke_test_runner_fails_gracefully_not_a_crash(base_setup):
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=None)
    smoke = next(c for c in report.checks if c.name == "smoke_test_forward_backward")
    assert smoke.passed is False
    assert "Docker unreachable" in smoke.message or "No smoke-test runner" in smoke.message


def test_run_smoke_test_false_skips_the_smoke_checks_entirely(base_setup):
    report = run_preflight(
        **base_setup, dataset_hash="h", require_gpu=False, run_smoke_test=False, smoke_test_runner=None,
    )
    names = [c.name for c in report.checks]
    assert "smoke_test_forward_backward" not in names
    assert "checkpoint_save_reload_round_trip" not in names


def test_a_check_that_raises_is_recorded_as_failed_not_crashing_the_whole_preflight(base_setup, monkeypatch):
    import archivetrust.htr.training.full_run.preflight as preflight_mod

    def _broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(preflight_mod, "_check_disk_space", _broken)
    report = run_preflight(**base_setup, dataset_hash="h", require_gpu=False, smoke_test_runner=_FakeSmokeTestRunner())
    check = next(c for c in report.checks if c.name == "output_dir_writable")
    assert check.passed is False
    assert "boom" in check.message
