"""`TrainingDashboardViewModel`: end-to-end against real fixture run directories built through the
actual training engine (`run_training_session` + a `FakeEpochRunner`) -- never a hand-typed JSON
fixture standing in for what the trainer really writes.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from archivetrust.htr.training.training_identity import TrainingConfiguration, create_or_load_identity
from archivetrust.htr.training.training_session import EpochResult, run_training_session
from archivetrust.presentation.training_dashboard_viewmodel import TrainingDashboardViewModel


class _FakeEpochRunner:
    def __init__(self, val_cers: list[float]):
        self._val_cers = val_cers
        self.calls = 0

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        self.calls += 1
        val_cer = self._val_cers[self.calls - 1]
        ckpt_dir = Path(output_dir) / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
            zf.writestr("config.json", "{}")
        return EpochResult(
            ok=True, val_cer=val_cer, train_cer=val_cer + 0.1, train_loss=50.0, val_loss=55.0,
            duration_seconds=10.0, checkpoint_dir=str(ckpt_dir), best_val_checkpoint_dir=str(ckpt_dir),
            exit_code=0,
        )


def _config() -> TrainingConfiguration:
    return TrainingConfiguration(
        train_manifest_hash="t", val_manifest_hash="v", charlist_hash="c",
        preprocessing_version="byte_identical_from_source_parquet", model_architecture="new10",
        parent_checkpoint_hash="p", learning_rate_policy="constant_0.0001", optimizer="adam",
        augmentation_policy="none",
    )


def _build_fixture_run(training_root: Path, *, run_name: str, val_cers: list[float], parent_checkpoint_dir: Path) -> Path:
    run_dir = training_root / run_name
    run_state_dir = run_dir / "run-state"
    identity, config_hash = create_or_load_identity(
        run_state_dir=run_state_dir, parent_checkpoint="generic-2023-02-15@abc", configuration=_config()
    )
    runner = _FakeEpochRunner(val_cers)
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=run_state_dir / "checkpoint_index.json",
        epoch_runner=runner, train_list_path="train.txt", validation_list_path="val.txt",
        parent_checkpoint_dir=str(parent_checkpoint_dir), run_id=identity.run_id,
        configuration_hash=config_hash, random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=len(val_cers), early_stopping_patience=5,
    )
    return run_dir


@pytest.fixture()
def parent_checkpoint_dir(tmp_path):
    d = tmp_path / "parent"
    d.mkdir()
    (d / "model.keras").write_bytes(b"fake")
    return d


def test_discovers_a_real_run_directory(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5, 0.4], parent_checkpoint_dir=parent_checkpoint_dir)
    vm = TrainingDashboardViewModel(training_root=training_root)
    dirs = vm.discover_run_dirs()
    assert len(dirs) == 1
    assert dirs[0].name == "run-a"


def test_discovers_multiple_run_directories_sorted(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    _build_fixture_run(training_root, run_name="run-b", val_cers=[0.5], parent_checkpoint_dir=parent_checkpoint_dir)
    _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5], parent_checkpoint_dir=parent_checkpoint_dir)
    vm = TrainingDashboardViewModel(training_root=training_root)
    dirs = vm.discover_run_dirs()
    assert [d.name for d in dirs] == ["run-a", "run-b"]


def test_ignores_directories_without_a_training_identity(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    _build_fixture_run(training_root, run_name="real-run", val_cers=[0.5], parent_checkpoint_dir=parent_checkpoint_dir)
    (training_root / "not-a-run").mkdir()
    vm = TrainingDashboardViewModel(training_root=training_root)
    dirs = vm.discover_run_dirs()
    assert [d.name for d in dirs] == ["real-run"]


def test_discover_run_dirs_on_a_missing_training_root_returns_empty(tmp_path):
    vm = TrainingDashboardViewModel(training_root=tmp_path / "does_not_exist")
    assert vm.discover_run_dirs() == ()


def test_run_summary_rows_reflect_real_progress(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5, 0.4, 0.35], parent_checkpoint_dir=parent_checkpoint_dir)
    vm = TrainingDashboardViewModel(training_root=training_root)
    rows = vm.run_summary_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.cumulative_epoch == 3
    assert row.best_val_cer == 0.35
    assert row.best_epoch == 3
    assert row.test_cer is None  # never evaluated this phase


def test_run_detail_includes_status_and_health(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    run_dir = _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5, 0.4], parent_checkpoint_dir=parent_checkpoint_dir)
    vm = TrainingDashboardViewModel(training_root=training_root)
    detail = vm.run_detail(run_dir)
    assert detail.status.cumulative_epoch == 2
    assert detail.health.level in ("healthy", "completed", "warning")
    assert len(detail.validation_history) == 2


def test_val_cer_series_carries_best_epoch_and_patience_markers(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    run_dir = _build_fixture_run(
        training_root, run_name="run-a", val_cers=[0.5, 0.3, 0.35], parent_checkpoint_dir=parent_checkpoint_dir
    )
    vm = TrainingDashboardViewModel(training_root=training_root)
    series = vm.val_cer_series(run_dir)
    assert series.epochs == (1, 2, 3)
    assert series.val_cer == (0.5, 0.3, 0.35)
    assert series.best_epoch == 2
    assert series.best_val_cer == 0.3
    assert series.early_stopping_patience == 5


def test_epoch_metric_series_reads_real_loss_values(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    run_dir = _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5], parent_checkpoint_dir=parent_checkpoint_dir)
    vm = TrainingDashboardViewModel(training_root=training_root)
    series = vm.epoch_metric_series(run_dir, metric="val_loss", label="Val loss")
    assert series.x == (1.0,)
    assert series.y == (55.0,)
    assert series.label == "Val loss"


def test_throughput_series_never_fabricates_a_value_without_a_known_line_count(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    run_dir = _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5], parent_checkpoint_dir=parent_checkpoint_dir)
    vm = TrainingDashboardViewModel(training_root=training_root)
    series = vm.throughput_series(run_dir, lines_per_epoch=None)
    assert series.y == (None,)


def test_throughput_series_computes_a_real_rate_when_line_count_is_known(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    run_dir = _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5], parent_checkpoint_dir=parent_checkpoint_dir)
    vm = TrainingDashboardViewModel(training_root=training_root)
    series = vm.throughput_series(run_dir, lines_per_epoch=1000)
    assert series.y[0] == pytest.approx(100.0)  # 1000 lines / 10.0s


def test_gpu_series_degrade_gracefully_with_no_gpu_samples():
    vm = TrainingDashboardViewModel()
    assert vm.gpu_utilization_series(()).y == ()
    assert vm.vram_series(()).y == ()
    assert vm.gpu_temperature_series(()).y == ()


def test_gpu_series_extract_the_right_nested_field():
    vm = TrainingDashboardViewModel()
    samples = (
        {"gpu": {"utilization_pct": 10.0, "memory_used_mb": 1000.0, "temperature_c": 50.0}, "system": {}},
        {"gpu": {"utilization_pct": 20.0, "memory_used_mb": 2000.0, "temperature_c": 60.0}, "system": {}},
    )
    assert vm.gpu_utilization_series(samples).y == (10.0, 20.0)
    assert vm.vram_series(samples).y == (1000.0, 2000.0)
    assert vm.gpu_temperature_series(samples).y == (50.0, 60.0)


def test_read_new_gpu_samples_reads_a_real_telemetry_file(tmp_path, parent_checkpoint_dir):
    training_root = tmp_path / "training"
    run_dir = _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5], parent_checkpoint_dir=parent_checkpoint_dir)
    telemetry_dir = run_dir / "run-state" / "telemetry"
    telemetry_dir.mkdir(parents=True)
    (telemetry_dir / "gpu_samples.jsonl").write_text(
        json.dumps({"epoch": 1, "gpu": {"utilization_pct": 5.0}, "system": {}}) + "\n", encoding="utf-8"
    )
    vm = TrainingDashboardViewModel(training_root=training_root)
    result = vm.read_new_gpu_samples(run_dir)
    assert len(result.rows) == 1
    assert result.rows[0]["gpu"]["utilization_pct"] == 5.0


def test_loading_an_interrupted_run_does_not_crash(tmp_path, parent_checkpoint_dir):
    """Simulates a real kill-mid-run: `_save_session_state` persists progress after every epoch, but
    `last_stop_reason`/`last_session_ended_at` are only ever set once, at the very end of
    `run_training_session` -- a process genuinely killed mid-session has real epoch progress on disk
    but neither of those two fields. Must load and classify cleanly as `interrupted`, not raise."""
    import time as time_mod

    training_root = tmp_path / "training"
    run_dir = _build_fixture_run(training_root, run_name="run-a", val_cers=[0.5, 0.4], parent_checkpoint_dir=parent_checkpoint_dir)

    session_state_path = run_dir / "run-state" / "session_state.json"
    state_payload = json.loads(session_state_path.read_text(encoding="utf-8"))
    state_payload["last_stop_reason"] = None
    state_payload["last_stopped_mid_epoch"] = None
    state_payload["last_stop_boundary"] = None
    state_payload["last_session_ended_at"] = None
    session_state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    telemetry_dir = run_dir / "run-state" / "telemetry"
    telemetry_dir.mkdir(parents=True)
    stale_time = time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", time_mod.gmtime(time_mod.time() - 600))
    (telemetry_dir / "status.json").write_text(
        json.dumps({"epoch": 2, "last_sample_at": stale_time, "container_alive": True, "gpu": {}, "system": {}}),
        encoding="utf-8",
    )
    vm = TrainingDashboardViewModel(training_root=training_root)
    detail = vm.run_detail(run_dir)
    assert detail.status.status == "interrupted"
    assert detail.health.level == "stalled"


def test_run_summary_rows_never_mixes_val_cer_and_test_cer():
    """Structural check: `RunSummaryRow.best_val_cer` and `.test_cer` are always distinct fields,
    never the same value under two names -- the comparison tab must never blur the two."""
    from archivetrust.presentation.training_dashboard_viewmodel import RunSummaryRow

    assert "best_val_cer" in RunSummaryRow.model_fields
    assert "test_cer" in RunSummaryRow.model_fields
    assert RunSummaryRow.model_fields["best_val_cer"].annotation != RunSummaryRow.model_fields["test_cer"].annotation
