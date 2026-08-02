from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from archivetrust.htr.training.full_run.corpus_sharding import ShardInfo
from archivetrust.htr.training.full_run.monitoring_config import FullRunMonitoringConfig
from archivetrust.htr.training.full_run.orchestrator import run_full_corpus_session
from archivetrust.htr.training.full_run.run_state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_STOPPED,
    create_initial_run_state,
    load_run_state,
    save_run_state,
)
from archivetrust.htr.training.training_session import EpochResult


class ScriptedFullRunEpochRunner:
    """Scripted per-call `EpochResult`s -- writes a real, valid `.keras` zip so `verify_checkpoint`
    genuinely passes, exactly like `test_training_session.py`'s own `FakeEpochRunner`."""

    def __init__(self, results: list[EpochResult]):
        self._results = results
        self.calls: list[dict] = []

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        self.calls.append({
            "train_list_path": train_list_path,
            "validation_list_path": validation_list_path,
            "output_dir": output_dir,
        })
        n = len(self.calls)
        result = self._results[min(n - 1, len(self._results) - 1)]
        if result.ok and result.checkpoint_dir is None:
            ckpt_dir = Path(output_dir) / "checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
                zf.writestr("config.json", "{}")
            result = result.model_copy(update={"checkpoint_dir": str(ckpt_dir), "best_val_checkpoint_dir": str(ckpt_dir)})
        return result


def _ok_result(val_cer, **overrides):
    base = dict(ok=True, val_cer=val_cer, train_cer=(val_cer + 0.1) if val_cer is not None else None, duration_seconds=10.0)
    base.update(overrides)
    return EpochResult(**base)


def _shards(n, line_count=100):
    return tuple(ShardInfo(shard_index=i, lap=0, line_count=line_count, manifest_path=f"shard_{i}.parquet") for i in range(n))


def _lists(n):
    """The container-readable training-list file per shard -- deliberately a DIFFERENT path from
    the shard's Parquet manifest, so a regression that feeds the manifest to the trainer (the real
    Gate 4 failure) shows up as a wrong path rather than passing silently."""
    return tuple(f"shard_{i}_list.txt" for i in range(n))


def _monitoring_config(**overrides):
    base = dict(
        pilot_run_id="r1", derived_at="2026-08-01T00:00:00Z", shard_line_count=100, steps_per_shard=10,
        min_exposure_steps=None, min_exposure_basis="none", meaningful_improvement_threshold=0.001,
        recommended_patience=2, patience_basis="test", max_full_run_epochs=1000,
        warning_thresholds={"loss_explosion_multiplier": 10.0, "loss_explosion_absolute_reference": None,
                             "stall_timeout_seconds": None, "check_nan_or_inf": True},
        assumptions_and_method="test",
    )
    base.update(overrides)
    return FullRunMonitoringConfig(**base)


@pytest.fixture()
def run_dirs(tmp_path):
    run_state_dir = tmp_path / "run_state"
    checkpoint_index_path = run_state_dir / "checkpoint_index.json"
    save_run_state(run_state_dir, create_initial_run_state(run_id="r1", configuration_hash="h1"))
    return run_state_dir, checkpoint_index_path


@pytest.fixture()
def parent_checkpoint_dir(tmp_path):
    """A real, on-disk directory standing in for the pinned base checkpoint --
    `_stage_writable_checkpoint` genuinely copies its contents, so it must genuinely exist."""
    d = tmp_path / "parent_checkpoint"
    d.mkdir()
    (d / "model.keras").write_bytes(b"fake pretrained weights")
    return str(d)


def test_raises_without_a_prior_prepared_run_state(tmp_path, parent_checkpoint_dir):
    with pytest.raises(RuntimeError):
        run_full_corpus_session(
            run_state_dir=tmp_path / "never_prepared", checkpoint_index_path=tmp_path / "idx.json",
            epoch_runner=ScriptedFullRunEpochRunner([_ok_result(0.5)]), shards=_shards(3), shard_train_list_paths=_lists(3),
            validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
            configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
            max_wall_clock_seconds=1e9, stop_requested=lambda: False,
        )


def test_completes_when_all_shards_are_consumed(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5 - i * 0.01) for i in range(3)])
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(3), shard_train_list_paths=_lists(3), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(recommended_patience=100),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "max_full_run_epochs_reached"
    assert summary.completed is True
    assert summary.cumulative_shards_completed == 3

    final_state = load_run_state(run_state_dir)
    assert final_state.status == STATUS_COMPLETED


def test_never_early_stops_before_minimum_exposure(run_dirs, parent_checkpoint_dir):
    """Real requirement: patience alone (reached quickly) must not stop the run before
    `min_exposure_steps` -- val_cer plateaus immediately but the run must keep going."""
    run_state_dir, checkpoint_index_path = run_dirs
    # constant val_cer from epoch 1 onward -- patience=2 would trigger almost immediately if not gated
    runner = ScriptedFullRunEpochRunner([_ok_result(0.3) for _ in range(6)])
    config = _monitoring_config(recommended_patience=2, min_exposure_steps=50, steps_per_shard=10)  # 5 shards' worth
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(6), shard_train_list_paths=_lists(6), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    # must have run at least 5 shards (min exposure) before patience was even allowed to apply
    assert summary.cumulative_shards_completed >= 5


def test_patience_triggers_after_minimum_exposure_and_a_completed_lap(run_dirs, parent_checkpoint_dir):
    """Patience now requires BOTH minimum exposure and one completed corpus lap. Previously a
    single-lap plan let it fire mid-lap; with a 5-shard lap inside a 20-shard plan it must wait for
    the lap boundary and then act."""
    run_state_dir, checkpoint_index_path = run_dirs
    # improves once, then plateaus forever
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] + [_ok_result(0.3)] * 25)
    config = _monitoring_config(recommended_patience=2, min_exposure_steps=10, steps_per_shard=10)
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(20, 5), shard_train_list_paths=tuple(f"s{i}_list.txt" for i in range(20)),
        validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "no_val_cer_improvement"
    assert summary.completed is True
    assert summary.cumulative_shards_completed >= 5, "must not stop before the first lap completes"
    assert summary.cumulative_shards_completed < 20, "must stop early once past the lap gate"
    final_state = load_run_state(run_state_dir)
    assert final_state.status == STATUS_COMPLETED


def test_real_epoch_failure_marks_the_run_failed_not_completed(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5), EpochResult(ok=False, duration_seconds=1.0, error_message="boom")])
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(5), shard_train_list_paths=_lists(5), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "epoch_failed"
    assert summary.completed is False
    assert load_run_state(run_state_dir).status == STATUS_FAILED


def test_nan_val_cer_is_detected_and_marks_the_run_failed(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(float("nan"))])
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(5), shard_train_list_paths=_lists(5), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "nan_or_inf_detected"
    assert summary.completed is False
    assert load_run_state(run_state_dir).status == STATUS_FAILED


def test_infinite_train_loss_is_detected(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5, train_loss=float("inf"))])
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(5), shard_train_list_paths=_lists(5), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "nan_or_inf_detected"


def test_oom_signature_in_stderr_is_detected(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner(
        [_ok_result(0.5, stderr_tail="tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM")]
    )
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(5), shard_train_list_paths=_lists(5), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "likely_oom"
    assert load_run_state(run_state_dir).status == STATUS_FAILED


def test_stalled_data_loading_is_detected_and_reported_not_marked_failed(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5, duration_seconds=9999.0)])
    config = _monitoring_config(warning_thresholds={
        "loss_explosion_multiplier": 10.0, "loss_explosion_absolute_reference": None,
        "stall_timeout_seconds": 100.0, "check_nan_or_inf": True,
    })
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(5), shard_train_list_paths=_lists(5), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "stalled_data_loading"
    assert summary.completed is False
    # detected and reported, but not necessarily an unrecoverable failure -- "stopped", not "failed"
    assert load_run_state(run_state_dir).status == STATUS_STOPPED


def test_missing_validation_result_is_detected(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(None)])
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(5), shard_train_list_paths=_lists(5), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "missing_validation_result"


def test_graceful_stop_request_halts_before_the_next_shard(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5), _ok_result(0.4), _ok_result(0.3)])
    calls_seen = []

    def stop_after_one():
        calls_seen.append(len(runner.calls))
        return len(runner.calls) >= 1

    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(5), shard_train_list_paths=_lists(5), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=stop_after_one,
    )
    assert summary.stop_reason == "stop_requested"
    assert summary.completed is False
    assert load_run_state(run_state_dir).status == STATUS_STOPPED


def test_resuming_continues_from_the_real_persisted_shard_index(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5 - i * 0.01) for i in range(6)])
    config = _monitoring_config(recommended_patience=100)

    first = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(6), shard_train_list_paths=_lists(6), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: len(runner.calls) >= 2,
    )
    assert first.cumulative_shards_completed == 2

    # a genuinely new call -- simulates a fresh process resuming
    second = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(6), shard_train_list_paths=_lists(6), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert second.cumulative_shards_completed == 6  # continued from 2, not restarted from 0
    # confirms the shard used for the 3rd call was shards[2], not shards[0] again
    assert runner.calls[2]["train_list_path"] == "shard_2_list.txt"


def test_trainer_receives_the_list_file_never_the_parquet_manifest(run_dirs, parent_checkpoint_dir):
    """Regression test for the real Gate 4 failure: the orchestrator handed the container
    `shard_00000.parquet`, which `data/manager.py` opens as UTF-8 text and cannot parse, so the epoch
    died in 53s having produced nothing. Every train_list_path must be a list file."""
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] * 3)
    run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(3), shard_train_list_paths=_lists(3), validation_list_path="val_list.txt",
        parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1", configuration_hash="h1", random_seed=1,
        monitoring_config=_monitoring_config(), max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert runner.calls, "no epoch was ever attempted"
    for call in runner.calls:
        assert not call["train_list_path"].endswith(".parquet"), (
            f"trainer was handed a Parquet manifest ({call['train_list_path']}) -- the container "
            "cannot read Parquet; this is the exact defect that failed Gate 4"
        )
        assert call["train_list_path"].endswith("_list.txt")
        assert not call["validation_list_path"].endswith(".parquet")


def test_time_budget_reached_stops_the_call(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] * 10)
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(10), shard_train_list_paths=_lists(10), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(recommended_patience=100),
        max_wall_clock_seconds=0.0001, stop_requested=lambda: False,
    )
    assert summary.stop_reason in ("time_budget_reached", "epoch_would_not_fit")
    assert summary.completed is False


def test_heartbeat_reflects_real_latest_and_best_metrics(run_dirs, parent_checkpoint_dir):
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5), _ok_result(0.3), _ok_result(0.4)])
    run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_shards(3), shard_train_list_paths=_lists(3), validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir, run_id="r1",
        configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(recommended_patience=100),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    final_state = load_run_state(run_state_dir)
    assert final_state.latest_metrics["val_cer"] == 0.4  # the last epoch's real value
    assert final_state.best_metrics["val_cer"] == 0.3  # the real best, not the latest


def _lap_shards(total, per_epoch):
    """A plan with real lap structure, so epoch boundaries actually exist."""
    from archivetrust.htr.training.full_run.corpus_sharding import ShardInfo
    return tuple(
        ShardInfo(shard_index=i, lap=i // per_epoch, line_count=100, manifest_path=f"shard_{i}.parquet")
        for i in range(total)
    )


def test_end_of_epoch_checkpoint_is_recorded_only_at_a_real_epoch_boundary(run_dirs, parent_checkpoint_dir):
    """3 shards per epoch: an end_of_epoch entry must appear after shard 3, not after shards 1 or 2."""
    from archivetrust.htr.training.checkpoint_index import load_index

    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5 - i * 0.01) for i in range(6)])
    run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(6, 3), shard_train_list_paths=tuple(f"shard_{i}_list.txt" for i in range(6)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    eoe = [e for e in load_index(checkpoint_index_path) if e.checkpoint_kind == "end_of_epoch"]
    assert len(eoe) == 2, f"expected one end_of_epoch per completed epoch, got {len(eoe)}"
    assert [e.epoch for e in eoe] == [1, 2], "end_of_epoch entries must carry the EPOCH number, not shard index"
    assert [e.global_step for e in eoe] == [3, 6], "global_step must record shards completed"
    assert all(e.resumable for e in eoe), "epoch boundaries must be verified before being resumable"


def test_no_end_of_epoch_checkpoint_before_the_epoch_is_actually_complete(run_dirs, parent_checkpoint_dir):
    from archivetrust.htr.training.checkpoint_index import load_index

    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] * 2)
    run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(6, 3), shard_train_list_paths=tuple(f"shard_{i}_list.txt" for i in range(6)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False, 
    )
    # only 2 of 3 shards ran (results exhausted -> the third reuses the last result, so cap by shards)
    eoe = [e for e in load_index(checkpoint_index_path) if e.checkpoint_kind == "end_of_epoch"]
    assert all(e.epoch >= 1 for e in eoe)


def test_run_state_records_epoch_position_not_shards_as_epochs(run_dirs, parent_checkpoint_dir):
    from archivetrust.htr.training.full_run.run_state import load_run_state

    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] * 6)
    # Stop after exactly 2 of the 3 shards in epoch 1 -- the runner repeats its last scripted result
    # when exhausted, so the shard count must be bounded by the stop signal, not by the script length.
    run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(6, 3), shard_train_list_paths=tuple(f"shard_{i}_list.txt" for i in range(6)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: len(runner.calls) >= 2,
    )
    st = load_run_state(run_state_dir)
    assert st.shards_per_epoch == 3
    assert st.global_shards_completed == 2
    assert st.epochs_completed == 0, "2 of 3 shards is not an epoch"
    assert st.shards_completed_in_current_epoch == 2
    assert st.epoch_progress == pytest.approx(2 / 3)


def test_early_stopping_cannot_fire_before_one_full_corpus_lap(run_dirs, parent_checkpoint_dir):
    """Operator requirement: patience is informational until a full lap completes. min_exposure is
    satisfied after ~20 shards (it was derived from a 9,999-line pilot), so without this gate the run
    would stop on evidence from a fraction of the corpus."""
    run_state_dir, checkpoint_index_path = run_dirs
    # Improves once then plateaus forever -- patience=2 would trigger almost immediately.
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] + [_ok_result(0.4)] * 20)
    config = _monitoring_config(recommended_patience=2, min_exposure_steps=10, steps_per_shard=10)
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(20, 10), shard_train_list_paths=tuple(f"s{i}_list.txt" for i in range(20)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    # 10 shards per lap: it must reach at least shard 10, not stop at ~3 where patience would bite.
    assert summary.cumulative_shards_completed >= 10, (
        f"stopped after {summary.cumulative_shards_completed} shards -- before completing a lap")


def test_patience_counter_still_accumulates_while_it_is_only_informational(run_dirs, parent_checkpoint_dir):
    """The counter must keep being calculated and displayed -- only its authority to stop is withheld."""
    from archivetrust.htr.training.training_session import load_session_state

    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] + [_ok_result(0.4)] * 8)
    config = _monitoring_config(recommended_patience=2, min_exposure_steps=10, steps_per_shard=10)
    run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(20, 10), shard_train_list_paths=tuple(f"s{i}_list.txt" for i in range(20)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: len(runner.calls) >= 6,
    )
    state = load_session_state(run_state_dir)
    assert state.epochs_since_improvement > 0, "counter must still accumulate for reporting"


def test_stop_at_epoch_boundary_hands_back_at_the_lap_and_stays_resumable(run_dirs, parent_checkpoint_dir):
    """Operator requirement: at the lap boundary, stop at the human decision gate and do not begin
    lap 2. The run must remain resumable, so this is a `stopped` state, never `completed`."""
    from archivetrust.htr.training.full_run.run_state import STATUS_STOPPED, load_run_state

    run_state_dir, checkpoint_index_path = run_dirs
    # Keeps improving, so nothing else would ever stop it before the plan is exhausted.
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5 - i * 0.01) for i in range(30)])
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(20, 5), shard_train_list_paths=tuple(f"s{i}_list.txt" for i in range(20)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
        stop_at_epoch_boundary=True,
    )
    assert summary.cumulative_shards_completed == 5, "must stop at the first lap boundary exactly"
    assert summary.stop_reason == "epoch_boundary_reached"
    assert summary.completed is False, "a lap boundary is a decision gate, not run completion"
    assert load_run_state(run_state_dir).status == STATUS_STOPPED


def test_without_the_flag_a_lap_boundary_does_not_stop_the_run(run_dirs, parent_checkpoint_dir):
    """The gate is opt-in: the default multi-lap plan still rolls straight into the next lap."""
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5 - i * 0.01) for i in range(30)])
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(20, 5), shard_train_list_paths=tuple(f"s{i}_list.txt" for i in range(20)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=_monitoring_config(),
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.cumulative_shards_completed == 20


def test_early_stopping_still_works_after_a_lap_is_complete(run_dirs, parent_checkpoint_dir):
    """The gate delays patience; it must not disable it permanently."""
    run_state_dir, checkpoint_index_path = run_dirs
    runner = ScriptedFullRunEpochRunner([_ok_result(0.5)] + [_ok_result(0.4)] * 30)
    config = _monitoring_config(recommended_patience=2, min_exposure_steps=10, steps_per_shard=10)
    summary = run_full_corpus_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        shards=_lap_shards(30, 5), shard_train_list_paths=tuple(f"s{i}_list.txt" for i in range(30)),
        validation_list_path="val_list.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, monitoring_config=config,
        max_wall_clock_seconds=1e9, stop_requested=lambda: False,
    )
    assert summary.stop_reason == "no_val_cer_improvement"
    assert summary.cumulative_shards_completed >= 5, "must have completed at least one 5-shard lap first"
    assert summary.cumulative_shards_completed < 30, "must have stopped early once past the lap gate"
