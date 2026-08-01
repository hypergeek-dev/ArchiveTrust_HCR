from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from archivetrust.htr.training.checkpoint_index import load_index
from archivetrust.htr.training.training_session import (
    EpochResult,
    load_session_state,
    prove_full_state_resume,
    run_training_session,
)


class FakeEpochRunner:
    """Scripted `EpochRunner` -- writes a real, valid `.keras` zip (so `verify_checkpoint` genuinely
    passes) but never touches Docker/TensorFlow. Records every call for assertions."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls: list[dict] = []
        self._fail_after = fail_after

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        self.calls.append(
            {
                "existing_model_dir": existing_model_dir,
                "output_dir": output_dir,
                "epoch_seed": epoch_seed,
            }
        )
        n = len(self.calls)
        if self._fail_after is not None and n > self._fail_after:
            return EpochResult(ok=False, duration_seconds=1.0, error_message="scripted failure")

        from pathlib import Path

        ckpt_dir = Path(output_dir) / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
            zf.writestr("config.json", "{}")
        return EpochResult(
            ok=True,
            train_cer=0.9 - n * 0.05,
            val_cer=0.8 - n * 0.05,
            train_loss=50.0 - n * 2.0,
            val_loss=55.0 - n * 2.0,
            duration_seconds=2.0,
            checkpoint_dir=str(ckpt_dir),
            best_val_checkpoint_dir=str(ckpt_dir),
            exit_code=0,
        )


class ScriptedValCerRunner:
    """Returns a caller-supplied sequence of `val_cer` values, one per call -- lets early-stopping
    tests script a plateau/regression deterministically, unlike `FakeEpochRunner`'s always-improving
    sequence."""

    def __init__(self, val_cers: list[float]) -> None:
        self._val_cers = val_cers
        self.calls: list[dict] = []

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        self.calls.append({"existing_model_dir": existing_model_dir, "output_dir": output_dir})
        n = len(self.calls)
        val_cer = self._val_cers[n - 1]

        ckpt_dir = Path(output_dir) / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
            zf.writestr("config.json", "{}")
        return EpochResult(
            ok=True, val_cer=val_cer, duration_seconds=1.0,
            checkpoint_dir=str(ckpt_dir), best_val_checkpoint_dir=str(ckpt_dir), exit_code=0,
        )


@pytest.fixture()
def run_state_dir(tmp_path):
    return tmp_path / "run_state"


@pytest.fixture()
def checkpoint_index_path(tmp_path):
    return tmp_path / "checkpoint_index.json"


@pytest.fixture()
def parent_checkpoint_dir(tmp_path):
    """A real, on-disk directory standing in for the pinned `generic-2023-02-15` checkpoint --
    `_stage_writable_checkpoint` genuinely copies its contents, so it must genuinely exist."""
    d = tmp_path / "parent_checkpoint"
    d.mkdir()
    (d / "model.keras").write_bytes(b"fake pretrained weights")
    (d / "charlist.txt").write_text("abc", encoding="utf-8")
    return str(d)


def test_session_starts_at_epoch_zero_for_a_fresh_run(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    assert summary.initial_epoch == 0
    assert summary.final_epoch == 1


def test_never_trains_from_random_initialization_first_epoch_uses_real_parent_checkpoint(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    # never handed the raw parent path directly -- staged to a writable copy first (see
    # training_session.py::_stage_writable_checkpoint's docstring for why)
    assert "the_real_parent_checkpoint" not in runner.calls[0]["existing_model_dir"]
    assert "staged_parent_checkpoint" in runner.calls[0]["existing_model_dir"]


def test_global_step_continues_across_a_simulated_new_process(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    session_1 = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=2,
    )
    # a genuinely new call to load_session_state, simulating a fresh process
    reloaded = load_session_state(run_state_dir)
    assert reloaded.cumulative_epoch == session_1.final_epoch

    session_2 = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    assert session_2.initial_epoch == session_1.final_epoch  # continues, does not restart at 0


def test_chains_existing_model_dir_from_the_previous_epochs_checkpoint(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    from pathlib import Path

    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=2,
    )
    expected = str(Path(runner.calls[0]["output_dir"]) / "checkpoint")
    assert runner.calls[1]["existing_model_dir"] == expected


def test_refuses_to_resume_under_a_changed_configuration_hash(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    with pytest.raises(ValueError):
        run_training_session(
            run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
            train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
            run_id="r1", configuration_hash="DIFFERENT_HASH", random_seed=1, max_wall_clock_seconds=1e9,
            stop_requested=lambda: False, max_epochs_this_call=1,
        )


def test_stop_requested_halts_before_the_next_epoch(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    calls_before_stop = []

    def stop_after_one():
        return len(runner.calls) >= 1

    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=stop_after_one,
    )
    assert summary.stop_reason == "stop_requested"
    assert summary.epochs_completed_this_session == 1


def test_time_budget_stop_never_starts_an_epoch_unlikely_to_fit(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir, monkeypatch
):
    """Real wall-clock timing is too noisy for a deterministic assertion here, so `time.monotonic`
    is controlled directly: session start at t=0, first budget check also at t=0 (first epoch always
    runs -- there is no duration estimate yet to reject it with), second check at t=99 against a
    budget of 100 -- 1 second remaining can never fit a 2-second estimated epoch."""
    import archivetrust.htr.training.training_session as ts_mod

    times = iter([0.0, 0.0, 99.0])
    monkeypatch.setattr(ts_mod.time, "monotonic", lambda: next(times))

    runner = FakeEpochRunner()
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=100.0,
        stop_requested=lambda: False,
    )
    assert summary.stop_reason == "epoch_would_not_fit"
    assert summary.epochs_completed_this_session == 1


def test_epoch_failure_stops_the_session_and_is_recorded(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner(fail_after=1)
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False,
    )
    assert summary.stop_reason == "epoch_failed"
    assert summary.epoch_results[-1].ok is False


def test_checkpoint_index_entries_are_never_labeled_resumable_before_verification(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    entries = load_index(checkpoint_index_path)
    assert entries  # at least one entry written
    for entry in entries:
        # every entry this fake produces has a real, valid .keras zip, so verification really passed
        assert entry.resumable is True
        assert entry.verification_status == "verified"


def test_session_produces_latest_best_and_end_of_session_checkpoint_kinds(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=2,
    )
    kinds = {e.checkpoint_kind for e in load_index(checkpoint_index_path)}
    assert {"latest", "best_val", "end_of_session"}.issubset(kinds)


def test_append_only_session_records_never_overwrite_earlier_sessions(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    entries_after_two_sessions = load_index(checkpoint_index_path)
    session_ids = {e.session_id for e in entries_after_two_sessions}
    assert len(session_ids) == 2  # both sessions' entries preserved, neither overwritten


def test_full_state_resume_proof_passes_with_a_fake_runner(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner()
    result = prove_full_state_resume(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1,
    )
    assert result["proof_passed"] is True
    assert result["epoch_continued_not_restarted"] is True
    assert result["global_step_continued_not_restarted"] is True


def test_report_reconstruction_without_rerunning_training(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    """`session_report.py`'s whole point -- state is entirely reconstructible from durable files."""
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=2,
    )
    # a fresh load, no runner, no training re-invoked
    reloaded = load_session_state(run_state_dir)
    assert reloaded.cumulative_epoch == 2
    assert len(reloaded.validation_history) == 2


def test_early_stopping_halts_after_patience_consecutive_non_improving_epochs(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    # improves for 2 epochs (0.5, 0.4), then plateaus for patience=3 epochs in a row (0.45, 0.42, 0.41
    # -- all worse than the 0.4 best) -- the session must stop right after the 3rd non-improving epoch,
    # never running a 6th.
    runner = ScriptedValCerRunner([0.5, 0.4, 0.45, 0.42, 0.41, 0.39])
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, early_stopping_patience=3,
    )
    assert summary.stop_reason == "no_val_cer_improvement"
    assert summary.epochs_completed_this_session == 5  # stopped right after the 3rd plateau epoch
    assert len(runner.calls) == 5  # the would-be-improving 6th epoch (0.39) was never even attempted


def test_early_stopping_resets_the_patience_counter_on_a_genuine_improvement(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    # ep1 0.5 (first val -- improvement, count resets to 0), ep2 0.6 (worse, count=1), ep3 0.3 (a
    # genuine improvement over 0.5 -- count MUST reset to 0 here, not continue accumulating), ep4 0.35
    # (worse than 0.3, count=1), ep5 0.4 (worse than 0.3, count=2 -> patience=2 reached, stop). If the
    # counter were *not* reset at ep3, patience=2 would already have been hit at ep2->ep3 (stopping
    # after only 3 epochs) instead of the correct 5.
    runner = ScriptedValCerRunner([0.5, 0.6, 0.3, 0.35, 0.4, 0.99])
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, early_stopping_patience=2,
    )
    assert summary.stop_reason == "no_val_cer_improvement"
    assert summary.epochs_completed_this_session == 5  # not 3 -- the reset at ep3 genuinely mattered


def test_early_stopping_disabled_by_default_never_stops_on_a_plateau(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    runner = ScriptedValCerRunner([0.5, 0.5, 0.5, 0.5])
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=4,
    )
    assert summary.stop_reason == "target_epochs_reached"  # never early-stopped -- patience wasn't set
    assert summary.epochs_completed_this_session == 4


def test_early_stopping_patience_persists_across_a_simulated_new_process(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    """`epochs_since_improvement` must survive a session boundary -- otherwise a sequence of
    1-epoch-per-call sessions could never accumulate enough non-improving epochs to ever stop."""
    runner = ScriptedValCerRunner([0.5, 0.6, 0.6])
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1, early_stopping_patience=2,
    )
    reloaded = load_session_state(run_state_dir)
    assert reloaded.epochs_since_improvement == 0  # epoch 1 (0.5) was the first val_cer -- an improvement

    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1, early_stopping_patience=2,
    )
    reloaded = load_session_state(run_state_dir)
    assert reloaded.epochs_since_improvement == 1  # epoch 2 (0.6) did not improve on 0.5

    # a genuinely new session -- if the counter had been reset by the new process/session boundary,
    # this 3rd non-improving epoch alone would not be enough to trigger patience=2.
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, early_stopping_patience=2,
    )
    assert summary.stop_reason == "no_val_cer_improvement"
    assert summary.epochs_completed_this_session == 1


def test_a_clean_stop_is_reported_as_after_validation_never_mid_epoch(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    runner = FakeEpochRunner()
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=2,
    )
    assert summary.stop_reason == "target_epochs_reached"
    assert summary.stopped_mid_epoch is False
    assert summary.stop_boundary == "after_validation_at_epoch_boundary"


def test_an_epoch_failure_is_reported_as_mid_epoch(run_state_dir, checkpoint_index_path, parent_checkpoint_dir):
    runner = FakeEpochRunner(fail_after=0)  # fails on the very first epoch attempted
    summary = run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False,
    )
    assert summary.stop_reason == "epoch_failed"
    assert summary.stopped_mid_epoch is True
    assert summary.stop_boundary == "epoch_failed_before_completion"


def test_stop_boundary_is_persisted_and_survives_a_simulated_new_process(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    reloaded = load_session_state(run_state_dir)
    assert reloaded.last_stop_reason == "target_epochs_reached"
    assert reloaded.last_stopped_mid_epoch is False
    assert reloaded.last_stop_boundary == "after_validation_at_epoch_boundary"


def test_train_and_val_loss_flow_into_validation_history_and_checkpoint_entries(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    reloaded = load_session_state(run_state_dir)
    assert reloaded.validation_history[0]["train_loss"] == 48.0  # 50.0 - 1*2.0
    assert reloaded.validation_history[0]["val_loss"] == 53.0  # 55.0 - 1*2.0

    latest_entry = next(e for e in load_index(checkpoint_index_path) if e.checkpoint_kind == "latest")
    assert latest_entry.validation_metrics["train_loss"] == 48.0
    assert latest_entry.validation_metrics["val_loss"] == 53.0


def test_configured_launch_budget_is_persisted_for_the_dashboards_safety_cap_panel(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=3 * 3600.0,
        stop_requested=lambda: False, max_epochs_this_call=1, early_stopping_patience=5,
    )
    reloaded = load_session_state(run_state_dir)
    assert reloaded.configured_wall_clock_hours == 3.0
    assert reloaded.configured_max_epochs_this_call == 1
    assert reloaded.configured_early_stopping_patience == 5


def test_last_session_ended_at_is_set_and_survives_a_simulated_new_process(
    run_state_dir, checkpoint_index_path, parent_checkpoint_dir
):
    runner = FakeEpochRunner()
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=runner,
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=1,
    )
    reloaded = load_session_state(run_state_dir)
    assert reloaded.last_session_ended_at is not None
