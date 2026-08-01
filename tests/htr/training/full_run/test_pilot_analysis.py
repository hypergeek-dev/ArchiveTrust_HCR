from __future__ import annotations

import pytest

from archivetrust.htr.training.full_run.pilot_analysis import (
    _estimate_plateau_epoch,
    _max_near_flat_streak,
    _meaningful_improvement_threshold,
    _val_cer_deltas,
    analyze_pilot_run,
    write_pilot_analysis,
)
from tests.htr.training.full_run.conftest import build_pilot_fixture


def test_analyzes_a_real_synthetic_pilot_directory(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35, 0.34, 0.335])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.pilot_epoch_count == 5
    assert analysis.batch_size == 16
    assert analysis.steps_per_pilot_epoch == 625  # ceil(9999/16)
    assert analysis.best_pilot_epoch == 5
    assert analysis.best_pilot_val_cer == 0.335
    assert analysis.last_checkpoint_epoch == 5


def test_val_cer_deltas_skips_pairs_missing_either_value():
    obs = [{"epoch": 1, "val_cer": 0.5}, {"epoch": 2, "val_cer": None}, {"epoch": 3, "val_cer": 0.4}]
    deltas = _val_cer_deltas(obs)
    assert deltas == []  # neither pair has both values present


def test_val_cer_deltas_computes_real_differences():
    obs = [{"epoch": 1, "val_cer": 0.5}, {"epoch": 2, "val_cer": 0.4}, {"epoch": 3, "val_cer": 0.35}]
    deltas = _val_cer_deltas(obs)
    assert [e for e, _ in deltas] == [2, 3]
    assert deltas[0][1] == pytest.approx(-0.1)
    assert deltas[1][1] == pytest.approx(-0.05)


def test_meaningful_improvement_threshold_is_the_trailing_median():
    deltas = [(2, -0.1), (3, -0.05), (4, -0.02), (5, -0.01)]
    threshold = _meaningful_improvement_threshold(deltas)
    assert threshold == 0.035  # median of [0.1, 0.05, 0.02, 0.01]


def test_meaningful_improvement_threshold_is_none_with_no_deltas():
    assert _meaningful_improvement_threshold([]) is None


def test_plateau_estimation_finds_a_real_observed_stable_window():
    # a clean, real plateau: deltas shrink then stabilize
    deltas = [(2, -0.05), (3, -0.03), (4, -0.001), (5, -0.0008), (6, -0.0009)]
    threshold = _meaningful_improvement_threshold(deltas)
    epoch, extrapolated, note = _estimate_plateau_epoch(deltas, threshold)
    assert extrapolated is False
    assert epoch is not None
    assert "Real, observed plateau" in note


def test_plateau_estimation_extrapolates_when_never_observed():
    """Mirrors this project's own real pilot: still improving at the last recorded epoch, no
    observed stable window -- must extrapolate, clearly labeled."""
    deltas = [(2, -0.02), (3, -0.015), (4, -0.011), (5, -0.009), (6, -0.007), (7, -0.006)]
    threshold = 0.0005  # deliberately far below anything observed -- forces extrapolation
    epoch, extrapolated, note = _estimate_plateau_epoch(deltas, threshold)
    assert extrapolated is True
    assert epoch is not None
    assert epoch > 7  # projected past the last real observation
    assert "extrapolat" in note.lower()


def test_plateau_estimation_refuses_to_extrapolate_a_non_decaying_trend():
    deltas = [(2, -0.01), (3, -0.02), (4, -0.03), (5, -0.05)]  # getting *worse* each epoch
    epoch, extrapolated, note = _estimate_plateau_epoch(deltas, 0.001)
    assert epoch is None
    assert "non-decaying" in note or "not shown any sign" in note


def test_plateau_estimation_none_with_too_few_deltas():
    epoch, extrapolated, note = _estimate_plateau_epoch([(2, -0.01)], 0.001)
    assert epoch is None
    assert extrapolated is False


def test_max_near_flat_streak_counts_consecutive_small_deltas():
    # mirrors the real pilot's epoch 15 (near-flat) then epoch 16 (recovers)
    deltas = [(2, -0.01), (3, -0.001), (4, -0.0009), (5, -0.0008), (6, -0.02)]
    threshold = 0.0015
    streak = _max_near_flat_streak(deltas, threshold)
    assert streak == 3  # epochs 3, 4, 5


def test_max_near_flat_streak_is_zero_with_no_threshold():
    assert _max_near_flat_streak([(2, -0.01)], None) == 0


def test_train_val_divergence_uses_only_epochs_with_both_losses_present(pilot_fixture_dir):
    build_pilot_fixture(
        pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35],
        train_losses=[None, 30.0, 28.0], val_losses=[None, 32.0, 29.0],
    )
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.train_val_divergence["sample_epoch_count"] == 2
    assert analysis.train_val_divergence["epochs_used"] == [2, 3]


def test_train_val_divergence_reports_zero_samples_when_no_losses_recorded(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.train_val_divergence["sample_epoch_count"] == 0


def test_negative_shrinking_gap_is_classified_narrowing_not_widening(pilot_fixture_dir):
    """Real bug this module's own real-pilot sanity check caught: a negative gap shrinking in
    magnitude (e.g. -4.0 -> -2.0) means train/val are converging, not diverging -- must compare by
    |gap|, never the signed value."""
    build_pilot_fixture(
        pilot_fixture_dir, val_cers=[0.5, 0.45, 0.4, 0.38],
        train_losses=[30.0, 29.0, 28.0, 27.0], val_losses=[26.0, 26.5, 26.8, 27.2],
        # gaps: -4.0, -2.5, -1.2, +0.2 -- shrinking in magnitude
    )
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.train_val_divergence["trend"] == "narrowing"


def test_checkpoint_sizes_are_real_file_sizes_not_guessed(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.checkpoint_size_bytes["latest"] is not None
    assert analysis.checkpoint_size_bytes["latest"] > 0


def test_gpu_memory_high_water_is_none_when_no_telemetry_recorded(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.gpu_memory_high_water_mb is None


def test_gpu_memory_high_water_reads_real_samples_when_present(pilot_fixture_dir):
    import json

    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    telemetry_dir = pilot_fixture_dir / "run-state" / "telemetry"
    telemetry_dir.mkdir(parents=True)
    with (telemetry_dir / "gpu_samples.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"gpu": {"memory_used_mb": 2000.0}}) + "\n")
        f.write(json.dumps({"gpu": {"memory_used_mb": 3500.0}}) + "\n")
        f.write(json.dumps({"gpu": {"memory_used_mb": 2800.0}}) + "\n")
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.gpu_memory_high_water_mb == 3500.0


def test_runtime_stability_reports_real_mean_and_stdev(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35], durations=[400.0, 500.0, 480.0])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.runtime_stability["epoch_duration_mean_seconds"] == (400.0 + 500.0 + 480.0) / 3
    assert analysis.runtime_stability["epoch_count_observed"] == 3


def test_missing_session_state_raises_not_fabricates(tmp_path):
    import pytest

    empty_dir = tmp_path / "empty-pilot"
    (empty_dir / "run-state").mkdir(parents=True)
    (empty_dir / "run-state" / "training_identity.json").write_text(
        '{"identity": {"run_id": "r1"}, "configuration_hash": "h", "configuration": {}}', encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        analyze_pilot_run(empty_dir)


def test_write_pilot_analysis_is_atomic_and_real_json(pilot_fixture_dir, tmp_path):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    output_path = tmp_path / "config" / "pilot_derived_monitoring.json"
    write_pilot_analysis(analysis, output_path)
    assert output_path.exists()

    import json

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["pilot_epoch_count"] == 2


def test_epoch_observations_carry_the_real_verbatim_deltas(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.epoch_observations[0].val_cer_delta is None  # first epoch, no prior
    assert analysis.epoch_observations[1].val_cer_delta == 0.4 - 0.5
    assert analysis.epoch_observations[2].val_cer_delta == 0.35 - 0.4


def test_epoch_observations_carry_real_relative_deltas(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.epoch_observations[0].val_cer_relative_delta is None
    assert analysis.epoch_observations[1].val_cer_relative_delta == pytest.approx((0.4 - 0.5) / 0.5)
    assert analysis.epoch_observations[2].val_cer_relative_delta == pytest.approx((0.35 - 0.4) / 0.4)


def test_base_checkpoint_identity_reflects_the_real_pilot_parent(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.base_checkpoint_identity == "generic-2023-02-15@abc"


def test_near_flat_epochs_lists_every_matching_epoch_not_just_the_streak(pilot_fixture_dir):
    # val_cers chosen so deltas after the initial two large drops are all tiny (near-flat)
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35, 0.349, 0.3485, 0.348])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert set(analysis.near_flat_epochs).issubset({4, 5, 6})
    assert len(analysis.near_flat_epochs) == analysis.max_near_flat_streak


def test_genuine_plateau_occurred_true_only_when_plateau_is_not_extrapolated(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.45, 0.40, 0.399, 0.3985, 0.3982])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.genuine_plateau_occurred == (analysis.estimated_plateau_epoch is not None and not analysis.plateau_is_extrapolated)


def test_early_stopping_not_triggered_when_stop_reason_is_something_else(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.early_stopping_triggered is False
    assert analysis.stop_reason is None  # fixture never records a last_stop_reason


def test_throughput_is_train_line_count_over_mean_epoch_duration(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4], durations=[400.0, 500.0], train_line_count=9999)
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.throughput_lines_per_second == pytest.approx(9999 / 450.0)


def test_gpu_observations_available_reflects_whether_telemetry_was_recorded(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.gpu_observations_available is False


def test_checkpoint_behavior_summary_mentions_real_entry_count(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert "3" in analysis.checkpoint_behavior_summary or "latest" in analysis.checkpoint_behavior_summary


def test_measured_vs_extrapolated_summary_is_never_empty(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.measured_vs_extrapolated_summary
    assert "MEASURED" in analysis.measured_vs_extrapolated_summary
