from __future__ import annotations

from archivetrust.htr.training.full_run.monitoring_config import (
    derive_monitoring_config,
    load_monitoring_config,
    write_monitoring_config,
)
from archivetrust.htr.training.full_run.pilot_analysis import analyze_pilot_run
from tests.htr.training.full_run.conftest import build_pilot_fixture


def test_shard_line_count_defaults_to_the_exact_pilot_train_line_count(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35], train_line_count=9999, batch_size=16)
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    assert config.shard_line_count == 9999  # never a rounded reconstruction like 10000


def test_shard_line_count_can_be_explicitly_overridden(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis, shard_line_count=20000)
    assert config.shard_line_count == 20000


def test_min_exposure_steps_uses_the_plateau_epoch_never_a_raw_epoch_count(pilot_fixture_dir):
    # a real, observed plateau
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.3, 0.299, 0.2985, 0.298], train_line_count=1000, batch_size=10)
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.estimated_plateau_epoch is not None
    config = derive_monitoring_config(analysis)
    steps_per_epoch = 100  # ceil(1000/10)
    assert config.min_exposure_steps == steps_per_epoch * analysis.estimated_plateau_epoch
    assert str(analysis.estimated_plateau_epoch) in config.min_exposure_basis


def test_min_exposure_falls_back_to_best_epoch_when_no_plateau_available(pilot_fixture_dir):
    # monotonically WORSENING deltas -- _estimate_plateau_epoch refuses to extrapolate
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.48, 0.44, 0.38], train_line_count=1000, batch_size=10)
    analysis = analyze_pilot_run(pilot_fixture_dir)
    assert analysis.estimated_plateau_epoch is None
    config = derive_monitoring_config(analysis)
    assert config.min_exposure_steps is not None
    assert "best_pilot_epoch" in config.min_exposure_basis


def test_min_exposure_is_none_when_steps_per_pilot_epoch_unknown(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    stripped = analysis.model_copy(update={"steps_per_pilot_epoch": None, "batch_size": None})
    config = derive_monitoring_config(stripped)
    assert config.min_exposure_steps is None
    assert "Could not be computed" in config.min_exposure_basis


def test_patience_is_derived_from_the_real_near_flat_streak(pilot_fixture_dir):
    # near-flat streak of length 3 -> patience = max(3, 3+2) = 5
    build_pilot_fixture(
        pilot_fixture_dir, val_cers=[0.5, 0.3, 0.299, 0.2985, 0.298, 0.25], train_line_count=1000, batch_size=10
    )
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    assert config.recommended_patience == max(3, analysis.max_near_flat_streak + 2)


def test_patience_override_takes_precedence(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis, patience_override=8)
    assert config.recommended_patience == 8
    assert "overridden" in config.patience_basis.lower()


def test_patience_falls_back_to_5_when_no_streak_observed(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5])  # single epoch, no deltas at all
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    assert config.recommended_patience == 5


def test_checkpoint_and_validation_frequency_are_always_one_shard(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    assert config.checkpoint_frequency_shards == 1
    assert config.validation_frequency_shards == 1


def test_max_full_run_epochs_defaults_to_a_fixed_ceiling_without_corpus_size(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    assert config.max_full_run_epochs == 500


def test_max_full_run_epochs_derived_from_corpus_size_covers_at_least_one_lap(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4], train_line_count=1000)
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis, total_valid_corpus_line_count=50000)
    one_lap = -(-50000 // config.shard_line_count)
    assert config.max_full_run_epochs == one_lap * 3


def test_max_full_run_epochs_override_wins(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis, total_valid_corpus_line_count=50000, max_full_run_epochs_override=42)
    assert config.max_full_run_epochs == 42


def test_warning_thresholds_reference_a_real_observed_max_train_loss(pilot_fixture_dir):
    build_pilot_fixture(
        pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35],
        train_losses=[40.0, 35.0, 30.0], val_losses=[42.0, 36.0, 31.0],
    )
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    assert config.warning_thresholds["loss_explosion_absolute_reference"] == 40.0


def test_warning_thresholds_reference_is_none_with_no_loss_data(pilot_fixture_dir):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    assert config.warning_thresholds["loss_explosion_absolute_reference"] is None


def test_stall_timeout_scales_with_shard_size_relative_to_pilot_epoch(pilot_fixture_dir):
    build_pilot_fixture(
        pilot_fixture_dir, val_cers=[0.5, 0.4], durations=[500.0, 500.0], train_line_count=1000, batch_size=10,
    )
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config_same_size = derive_monitoring_config(analysis, shard_line_count=1000)
    config_double_size = derive_monitoring_config(analysis, shard_line_count=2000)
    assert config_double_size.warning_thresholds["stall_timeout_seconds"] == \
        config_same_size.warning_thresholds["stall_timeout_seconds"] * 2


def test_write_and_load_monitoring_config_round_trips(pilot_fixture_dir, tmp_path):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    path = tmp_path / "config" / "full_run_monitoring.json"
    write_monitoring_config(config, path)
    reloaded = load_monitoring_config(path)
    assert reloaded == config


def test_every_derived_value_is_a_plain_editable_field(pilot_fixture_dir):
    """Structural check: `prepare`/the GUI must be able to show and override every derived value --
    confirms the model has no computed-only properties hiding fields from `model_dump()`."""
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    analysis = analyze_pilot_run(pilot_fixture_dir)
    config = derive_monitoring_config(analysis)
    dumped = config.model_dump()
    for field in ("shard_line_count", "min_exposure_steps", "recommended_patience", "max_full_run_epochs", "warning_thresholds"):
        assert field in dumped
