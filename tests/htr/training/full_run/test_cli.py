from __future__ import annotations

import json

import pytest

import archivetrust.htr.training.full_run.cli as cli
from archivetrust.htr.training.full_run.run_state import STATUS_PREPARED, create_initial_run_state, save_run_state
from tests.htr.training.full_run.conftest import build_pilot_fixture


def test_parser_exposes_every_required_subcommand():
    parser = cli.build_parser()
    subcommands = {a.dest: a.choices for a in parser._subparsers._group_actions if a.dest == "command"}
    choices = set(subcommands["command"])
    assert choices == {"analyze-pilot", "preflight", "prepare", "start", "status", "stop", "resume", "gui"}


def test_main_dispatches_to_the_right_command_function(monkeypatch):
    called = {}

    def _fake_cmd_status(args):
        called["ran"] = args.run
        return 0

    monkeypatch.setattr(cli, "cmd_status", _fake_cmd_status)
    result = cli.main(["status", "--run", "/some/run"])
    assert result == 0
    assert called["ran"] == "/some/run"


def test_analyze_pilot_end_to_end(pilot_fixture_dir, tmp_path, monkeypatch):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35, 0.34])
    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)

    result = cli.main(["analyze-pilot", "--pilot-run", str(pilot_fixture_dir)])
    assert result == 0
    assert (config_dir / "pilot_analysis.json").exists()
    assert (config_dir / "pilot_derived_monitoring.json").exists()

    payload = json.loads((config_dir / "pilot_analysis.json").read_text(encoding="utf-8"))
    assert payload["pilot_epoch_count"] == 4


def test_analyze_pilot_respects_explicit_overrides(pilot_fixture_dir, tmp_path, monkeypatch):
    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)

    cli.main([
        "analyze-pilot", "--pilot-run", str(pilot_fixture_dir),
        "--shard-line-count", "12345", "--patience", "9", "--max-epochs", "77",
    ])
    payload = json.loads((config_dir / "pilot_derived_monitoring.json").read_text(encoding="utf-8"))
    assert payload["shard_line_count"] == 12345
    assert payload["recommended_patience"] == 9
    assert payload["max_full_run_epochs"] == 77


def test_status_prints_the_real_state(tmp_path, capsys):
    run_dir = tmp_path / "myrun"
    save_run_state(run_dir / "run-state", create_initial_run_state(run_id="r1", configuration_hash="h1"))
    result = cli.main(["status", "--run", str(run_dir)])
    assert result == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] == STATUS_PREPARED
    assert payload["run_id"] == "r1"


def test_status_on_a_never_prepared_run_returns_nonzero(tmp_path):
    result = cli.main(["status", "--run", str(tmp_path / "never_prepared")])
    assert result == 1


def test_stop_writes_the_sentinel_file(tmp_path):
    run_dir = tmp_path / "myrun"
    save_run_state(run_dir / "run-state", create_initial_run_state(run_id="r1", configuration_hash="h1"))
    result = cli.main(["stop", "--run", str(run_dir)])
    assert result == 0
    assert (run_dir / "run-state" / cli.STOP_SENTINEL_NAME).exists()


def test_dataset_hash_is_none_when_inventory_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INVENTORY_PATH", tmp_path / "does_not_exist.parquet")
    assert cli._dataset_hash() is None


def test_dataset_hash_is_a_real_sha256_when_inventory_present(tmp_path, monkeypatch):
    inventory_path = tmp_path / "inventory.parquet"
    inventory_path.write_bytes(b"fake inventory content")
    monkeypatch.setattr(cli, "INVENTORY_PATH", inventory_path)
    digest = cli._dataset_hash()
    assert digest is not None
    assert len(digest) == 64


def test_prepare_end_to_end_with_monkeypatched_paths(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist):
    """A full, real `prepare` invocation against synthetic fixtures -- confirms the CLI wiring
    genuinely produces a real run directory, real shards, and a real run_state.json, never touching
    the real repo's `training/` directory."""
    from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory

    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])

    training_root = tmp_path / "training"
    inventory_path = tmp_path / "inventory.parquet"
    build_source_inventory(dataset_root=synthetic_dataset_root, output_path=inventory_path, charlist_path=synthetic_charlist)

    parent_checkpoint_dir = tmp_path / "parent_checkpoint"
    parent_checkpoint_dir.mkdir()
    import zipfile

    with zipfile.ZipFile(parent_checkpoint_dir / "model.keras", "w") as zf:
        zf.writestr("config.json", "{}")
    (parent_checkpoint_dir / "charlist.txt").write_text("abc", encoding="utf-8")

    monkeypatch.setattr(cli, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(cli, "DEFAULT_PILOT_RUN_DIR", pilot_fixture_dir)
    monkeypatch.setattr(cli, "INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(cli, "PARENT_CHECKPOINT_DIR", parent_checkpoint_dir)
    monkeypatch.setattr(cli, "CHARLIST_PATH", parent_checkpoint_dir / "charlist.txt")
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path / "config")

    cli.main(["analyze-pilot", "--pilot-run", str(pilot_fixture_dir), "--shard-line-count", "3", "--max-epochs", "2"])
    result = cli.main(["prepare", "--config", str(tmp_path / "config" / "pilot_derived_monitoring.json"), "--run-name", "test-full-run"])
    assert result == 0

    run_dir = training_root / "test-full-run"
    assert (run_dir / "run-state" / "run_state.json").exists()
    assert (run_dir / "run-state" / "training_identity.json").exists()
    assert (run_dir / "shards" / "sharding_summary.json").exists()
    assert (run_dir / "config" / "full_run_monitoring.json").exists()


def test_prepare_never_reuses_an_existing_run_name(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist):
    from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory
    import zipfile

    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4])
    training_root = tmp_path / "training"
    inventory_path = tmp_path / "inventory.parquet"
    build_source_inventory(dataset_root=synthetic_dataset_root, output_path=inventory_path, charlist_path=synthetic_charlist)
    parent_checkpoint_dir = tmp_path / "parent_checkpoint"
    parent_checkpoint_dir.mkdir()
    with zipfile.ZipFile(parent_checkpoint_dir / "model.keras", "w") as zf:
        zf.writestr("config.json", "{}")
    (parent_checkpoint_dir / "charlist.txt").write_text("abc", encoding="utf-8")

    monkeypatch.setattr(cli, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(cli, "DEFAULT_PILOT_RUN_DIR", pilot_fixture_dir)
    monkeypatch.setattr(cli, "INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(cli, "PARENT_CHECKPOINT_DIR", parent_checkpoint_dir)
    monkeypatch.setattr(cli, "CHARLIST_PATH", parent_checkpoint_dir / "charlist.txt")
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path / "config")

    cli.main(["analyze-pilot", "--pilot-run", str(pilot_fixture_dir), "--shard-line-count", "3", "--max-epochs", "2"])
    config_path = str(tmp_path / "config" / "pilot_derived_monitoring.json")
    cli.main(["prepare", "--config", config_path, "--run-name", "dup-run"])
    with pytest.raises(Exception):  # RunDirectoryAlreadyExists, propagated through the CLI
        cli.main(["prepare", "--config", config_path, "--run-name", "dup-run"])


def test_start_refuses_a_run_that_is_not_in_prepared_state(tmp_path, capsys):
    from archivetrust.htr.training.full_run.run_state import mark_running

    run_dir = tmp_path / "myrun"
    state = create_initial_run_state(run_id="r1", configuration_hash="h1")
    save_run_state(run_dir / "run-state", mark_running(state))  # already running, not "prepared"

    class _Args:
        run = str(run_dir)
        hours = 1.0
        batch_size = 16

    result = cli._run_session(_Args(), resume=False)
    assert result == 1
    assert "use `resume`" in capsys.readouterr().err
