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
    assert "display_status: PREPARED_NOT_STARTED" in out
    json_text = out.split("\n", 1)[1]
    payload = json.loads(json_text)
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


def _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist, *, run_name="test-full-run"):
    from archivetrust.htr.training.swedish_dataset_inventory import build_source_inventory
    import zipfile

    build_pilot_fixture(pilot_fixture_dir, val_cers=[0.5, 0.4, 0.35])
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
    result = cli.main(["prepare", "--config", config_path, "--run-name", run_name])
    assert result == 0
    return training_root / run_name


def test_prepare_writes_launch_manifest_and_prepared_marker(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist):
    from archivetrust.htr.training.full_run.launch_manifest import PREPARED_MARKER_NAME, load_launch_manifest
    from archivetrust.htr.training.full_run.run_state import load_run_state, display_status

    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)

    assert (run_dir / "launch_manifest.json").exists()
    assert (run_dir / PREPARED_MARKER_NAME).exists()
    assert (run_dir / "LAUNCH_COMMANDS.txt").exists()

    manifest = load_launch_manifest(run_dir / "launch_manifest.json")
    assert manifest.pilot_checkpoint_used_as_parent is False
    assert manifest.exact_launch_command
    assert "--confirm-full-corpus-run" in manifest.exact_launch_command

    state = load_run_state(run_dir / "run-state")
    assert display_status(state) == "PREPARED_NOT_STARTED"


def test_prepare_writes_launch_commands_file_with_do_not_execute_banner(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist):
    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)
    text = (run_dir / "LAUNCH_COMMANDS.txt").read_text(encoding="utf-8")
    assert text.startswith("# DO NOT EXECUTE AUTOMATICALLY")
    assert "--confirm-full-corpus-run" in text


def test_start_without_confirm_flag_is_rejected_by_guard(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist, capsys):
    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)
    result = cli.main(["start", "--run", str(run_dir), "--no-docker-check"])  # deliberately no --confirm-full-corpus-run
    assert result == 1
    assert "Missing explicit confirmation" in capsys.readouterr().err


def test_start_with_confirm_but_no_preflight_recorded_is_rejected(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist, capsys):
    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)
    result = cli.main(["start", "--run", str(run_dir), "--no-docker-check", "--confirm-full-corpus-run"])
    assert result == 1
    assert "No passing preflight recorded" in capsys.readouterr().err


def test_start_dry_run_reports_guard_problems_but_never_invokes_the_trainer(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist, capsys):
    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)

    def _must_not_be_called(*a, **k):
        raise AssertionError("run_full_corpus_session must never be invoked during --dry-run")

    monkeypatch.setattr("archivetrust.htr.training.full_run.orchestrator.run_full_corpus_session", _must_not_be_called)

    result = cli.main(["start", "--run", str(run_dir), "--no-docker-check", "--dry-run"])
    assert result == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "STOPPING HERE" in out
    assert "before any optimizer step" in out


def test_start_dry_run_does_not_require_confirmation_to_report(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist, capsys):
    """A dry run's whole purpose is telling the operator what the guard *would* say -- it must not
    itself require --confirm-full-corpus-run to produce that report."""
    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)
    result = cli.main(["start", "--run", str(run_dir), "--no-docker-check", "--dry-run"])
    assert result == 0
    out = capsys.readouterr().out
    assert "[WOULD REJECT] Missing explicit confirmation" in out


def test_start_rejects_code_revision_drift_since_preparation(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist, capsys):
    """Reproduces a real audit finding end to end: a run prepared at one commit must be rejected by
    `start` if the code has since moved to a different commit, unless explicitly overridden."""
    import archivetrust.htr.training.full_run.run_state as run_state_mod

    monkeypatch.setattr(run_state_mod, "get_code_revision", lambda: "commit-at-prepare-time")
    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)

    # Write a passing preflight status so the drift check (not preflight) is what triggers rejection.
    (tmp_path / "config" / "preflight_status.json").write_text(
        '{"all_critical_passed": true, "summary": "1/1", "checked_at": "2026-08-01T00:00:00Z"}', encoding="utf-8"
    )

    monkeypatch.setattr(run_state_mod, "get_code_revision", lambda: "commit-after-a-real-bugfix")
    result = cli.main(["start", "--run", str(run_dir), "--no-docker-check", "--confirm-full-corpus-run"])
    assert result == 1
    assert "Code commit changed since preparation" in capsys.readouterr().err

    result_overridden = cli.main([
        "start", "--run", str(run_dir), "--no-docker-check", "--confirm-full-corpus-run", "--allow-code-revision-drift",
    ])
    # allow_code_revision_drift removes that one rejection; the call may still fail later for
    # unrelated reasons (no real Docker/container in this test), but must NOT fail on drift.
    if result_overridden != 0:
        assert "Code commit changed since preparation" not in capsys.readouterr().err


def test_start_dry_run_never_creates_a_running_status(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist):
    from archivetrust.htr.training.full_run.run_state import STATUS_PREPARED, load_run_state

    run_dir = _prepare_a_real_run(pilot_fixture_dir, tmp_path, monkeypatch, synthetic_dataset_root, synthetic_charlist)
    cli.main(["start", "--run", str(run_dir), "--no-docker-check", "--dry-run"])
    state = load_run_state(run_dir / "run-state")
    assert state.status == STATUS_PREPARED
    assert state.pid is None
