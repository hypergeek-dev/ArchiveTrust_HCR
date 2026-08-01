"""Tests the GUI's subprocess-command construction -- pure functions, no `QWidget`/`QApplication`
instantiated anywhere, so this file runs in the default headless suite exactly like every other test
in this project."""

from __future__ import annotations

from archivetrust.htr.training.full_run.gui.commands import (
    build_analyze_pilot_command,
    build_prepare_command,
    build_preflight_command,
    build_resume_command,
    build_start_command,
    build_status_command,
    build_stop_command,
)


def test_analyze_pilot_command_uses_the_real_module_path():
    argv = build_analyze_pilot_command(python_exe="python.exe", pilot_run="training/loghi-swedish-v1")
    assert argv[:3] == ["python.exe", "-m", "archivetrust.htr.training.full_run"]
    assert "analyze-pilot" in argv
    assert "--pilot-run" in argv
    assert argv[argv.index("--pilot-run") + 1] == "training/loghi-swedish-v1"


def test_analyze_pilot_command_omits_optional_flags_when_not_given():
    argv = build_analyze_pilot_command(python_exe="python.exe", pilot_run="p")
    assert "--shard-line-count" not in argv
    assert "--patience" not in argv
    assert "--max-epochs" not in argv


def test_analyze_pilot_command_includes_every_provided_override():
    argv = build_analyze_pilot_command(
        python_exe="python.exe", pilot_run="p", shard_line_count=1000, corpus_line_count=500000,
        patience=7, max_epochs=42,
    )
    assert argv[argv.index("--shard-line-count") + 1] == "1000"
    assert argv[argv.index("--corpus-line-count") + 1] == "500000"
    assert argv[argv.index("--patience") + 1] == "7"
    assert argv[argv.index("--max-epochs") + 1] == "42"


def test_defaults_to_sys_executable_when_no_python_exe_given():
    import sys

    argv = build_analyze_pilot_command(pilot_run="p")
    assert argv[0] == sys.executable


def test_preflight_command_structure():
    argv = build_preflight_command(python_exe="python.exe", config_path="config.json", run_dir="training/run1")
    assert "preflight" in argv
    assert argv[argv.index("--config") + 1] == "config.json"
    assert argv[argv.index("--run") + 1] == "training/run1"
    assert "--no-gpu" not in argv
    assert "--no-smoke-test" not in argv


def test_preflight_command_includes_boolean_flags_only_when_set():
    argv = build_preflight_command(python_exe="python.exe", config_path="c.json", no_gpu=True, no_smoke_test=True)
    assert "--no-gpu" in argv
    assert "--no-smoke-test" in argv


def test_prepare_command_structure():
    argv = build_prepare_command(python_exe="python.exe", config_path="c.json", run_name="my-run")
    assert "prepare" in argv
    assert argv[argv.index("--config") + 1] == "c.json"
    assert argv[argv.index("--run-name") + 1] == "my-run"


def test_prepare_command_omits_run_name_when_not_given():
    argv = build_prepare_command(python_exe="python.exe", config_path="c.json")
    assert "--run-name" not in argv


def test_start_command_carries_hours_and_batch_size():
    argv = build_start_command(python_exe="python.exe", run_dir="training/run1", hours=2.5, batch_size=8)
    assert "start" in argv
    assert argv[argv.index("--run") + 1] == "training/run1"
    assert argv[argv.index("--hours") + 1] == "2.5"
    assert argv[argv.index("--batch-size") + 1] == "8"


def test_resume_command_uses_the_resume_subcommand_not_start():
    argv = build_resume_command(python_exe="python.exe", run_dir="training/run1")
    assert "resume" in argv
    assert "start" not in argv


def test_status_command_structure():
    argv = build_status_command(python_exe="python.exe", run_dir="training/run1")
    assert argv[-2:] == ["--run", "training/run1"]
    assert "status" in argv


def test_stop_command_structure():
    argv = build_stop_command(python_exe="python.exe", run_dir="training/run1")
    assert "stop" in argv
    assert argv[-2:] == ["--run", "training/run1"]


def test_every_command_is_a_flat_list_of_strings():
    """Structural check -- `QProcess.start(program, arguments)` needs a flat list, never a nested
    structure or a non-string element."""
    argv = build_start_command(python_exe="python.exe", run_dir="r", hours=1.0, batch_size=16)
    assert all(isinstance(a, str) for a in argv)
