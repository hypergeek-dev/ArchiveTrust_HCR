"""Pure subprocess-command construction for the GUI -- no Qt import, no `QProcess`, so every
function here is directly unit-testable without a display or an event loop. `gui/app.py` calls these
and launches the result via `QProcess`; it never constructs an argv itself and never runs training
in-process ("run the training process as a subprocess rather than blocking the GUI thread").
"""

from __future__ import annotations

import sys

_MODULE = "archivetrust.htr.training.full_run"


def _base(python_exe: str | None) -> list[str]:
    return [python_exe or sys.executable, "-m", _MODULE]


def build_analyze_pilot_command(
    *,
    python_exe: str | None = None,
    pilot_run: str,
    shard_line_count: int | None = None,
    corpus_line_count: int | None = None,
    patience: int | None = None,
    max_epochs: int | None = None,
) -> list[str]:
    argv = _base(python_exe) + ["analyze-pilot", "--pilot-run", pilot_run]
    if shard_line_count is not None:
        argv += ["--shard-line-count", str(shard_line_count)]
    if corpus_line_count is not None:
        argv += ["--corpus-line-count", str(corpus_line_count)]
    if patience is not None:
        argv += ["--patience", str(patience)]
    if max_epochs is not None:
        argv += ["--max-epochs", str(max_epochs)]
    return argv


def build_preflight_command(
    *,
    python_exe: str | None = None,
    config_path: str,
    run_dir: str | None = None,
    batch_size: int = 16,
    no_gpu: bool = False,
    no_smoke_test: bool = False,
) -> list[str]:
    argv = _base(python_exe) + ["preflight", "--config", config_path, "--batch-size", str(batch_size)]
    if run_dir is not None:
        argv += ["--run", run_dir]
    if no_gpu:
        argv.append("--no-gpu")
    if no_smoke_test:
        argv.append("--no-smoke-test")
    return argv


def build_prepare_command(
    *, python_exe: str | None = None, config_path: str, run_name: str | None = None
) -> list[str]:
    argv = _base(python_exe) + ["prepare", "--config", config_path]
    if run_name is not None:
        argv += ["--run-name", run_name]
    return argv


def build_start_command(
    *, python_exe: str | None = None, run_dir: str, hours: float = 5.0, batch_size: int = 16
) -> list[str]:
    return _base(python_exe) + ["start", "--run", run_dir, "--hours", str(hours), "--batch-size", str(batch_size)]


def build_resume_command(
    *, python_exe: str | None = None, run_dir: str, hours: float = 5.0, batch_size: int = 16
) -> list[str]:
    return _base(python_exe) + ["resume", "--run", run_dir, "--hours", str(hours), "--batch-size", str(batch_size)]


def build_status_command(*, python_exe: str | None = None, run_dir: str) -> list[str]:
    return _base(python_exe) + ["status", "--run", run_dir]


def build_stop_command(*, python_exe: str | None = None, run_dir: str) -> list[str]:
    return _base(python_exe) + ["stop", "--run", run_dir]
