"""The subprocess seam between `SatrnAdapter` (running in the main project venv) and the real
SATRN inference worker (`providers/satrn/_worker.py`, running under the isolated `.venv-satrn`
interpreter -- see that module's docstring for why this isolation exists).

Follows the same "narrow Protocol standing in for a real, hard-to-install dependency" discipline
`runtime/transformers_runtime.py::TorchAndTransformersFacade` uses for torch/transformers: tests
inject `_FakeSatrnWorkerFacade`-shaped objects (fast, no subprocess, no GPU); production code uses
`subprocess_satrn_facade()` (slow, real subprocess, real model).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

_DEFAULT_TIMEOUT_SECONDS = 300.0
_WORKER_SCRIPT = Path(__file__).with_name("_worker.py")


class SatrnWorkerResult(dict):
    """The parsed JSON object the worker prints -- kept as a plain dict (not a pydantic model)
    since this is an internal transport type between this module and `adapter.py`, which is what
    translates it into the public `RecognitionResult`/`Evidence`/`FailureRecord` types."""


class SatrnWorkerFacade(Protocol):
    """The exact surface `SatrnAdapter` needs from a SATRN inference run, isolated behind one
    Protocol so it can be the real subprocess (production) or a fake (tests)."""

    def run_inference(
        self, *, image_path: str, device_request: str, model_id: str, revision: str | None
    ) -> SatrnWorkerResult: ...


class SatrnWorkerTimeout(RuntimeError):
    """Raised when the worker subprocess does not complete within the configured timeout --
    mapped to `FailureRecord.category == "timeout"` by `adapter.py`."""


def _find_satrn_python() -> Path:
    """Locates the isolated `.venv-satrn` interpreter. Resolution order (documented in
    `providers/satrn/README.md`): the `ARCHIVETRUST_SATRN_PYTHON` environment variable (explicit
    override, e.g. for a deployment where the isolated venv lives elsewhere), then
    `<repo_root>/.venv-satrn/Scripts/python.exe` (Windows) / `.../bin/python` (POSIX) relative to
    this repository's layout -- never silently falls back to the main interpreter, since that
    interpreter does not have `mmocr`/`mmcv`/`mmdet`/`mmengine` installed and would fail with a
    confusing `ModuleNotFoundError` deep inside the worker instead of a clear "isolated venv not
    found" message here.
    """
    import os

    override = os.environ.get("ARCHIVETRUST_SATRN_PYTHON")
    if override:
        return Path(override)

    # src/archivetrust/providers/satrn/facade.py -> repo root is 4 parents up.
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / ".venv-satrn" / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    candidate = repo_root / ".venv-satrn" / "bin" / "python"
    return candidate


def satrn_python_available() -> tuple[bool, str]:
    """Cheap existence check (no subprocess launch) -- used by `validate_environment()` /
    `health_check()`, which must not pay the multi-second worker-startup cost just to answer
    "is this method usable at all"."""
    python_path = _find_satrn_python()
    if not python_path.exists():
        return False, (
            f"Isolated SATRN interpreter not found at {python_path} -- see "
            "providers/satrn/README.md 'Install / Setup' for how to create .venv-satrn, or set "
            "ARCHIVETRUST_SATRN_PYTHON to point at an existing one."
        )
    return True, f"Found isolated SATRN interpreter at {python_path}"


class _SubprocessSatrnWorkerFacade:
    def __init__(self, *, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def run_inference(
        self, *, image_path: str, device_request: str, model_id: str, revision: str | None
    ) -> SatrnWorkerResult:
        python_path = _find_satrn_python()
        available, message = satrn_python_available()
        if not available:
            raise FileNotFoundError(message)

        args = [
            str(python_path),
            str(_WORKER_SCRIPT),
            image_path,
            "--device",
            device_request,
            "--model-id",
            model_id,
        ]
        if revision:
            args.extend(["--revision", revision])

        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SatrnWorkerTimeout(
                f"SATRN worker did not complete within {self._timeout_seconds}s"
            ) from exc

        stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            return SatrnWorkerResult(
                ok=False,
                category="unknown_error",
                message=(
                    f"worker produced no stdout (exit={completed.returncode}); "
                    f"stderr tail: {completed.stderr[-2000:]}"
                ),
            )
        try:
            payload = json.loads(stdout_lines[-1])
        except json.JSONDecodeError as exc:
            return SatrnWorkerResult(
                ok=False,
                category="unknown_error",
                message=f"worker stdout was not valid JSON ({exc}); raw tail: {stdout_lines[-1][:2000]}",
            )
        return SatrnWorkerResult(payload)


def subprocess_satrn_facade(*, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> SatrnWorkerFacade:
    """Constructs the real, subprocess-backed facade. Not imported/constructed at module load
    time by `adapter.py` -- only when a caller actually wants to run real inference (mirrors
    `transformers_runtime.py::real_torch_and_transformers_facade`'s lazy-construction discipline)."""
    return _SubprocessSatrnWorkerFacade(timeout_seconds=timeout_seconds)
