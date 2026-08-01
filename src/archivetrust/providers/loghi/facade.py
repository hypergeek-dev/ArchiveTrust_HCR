"""The container-invocation seam between `LoghiAdapter` and a real Loghi pipeline run.

Same "narrow Protocol standing in for a real, hard-to-install dependency" discipline as
`providers/satrn/facade.py` -- tests inject `FakeLoghiWorkerFacade` (fast, no Docker, no GPU);
production code uses `subprocess_loghi_facade()` (slow, real `docker run`, real containers). Argv
construction is deterministic and never uses `shell=True` or an interactive shell -- every value
(image reference, mounted paths, beam width, GPU selection) comes from an explicit
`LoghiComponentVersions`/`LoghiEnvironmentReport` pair, never a floating default.

**What is, and is not, asserted here.** No real Loghi Docker image has been pulled or run as part of
this integration (`docs/loghi-integration-audit.md` §13). The argv this module builds is a structurally
correct `docker run` invocation shaped from `LoghiComponentVersions` (image tag/digest, mounted input/
output paths, GPU device, beam width) -- it is **not** asserted to match Loghi's own published
entrypoint contract byte-for-byte, since that contract has not been read from a real pulled image in
this session. `_build_docker_argv`'s docstring states exactly which parts are structural (mount syntax,
`docker run` flag shapes -- standard Docker CLI, verifiable independent of Loghi) versus which parts
are placeholders for Loghi's actual documented entrypoint/args (must be confirmed against the pinned
image's own documentation before the first real run).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Protocol

from archivetrust.providers.loghi.models import (
    LoghiComponentVersions,
    LoghiEnvironmentReport,
    LoghiExecutionMode,
)
from archivetrust.providers.loghi.stage_results import LoghiPipelineResult, LoghiStageResult

_DEFAULT_TIMEOUT_SECONDS = 1800.0
"""Multi-stage containerized pipeline -- generously longer than SATRN's 300s single-recognizer
timeout, since Laypa + Tooling + HTR run sequentially inside the container(s)."""


class LoghiPipelineTimeout(RuntimeError):
    """Raised when the pipeline invocation does not complete within the configured timeout --
    mapped to `FailureRecord.category == "timeout"` by `adapter.py`."""


class LoghiWorkerFacade(Protocol):
    """The exact surface `LoghiAdapter` needs from one page's Loghi run, isolated behind one
    Protocol so it can be the real container invocation (production) or a fake (tests)."""

    def run_pipeline(
        self,
        *,
        input_image_path: str,
        output_dir: str,
        component_versions: LoghiComponentVersions,
        environment: LoghiEnvironmentReport,
        timeout_seconds: float,
    ) -> LoghiPipelineResult: ...


def _to_wsl_path(windows_path: str) -> str:
    """`C:\\Users\\x\\y` -> `/mnt/c/Users/x/y` -- the standard WSL2 path-translation convention, used
    when mounting a Windows-side directory into a container run through `wsl.exe -- docker run`.
    Applied only under `LoghiExecutionMode.DOCKER_WSL2`; Linux-native paths pass through untouched."""
    path = Path(windows_path).resolve()
    drive = path.drive.rstrip(":").lower()
    rest = path.as_posix()[len(path.drive) :].lstrip("/")
    return f"/mnt/{drive}/{rest}"


def _build_docker_argv(
    *,
    input_image_path: str,
    output_dir: str,
    component_versions: LoghiComponentVersions,
    environment: LoghiEnvironmentReport,
) -> list[str]:
    """Builds one deterministic `docker run` argv. The `docker run`/`-v`/`--gpus`/`--rm` shape is
    standard Docker CLI syntax, verifiable independent of Loghi. The image reference and mounted
    paths come entirely from `component_versions`/the caller's paths -- never a floating tag. The
    container-side entrypoint/argument names (`/input`, `/output`, `--beam-width`) are this
    integration's placeholder convention pending confirmation against the pinned image's actual
    documented usage (see module docstring) -- structurally present so the rest of the pipeline
    (timeout handling, stdout/stderr capture, exit-code mapping) can be built and tested now, not a
    claim that these exact flags are Loghi's real ones.
    """
    input_path = Path(input_image_path).resolve()
    output_path = Path(output_dir).resolve()

    if environment.execution_mode is LoghiExecutionMode.DOCKER_WSL2:
        host_input = _to_wsl_path(str(input_path))
        host_output = _to_wsl_path(str(output_path))
        prefix = ["wsl.exe", "--"]
    else:
        host_input = str(input_path)
        host_output = str(output_path)
        prefix = []

    image_ref = (
        f"{component_versions.docker_image_tag}@{component_versions.docker_image_digest}"
        if component_versions.docker_image_digest
        else component_versions.docker_image_tag
    )

    argv = [
        *prefix,
        "docker",
        "run",
        "--rm",
        "-v",
        f"{host_input}:/input:ro",
        "-v",
        f"{host_output}:/output",
    ]
    if component_versions.gpu_selection and component_versions.gpu_selection.lower() != "none":
        argv += ["--gpus", component_versions.gpu_selection]
    argv += [
        image_ref,
        "--input",
        "/input",
        "--output",
        "/output",
        "--beam-width",
        str(component_versions.beam_width),
        "--reading-order",
        component_versions.reading_order_settings,
        "--language-detection",
        component_versions.language_detection_settings,
    ]
    return argv


class _SubprocessLoghiWorkerFacade:
    def run_pipeline(
        self,
        *,
        input_image_path: str,
        output_dir: str,
        component_versions: LoghiComponentVersions,
        environment: LoghiEnvironmentReport,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> LoghiPipelineResult:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        start_monotonic = time.monotonic()

        if component_versions.is_placeholder():
            return LoghiPipelineResult(
                ok=False,
                stages=(
                    LoghiStageResult(
                        stage_name="environment_validation",
                        started_at=started_at,
                        completed_at=started_at,
                        ok=False,
                        errors=(
                            "LoghiComponentVersions still carries placeholder pins -- see "
                            "providers/loghi/pinned_versions.py and docs/methods/loghi.md",
                        ),
                    ),
                ),
            )

        if environment.execution_mode is None:
            return LoghiPipelineResult(
                ok=False,
                stages=(
                    LoghiStageResult(
                        stage_name="environment_validation",
                        started_at=started_at,
                        completed_at=started_at,
                        ok=False,
                        errors=(
                            "No supported Loghi execution mode available on this host -- see "
                            "LoghiEnvironmentReport (docker/WSL2 status).",
                        ),
                    ),
                ),
            )

        argv = _build_docker_argv(
            input_image_path=input_image_path,
            output_dir=output_dir,
            component_versions=component_versions,
            environment=environment,
        )

        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise LoghiPipelineTimeout(
                f"Loghi pipeline did not complete within {timeout_seconds}s"
            ) from exc

        duration_ms = (time.monotonic() - start_monotonic) * 1000.0
        completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        pipeline_stage = LoghiStageResult(
            stage_name="loghi_pipeline",
            started_at=started_at,
            completed_at=completed_at,
            ok=completed.returncode == 0,
            stdout_tail=(completed.stdout or "")[-4000:],
            stderr_tail=(completed.stderr or "")[-4000:],
            exit_code=completed.returncode,
            duration_ms=duration_ms,
        )

        output_page_xml = None
        output_dir_path = Path(output_dir)
        if completed.returncode == 0 and output_dir_path.exists():
            xml_candidates = sorted(output_dir_path.glob("*.xml"))
            if xml_candidates:
                output_page_xml = xml_candidates[0].read_text(encoding="utf-8")

        return LoghiPipelineResult(
            ok=completed.returncode == 0 and output_page_xml is not None,
            stages=(pipeline_stage,),
            final_page_xml=output_page_xml,
            total_duration_ms=duration_ms,
        )


class FakeLoghiWorkerFacade:
    """Scripted facade for tests -- returns whatever `LoghiPipelineResult` it was constructed with,
    no Docker, no subprocess, no GPU. `calls` records every invocation's arguments for assertions."""

    def __init__(self, *, result: LoghiPipelineResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def run_pipeline(
        self,
        *,
        input_image_path: str,
        output_dir: str,
        component_versions: LoghiComponentVersions,
        environment: LoghiEnvironmentReport,
        timeout_seconds: float,
    ) -> LoghiPipelineResult:
        self.calls.append(
            {
                "input_image_path": input_image_path,
                "output_dir": output_dir,
                "component_versions": component_versions,
                "environment": environment,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._result


def subprocess_loghi_facade() -> LoghiWorkerFacade:
    """Constructs the real, subprocess-backed facade. Not imported/constructed at module load time
    by `adapter.py` -- only when a caller actually wants to run a real pipeline."""
    return _SubprocessLoghiWorkerFacade()
