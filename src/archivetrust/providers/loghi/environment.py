"""Read-only environment probes for Loghi's execution boundary (docs/methods/loghi.md §"Execution
architecture"). Every function here inspects the host -- it never starts, stops, or runs a container.

Loghi's own tooling assumes Bash, Linux paths and NVIDIA container tooling; this module's whole job is
answering "which of Docker-on-Linux / Docker-through-WSL2 / native-Linux is actually available right
now" without ever pretending native Windows execution works.
"""

from __future__ import annotations

import platform
import re
import subprocess

from archivetrust.providers.loghi.models import LoghiEnvironmentReport, LoghiExecutionMode

_PROBE_TIMEOUT_SECONDS = 10.0


def _run(args: list[str]) -> tuple[bool, str]:
    """Runs one short-lived, read-only probe command. Never `shell=True`, never a stage of the
    actual pipeline -- just "does this command exist and what does it print." Returns
    `(succeeded, combined_output)`; a missing executable or non-zero exit is reported, not raised,
    since "the tool isn't there" is exactly the fact this module exists to detect.

    Captured as raw bytes, not `text=True`: `wsl.exe` prints UTF-16LE on Windows regardless of the
    console codepage, which `text=True`'s locale-guessed decoding turns into a string with a stray
    `\\x00` after every character -- a real bug hit while smoke-testing this module (embedded-null
    `ValueError` from a *second* subprocess call whose argv was built from that corrupted text). UTF-16
    output is detected by its telltale null-byte density and decoded explicitly; anything else falls
    back to UTF-8 with lossy replacement, since a probe must never raise on unexpected output."""
    try:
        completed = subprocess.run(
            args, capture_output=True, text=False, timeout=_PROBE_TIMEOUT_SECONDS, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""
    raw = (completed.stdout or b"") + (completed.stderr or b"")
    if raw.count(b"\x00") > len(raw) // 4:
        output = raw.decode("utf-16-le", errors="replace")
    else:
        output = raw.decode("utf-8", errors="replace")
    output = output.replace("\x00", "")
    return completed.returncode == 0, output


def _docker_status() -> tuple[bool, str | None]:
    ok, output = _run(["docker", "--version"])
    if not ok:
        return False, None
    return True, output.strip().splitlines()[0] if output.strip() else None


def _wsl_status() -> tuple[bool, tuple[str, ...]]:
    """Parses `wsl.exe -l -v` output. Distro names are read from the first whitespace-delimited
    column after stripping the `*` default-distro marker -- the command's own table formatting is
    not a stable machine format, so this is a best-effort parse that returns `()` (never raises) if
    the output does not look like the expected table."""
    ok, output = _run(["wsl.exe", "-l", "-v"])
    if not ok:
        return False, ()
    distros: list[str] = []
    for line in output.splitlines():
        stripped = line.strip().lstrip("*").strip()
        if not stripped or stripped.upper().startswith("NAME"):
            continue
        name = stripped.split()[0] if stripped.split() else None
        if name:
            distros.append(name)
    return True, tuple(distros)


def _nvidia_toolkit_version() -> str | None:
    ok, output = _run(["nvidia-ctk", "--version"])
    if not ok:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", output)
    return match.group(1) if match else output.strip().splitlines()[0] if output.strip() else None


def _parse_pretty_name(os_release_text: str) -> str | None:
    for line in os_release_text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def _linux_distribution_via_wsl(distro: str) -> str | None:
    ok, output = _run(["wsl.exe", "-d", distro, "--", "cat", "/etc/os-release"])
    return _parse_pretty_name(output) if ok else None


def _linux_distribution_native() -> str | None:
    ok, output = _run(["cat", "/etc/os-release"])
    return _parse_pretty_name(output) if ok else None


def probe_loghi_environment() -> LoghiEnvironmentReport:
    """The one function `LoghiAdapter.validate_environment()` calls. Read-only: `docker --version`,
    `wsl.exe -l -v`, `nvidia-ctk --version`, at most one `wsl.exe -d <distro> -- cat /etc/os-release`
    -- no image pull, no container start, no model load.
    """
    host_os = platform.system()

    docker_present, docker_version = _docker_status()
    wsl_present, wsl_distros = _wsl_status() if host_os == "Windows" else (False, ())
    nvidia_toolkit_version = _nvidia_toolkit_version()

    linux_distribution: str | None = None
    if host_os == "Linux":
        linux_distribution = _linux_distribution_native()
    elif wsl_distros:
        running_distro = next((d for d in wsl_distros if d.lower() != "docker-desktop-data"), None)
        if running_distro:
            linux_distribution = _linux_distribution_via_wsl(running_distro)

    execution_mode: LoghiExecutionMode | None = None
    if host_os == "Linux" and docker_present:
        execution_mode = LoghiExecutionMode.DOCKER_LINUX
    elif host_os == "Windows" and docker_present and wsl_present and wsl_distros:
        execution_mode = LoghiExecutionMode.DOCKER_WSL2
    elif host_os == "Linux":
        execution_mode = LoghiExecutionMode.NATIVE_LINUX

    return LoghiEnvironmentReport(
        host_os=host_os,
        docker_cli_present=docker_present,
        docker_version=docker_version,
        wsl_present=wsl_present,
        wsl_distros=wsl_distros,
        linux_distribution=linux_distribution,
        nvidia_toolkit_version=nvidia_toolkit_version,
        cuda_visible_devices=None,
        execution_mode=execution_mode,
    )
