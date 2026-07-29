"""Preflight checks for the Windows bootstrap (`scripts/setup-archivetrust.ps1`).

Every check that needs to run an external command takes an injectable `CommandRunner` so unit
tests never actually invoke Docker, WSL, or PowerShell -- see `tests/bootstrap/test_preflight.py`.
Checks that are safely computable in pure Python (disk space, port availability, path safety) run
for real; nothing here requires administrator rights or mutates the machine.

This module intentionally covers a curated subset of the full preflight list a production
installer would eventually need (Docker/provider-container-conflict detail, full virtualization
probing, and PowerShell-version/execution-policy checks are left to the caller to supply via
`CommandRunner` results, since Python cannot reliably self-report them) -- see
`docs/WINDOWS_BOOTSTRAP.md` for what is deferred to later bootstrap work.
"""
from __future__ import annotations

import contextlib
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from archivetrust.worker.acl_policy import is_dangerous_root


class PreflightStatus(str, Enum):
    OK = "ok"
    MISSING_PREREQUISITE = "missing_prerequisite"
    INCOMPATIBLE_VERSION = "incompatible_version"
    INSTALLED_NOT_RUNNING = "installed_not_running"
    INSTALLED_UNHEALTHY = "installed_unhealthy"
    INSUFFICIENT_PRIVILEGE = "insufficient_privilege"
    REBOOT_REQUIRED = "reboot_required"
    PORT_CONFLICT = "port_conflict"
    CONFIGURATION_CONFLICT = "configuration_conflict"
    UNSUPPORTED_ENVIRONMENT = "unsupported_environment"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: PreflightStatus
    detail: str


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str]], CommandResult]
"""`runner(["docker", "--version"]) -> CommandResult`. The default runner (`real_command_runner`)
shells out with a timeout; tests pass a fake that never touches the real machine."""


def _decode(raw: bytes) -> str:
    """`wsl.exe` (and some other native Windows console tools) writes UTF-16LE to redirected
    pipes regardless of console codepage. A UTF-8 decode of that output doesn't raise -- every
    ASCII byte is null-interleaved, which UTF-8 happily decodes as garbage embedded nulls -- so
    detect the interleave pattern before falling back to UTF-16LE."""
    if len(raw) >= 4 and raw[1::2].count(0) > len(raw) // 4:
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def real_command_runner(args: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
    try:
        completed = subprocess.run(
            list(args), capture_output=True, timeout=timeout, check=False
        )
        return CommandResult(returncode=completed.returncode, stdout=_decode(completed.stdout), stderr=_decode(completed.stderr))
    except FileNotFoundError:
        return CommandResult(returncode=127, stdout="", stderr="executable not found")
    except subprocess.TimeoutExpired:
        return CommandResult(returncode=124, stdout="", stderr="command timed out")


# -- Pure-Python checks (no subprocess, safe to always run) -------------------------------------


def check_windows_platform() -> PreflightCheck:
    if platform.system() != "Windows":
        return PreflightCheck("windows_platform", PreflightStatus.UNSUPPORTED_ENVIRONMENT, f"platform.system()={platform.system()!r}, not Windows")
    release, version, csd, ptype = platform.win32_ver()
    return PreflightCheck("windows_platform", PreflightStatus.OK, f"Windows {release} (version {version}, {ptype})")


def check_64_bit_architecture() -> PreflightCheck:
    architecture = platform.machine()
    is_64_bit = architecture.lower() in {"amd64", "x86_64", "arm64"}
    if not is_64_bit:
        return PreflightCheck("architecture_64_bit", PreflightStatus.UNSUPPORTED_ENVIRONMENT, f"machine()={architecture!r}, expected a 64-bit architecture")
    return PreflightCheck("architecture_64_bit", PreflightStatus.OK, architecture)


def check_administrator_state() -> PreflightCheck:
    if platform.system() != "Windows":
        return PreflightCheck("administrator_state", PreflightStatus.UNSUPPORTED_ENVIRONMENT, "not Windows")
    try:
        import ctypes

        is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception as error:  # pragma: no cover - platform-dependent
        return PreflightCheck("administrator_state", PreflightStatus.UNSUPPORTED_ENVIRONMENT, f"could not determine elevation: {error}")
    detail = "running elevated (Administrator)" if is_admin else "running as a standard (non-elevated) user"
    return PreflightCheck("administrator_state", PreflightStatus.OK, detail)


def check_disk_space(path: Path, *, minimum_free_gb: float = 20.0) -> PreflightCheck:
    probe = path.resolve()
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return PreflightCheck("disk_space", PreflightStatus.MISSING_PREREQUISITE, f"no existing ancestor found for {path}")
        probe = parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as error:
        return PreflightCheck("disk_space", PreflightStatus.MISSING_PREREQUISITE, f"cannot inspect {probe}: {error}")
    free_gb = usage.free / (1024**3)
    if free_gb < minimum_free_gb:
        return PreflightCheck("disk_space", PreflightStatus.CONFIGURATION_CONFLICT, f"{free_gb:.1f} GiB free at {probe} (nearest existing ancestor of {path}), need at least {minimum_free_gb:.0f} GiB")
    return PreflightCheck("disk_space", PreflightStatus.OK, f"{free_gb:.1f} GiB free at {probe} (nearest existing ancestor of {path})")


def check_total_ram(*, minimum_gb: float = 8.0) -> PreflightCheck:
    if platform.system() != "Windows":
        return PreflightCheck("total_ram", PreflightStatus.UNSUPPORTED_ENVIRONMENT, "not Windows")
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            raise OSError("GlobalMemoryStatusEx failed")
        total_gb = status.ullTotalPhys / (1024**3)
    except Exception as error:  # pragma: no cover - platform-dependent
        return PreflightCheck("total_ram", PreflightStatus.UNSUPPORTED_ENVIRONMENT, f"could not determine total RAM: {error}")
    if total_gb < minimum_gb:
        return PreflightCheck("total_ram", PreflightStatus.CONFIGURATION_CONFLICT, f"{total_gb:.1f} GiB total RAM, recommend at least {minimum_gb:.0f} GiB")
    return PreflightCheck("total_ram", PreflightStatus.OK, f"{total_gb:.1f} GiB total RAM")


def check_python_version(*, minimum: tuple[int, int] = (3, 11)) -> PreflightCheck:
    current = sys.version_info[:2]
    if current < minimum:
        return PreflightCheck("python_version", PreflightStatus.INCOMPATIBLE_VERSION, f"Python {'.'.join(map(str, current))}, need >= {'.'.join(map(str, minimum))}")
    return PreflightCheck("python_version", PreflightStatus.OK, f"Python {platform.python_version()}")


def check_port_available(port: int, *, host: str = "127.0.0.1") -> PreflightCheck:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.settimeout(0.5)
        result = probe.connect_ex((host, port))
    if result == 0:
        return PreflightCheck(f"port_{port}", PreflightStatus.PORT_CONFLICT, f"port {port} on {host} is already in use")
    return PreflightCheck(f"port_{port}", PreflightStatus.OK, f"port {port} on {host} is free")


def check_installation_path(path: Path) -> PreflightCheck:
    if is_dangerous_root(path):
        return PreflightCheck("installation_path", PreflightStatus.CONFIGURATION_CONFLICT, f"{path} is a dangerous root (drive root, Windows dir, or user profile root)")
    if path.exists() and not path.is_dir():
        return PreflightCheck("installation_path", PreflightStatus.CONFIGURATION_CONFLICT, f"{path} exists and is not a directory")
    return PreflightCheck("installation_path", PreflightStatus.OK, str(path))


def check_env_file_safety(env_path: Path) -> PreflightCheck:
    if not env_path.exists():
        return PreflightCheck("env_file_safety", PreflightStatus.OK, f"no {env_path.name} present")
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return PreflightCheck("env_file_safety", PreflightStatus.CONFIGURATION_CONFLICT, f"cannot read {env_path}: {error}")
    unsafe_markers = ("PASSWORD=changeme", "SECRET=changeme", "PASSWORD=password", "TOKEN=test")
    hit = next((marker for marker in unsafe_markers if marker in text), None)
    if hit:
        return PreflightCheck("env_file_safety", PreflightStatus.CONFIGURATION_CONFLICT, f"{env_path.name} contains an unsafe default value ({hit.split('=')[0]})")
    return PreflightCheck("env_file_safety", PreflightStatus.OK, f"{env_path.name} present, no known-unsafe defaults found")


# -- Checks requiring an injected command runner -------------------------------------------------


def check_git_present(runner: CommandRunner) -> PreflightCheck:
    result = runner(["git", "--version"])
    if result.returncode == 127:
        return PreflightCheck("git_present", PreflightStatus.MISSING_PREREQUISITE, "git not found on PATH")
    if result.returncode != 0:
        return PreflightCheck("git_present", PreflightStatus.INSTALLED_UNHEALTHY, result.stderr.strip() or "git --version failed")
    return PreflightCheck("git_present", PreflightStatus.OK, result.stdout.strip())


def check_docker_desktop_present(runner: CommandRunner) -> PreflightCheck:
    result = runner(["docker", "--version"])
    if result.returncode == 127:
        return PreflightCheck("docker_desktop_present", PreflightStatus.MISSING_PREREQUISITE, "docker not found on PATH")
    if result.returncode != 0:
        return PreflightCheck("docker_desktop_present", PreflightStatus.INSTALLED_UNHEALTHY, result.stderr.strip() or "docker --version failed")
    return PreflightCheck("docker_desktop_present", PreflightStatus.OK, result.stdout.strip())


def check_docker_daemon_running(runner: CommandRunner) -> PreflightCheck:
    result = runner(["docker", "info"])
    if result.returncode == 127:
        return PreflightCheck("docker_daemon_running", PreflightStatus.MISSING_PREREQUISITE, "docker not found on PATH")
    if result.returncode != 0:
        return PreflightCheck("docker_daemon_running", PreflightStatus.INSTALLED_NOT_RUNNING, result.stderr.strip() or "docker info failed -- Docker Desktop is installed but not running")
    return PreflightCheck("docker_daemon_running", PreflightStatus.OK, "docker daemon reachable")


def check_docker_compose_available(runner: CommandRunner) -> PreflightCheck:
    result = runner(["docker", "compose", "version"])
    if result.returncode == 127:
        return PreflightCheck("docker_compose_available", PreflightStatus.MISSING_PREREQUISITE, "docker compose not found")
    if result.returncode != 0:
        return PreflightCheck("docker_compose_available", PreflightStatus.INSTALLED_UNHEALTHY, result.stderr.strip() or "docker compose version failed")
    return PreflightCheck("docker_compose_available", PreflightStatus.OK, result.stdout.strip())


def check_wsl2_available(runner: CommandRunner) -> PreflightCheck:
    result = runner(["wsl", "--status"])
    if result.returncode == 127:
        return PreflightCheck("wsl2_available", PreflightStatus.MISSING_PREREQUISITE, "wsl not found on PATH")
    if result.returncode != 0:
        return PreflightCheck("wsl2_available", PreflightStatus.INSTALLED_UNHEALTHY, result.stderr.strip() or "wsl --status failed")
    return PreflightCheck("wsl2_available", PreflightStatus.OK, result.stdout.strip() or "wsl reports OK")


def check_gpu_driver(runner: CommandRunner) -> PreflightCheck:
    result = runner(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if result.returncode == 127:
        return PreflightCheck("gpu_driver", PreflightStatus.MISSING_PREREQUISITE, "nvidia-smi not found -- no NVIDIA GPU/driver detected; CPU-fallback providers only")
    if result.returncode != 0:
        return PreflightCheck("gpu_driver", PreflightStatus.INSTALLED_UNHEALTHY, result.stderr.strip() or "nvidia-smi failed")
    return PreflightCheck("gpu_driver", PreflightStatus.OK, result.stdout.strip())


def run_all(
    *,
    installation_path: Path,
    deployment_root: Path,
    env_path: Path,
    ports: Sequence[int] = (),
    runner: CommandRunner = real_command_runner,
) -> tuple[PreflightCheck, ...]:
    """Runs every check this module implements and returns them in a stable order. Does not
    raise on individual check failure -- callers (the CLI, the PowerShell entry point) decide
    what to do with a non-`OK` result; this function only observes."""
    checks = [
        check_windows_platform(),
        check_64_bit_architecture(),
        check_administrator_state(),
        check_python_version(),
        check_disk_space(installation_path),
        check_total_ram(),
        check_installation_path(installation_path),
        check_installation_path(deployment_root),
        check_env_file_safety(env_path),
        check_git_present(runner),
        check_docker_desktop_present(runner),
        check_docker_daemon_running(runner),
        check_docker_compose_available(runner),
        check_wsl2_available(runner),
        check_gpu_driver(runner),
    ]
    checks.extend(check_port_available(port) for port in ports)
    return tuple(checks)
