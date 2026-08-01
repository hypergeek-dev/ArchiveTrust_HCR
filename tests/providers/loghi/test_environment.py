"""`environment.py` -- read-only probes. Tests exercise the parsing/decision logic with monkeypatched
subprocess output rather than real `docker`/`wsl.exe` calls, so they run everywhere regardless of what
is actually installed."""

from __future__ import annotations

import subprocess

from archivetrust.providers.loghi import environment
from archivetrust.providers.loghi.models import LoghiExecutionMode


class _FakeCompleted:
    def __init__(self, *, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_decodes_utf16le_wsl_output_without_embedded_nulls(monkeypatch) -> None:
    """Regression test for the real bug hit while building this module: `wsl.exe` prints UTF-16LE on
    Windows; naive `text=True` capture corrupts it with a stray `\\x00` after every character, which
    then breaks a *second* subprocess call built from the corrupted string."""
    utf16_output = "docker-desktop    Stopped    2\n".encode("utf-16-le")

    def fake_run(args, **kwargs):
        assert kwargs.get("text") is False
        return _FakeCompleted(returncode=0, stdout=utf16_output)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, output = environment._run(["wsl.exe", "-l", "-v"])
    assert ok is True
    assert "\x00" not in output
    assert "docker-desktop" in output


def test_run_falls_back_to_utf8_for_normal_output(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        return _FakeCompleted(returncode=0, stdout=b"Docker version 29.6.1, build abc123\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, output = environment._run(["docker", "--version"])
    assert ok is True
    assert "Docker version 29.6.1" in output


def test_run_reports_missing_executable_without_raising(monkeypatch) -> None:
    def fake_run(args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, output = environment._run(["nonexistent-tool"])
    assert ok is False
    assert output == ""


def test_probe_resolves_docker_wsl2_mode_when_windows_docker_and_distro_present(monkeypatch) -> None:
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment, "_docker_status", lambda: (True, "Docker version 29.6.1"))
    monkeypatch.setattr(environment, "_wsl_status", lambda: (True, ("docker-desktop",)))
    monkeypatch.setattr(environment, "_nvidia_toolkit_version", lambda: None)
    monkeypatch.setattr(environment, "_linux_distribution_via_wsl", lambda distro: "Docker Desktop")

    report = environment.probe_loghi_environment()
    assert report.execution_mode is LoghiExecutionMode.DOCKER_WSL2
    assert report.docker_cli_present is True
    assert report.wsl_distros == ("docker-desktop",)


def test_probe_reports_no_execution_mode_when_docker_absent(monkeypatch) -> None:
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment, "_docker_status", lambda: (False, None))
    monkeypatch.setattr(environment, "_wsl_status", lambda: (False, ()))
    monkeypatch.setattr(environment, "_nvidia_toolkit_version", lambda: None)

    report = environment.probe_loghi_environment()
    assert report.execution_mode is None
    assert report.docker_cli_present is False


def test_probe_resolves_docker_linux_mode_when_docker_present(monkeypatch) -> None:
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment, "_docker_status", lambda: (True, "Docker version 29.6.1"))
    monkeypatch.setattr(environment, "_wsl_status", lambda: (False, ()))
    monkeypatch.setattr(environment, "_nvidia_toolkit_version", lambda: "1.15.0")
    monkeypatch.setattr(environment, "_linux_distribution_native", lambda: "Ubuntu 22.04")

    report = environment.probe_loghi_environment()
    assert report.execution_mode is LoghiExecutionMode.DOCKER_LINUX
    assert report.linux_distribution == "Ubuntu 22.04"


def test_probe_resolves_native_linux_mode_when_docker_absent(monkeypatch) -> None:
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment, "_docker_status", lambda: (False, None))
    monkeypatch.setattr(environment, "_wsl_status", lambda: (False, ()))
    monkeypatch.setattr(environment, "_nvidia_toolkit_version", lambda: None)
    monkeypatch.setattr(environment, "_linux_distribution_native", lambda: "Ubuntu 22.04")

    report = environment.probe_loghi_environment()
    assert report.execution_mode is LoghiExecutionMode.NATIVE_LINUX
    assert report.linux_distribution == "Ubuntu 22.04"
