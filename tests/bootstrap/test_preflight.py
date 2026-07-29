from __future__ import annotations

from archivetrust.bootstrap.preflight import (
    CommandResult,
    PreflightStatus,
    check_disk_space,
    check_docker_daemon_running,
    check_docker_desktop_present,
    check_env_file_safety,
    check_git_present,
    check_installation_path,
    check_port_available,
    check_python_version,
    check_wsl2_available,
    run_all,
)


def fake_runner(responses: dict[tuple, CommandResult]):
    def runner(args):
        key = tuple(args)
        if key in responses:
            return responses[key]
        for prefix, result in responses.items():
            if key[: len(prefix)] == prefix:
                return result
        return CommandResult(returncode=127, stdout="", stderr="unmapped command in fake runner")

    return runner


def test_docker_missing_prerequisite() -> None:
    runner = fake_runner({})
    check = check_docker_desktop_present(runner)
    assert check.status is PreflightStatus.MISSING_PREREQUISITE


def test_docker_present_but_daemon_not_running() -> None:
    runner = fake_runner(
        {
            ("docker", "--version"): CommandResult(0, "Docker version 27.0.0", ""),
            ("docker", "info"): CommandResult(1, "", "error during connect: this error may indicate the docker daemon is not running"),
        }
    )
    present = check_docker_desktop_present(runner)
    assert present.status is PreflightStatus.OK
    running = check_docker_daemon_running(runner)
    assert running.status is PreflightStatus.INSTALLED_NOT_RUNNING


def test_git_present_ok() -> None:
    runner = fake_runner({("git", "--version"): CommandResult(0, "git version 2.44.0", "")})
    check = check_git_present(runner)
    assert check.status is PreflightStatus.OK


def test_git_unhealthy_when_nonzero_exit() -> None:
    runner = fake_runner({("git", "--version"): CommandResult(1, "", "some internal git error")})
    check = check_git_present(runner)
    assert check.status is PreflightStatus.INSTALLED_UNHEALTHY


def test_wsl_unavailable() -> None:
    runner = fake_runner({})
    check = check_wsl2_available(runner)
    assert check.status is PreflightStatus.MISSING_PREREQUISITE


def test_disk_space_reports_ok_for_roomy_path(tmp_path) -> None:
    check = check_disk_space(tmp_path, minimum_free_gb=0.001)
    assert check.status is PreflightStatus.OK


def test_disk_space_walks_up_to_existing_ancestor_for_missing_path(tmp_path) -> None:
    missing = tmp_path / "does" / "not" / "exist" / "yet"
    check = check_disk_space(missing, minimum_free_gb=0.001)
    assert check.status is PreflightStatus.OK
    assert str(tmp_path) in check.detail


def test_disk_space_flags_insufficient_space(tmp_path) -> None:
    check = check_disk_space(tmp_path, minimum_free_gb=10_000_000)
    assert check.status is PreflightStatus.CONFIGURATION_CONFLICT


def test_python_version_ok_for_current_interpreter() -> None:
    check = check_python_version(minimum=(3, 8))
    assert check.status is PreflightStatus.OK


def test_python_version_incompatible_when_minimum_too_high() -> None:
    check = check_python_version(minimum=(99, 0))
    assert check.status is PreflightStatus.INCOMPATIBLE_VERSION


def test_port_available_reports_ok_for_unused_high_port() -> None:
    check = check_port_available(58211)
    assert check.status is PreflightStatus.OK


def test_port_conflict_detected_for_bound_port() -> None:
    import contextlib
    import socket

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        check = check_port_available(port)
    assert check.status is PreflightStatus.PORT_CONFLICT


def test_installation_path_rejects_dangerous_root() -> None:
    from pathlib import Path

    check = check_installation_path(Path("C:\\"))
    assert check.status is PreflightStatus.CONFIGURATION_CONFLICT


def test_installation_path_ok_for_normal_path(tmp_path) -> None:
    check = check_installation_path(tmp_path / "archivetrust_data")
    assert check.status is PreflightStatus.OK


def test_env_file_safety_ok_when_absent(tmp_path) -> None:
    check = check_env_file_safety(tmp_path / ".env")
    assert check.status is PreflightStatus.OK


def test_env_file_safety_flags_unsafe_default(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ADMIN_PASSWORD=changeme\n", encoding="utf-8")
    check = check_env_file_safety(env_path)
    assert check.status is PreflightStatus.CONFIGURATION_CONFLICT


def test_run_all_returns_a_check_per_requested_port(tmp_path) -> None:
    runner = fake_runner(
        {
            ("git", "--version"): CommandResult(0, "git version 2.44.0", ""),
            ("docker", "--version"): CommandResult(0, "Docker version 27.0.0", ""),
            ("docker", "info"): CommandResult(0, "", ""),
            ("docker", "compose", "version"): CommandResult(0, "Docker Compose version v2.0", ""),
            ("wsl", "--status"): CommandResult(0, "Default Version: 2", ""),
            ("nvidia-smi",): CommandResult(127, "", ""),
        }
    )
    checks = run_all(
        installation_path=tmp_path,
        deployment_root=tmp_path,
        env_path=tmp_path / ".env",
        ports=(58212, 58213),
        runner=runner,
    )
    port_checks = [c for c in checks if c.name.startswith("port_")]
    assert {c.name for c in port_checks} == {"port_58212", "port_58213"}
