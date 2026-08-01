"""`facade.py` -- deterministic argv construction and Windows->WSL path translation. No real Docker
subprocess is invoked; `_build_docker_argv`/`_to_wsl_path` are pure functions tested directly."""

from __future__ import annotations

from archivetrust.providers.loghi.facade import _build_docker_argv, _to_wsl_path
from archivetrust.providers.loghi.models import (
    LoghiComponentVersions,
    LoghiEnvironmentReport,
    LoghiExecutionMode,
)

_PINS = LoghiComponentVersions(
    loghi_repo_commit="a" * 40,
    submodule_commits={},
    laypa_commit="b" * 40,
    loghi_tooling_commit="c" * 40,
    loghi_htr_commit="d" * 40,
    model_checkpoint_id="dutch-v1",
    model_checkpoint_hash="e" * 64,
    docker_image_tag="ghcr.io/knaw-huc/loghi:v2",
    docker_image_digest="sha256:" + "f" * 64,
    inference_script_version="1.0",
    beam_width=3,
    reading_order_settings="default",
    language_detection_settings="nl",
    gpu_selection="all",
    container_runtime_version="29.6.1",
)

_WSL2_REPORT = LoghiEnvironmentReport(
    host_os="Windows", docker_cli_present=True, docker_version="x", wsl_present=True,
    wsl_distros=("docker-desktop",), linux_distribution="Docker Desktop", nvidia_toolkit_version=None,
    cuda_visible_devices=None, execution_mode=LoghiExecutionMode.DOCKER_WSL2,
)

_LINUX_REPORT = LoghiEnvironmentReport(
    host_os="Linux", docker_cli_present=True, docker_version="x", wsl_present=False,
    wsl_distros=(), linux_distribution="Ubuntu", nvidia_toolkit_version=None,
    cuda_visible_devices=None, execution_mode=LoghiExecutionMode.DOCKER_LINUX,
)


def test_to_wsl_path_translates_windows_drive_path() -> None:
    result = _to_wsl_path(r"C:\Users\researcher\input.jpg")
    assert result.startswith("/mnt/c/")
    assert "Users/researcher/input.jpg" in result


def test_build_docker_argv_uses_digest_when_present() -> None:
    argv = _build_docker_argv(
        input_image_path="C:\\tmp\\in", output_dir="C:\\tmp\\out",
        component_versions=_PINS, environment=_WSL2_REPORT,
    )
    joined = " ".join(argv)
    assert "ghcr.io/knaw-huc/loghi:v2@sha256:" in joined
    assert "--rm" in argv
    assert "docker" in argv and "run" in argv


def test_build_docker_argv_prefixes_wsl_exe_under_wsl2_mode() -> None:
    argv = _build_docker_argv(
        input_image_path="C:\\tmp\\in", output_dir="C:\\tmp\\out",
        component_versions=_PINS, environment=_WSL2_REPORT,
    )
    assert argv[0] == "wsl.exe"
    assert argv[1] == "--"
    # mounted paths are WSL-translated, not raw Windows paths
    mount_flag_index = argv.index("-v")
    assert "/mnt/c/" in argv[mount_flag_index + 1]


def test_build_docker_argv_no_wsl_prefix_under_native_linux() -> None:
    argv = _build_docker_argv(
        input_image_path="/tmp/in", output_dir="/tmp/out",
        component_versions=_PINS, environment=_LINUX_REPORT,
    )
    assert argv[0] == "docker"
    assert "wsl.exe" not in argv


def test_build_docker_argv_omits_gpus_flag_when_selection_is_none() -> None:
    pins = _PINS.model_copy(update={"gpu_selection": "none"})
    argv = _build_docker_argv(
        input_image_path="/tmp/in", output_dir="/tmp/out", component_versions=pins, environment=_LINUX_REPORT,
    )
    assert "--gpus" not in argv


def test_build_docker_argv_includes_beam_width_and_reading_order() -> None:
    argv = _build_docker_argv(
        input_image_path="/tmp/in", output_dir="/tmp/out", component_versions=_PINS, environment=_LINUX_REPORT,
    )
    assert "--beam-width" in argv
    assert argv[argv.index("--beam-width") + 1] == "3"
    assert "--reading-order" in argv
    assert argv[argv.index("--reading-order") + 1] == "default"


def test_is_placeholder_true_for_an_explicit_placeholder_instance() -> None:
    from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS, PLACEHOLDER_SENTINEL

    placeholder_pins = CURRENT_PINNED_VERSIONS.model_copy(
        update={"loghi_repo_commit": PLACEHOLDER_SENTINEL}
    )
    assert placeholder_pins.is_placeholder() is True


def test_is_placeholder_false_for_current_pinned_versions() -> None:
    """`CURRENT_PINNED_VERSIONS` is a real, resolved environment as of 2026-08-01 (the Swedish
    fine-tuning environment install -- docs/methods/loghi-swedish-finetuning.md), not the placeholder
    it used to default to."""
    from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

    assert CURRENT_PINNED_VERSIONS.is_placeholder() is False


def test_is_placeholder_false_for_real_pins() -> None:
    assert _PINS.is_placeholder() is False
