from __future__ import annotations

import pytest

from archivetrust.htr.training.scratch_epoch_runner import (
    RECOMMENDED_VGSL_SPEC,
    PretrainedCheckpointRejected,
    ScratchEpochRunner,
)
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS


@pytest.fixture()
def runner():
    return ScratchEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001,
    )


@pytest.fixture()
def paths(tmp_path):
    output_dir = tmp_path / "output"
    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()
    train_list = lists_dir / "train_list.txt"
    train_list.write_text("", encoding="utf-8")
    val_list = lists_dir / "val_list.txt"
    val_list.write_text("", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "train_list_path": str(train_list),
        "validation_list_path": str(val_list),
    }


# --- Test 1: Experiment 2 rejects pretrained checkpoints ---
def test_run_epoch_rejects_a_pretrained_checkpoint_path(runner, paths, tmp_path):
    with pytest.raises(PretrainedCheckpointRejected):
        runner.run_epoch(existing_model_dir=str(tmp_path / "some_checkpoint"), epoch_seed=1, **paths)


def test_run_epoch_accepts_none_existing_model_dir_without_raising_the_guard(runner, paths, monkeypatch):
    """The guard fires only for a non-None value -- None is the expected scratch-training input."""
    import archivetrust.htr.training.scratch_epoch_runner as ser_mod
    from archivetrust.providers.loghi.environment import LoghiEnvironmentReport

    monkeypatch.setattr(
        ser_mod, "probe_loghi_environment",
        lambda: LoghiEnvironmentReport(
            host_os="Windows", docker_cli_present=False, docker_version=None, wsl_present=False,
            wsl_distros=(), linux_distribution=None, nvidia_toolkit_version=None,
            cuda_visible_devices=None, execution_mode=None,
        ),
    )
    result = runner.run_epoch(existing_model_dir=None, epoch_seed=1, **paths)
    # Docker unavailable is a separate, honest failure -- but it's NOT PretrainedCheckpointRejected,
    # proving the guard itself did not trip for None.
    assert not result.ok
    assert "Docker CLI not available" in (result.error_message or "")


# --- Test 2: random initialization is real (no /model mount, --model recommended literally) ---
def test_never_mounts_a_model_directory(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert not any(":/model" in a for a in argv)


def test_model_flag_is_the_literal_library_key_never_a_path(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    model_index = argv.index("--model")
    assert argv[model_index + 1] == "recommended"
    assert "/model" not in argv[model_index + 1]
    assert "--existing_model" not in argv


def test_recommended_vgsl_spec_is_the_real_pinned_library_value():
    """Cross-checked against management.py::get_model_library() directly, not retyped from memory --
    see scratch_epoch_runner.py's own module docstring for how this was verified."""
    assert RECOMMENDED_VGSL_SPEC.startswith("None,None,64,1")
    assert "Fs92" in RECOMMENDED_VGSL_SPEC  # the literal spec string; real vocab size is applied later


def test_epochs_is_always_exactly_one(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert argv[argv.index("--epochs") + 1] == "1"


def test_image_reference_uses_the_real_pinned_digest(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    image_ref = argv[argv.index("--entrypoint") + 2]
    assert CURRENT_PINNED_VERSIONS.docker_image_digest in image_ref


def test_do_validate_and_output_checkpoints_flags_present(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert "--do_validate" in argv
    assert "--output_checkpoints" in argv


def test_extra_volume_mounts_are_appended(paths, tmp_path):
    override_file = tmp_path / "override.py"
    override_file.write_text("", encoding="utf-8")
    runner = ScratchEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001,
        extra_volume_mounts=((str(override_file), "/src/loghi-htr/src/modes/training.py"),),
    )
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert any("/src/loghi-htr/src/modes/training.py" in a for a in argv)
