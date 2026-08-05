from __future__ import annotations

import csv

import pytest

from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner, _parse_latest_metrics
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS


@pytest.fixture()
def runner():
    return ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001,
    )


@pytest.fixture()
def paths(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "output"
    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()
    train_list = lists_dir / "train_list.txt"
    train_list.write_text("", encoding="utf-8")
    val_list = lists_dir / "val_list.txt"
    val_list.write_text("", encoding="utf-8")
    return {
        "existing_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "train_list_path": str(train_list),
        "validation_list_path": str(val_list),
    }


def test_never_invokes_wsl_exe(runner, paths):
    """A real bug this integration hit: `wsl.exe -- docker ...` is explicitly refused by Docker
    Desktop's own WSL2 backend distro. The fix was to call `docker` directly."""
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert argv[0] == "docker"
    assert "wsl.exe" not in argv


def test_entrypoint_is_main_py_not_src_main_py(runner, paths):
    """The container's own WORKDIR is already `/src/loghi-htr/src` -- `src/main.py` doubles the
    segment and fails with `can't open file '/src/loghi-htr/src/src/main.py'`."""
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert "main.py" in argv
    assert "src/main.py" not in argv


def test_gpus_flag_is_all_not_a_bare_device_index(runner, paths):
    """`--gpus 0` is Docker CLI syntax for "allocate zero GPUs" (a count), not "use device 0" -- a
    real silent-failure bug this integration caught (TensorFlow fell back to CPU with no error)."""
    argv = runner._build_argv(epoch_seed=1, **paths)
    gpus_index = argv.index("--gpus")
    assert argv[gpus_index + 1] == "all"


def test_model_mount_is_read_write_not_read_only(runner, paths):
    """`Tokenizer.load_from_file` writes a converted `tokenizer.json` back into the model directory
    on load -- a `:ro` mount crashes with `Read-only file system`."""
    argv = runner._build_argv(epoch_seed=1, **paths)
    model_mount = next(a for a in argv if a.endswith(":/model") or ":/model:" in a)
    assert not model_mount.endswith(":/model:ro")


def test_uses_the_model_flag_never_existing_model(runner, paths):
    """The pinned `loghi-htr` commit's real `arg_parser.py` defines no `--existing_model` flag at
    all -- passing it would silently be ignored and training would start from random init."""
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert "--model" in argv
    assert "--existing_model" not in argv
    model_index = argv.index("--model")
    assert argv[model_index + 1] == "/model"


def test_never_passes_a_vgsl_spec_string_as_the_model_argument(runner, paths):
    """Passing a checkpoint *directory* (not a VGSL spec) for `--model` is what makes
    `model/management.py::load_or_create_model` load rather than create-from-scratch -- structurally
    enforcing "do not train from random initialization"."""
    argv = runner._build_argv(epoch_seed=1, **paths)
    model_index = argv.index("--model")
    assert argv[model_index + 1] == "/model"  # a mount point, not something like "new10" or a spec


def test_batch_size_and_learning_rate_come_from_the_configured_runner(paths):
    runner = ContainerEpochRunner(
        batch_size=4, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="sgd", learning_rate=0.005,
    )
    argv = runner._build_argv(epoch_seed=7, **paths)
    assert argv[argv.index("--batch_size") + 1] == "4"
    assert argv[argv.index("--learning_rate") + 1] == "0.005"
    assert argv[argv.index("--optimizer") + 1] == "sgd"
    assert argv[argv.index("--seed") + 1] == "7"


def test_epochs_defaults_to_one_container_invocation_is_one_epoch(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert argv[argv.index("--epochs") + 1] == "1"


def test_epochs_per_invocation_is_configurable(paths):
    """`--epochs 2` (or more) in ONE container invocation keeps the same Keras optimizer object alive
    across every one of those real epochs -- `model.fit(epochs=N)` never rebuilds it between them --
    a strictly stronger continuity guarantee than chaining N separate single-epoch invocations."""
    runner = ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001, epochs_per_invocation=2,
    )
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert argv[argv.index("--epochs") + 1] == "2"


def test_latest_checkpoint_is_the_highest_epoch_number_not_glob_order(runner, paths, tmp_path, monkeypatch):
    """A real, previously-latent bug: with only ever one epoch_N dir per invocation (every caller
    before epochs_per_invocation existed), whichever non-best_val match `rglob` visited last was
    always the only match, so iteration order never mattered. With 2+ real epochs in one invocation,
    multiple epoch_N dirs can exist, and filesystem enumeration order does not reliably follow epoch
    order. Deliberately creates them so the numerically-earlier epoch is the LAST one `Path.rglob`
    actually visits (directory created last), and asserts the real `run_epoch()` still picks the true
    latest by parsed epoch number, not by visit order."""
    import zipfile

    import archivetrust.htr.training.container_epoch_runner as cer_mod
    from archivetrust.providers.loghi.environment import LoghiEnvironmentReport

    monkeypatch.setattr(
        cer_mod, "probe_loghi_environment",
        lambda: LoghiEnvironmentReport(
            host_os="Windows", docker_cli_present=True, docker_version="1", wsl_present=True,
            wsl_distros=(), linux_distribution=None, nvidia_toolkit_version=None,
            cuda_visible_devices=None, execution_mode=None,
        ),
    )

    output_dir = tmp_path / "output"

    def _fake_run(*args, **kwargs):
        # Simulate what a real --epochs 2 invocation produces: two epoch_N checkpoint dirs. Created
        # in an order where the numerically-EARLIER epoch is created (and thus glob-visited) LAST.
        model_root = output_dir / "model_name"
        for name in ("epoch_10_CER_0.10_val_0.10", "epoch_9_CER_0.20_val_0.20"):
            d = model_root / name
            d.mkdir(parents=True)
            with zipfile.ZipFile(d / "model.keras", "w") as zf:
                zf.writestr("config.json", "{}")

        class _FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        return _FakeCompleted()

    monkeypatch.setattr(cer_mod.subprocess, "run", _fake_run)

    result = runner.run_epoch(
        existing_model_dir=paths["existing_model_dir"], output_dir=str(output_dir),
        train_list_path=paths["train_list_path"], validation_list_path=paths["validation_list_path"],
        epoch_seed=1,
    )
    assert result.ok
    assert "epoch_10_" in result.checkpoint_dir
    assert "epoch_9_" not in result.checkpoint_dir


def test_float32_flag_only_appears_for_float32_precision(paths):
    fp32_runner = ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="float32", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001,
    )
    mixed_runner = ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001,
    )
    assert "--use_float32" in fp32_runner._build_argv(epoch_seed=1, **paths)
    assert "--use_float32" not in mixed_runner._build_argv(epoch_seed=1, **paths)


def test_train_and_validation_lists_are_mounted_container_relative_not_host_paths(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    train_list_arg = argv[argv.index("--train_list") + 1]
    val_list_arg = argv[argv.index("--validation_list") + 1]
    assert train_list_arg == "/lists/train_list.txt"
    assert val_list_arg == "/lists/val_list.txt"
    assert "C:" not in train_list_arg and "C:" not in val_list_arg


def test_image_reference_uses_the_real_pinned_digest_not_a_floating_tag(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    image_ref = argv[argv.index("--entrypoint") + 2]
    assert CURRENT_PINNED_VERSIONS.docker_image_digest in image_ref
    assert "latest" not in image_ref or "@sha256:" in image_ref


def test_do_validate_flag_always_present(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert "--do_validate" in argv


def test_output_checkpoints_flag_always_present(runner, paths):
    argv = runner._build_argv(epoch_seed=1, **paths)
    assert "--output_checkpoints" in argv


def test_parse_latest_metrics_reads_the_most_recent_row(tmp_path):
    log_path = tmp_path / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "CER_metric", "val_CER_metric", "WER_metric", "val_WER_metric", "loss", "val_loss"]
        )
        writer.writeheader()
        writer.writerow({"epoch": 0, "CER_metric": 0.5, "val_CER_metric": 0.6, "WER_metric": 0.7, "val_WER_metric": 0.8, "loss": 40.0, "val_loss": 45.0})
        writer.writerow({"epoch": 1, "CER_metric": 0.3, "val_CER_metric": 0.35, "WER_metric": 0.4, "val_WER_metric": 0.45, "loss": 30.0, "val_loss": 32.0})

    train_cer, val_cer, train_wer, val_wer, train_loss, val_loss = _parse_latest_metrics(tmp_path)
    assert train_cer == 0.3
    assert val_cer == 0.35
    assert train_wer == 0.4
    assert val_wer == 0.45
    assert train_loss == 30.0
    assert val_loss == 32.0


def test_parse_latest_metrics_returns_none_when_no_log_present(tmp_path):
    assert _parse_latest_metrics(tmp_path) == (None, None, None, None, None, None)


def test_parse_latest_metrics_never_fabricates_a_zero_for_a_missing_column(tmp_path):
    log_path = tmp_path / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "CER_metric"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "CER_metric": 0.5})

    train_cer, val_cer, train_wer, val_wer, train_loss, val_loss = _parse_latest_metrics(tmp_path)
    assert train_cer == 0.5
    assert val_cer is None
    assert train_wer is None
    assert val_wer is None
    assert train_loss is None
    assert val_loss is None


def test_no_telemetry_sampler_runs_when_run_state_dir_is_not_configured(tmp_path, monkeypatch):
    """The default -- no `run_state_dir` passed -- must not create a `telemetry/` directory
    anywhere, so a runner constructed the way every other test in this file constructs it (no
    GPU/psutil dependency) genuinely never spawns a sampler thread."""
    import archivetrust.htr.training.container_epoch_runner as cer_mod
    from archivetrust.providers.loghi.environment import LoghiEnvironmentReport

    monkeypatch.setattr(
        cer_mod, "probe_loghi_environment",
        lambda: LoghiEnvironmentReport(
            host_os="Windows", docker_cli_present=True, docker_version="1", wsl_present=True,
            wsl_distros=(), linux_distribution=None, nvidia_toolkit_version=None,
            cuda_visible_devices=None, execution_mode=None,
        ),
    )

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(cer_mod.subprocess, "run", lambda *a, **k: _FakeCompleted())

    runner = ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001,
    )
    output_dir = tmp_path / "run_state" / "epoch_output" / "epoch_1"
    runner.run_epoch(
        existing_model_dir=str(tmp_path / "model"), output_dir=str(output_dir),
        train_list_path=str(tmp_path / "train.txt"), validation_list_path=str(tmp_path / "val.txt"),
        epoch_seed=1,
    )
    assert not (tmp_path / "run_state" / "telemetry").exists()


def test_telemetry_sampler_runs_when_run_state_dir_is_configured(tmp_path, monkeypatch):
    import archivetrust.htr.training.container_epoch_runner as cer_mod
    from archivetrust.providers.loghi.environment import LoghiEnvironmentReport

    monkeypatch.setattr(
        cer_mod, "probe_loghi_environment",
        lambda: LoghiEnvironmentReport(
            host_os="Windows", docker_cli_present=True, docker_version="1", wsl_present=True,
            wsl_distros=(), linux_distribution=None, nvidia_toolkit_version=None,
            cuda_visible_devices=None, execution_mode=None,
        ),
    )
    import archivetrust.htr.training.telemetry_sampler as ts_mod
    monkeypatch.setattr(ts_mod.shutil, "which", lambda name: None)  # no real nvidia-smi call needed

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(cer_mod.subprocess, "run", lambda *a, **k: _FakeCompleted())

    run_state_dir = tmp_path / "run_state"
    runner = ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16", max_image_width=2000,
        optimizer="adam", learning_rate=0.0001, run_state_dir=run_state_dir,
    )
    output_dir = run_state_dir / "epoch_output" / "epoch_5"
    runner.run_epoch(
        existing_model_dir=str(tmp_path / "model"), output_dir=str(output_dir),
        train_list_path=str(tmp_path / "train.txt"), validation_list_path=str(tmp_path / "val.txt"),
        epoch_seed=1,
    )
    status_path = run_state_dir / "telemetry" / "status.json"
    assert status_path.exists()

    import json

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["epoch"] == 5  # parsed from the "epoch_5" output_dir leaf
