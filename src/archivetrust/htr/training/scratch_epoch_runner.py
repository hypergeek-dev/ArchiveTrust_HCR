"""`ScratchEpochRunner`: the from-scratch counterpart to `container_epoch_runner.py::ContainerEpochRunner`.

That module's own docstring says, correctly and deliberately: "'do not train from random
initialization' is enforced structurally" by always mounting a real checkpoint directory at `/model`
and never passing a bare VGSL spec string. Experiment 2 needs exactly the opposite guarantee --
never mount a pretrained checkpoint, always pass `--model recommended` literally -- so reusing
`ContainerEpochRunner` (even by parameter) would fight its own stated safety design. This is a
parallel, equally narrow class instead: it mounts NO `/model` directory at all, and raises if handed
one, so "no pretrained checkpoint was used" is a structural guarantee here too, not a convention.

Confirmed from the pinned commit (`model/management.py::load_or_create_model`,
`build_predefined_model`, `get_model_library`): passing a `--model` value that is neither a directory
nor `"custom"` resolves it against the predefined model library and builds a fresh, randomly
initialized model from the corresponding VGSL spec string via `VGSLModelGenerator`. `"recommended"`
resolves to `None,None,64,1 Cr3,3,24 Mp2,2,2,2 Bn Cr3,3,48 Bn Cr3,3,96 Mp2,2,2,2 Bn Cr3,3,96 Mp2,2,2,2
Bn Rc3 Bl512 D50 Bl512 D50 Bl512 D50 Bl512 D50 Bl512 D50 Fs92` (verified via `get_model_library()`,
not guessed). The literal `Fs92` is irrelevant: `customize_model` always resizes the final layer to
`len(tokenizer)` whenever `--model` is not a directory, regardless of what the spec string says.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from archivetrust.htr.training.telemetry_sampler import TelemetrySampler
from archivetrust.htr.training.training_session import EpochResult
from archivetrust.providers.loghi.environment import probe_loghi_environment
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

DEFAULT_EPOCH_TIMEOUT_SECONDS = 7200.0
RECOMMENDED_VGSL_SPEC = (
    "None,None,64,1 Cr3,3,24 Mp2,2,2,2 Bn Cr3,3,48 Bn Cr3,3,96 Mp2,2,2,2 Bn Cr3,3,96 Mp2,2,2,2 Bn "
    "Rc3 Bl512 D50 Bl512 D50 Bl512 D50 Bl512 D50 Bl512 D50 Fs92"
)
"""Recorded here as documentation/cross-check only -- the container resolves "recommended" itself via
its own pinned `get_model_library()`; this value is not passed to the container and is never used to
construct the model. It exists so a reader (or a test) can confirm what "recommended" means without
having to open the pinned checkout."""

_EPOCH_NUMBER_FROM_DIR_NAME = re.compile(r"epoch_(\d+)")


class PretrainedCheckpointRejected(RuntimeError):
    """Raised if anything ever tries to hand this runner an existing model directory -- Experiment 2's
    entire premise is training from random initialization, so accepting a checkpoint path here would
    silently defeat it."""


class ScratchEpochRunner:
    """The from-scratch `EpochRunner`. Structurally cannot mount or reference a pretrained checkpoint:
    there is no `/model` bind mount at all, and `--model recommended` is always passed literally."""

    def __init__(
        self,
        *,
        batch_size: int,
        gradient_accumulation: int,
        precision: str,
        max_image_width: int,
        optimizer: str,
        learning_rate: float,
        beam_width: int = 1,
        timeout_seconds: float = DEFAULT_EPOCH_TIMEOUT_SECONDS,
        run_state_dir: str | Path | None = None,
        extra_volume_mounts: tuple[tuple[str, str], ...] = (),
        architecture: str = "recommended",
    ) -> None:
        self._batch_size = batch_size
        self._gradient_accumulation = gradient_accumulation
        self._precision = precision
        self._max_image_width = max_image_width
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._beam_width = beam_width
        self._timeout_seconds = timeout_seconds
        self._run_state_dir = run_state_dir
        self._extra_volume_mounts = extra_volume_mounts
        self._architecture = architecture

    def _build_argv(
        self, *, output_dir: str, train_list_path: str, validation_list_path: str, epoch_seed: int
    ) -> list[str]:
        pins = CURRENT_PINNED_VERSIONS
        image_ref = (
            f"{pins.docker_image_tag}@{pins.docker_image_digest}" if pins.docker_image_digest else pins.docker_image_tag
        )

        output_host = Path(output_dir).resolve()
        output_host.mkdir(parents=True, exist_ok=True)
        train_list_host = Path(train_list_path).resolve()
        val_list_host = Path(validation_list_path).resolve()

        gpu_flag = "all" if pins.gpu_selection in ("all", "0", "none", "") or not pins.gpu_selection else f'"device={pins.gpu_selection}"'

        argv = [
            "docker", "run", "--rm", "--gpus", gpu_flag,
            "-v", f"{str(output_host)}:/output",
            "-v", f"{str(train_list_host.parent)}:/lists:ro",
            # Deliberately NO "-v ...:/model" mount -- there is no pretrained checkpoint directory to
            # give the container access to. This is the structural guarantee, not just the
            # --model recommended flag below.
        ]
        for host_path, container_path in self._extra_volume_mounts:
            argv += ["-v", f"{str(Path(host_path).resolve())}:{container_path}"]
        argv += [
            "--entrypoint", "python3",
            image_ref,
            "main.py",
            "--model", self._architecture,  # e.g. "recommended" -- a predefined-library key, never a
                                             # directory path, so load_or_create_model's os.path.isdir
                                             # check always takes the from-scratch build branch.
            "--train_list", f"/lists/{train_list_host.name}",
            "--validation_list", f"/lists/{val_list_host.name}",
            "--do_validate",
            "--output", "/output",
            "--epochs", "1",
            "--batch_size", str(self._batch_size),
            "--seed", str(epoch_seed),
            "--learning_rate", str(self._learning_rate),
            "--optimizer", self._optimizer,
            "--beam_width", str(self._beam_width),
            "--output_checkpoints",
            "--gpu", "0",
        ]
        if self._precision == "float32":
            argv.append("--use_float32")
        return argv

    def run_epoch(
        self, *, existing_model_dir: str | None, output_dir: str, train_list_path: str,
        validation_list_path: str, epoch_seed: int
    ) -> EpochResult:
        if existing_model_dir:
            raise PretrainedCheckpointRejected(
                f"ScratchEpochRunner refuses to run with existing_model_dir={existing_model_dir!r} -- "
                "Experiment 2 must start from random initialization, never a pretrained or prior "
                "checkpoint. Use ContainerEpochRunner for fine-tuning runs instead."
            )

        started = time.monotonic()
        report = probe_loghi_environment()
        if not report.docker_cli_present:
            return EpochResult(
                ok=False, duration_seconds=time.monotonic() - started,
                error_message=f"Docker CLI not available on this host: {report.model_dump()}",
            )
        argv = self._build_argv(
            output_dir=output_dir, train_list_path=train_list_path,
            validation_list_path=validation_list_path, epoch_seed=epoch_seed,
        )
        epoch_match = _EPOCH_NUMBER_FROM_DIR_NAME.search(Path(output_dir).name)
        epoch_number = int(epoch_match.group(1)) if epoch_match else 0
        sampler = (
            TelemetrySampler(run_state_dir=self._run_state_dir, epoch=epoch_number, output_dir=output_dir)
            if self._run_state_dir is not None else None
        )
        if sampler is not None:
            sampler.start()
        try:
            completed = subprocess.run(
                argv, capture_output=True, encoding="utf-8", errors="replace",
                timeout=self._timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return EpochResult(
                ok=False, duration_seconds=time.monotonic() - started,
                error_message=f"epoch container did not complete within {self._timeout_seconds}s: {exc}",
            )
        finally:
            if sampler is not None:
                sampler.stop()

        duration = time.monotonic() - started
        stdout_tail = (completed.stdout or "")[-4000:]
        stderr_tail = (completed.stderr or "")[-4000:]

        if completed.returncode != 0:
            return EpochResult(
                ok=False, duration_seconds=duration, stdout_tail=stdout_tail, stderr_tail=stderr_tail,
                exit_code=completed.returncode, error_message=f"epoch container exited {completed.returncode}",
            )

        output_root = Path(output_dir)
        model_dirs = list(output_root.rglob("*.keras"))
        latest_dir = None
        best_dir = None
        for model_file in model_dirs:
            parent = str(model_file.parent)
            if model_file.parent.name == "best_val":
                best_dir = parent
            else:
                latest_dir = parent

        train_cer, val_cer, train_wer, val_wer, train_loss, val_loss = _parse_latest_metrics(output_root)

        return EpochResult(
            ok=latest_dir is not None, train_cer=train_cer, val_cer=val_cer, train_wer=train_wer,
            val_wer=val_wer, train_loss=train_loss, val_loss=val_loss, duration_seconds=duration,
            checkpoint_dir=latest_dir, best_val_checkpoint_dir=best_dir, stdout_tail=stdout_tail,
            stderr_tail=stderr_tail, exit_code=completed.returncode,
            error_message=None if latest_dir is not None else "container exited 0 but produced no .keras checkpoint",
        )


def _parse_latest_metrics(output_root: Path):
    """Identical parsing to `container_epoch_runner.py::_parse_latest_metrics` -- same `log.csv`
    schema, same file regardless of scratch vs fine-tuning."""
    import csv

    log_paths = list(output_root.rglob("log.csv"))
    if not log_paths:
        return None, None, None, None, None, None
    with log_paths[0].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, None, None, None, None
    last = rows[-1]

    def _get(key: str):
        value = last.get(key)
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None

    return (_get("CER_metric"), _get("val_CER_metric"), _get("WER_metric"),
            _get("val_WER_metric"), _get("loss"), _get("val_loss"))
