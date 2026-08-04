"""The real, Docker-backed `EpochRunner` (`training_session.py`'s `EpochRunner` Protocol) --
production implementation of one training epoch as one deterministic container invocation.

Reuses `providers/loghi/pinned_versions.py::CURRENT_PINNED_VERSIONS` (the real, pinned `loghi/
docker.htr` image) and `providers/loghi/environment.py::probe_loghi_environment` for host capability
checks -- this module does not duplicate either, per docs/loghi-integration-audit.md's "do not
introduce a second Loghi adapter, second environment model" instruction. It *does* introduce its own
argv builder rather than reusing `providers/loghi/facade.py::_build_docker_argv`, because that one
builds an *inference* invocation (mounted page image in, PAGE XML out); this one builds a *training*
invocation of the pinned `loghi-htr` commit's real `src/main.py` (mounted training-list files and
checkpoint directories, `--model`/`--train_list`/`--validation_list`/`--epochs 1`/... out) -- a
different real CLI contract, confirmed by reading `.loghi-upstream/loghi-htr/src/setup/arg_parser.py`
directly, not assumed from inference-time conventions.

**Why `--model <checkpoint_dir>`, never `--existing_model`.** `setup/arg_parser.py` defines no
`--existing_model` flag at all -- `model/management.py::load_or_create_model` reads `config["model"]`
alone and checks `os.path.isdir(...)` to decide "load" vs. "create new". Every epoch this runner
executes passes a real checkpoint directory for `--model`, never a VGSL spec string: "do not train from
random initialization" is enforced structurally here, not just by convention.
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

_EPOCH_NUMBER_FROM_DIR_NAME = re.compile(r"epoch_(\d+)")


class ContainerEpochRunner:
    """Production `EpochRunner`. Every path handed to the container is mounted explicitly -- no bind
    mount wider than the specific directory needed, matching `providers/loghi/facade.py`'s existing
    "no interactive shell, deterministic subprocess" discipline.
    """

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
        """`None` (the default, and what every existing test's `ContainerEpochRunner()` construction
        gets) disables the background `TelemetrySampler` entirely -- no GPU/`psutil` dependency enters
        a test run that never opts in. The real launcher passes its real `RUN_STATE_DIR` here."""
        self._extra_volume_mounts = extra_volume_mounts
        """`(host_path, container_path)` pairs appended as additional `-v host:container` mounts, e.g.
        overriding one file inside the pinned image for a single run without editing the vendored,
        pinned `.loghi-upstream` checkout on disk. Defaults to `()` -- zero effect on every existing
        caller (Experiment 0's shard pipeline included)."""

    def _build_argv(
        self, *, existing_model_dir: str, output_dir: str, train_list_path: str, validation_list_path: str, epoch_seed: int
    ) -> list[str]:
        pins = CURRENT_PINNED_VERSIONS
        image_ref = (
            f"{pins.docker_image_tag}@{pins.docker_image_digest}" if pins.docker_image_digest else pins.docker_image_tag
        )

        model_host = Path(existing_model_dir).resolve()
        output_host = Path(output_dir).resolve()
        output_host.mkdir(parents=True, exist_ok=True)
        train_list_host = Path(train_list_path).resolve()
        val_list_host = Path(validation_list_path).resolve()

        # No `wsl.exe --` prefix, and no WSL path translation -- a real, empirically-confirmed
        # correction to the assumption `providers/loghi/facade.py` made without ever running a real
        # container (its own docstring said so honestly). Docker Desktop's Windows-native `docker` CLI
        # is directly callable from the host (confirmed working for `docker pull`/`docker inspect`/
        # `docker run --gpus all` earlier in this same integration) and handles `C:\...`-style host
        # paths in `-v` mounts itself. Actually invoking `wsl.exe -- docker ...` was tried first and
        # failed for real: "It looks like you have tried to invoke the docker CLI from the
        # docker-desktop WSL2 distribution. This is not supported" -- Docker Desktop's own WSL2
        # backend distro deliberately refuses to be driven this way.
        translate = lambda p: p  # noqa: E731 -- kept as a named seam in case a genuine WSL2 Linux
        # distro (not docker-desktop's own internal one) is used on a future machine.

        # `--gpus <bare digit>` is Docker CLI syntax for "allocate this many GPUs" (a *count*), not
        # "use device N" -- a real bug caught by this smoke test: `--gpus 0` silently allocated zero
        # GPUs, TensorFlow logged "Available GPU(s): []" and fell back to CPU with no error at all.
        # This host has exactly one GPU (confirmed via `nvidia-smi`), so "all" is both correct and
        # honest here; a genuine multi-GPU device selector would need Docker's `'"device=N"'` syntax.
        gpu_flag = "all" if pins.gpu_selection in ("all", "0", "none", "") or not pins.gpu_selection else f'"device={pins.gpu_selection}"'

        argv = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            gpu_flag,
            "-v",
            # Read-write, not `:ro` -- another real bug this smoke test caught: `Tokenizer.
            # load_from_file` writes a converted `tokenizer.json` back into the model directory when
            # loading a legacy `charlist.txt`-only checkpoint (`utils/text.py::load_from_file`), so a
            # read-only mount fails with "Read-only file system: '/model/tokenizer.json'". The pinned,
            # pristine parent checkpoint is never mounted directly here for exactly this reason --
            # `training_session.py::_stage_writable_checkpoint` copies it to a writable staging
            # directory before this runner ever sees the path, so "do not overwrite the generic Dutch
            # model" is enforced by never handing this function the pristine directory, not by hoping
            # the container behaves.
            f"{translate(str(model_host))}:/model",
            "-v",
            f"{translate(str(output_host))}:/output",
            "-v",
            f"{translate(str(train_list_host.parent))}:/lists:ro",
        ]
        for host_path, container_path in self._extra_volume_mounts:
            argv += ["-v", f"{translate(str(Path(host_path).resolve()))}:{container_path}"]
        argv += [
            "--entrypoint",
            "python3",
            image_ref,
            "main.py",  # the image's own WORKDIR is already /src/loghi-htr/src (confirmed via
            # `docker inspect --format '{{.Config.WorkingDir}}'`), so this is main.py's real location
            # relative to where the container's entrypoint actually starts -- "src/main.py" (this
            # repo's own path convention) doubled the "src" segment and failed for real.
            "--model",
            "/model",
            "--train_list",
            f"/lists/{train_list_host.name}",
            "--validation_list",
            f"/lists/{val_list_host.name}",
            "--do_validate",
            "--output",
            "/output",
            "--epochs",
            "1",
            "--batch_size",
            str(self._batch_size),
            "--seed",
            str(epoch_seed),
            "--learning_rate",
            str(self._learning_rate),
            "--optimizer",
            self._optimizer,
            "--beam_width",
            str(self._beam_width),
            "--output_checkpoints",
            "--gpu",
            "0",
        ]
        if self._precision == "float32":
            argv.append("--use_float32")
        return argv

    def run_epoch(
        self, *, existing_model_dir: str, output_dir: str, train_list_path: str, validation_list_path: str, epoch_seed: int
    ) -> EpochResult:
        started = time.monotonic()
        report = probe_loghi_environment()
        if not report.docker_cli_present:
            return EpochResult(
                ok=False,
                duration_seconds=time.monotonic() - started,
                error_message=f"Docker CLI not available on this host: {report.model_dump()}",
            )
        argv = self._build_argv(
            existing_model_dir=existing_model_dir,
            output_dir=output_dir,
            train_list_path=train_list_path,
            validation_list_path=validation_list_path,
            epoch_seed=epoch_seed,
        )
        epoch_match = _EPOCH_NUMBER_FROM_DIR_NAME.search(Path(output_dir).name)
        epoch_number = int(epoch_match.group(1)) if epoch_match else 0
        sampler = (
            TelemetrySampler(run_state_dir=self._run_state_dir, epoch=epoch_number, output_dir=output_dir)
            if self._run_state_dir is not None
            else None
        )
        if sampler is not None:
            sampler.start()
        try:
            # `encoding="utf-8", errors="replace"`, never bare `text=True` -- a real bug this smoke
            # test caught: `text=True` on Windows decodes with the system codepage (cp1252 on this
            # host), and the container's real stderr contains UTF-8 Swedish text (å/ä/ö in per-line
            # "Original text"/"Predicted text" log output) that cp1252 cannot decode, which crashed
            # Python's own stdout/stderr reader thread with `UnicodeDecodeError` mid-run. The same
            # class of bug `providers/loghi/environment.py::_run` already fixed once for `wsl.exe`'s
            # UTF-16LE output -- a different encoding, the same "never trust the platform default for
            # a container's real output" lesson.
            #
            # The sampler thread above runs *alongside* this unmodified blocking call -- it does not
            # read this process's stdout/stderr and does not change argv, so it cannot affect what
            # the container itself does.
            completed = subprocess.run(
                argv,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return EpochResult(
                ok=False,
                duration_seconds=time.monotonic() - started,
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
                ok=False,
                duration_seconds=duration,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                exit_code=completed.returncode,
                error_message=f"epoch container exited {completed.returncode}",
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
            ok=latest_dir is not None,
            train_cer=train_cer,
            val_cer=val_cer,
            train_wer=train_wer,
            val_wer=val_wer,
            train_loss=train_loss,
            val_loss=val_loss,
            duration_seconds=duration,
            checkpoint_dir=latest_dir,
            best_val_checkpoint_dir=best_dir,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            exit_code=completed.returncode,
            error_message=None if latest_dir is not None else "container exited 0 but produced no .keras checkpoint",
        )


def _parse_latest_metrics(
    output_root: Path,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Reads the real `log.csv` `CSVLogger` output (`modes/training.py::train_model`'s
    `logging_callback`) for the most recent epoch's metrics -- never estimated, `None` when the
    columns genuinely are not present (an honest absence, not a fabricated 0.0). `loss`/`val_loss`
    are real columns the container already writes (confirmed in a real completed epoch's `log.csv`)
    that were previously parsed here but silently dropped -- the dashboard's loss-by-epoch chart
    needs them."""
    import csv

    log_paths = list(output_root.rglob("log.csv"))
    if not log_paths:
        return None, None, None, None, None, None
    with log_paths[0].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, None, None, None, None
    last = rows[-1]

    def _get(key: str) -> float | None:
        value = last.get(key)
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None

    return (
        _get("CER_metric"),
        _get("val_CER_metric"),
        _get("WER_metric"),
        _get("val_WER_metric"),
        _get("loss"),
        _get("val_loss"),
    )
