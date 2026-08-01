"""`TelemetrySampler`: a background thread that samples GPU/system/disk telemetry while a training
epoch's container runs, for the read-only dashboard's "GPU utilization over time" / "VRAM usage over
time" / "GPU temperature over time" charts and its liveness heartbeat.

**Purely additive, never changes how a container runs.** `container_epoch_runner.py` starts this
*around* its existing, unmodified blocking `subprocess.run(...)` call (`with TelemetrySampler(...):`)
-- it does not read the container's stdout, does not change the argv, and does not affect the
already-running training session this was built alongside. `ContainerEpochRunner`'s own constructor
flag defaults this on for real runs and off for `FakeEpochRunner`-based tests, so no GPU/`psutil`
dependency enters the default `pytest -q` sweep.

**Real, `None`-on-failure, never guessed** -- the same discipline `runtime/gpu_resource_manager.py`'s
`GPUInfoProvider` and `memory_probe.py`'s `_poll_peak_vram` already use. `psutil` is imported lazily
and optionally (matching this repo's `transformers`/`gui`/`watch` heavy-optional-dependency
convention): its absence degrades RAM/CPU fields to `None`, it never raises.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_SAMPLE_INTERVAL_SECONDS = 5.0


class GpuSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = None
    utilization_pct: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None


class SystemSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    cpu_pct: float | None = None
    disk_free_gb: float | None = None


class TelemetrySample(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str
    epoch: int
    gpu: GpuSample
    system: SystemSample


def _query_gpu_sample() -> GpuSample:
    if shutil.which("nvidia-smi") is None:
        return GpuSample()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
        fields = [f.strip() for f in completed.stdout.strip().splitlines()[0].split(",")]
        name, util, mem_used, mem_total, temp = fields

        def _f(v: str) -> float | None:
            try:
                return float(v)
            except ValueError:
                return None

        return GpuSample(
            name=name or None,
            utilization_pct=_f(util),
            memory_used_mb=_f(mem_used),
            memory_total_mb=_f(mem_total),
            temperature_c=_f(temp),
        )
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return GpuSample()


def _query_system_sample(output_dir: str | Path) -> SystemSample:
    ram_used_mb = ram_total_mb = cpu_pct = None
    try:
        import psutil  # noqa: PLC0415 -- intentionally lazy/optional, see module docstring

        vm = psutil.virtual_memory()
        ram_used_mb = vm.used / (1024 * 1024)
        ram_total_mb = vm.total / (1024 * 1024)
        cpu_pct = psutil.cpu_percent(interval=None)
    except ImportError:
        pass

    disk_free_gb = None
    try:
        disk_dir = Path(output_dir)
        disk_dir.mkdir(parents=True, exist_ok=True)
        disk_free_gb = shutil.disk_usage(disk_dir).free / (1024**3)
    except OSError:
        pass

    return SystemSample(
        ram_used_mb=ram_used_mb, ram_total_mb=ram_total_mb, cpu_pct=cpu_pct, disk_free_gb=disk_free_gb
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-telemetry-status-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, ensure_ascii=False))
        os.replace(tmp_path, str(path))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


class TelemetrySampler:
    """Use as a context manager around one epoch's blocking container call:
    `with TelemetrySampler(run_state_dir=..., epoch=epoch, output_dir=output_dir): subprocess.run(...)`.
    """

    def __init__(
        self,
        *,
        run_state_dir: str | Path,
        epoch: int,
        output_dir: str | Path,
        interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._telemetry_dir = Path(run_state_dir) / "telemetry"
        self._epoch = epoch
        self._output_dir = output_dir
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _tick(self, *, container_alive: bool) -> None:
        gpu = _query_gpu_sample()
        system = _query_system_sample(self._output_dir)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sample = TelemetrySample(timestamp=now, epoch=self._epoch, gpu=gpu, system=system)
        _append_jsonl(self._telemetry_dir / "gpu_samples.jsonl", sample.model_dump())
        _atomic_write_json(
            self._telemetry_dir / "status.json",
            {
                "epoch": self._epoch,
                "last_sample_at": now,
                "container_alive": container_alive,
                "gpu": gpu.model_dump(),
                "system": system.model_dump(),
            },
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._tick(container_alive=True)
            self._stop_event.wait(self._interval)
        # One final tick marking the container no longer alive -- so a dashboard reading
        # `status.json` after this epoch finishes sees a fresh, honest "not running" snapshot
        # rather than a stale "alive=True" one from the last in-progress tick.
        self._tick(container_alive=False)

    def start(self) -> None:
        self._telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)

    def __enter__(self) -> "TelemetrySampler":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
