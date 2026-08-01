"""RTX 3070 (8GB) memory probe (Work Package 9) -- a bounded, real run over a handful of samples and
steps, never a guessed batch size.

Runs the real pinned container against a tiny slice of the prepared training list, sweeping batch size
downward from a starting guess until a configuration completes without CUDA OOM, sampling peak VRAM via
`nvidia-smi` polling during the run (a separate thread, since the container itself is opaque from the
host's perspective -- this is the same "read the real device, don't ask the process to self-report"
discipline `providers/swedish_lion/facade.py` already uses for GPU memory).
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner

CANDIDATE_BATCH_SIZES = (16, 8, 4, 2, 1)
PROBE_STEPS_PER_EPOCH = 5


class MemoryProbeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_size: int
    gradient_accumulation: int
    effective_batch_size: int
    precision: str
    ok: bool
    peak_vram_mb: float | None
    error_message: str | None
    duration_seconds: float


class MemoryProbeReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempts: tuple[MemoryProbeResult, ...]
    chosen_batch_size: int | None
    chosen_gradient_accumulation: int
    chosen_effective_batch_size: int | None
    chosen_precision: str
    chosen_peak_vram_mb: float | None
    safety_margin_mb: float | None
    total_vram_mb: float


def _query_total_vram_mb() -> float:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return float(completed.stdout.strip().splitlines()[0])


def _poll_peak_vram(stop_event: threading.Event, samples: list[float]) -> None:
    while not stop_event.is_set():
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if completed.returncode == 0:
                samples.append(float(completed.stdout.strip().splitlines()[0]))
        except (subprocess.TimeoutExpired, ValueError):
            pass
        time.sleep(1.0)


def run_memory_probe(
    *,
    probe_train_list_path: str,
    probe_validation_list_path: str,
    parent_checkpoint_dir: str,
    output_root: str | Path,
    candidate_batch_sizes: tuple[int, ...] = CANDIDATE_BATCH_SIZES,
    precision: str = "mixed_float16",
    gradient_accumulation: int = 1,
    learning_rate: float = 0.0001,
    optimizer: str = "adam",
) -> MemoryProbeReport:
    """Attempts each candidate batch size (largest first) for one bounded, real epoch over the probe
    list; the first one that completes without an OOM/error is chosen. Every attempt -- including
    failures -- is recorded, never only the winner."""
    output_root = Path(output_root)
    total_vram_mb = _query_total_vram_mb()
    attempts: list[MemoryProbeResult] = []
    chosen: MemoryProbeResult | None = None

    for batch_size in candidate_batch_sizes:
        attempt_dir = output_root / f"probe_batch_{batch_size}"
        runner = ContainerEpochRunner(
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            precision=precision,
            max_image_width=65536,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )

        vram_samples: list[float] = []
        stop_event = threading.Event()
        poller = threading.Thread(target=_poll_peak_vram, args=(stop_event, vram_samples), daemon=True)
        poller.start()

        started = time.monotonic()
        result = runner.run_epoch(
            existing_model_dir=parent_checkpoint_dir,
            output_dir=str(attempt_dir),
            train_list_path=probe_train_list_path,
            validation_list_path=probe_validation_list_path,
            epoch_seed=1,
        )
        stop_event.set()
        poller.join(timeout=5)

        peak_vram = max(vram_samples) if vram_samples else None
        attempt = MemoryProbeResult(
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            effective_batch_size=batch_size * gradient_accumulation,
            precision=precision,
            ok=result.ok,
            peak_vram_mb=peak_vram,
            error_message=result.error_message,
            duration_seconds=time.monotonic() - started,
        )
        attempts.append(attempt)
        if result.ok:
            chosen = attempt
            break

    return MemoryProbeReport(
        attempts=tuple(attempts),
        chosen_batch_size=chosen.batch_size if chosen else None,
        chosen_gradient_accumulation=gradient_accumulation,
        chosen_effective_batch_size=chosen.effective_batch_size if chosen else None,
        chosen_precision=precision,
        chosen_peak_vram_mb=chosen.peak_vram_mb if chosen else None,
        safety_margin_mb=(total_vram_mb - chosen.peak_vram_mb) if chosen and chosen.peak_vram_mb else None,
        total_vram_mb=total_vram_mb,
    )
