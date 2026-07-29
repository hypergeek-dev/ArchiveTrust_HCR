"""`GPUResourceManager` (Production Runtime Completion milestone, Part 1): the sole authority for
GPU memory budgeting across every vLLM-backed runtime in the process.

Before this milestone, `VLLMRuntime` hardcoded `gpu_memory_utilization=0.75` -- a value live-
verified safe for exactly one resident vLLM server on an 8GB card (see
`docs/PADDLEOCR_VL_VLLM_LIVE_VERIFICATION.md`), but fatal for a second: vLLM's own
`--gpu-memory-utilization` flag is a ceiling fraction of *total* device memory that one process
commits to at startup, so two independently-launched servers at 0.75 each demand 150% of the card.

This module fixes that by making GPU budgeting a shared, stateful decision: every runtime that
wants GPU memory asks here first (`reserve`), and gives it back when done (`release`). Individual
runtimes never invent their own utilization value again -- they receive one (Part 2: "The runtime
receives its allocation. It never invents one.").
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class GPUOversubscriptionError(RuntimeError):
    """Raised when a reservation cannot be satisfied even after the caller has evicted every other
    runtime it is willing to evict. `RuntimeManager` catches this to drive its LRU eviction policy
    (Part 9); it is a real, user-facing error only once eviction has been exhausted.
    """


class GPUInfoProvider(Protocol):
    """The exact GPU-introspection surface this module needs, isolated behind one Protocol so it
    can be backed by the real `nvidia-smi` CLI in production or a fake in tests (this sandbox has
    no GPU to query) -- mirrors `vllm_runtime.ContainerLifecycle`'s own seam.
    """

    def total_memory_bytes(self, gpu_device: str | None) -> int | None:
        """Total VRAM on the named device, in bytes -- `None` when it cannot be determined (no
        `nvidia-smi` on `PATH`, no GPU present, a parse failure), never guessed."""
        ...

    def used_memory_bytes(self, gpu_device: str | None) -> int | None:
        """VRAM currently in use on the named device, in bytes -- `None` when it cannot be
        measured. What makes GPU consumers *outside* this manager's own bookkeeping visible
        (Production Hardening Review, 2026-07-13: an orphaned vLLM container from a crashed prior
        session held 4.8GB of an 8GB card while `reserve` still believed the whole card minus its
        own reservations was free)."""
        ...


def real_gpu_info_provider() -> GPUInfoProvider:
    """Constructs the facade backed by `nvidia-smi` via `subprocess` -- no GPU-vendor SDK/Python
    binding dependency needed for a single `memory.total` query, exactly the same "shell out to the
    one CLI everyone already has" choice `vllm_runtime.real_container_lifecycle` makes for Docker.
    """
    import shutil
    import subprocess  # noqa: PLC0415 (intentionally lazy -- see module docstring)

    class _NvidiaSmiGPUInfoProvider:
        @staticmethod
        def _query_memory_field(field: str, gpu_device: str | None) -> int | None:
            if shutil.which("nvidia-smi") is None:
                return None
            index = gpu_device if gpu_device is not None else "0"
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={index}",
                        f"--query-gpu={field}",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=True,
                )
            except (subprocess.SubprocessError, OSError):
                return None
            try:
                mebibytes = int(result.stdout.strip().splitlines()[0])
            except (ValueError, IndexError):
                return None
            return mebibytes * 1024 * 1024

        def total_memory_bytes(self, gpu_device: str | None) -> int | None:
            return self._query_memory_field("memory.total", gpu_device)

        def used_memory_bytes(self, gpu_device: str | None) -> int | None:
            return self._query_memory_field("memory.used", gpu_device)

    return _NvidiaSmiGPUInfoProvider()


# Live-measured weight footprints (this session's real Docker/vLLM runs, RTX 3070 8GB -- see
# `docs/PADDLEOCR_VL_SURYA_LIVE_RAW_OUTPUTS.md` and `IMPLEMENTATION_STATUS.md`), rounded up to a
# whole-server budget (weights + KV cache + activation memory + CUDA graph pool -- everything
# `--gpu-memory-utilization` actually has to cover, not just the checkpoint size). These are
# starting estimates for known models, never a hard promise -- `reserve` always clamps to whatever
# VRAM is actually still free.
_KNOWN_MODEL_VRAM_BYTES: dict[str, int] = {
    # Production Investigation (vLLM startup failure, 2026-07-12): 3.0GB (utilization 0.3492 on an
    # 8GB card) live-crashed vLLM v0.20.1's engine-core with `ValueError: No available memory for
    # the cache blocks` -- `gpu_worker.py` logged "Available KV cache memory: -0.66 GiB" at that
    # budget. Retested at 3.4GB (util 0.3958): still negative (-0.29 GiB) -- the KV-cache-vs-budget
    # relationship isn't linear-safe near the margin (observed required-equivalent utilization
    # varied 0.3837/0.4302/0.5001 across otherwise-identical live runs of the same model). 4.0GB
    # (util ~0.466) finally cleared it with real headroom (Available KV cache: +0.27 GiB) -- paired
    # with `vllm_runtime._MAX_MODEL_LEN_OVERRIDES` capping this model's `--max-model-len` to 8192
    # (still far more than one OCR page needs), which is what made that 0.27 GiB enough to actually
    # start serving. Live-confirmed end-to-end: PaddleOCR-VL now starts, passes its health check,
    # and serves a real inference request, concurrently with another resident vLLM runtime on the
    # same 8GB card.
    "PaddlePaddle/PaddleOCR-VL": 4_000_000_000,
    "datalab-to/surya-ocr-2": 2_800_000_000,
    # Phase 29 (PaddleOCR-VL Pipeline Deployment Feasibility, 2026-07-16): live-measured on this
    # repo's own reference RTX 3070 via `nvidia-smi` polling correlated against timestamped markers
    # -- peak VRAM ~700-710MB over idle-desktop baseline during warm-up + steady-state inference,
    # fully released on `del`+`empty_cache()`. Rounded up to 750MB (never down from a live
    # measurement). This is a genuinely small in-process model (~15.6M params, an RT-DETR detector),
    # not a served vLLM chat endpoint -- included here so it is visible to `GPUResourceManager`'s
    # accounting for operators who want the operator-facing GPU-usage summary to show it, even
    # though non-"vllm" runtime kinds are never auto-reserved (opt-in, `pp_doclayout_runtime.py`
    # calls `reserve`/`release` explicitly around its own `warm_up`/`shut_down`).
    "PaddlePaddle/PP-DocLayoutV2": 750_000_000,
}
_DEFAULT_MODEL_VRAM_BYTES = 3_500_000_000
"""Conservative estimate for a model this table has no live measurement for -- large enough that an
unknown VLM checkpoint is unlikely to fail at startup, small enough that it still leaves room for a
second resident runtime on an 8GB card."""


def estimate_bytes_for(model_id: str) -> int:
    return _KNOWN_MODEL_VRAM_BYTES.get(model_id, _DEFAULT_MODEL_VRAM_BYTES)


class GPUReservation(BaseModel):
    """One runtime's current claim on GPU memory -- what `GPUResourceManager` tracks per
    `runtime_key` (the model id) until `release` is called."""

    model_config = ConfigDict(frozen=True)

    runtime_key: str
    model_id: str
    gpu_device: str | None
    reserved_bytes: int
    utilization: float
    """The `gpu_memory_utilization` fraction actually handed to the runtime -- always a fraction of
    *total* device memory, matching vLLM's own flag semantics."""


_FALLBACK_UTILIZATION = 0.75
"""Used only when VRAM cannot be detected at all (no `nvidia-smi`, e.g. this development sandbox,
or a GPU-less CI run) -- the exact value live-verified safe for one resident server before this
milestone existed. Multi-runtime coexistence is not attempted when VRAM cannot be measured; a
second reservation in that state raises `GPUOversubscriptionError` immediately (nothing to safely
subdivide), the same as if the whole card were already claimed."""


class GPUResourceManager:
    """Detects total VRAM, tracks active reservations, and computes a safe per-runtime
    `gpu_memory_utilization` on request -- the "authority for GPU allocation" Part 1 calls for.
    Stateful and shared: one instance per process (constructed once in `AppContext`), used by every
    `RuntimeManager` request so reservations are visible across every vLLM-backed provider, not
    just the one that happens to be starting up.
    """

    def __init__(
        self,
        *,
        gpu_info: GPUInfoProvider,
        safety_margin_bytes: int = 1_000_000_000,
        min_utilization: float = 0.2,
        max_utilization: float = 0.75,
        min_reservation_bytes: int = 1_500_000_000,
    ) -> None:
        self._gpu_info = gpu_info
        self._safety_margin_bytes = safety_margin_bytes
        self._min_utilization = min_utilization
        self._max_utilization = max_utilization
        self._min_reservation_bytes = min_reservation_bytes
        self._reservations: dict[str, GPUReservation] = {}

    def total_memory_bytes(self, gpu_device: str | None = None) -> int | None:
        return self._gpu_info.total_memory_bytes(gpu_device)

    def active_reservations(self) -> tuple[GPUReservation, ...]:
        return tuple(self._reservations.values())

    def reservation_for(self, runtime_key: str) -> GPUReservation | None:
        return self._reservations.get(runtime_key)

    def reserve(self, *, runtime_key: str, model_id: str, gpu_device: str | None = None) -> float:
        """Computes and records a `gpu_memory_utilization` fraction for `runtime_key` (a model id),
        accounting for every *other* currently-active reservation and a fixed safety margin.
        Re-reserving an already-held `runtime_key` (e.g. a caller retrying after a transient
        failure) recomputes against the *other* reservations only -- it never double-counts its own
        prior claim.

        Raises `GPUOversubscriptionError` when no VRAM measurement is available while a reservation
        is already active, or when what remains after other reservations and the safety margin
        falls below `min_reservation_bytes` -- the caller (`RuntimeManager`) is expected to evict a
        least-recently-used reservation and retry, never to receive a utilization too small to
        plausibly start a server.
        """
        total = self.total_memory_bytes(gpu_device)
        if total is None:
            if self._reservations:
                raise GPUOversubscriptionError(
                    "GPU memory could not be measured (no 'nvidia-smi' on PATH, or no GPU present) "
                    "while another runtime already holds a reservation -- refusing to guess a "
                    "second budget that might oversubscribe the same card."
                )
            reservation = GPUReservation(
                runtime_key=runtime_key,
                model_id=model_id,
                gpu_device=gpu_device,
                reserved_bytes=0,
                utilization=_FALLBACK_UTILIZATION,
            )
            self._reservations[runtime_key] = reservation
            return reservation.utilization

        other_reserved_bytes = sum(
            r.reserved_bytes for key, r in self._reservations.items() if key != runtime_key
        )
        # GPU consumers *outside* this manager's bookkeeping (Production Hardening Review,
        # 2026-07-13): an orphaned vLLM container from a crashed prior session, the desktop
        # compositor, another process entirely -- none of them appear in `self._reservations`, so
        # budgeting against reservations alone can double-book VRAM that is already physically
        # occupied. Take the *more pessimistic* of "what we have promised" and "what the card
        # actually reports in use". Measured usage legitimately includes our own already-running
        # reservations, so `max` (not a sum) is the non-double-counting combination; a `None`
        # measurement falls back to reservation-only accounting, exactly the previous behavior.
        used_probe = getattr(self._gpu_info, "used_memory_bytes", None)
        measured_used_bytes = used_probe(gpu_device) if used_probe is not None else None
        occupied_bytes = max(other_reserved_bytes, measured_used_bytes or 0)
        available_bytes = total - occupied_bytes - self._safety_margin_bytes
        if available_bytes < self._min_reservation_bytes:
            raise GPUOversubscriptionError(
                f"Only {max(available_bytes, 0) / 1e9:.2f}GB free of {total / 1e9:.2f}GB total VRAM "
                f"after {len(self._reservations)} existing reservation(s) and safety margin -- "
                f"cannot start another vLLM-backed runtime without evicting one."
            )

        estimated_bytes = estimate_bytes_for(model_id)
        reserved_bytes = min(estimated_bytes, available_bytes)
        utilization = max(
            self._min_utilization, min(self._max_utilization, reserved_bytes / total)
        )
        reserved_bytes = int(utilization * total)

        self._reservations[runtime_key] = GPUReservation(
            runtime_key=runtime_key,
            model_id=model_id,
            gpu_device=gpu_device,
            reserved_bytes=reserved_bytes,
            utilization=utilization,
        )
        return utilization

    def adopt(self, *, runtime_key: str, model_id: str, gpu_device: str | None = None) -> None:
        """Registers bookkeeping for GPU memory a runtime already holds -- discovered already
        running (Phase 24: Runtime Discovery and Reuse) rather than one this process is about to
        start. Unlike `reserve`, this never checks available capacity and can never raise
        `GPUOversubscriptionError`: the memory is already spent by a container outside this
        reservation entirely, so there is no new allocation to refuse, only an existing one to make
        visible to `reserve()`'s "other reservations" accounting and to the operator-facing GPU
        usage summary (`ProcessingCenterViewModel.gpu_usage()`, which reads `active_reservations()`
        and would otherwise silently under-report a resident, GPU-consuming runtime).
        `reserved_bytes` is a best estimate (the same live-measured table `reserve` uses) -- the
        true size the already-running container is spending is not directly knowable to this
        process, exactly as `reserve` already only estimates for a brand-new one.
        """
        total = self.total_memory_bytes(gpu_device)
        if total is None:
            self._reservations[runtime_key] = GPUReservation(
                runtime_key=runtime_key, model_id=model_id, gpu_device=gpu_device,
                reserved_bytes=0, utilization=_FALLBACK_UTILIZATION,
            )
            return
        estimated_bytes = min(estimate_bytes_for(model_id), total)
        utilization = max(self._min_utilization, min(self._max_utilization, estimated_bytes / total))
        self._reservations[runtime_key] = GPUReservation(
            runtime_key=runtime_key, model_id=model_id, gpu_device=gpu_device,
            reserved_bytes=int(utilization * total), utilization=utilization,
        )

    def release(self, runtime_key: str) -> None:
        """Frees `runtime_key`'s reservation, if any -- safe to call even when nothing is held
        (mirrors every other `shut_down`-shaped method in this codebase: idempotent, never an
        error to release twice or release the never-reserved)."""
        self._reservations.pop(runtime_key, None)
