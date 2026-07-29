from __future__ import annotations

import pytest

from archivetrust.runtime.gpu_resource_manager import (
    GPUOversubscriptionError,
    GPUResourceManager,
    estimate_bytes_for,
)


class FakeGPUInfoProvider:
    def __init__(self, *, total_bytes: int | None) -> None:
        self.total_bytes = total_bytes
        self.calls: list[str | None] = []

    def total_memory_bytes(self, gpu_device: str | None) -> int | None:
        self.calls.append(gpu_device)
        return self.total_bytes


_EIGHT_GB = 8_000_000_000


def _manager(*, total_bytes: int | None = _EIGHT_GB, **kwargs) -> tuple[GPUResourceManager, FakeGPUInfoProvider]:
    info = FakeGPUInfoProvider(total_bytes=total_bytes)
    return GPUResourceManager(gpu_info=info, **kwargs), info


def test_reserve_computes_a_utilization_fraction_of_total_vram() -> None:
    manager, _ = _manager()
    utilization = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    assert 0.0 < utilization <= 0.75
    reservation = manager.reservation_for("paddleocr")
    assert reservation is not None
    assert reservation.reserved_bytes == pytest.approx(utilization * _EIGHT_GB)


def test_two_known_models_coexist_on_an_8gb_card() -> None:
    """Live-verified regression scenario (Production Runtime Completion milestone): PaddleOCR-VL
    (~3.0GB estimated) and Surya (~2.8GB estimated) both requesting a reservation on the same 8GB
    card must both succeed, each with a utilization that reflects a bounded slice of the card, not
    vLLM's own oversubscribing 0.9-of-total default repeated twice.
    """
    manager, _ = _manager()
    paddle_utilization = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    surya_utilization = manager.reserve(runtime_key="surya", model_id="datalab-to/surya-ocr-2")

    assert paddle_utilization + surya_utilization < 1.0
    reservations = {r.runtime_key: r for r in manager.active_reservations()}
    assert reservations["paddleocr"].reserved_bytes + reservations["surya"].reserved_bytes < _EIGHT_GB


def test_reserve_raises_when_remaining_vram_is_too_small() -> None:
    manager, _ = _manager(total_bytes=4_000_000_000)
    manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    with pytest.raises(GPUOversubscriptionError):
        manager.reserve(runtime_key="surya", model_id="datalab-to/surya-ocr-2")


def test_release_frees_the_reservation_so_a_new_one_can_fit() -> None:
    manager, _ = _manager(total_bytes=4_000_000_000)
    manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    manager.release("paddleocr")
    assert manager.reservation_for("paddleocr") is None
    # Now fits, since the only prior reservation was released.
    manager.reserve(runtime_key="surya", model_id="datalab-to/surya-ocr-2")


def test_release_is_idempotent_and_never_raises_for_an_unknown_key() -> None:
    manager, _ = _manager()
    manager.release("never-reserved")  # must not raise


def test_re_reserving_the_same_key_does_not_double_count_its_own_prior_claim() -> None:
    manager, _ = _manager()
    first = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    second = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    assert first == second


def test_unknown_model_id_gets_the_conservative_default_estimate() -> None:
    assert estimate_bytes_for("some/unrecognized-vlm") == 3_500_000_000
    # Production Investigation (vLLM startup failure, 2026-07-12): 3.0GB and 3.4GB both live-
    # crashed vLLM's engine-core with negative KV-cache headroom; 4.0GB (paired with
    # `vllm_runtime.max_model_len_for`'s 8192 override for this model) is the live-verified-
    # sufficient estimate -- confirmed by starting a real container and serving a real request.
    assert estimate_bytes_for("PaddlePaddle/PaddleOCR-VL") == 4_000_000_000


def test_falls_back_to_the_previously_hardcoded_default_when_vram_cannot_be_measured() -> None:
    """No 'nvidia-smi' on PATH (this sandbox, or a GPU-less CI run) -- a single reservation must
    still succeed, at the exact value live-verified safe before this milestone (0.75), rather than
    refusing to start any vLLM-backed runtime at all.
    """
    manager, _ = _manager(total_bytes=None)
    utilization = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    assert utilization == 0.75


def test_a_second_reservation_is_refused_when_vram_cannot_be_measured_at_all() -> None:
    manager, _ = _manager(total_bytes=None)
    manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    with pytest.raises(GPUOversubscriptionError):
        manager.reserve(runtime_key="surya", model_id="datalab-to/surya-ocr-2")


class FakeGPUInfoProviderWithUsage(FakeGPUInfoProvider):
    """A `GPUInfoProvider` that also reports measured usage — the surface added by the Production
    Hardening Review (2026-07-13) so consumers *outside* the manager's own bookkeeping (an orphaned
    vLLM container from a crashed prior session, another process) become visible to `reserve`."""

    def __init__(self, *, total_bytes: int | None, used_bytes: int | None) -> None:
        super().__init__(total_bytes=total_bytes)
        self.used_bytes = used_bytes

    def used_memory_bytes(self, gpu_device: str | None) -> int | None:
        return self.used_bytes


def test_reserve_accounts_for_externally_used_vram() -> None:
    """Incident scenario (2026-07-13): an orphaned vLLM container from a crashed prior session held
    4.8GB of an 8GB card; a fresh process's manager had zero reservations and would previously have
    budgeted as though the whole card were free. With measured usage visible, the reservation must
    be clamped to what is physically left, never double-booked."""
    info = FakeGPUInfoProviderWithUsage(total_bytes=_EIGHT_GB, used_bytes=4_800_000_000)
    manager = GPUResourceManager(gpu_info=info)

    utilization = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")

    reservation = manager.reservation_for("paddleocr")
    assert reservation is not None
    # 8GB total - 4.8GB measured in use - 1GB safety margin = 2.2GB actually available; the 4GB
    # estimate for this model must have been clamped below that, not granted in full.
    assert reservation.reserved_bytes <= 2_200_000_000
    assert utilization < 0.3


def test_reserve_refuses_when_external_usage_leaves_too_little() -> None:
    info = FakeGPUInfoProviderWithUsage(total_bytes=_EIGHT_GB, used_bytes=6_500_000_000)
    manager = GPUResourceManager(gpu_info=info)
    with pytest.raises(GPUOversubscriptionError):
        manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")


def test_reserve_without_a_usage_probe_behaves_exactly_as_before() -> None:
    """`FakeGPUInfoProvider` (no `used_memory_bytes` at all) is the pre-existing provider shape --
    reservation-only accounting must remain byte-identical for it."""
    manager, _ = _manager()
    utilization = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    reservation = manager.reservation_for("paddleocr")
    assert reservation is not None
    assert reservation.reserved_bytes == pytest.approx(utilization * _EIGHT_GB)


def test_measured_usage_is_never_double_counted_against_own_reservations() -> None:
    """Measured usage legitimately includes this manager's own already-running runtimes -- the
    combination is max(reserved, measured), never a sum, so a second reservation alongside one
    running (and measured) runtime is not penalized twice."""
    info = FakeGPUInfoProviderWithUsage(total_bytes=_EIGHT_GB, used_bytes=None)
    manager = GPUResourceManager(gpu_info=info)
    first = manager.reserve(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    # The first runtime is now running and physically using roughly its reservation.
    info.used_bytes = manager.reservation_for("paddleocr").reserved_bytes

    second = manager.reserve(runtime_key="surya", model_id="datalab-to/surya-ocr-2")

    assert first + second < 1.0
    assert manager.reservation_for("surya") is not None


def test_adopt_registers_a_reservation_without_a_capacity_check() -> None:
    """Phase 24 (Runtime Discovery and Reuse): `adopt` records bookkeeping for a runtime already
    discovered running -- e.g. after a full card's worth of VRAM is already externally occupied by
    the very container being adopted -- and must never raise `GPUOversubscriptionError` for that,
    unlike `reserve`."""
    manager, _ = _manager(total_bytes=1_000_000_000)  # far below reserve()'s min_reservation_bytes
    manager.adopt(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")

    reservation = manager.reservation_for("paddleocr")
    assert reservation is not None
    assert reservation.reserved_bytes > 0


def test_adopt_is_visible_to_active_reservations_and_a_later_reserve_call() -> None:
    """The whole point of registering an adoption, not just skipping reservation silently: the
    operator-facing GPU usage summary (`active_reservations`) must still show the adopted runtime
    as consuming VRAM, and a *different* model's later `reserve()` must budget around it."""
    manager, _ = _manager(total_bytes=_EIGHT_GB)
    manager.adopt(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")

    assert {r.runtime_key for r in manager.active_reservations()} == {"paddleocr"}

    # A second, real reservation must account for the adopted one as "other reserved" capacity.
    surya_utilization = manager.reserve(runtime_key="surya", model_id="datalab-to/surya-ocr-2")
    paddle = manager.reservation_for("paddleocr")
    assert paddle.reserved_bytes + int(surya_utilization * _EIGHT_GB) < _EIGHT_GB


def test_adopt_never_raises_when_vram_cannot_be_measured() -> None:
    manager, _ = _manager(total_bytes=None)
    manager.adopt(runtime_key="paddleocr", model_id="PaddlePaddle/PaddleOCR-VL")
    assert manager.reservation_for("paddleocr") is not None
