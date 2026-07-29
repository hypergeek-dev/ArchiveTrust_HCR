"""Operational metrics tests (docs/htr-migration-plan.md Stage 9). Uses the real `build_evidence`
functions from `providers/satrn/adapter.py` and `providers/transkribus/adapter.py`, fed with real
recognition results from a fake facade / a real parsed export file (same seams the adapters' own
contract tests use) -- so `Evidence.execution_time_ms`/`gpu_memory_mb`/`execution_device` are
genuinely populated (or genuinely `None`) exactly the way the real adapters populate them, not
hand-built `Evidence` objects with invented timing numbers.
"""

from __future__ import annotations

from pathlib import Path

from archivetrust.htr.evaluation import definitions
from archivetrust.htr.evaluation.operational import (
    aggregate_operational_metrics,
    operational_metric_results,
)
from archivetrust.providers.htr_adapter import RecognitionInput
from archivetrust.providers.satrn.adapter import SatrnAdapter, build_evidence as satrn_build_evidence
from archivetrust.providers.satrn.facade import SatrnWorkerResult
from archivetrust.providers.transkribus.adapter import TranskribusAdapter
from archivetrust.providers.transkribus.adapter import build_evidence as transkribus_build_evidence

TRANSKRIBUS_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "transkribus"


class _FakeSatrnFacade:
    def __init__(self, result: dict) -> None:
        self._result = result

    def run_inference(self, *, image_path, device_request, model_id, revision):
        return SatrnWorkerResult(self._result)


def _real_satrn_evidence(*, elapsed_seconds: float, peak_gpu_memory_mb: float, device_used: str):
    adapter = SatrnAdapter(
        facade=_FakeSatrnFacade(
            {
                "ok": True,
                "text": "hejdå vän",
                "score": 0.71,
                "elapsed_seconds": elapsed_seconds,
                "device_used": device_used,
                "model_revision": "deadbeef",
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "software_environment": {"torch": "2.1.0+cu121"},
                "gpu_name": "NVIDIA GeForce RTX 3070",
            }
        )
    )
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    evidence = satrn_build_evidence(result)
    assert evidence is not None
    return evidence


def _real_transkribus_evidence(filename: str) -> object:
    adapter = TranskribusAdapter()
    configuration = {"export_file_path": str(TRANSKRIBUS_FIXTURES / filename)}
    result = adapter.recognize(RecognitionInput(configuration=configuration))
    evidence = transkribus_build_evidence(result)
    assert evidence is not None
    return evidence


# --- operational_metric_results: only populated fields become MetricResults -------------------


def test_real_satrn_evidence_produces_both_timing_and_memory_metric_results():
    evidence = _real_satrn_evidence(elapsed_seconds=1.31, peak_gpu_memory_mb=439.0, device_used="cuda")
    results = operational_metric_results(evidence, method_run_id="method_run_1")
    by_definition = {r.metric_definition_id: r.value for r in results}
    assert by_definition[definitions.EXECUTION_TIME_MS.metric_definition_id] == 1310.0
    assert by_definition[definitions.GPU_MEMORY_MB.metric_definition_id] == 439.0


def test_real_transkribus_evidence_produces_no_timing_or_memory_metric_results():
    """Transkribus's real `build_evidence` always leaves `execution_time_ms`/`gpu_memory_mb` as
    `None` (module docstring: "no local execution happened -- explicit None, never a misleading
    0/'cpu' default") -- this must produce zero `MetricResult`s for those fields, not fabricated
    `0.0`s."""
    evidence = _real_transkribus_evidence("sample_page.xml")
    assert evidence.execution_time_ms is None
    assert evidence.gpu_memory_mb is None
    results = operational_metric_results(evidence, method_run_id="method_run_1")
    assert results == ()


# --- aggregate_operational_metrics: honest handling of a mixed present/missing population -------


def test_aggregate_operational_metrics_over_real_satrn_evidence_across_two_devices():
    """Mirrors SATRN's own README table: the same real text/score measured on both cuda and cpu,
    with genuinely different elapsed time and (cuda-only) peak memory."""
    cuda_evidence = _real_satrn_evidence(elapsed_seconds=1.31, peak_gpu_memory_mb=439.0, device_used="cuda")
    cpu_evidence = _real_satrn_evidence(elapsed_seconds=10.66, peak_gpu_memory_mb=0.0, device_used="cpu")
    # cpu run genuinely reports no peak GPU memory (0.0 is what the fake facade sends here to
    # mimic a CPU run reporting no GPU usage) -- kept as a real recorded 0.0, not treated as
    # "missing"; a *missing* case is exercised separately via Transkribus's real None below.
    aggregate = aggregate_operational_metrics([cuda_evidence, cpu_evidence], method_id="satrn")
    assert aggregate.runs == 2
    assert aggregate.execution_time_ms_min == 1310.0
    assert aggregate.execution_time_ms_max == 10660.0
    assert aggregate.execution_time_ms_mean == (1310.0 + 10660.0) / 2
    assert aggregate.execution_time_ms_missing == 0
    assert set(aggregate.execution_devices_observed) == {"cuda", "cpu"}


def test_aggregate_operational_metrics_over_real_transkribus_evidence_reports_all_missing():
    evidence = _real_transkribus_evidence("sample_page.xml")
    aggregate = aggregate_operational_metrics([evidence], method_id="transkribus_swedish_lion_1")
    assert aggregate.runs == 1
    assert aggregate.execution_time_ms_missing == 1
    assert aggregate.execution_time_ms_mean is None  # never a fabricated 0.0 mean
    assert aggregate.gpu_memory_mb_missing == 1
    assert aggregate.execution_devices_observed == ()  # Transkribus never reports a device


def test_aggregate_operational_metrics_mixed_population_counts_missing_explicitly():
    """One method run with real timing, one without -- the aggregate must make the partial
    population visible (`*_missing`), not silently average over only the present values without
    saying so."""
    satrn_evidence = _real_satrn_evidence(elapsed_seconds=1.31, peak_gpu_memory_mb=439.0, device_used="cuda")
    no_timing_evidence = _real_transkribus_evidence("sample_page.xml")
    aggregate = aggregate_operational_metrics([satrn_evidence, no_timing_evidence], method_id="mixed")
    assert aggregate.runs == 2
    assert aggregate.execution_time_ms_missing == 1
    assert aggregate.execution_time_ms_mean == 1310.0  # only the one real value, not zero-padded
