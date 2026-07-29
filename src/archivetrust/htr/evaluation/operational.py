"""Operational metrics aggregation (docs/htr-migration-plan.md Stage 9). Pulls together timing/
resource fields already recorded on `Evidence` (`execution_time_ms`, `gpu_memory_mb`,
`software_environment`, `hardware_environment` -- added in Phase 3/Stage 3, populated for real by
SATRN/Florence-2's adapters in Stage 6/7) into `MetricResult`s and a per-method aggregate summary.
**No new timing capture** -- every value here is read verbatim from an already-constructed
`Evidence`, never measured freshly by this module.

Honest-absence discipline, exercised for real by the three adapters' actual behavior:
- SATRN/Florence-2 populate `execution_time_ms`/`gpu_memory_mb`/`execution_device` for real,
  measured runs (see `providers/satrn/adapter.py::build_evidence`, `providers/florence2_htr/
  adapter.py::build_evidence`).
- Transkribus's `build_evidence` always sets these three fields to `None` -- "no local execution
  happened -- explicit None, never a misleading 0/'cpu' default" (its own module docstring). This
  module's aggregation must therefore treat `None` as "not applicable to this method" (excluded
  from a mean, not counted as a measured `0.0`), and `count_missing` makes that exclusion visible
  rather than silent -- proven in `tests/htr/evaluation/test_operational.py` using Transkribus's
  real `build_evidence(...)` output as a fixture, not a synthetic stand-in.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.evidence.models import Evidence
from archivetrust.htr.evaluation import definitions
from archivetrust.htr.experiment.models import MetricResult


def operational_metric_results(evidence: Evidence, *, method_run_id: str) -> tuple[MetricResult, ...]:
    """`MetricResult`s for one `Evidence` record's timing/resource fields -- only for fields that
    are actually populated (never a fabricated `0.0` standing in for "not measured/not
    applicable")."""
    results: list[MetricResult] = []
    if evidence.execution_time_ms is not None:
        results.append(
            MetricResult.create(
                metric_definition_id=definitions.EXECUTION_TIME_MS.metric_definition_id,
                method_run_id=method_run_id,
                value=evidence.execution_time_ms,
            )
        )
    if evidence.gpu_memory_mb is not None:
        results.append(
            MetricResult.create(
                metric_definition_id=definitions.GPU_MEMORY_MB.metric_definition_id,
                method_run_id=method_run_id,
                value=evidence.gpu_memory_mb,
            )
        )
    return tuple(results)


class OperationalAggregate(BaseModel):
    model_config = ConfigDict(frozen=True)

    method_id: str
    runs: int
    execution_time_ms_mean: float | None
    execution_time_ms_min: float | None
    execution_time_ms_max: float | None
    execution_time_ms_missing: int
    """Evidence records for this method with `execution_time_ms is None` -- e.g. every Transkribus
    run, honestly. Never folded into the mean as a `0.0`."""
    gpu_memory_mb_mean: float | None
    gpu_memory_mb_min: float | None
    gpu_memory_mb_max: float | None
    gpu_memory_mb_missing: int
    execution_devices_observed: tuple[str, ...]
    """Distinct non-`None` `execution_device` values seen (e.g. `("cuda:0", "cpu")`) -- an SATRN/
    Florence-2 run reports one of these; a Transkribus run reports none, so this may legitimately
    be empty for a method that never executes locally."""
    software_environments_observed: tuple[dict, ...]
    """Distinct `software_environment` dicts observed, exactly as adapters recorded them (e.g.
    `{"torch": "2.1.0+cu121", "mmocr": "1.0.1", ...}`) -- not merged/summarized, since merging
    would silently claim one canonical environment for runs that may genuinely differ."""
    hardware_environments_observed: tuple[dict, ...]


def aggregate_operational_metrics(
    evidences: Sequence[Evidence], *, method_id: str
) -> OperationalAggregate:
    times = [e.execution_time_ms for e in evidences if e.execution_time_ms is not None]
    memory = [e.gpu_memory_mb for e in evidences if e.gpu_memory_mb is not None]
    devices = sorted({e.execution_device for e in evidences if e.execution_device is not None})
    software_seen: list[dict] = []
    hardware_seen: list[dict] = []
    for evidence in evidences:
        if evidence.software_environment is not None and evidence.software_environment not in software_seen:
            software_seen.append(evidence.software_environment)
        if evidence.hardware_environment is not None and evidence.hardware_environment not in hardware_seen:
            hardware_seen.append(evidence.hardware_environment)

    return OperationalAggregate(
        method_id=method_id,
        runs=len(evidences),
        execution_time_ms_mean=(sum(times) / len(times)) if times else None,
        execution_time_ms_min=min(times) if times else None,
        execution_time_ms_max=max(times) if times else None,
        execution_time_ms_missing=len(evidences) - len(times),
        gpu_memory_mb_mean=(sum(memory) / len(memory)) if memory else None,
        gpu_memory_mb_min=min(memory) if memory else None,
        gpu_memory_mb_max=max(memory) if memory else None,
        gpu_memory_mb_missing=len(evidences) - len(memory),
        execution_devices_observed=tuple(devices),
        software_environments_observed=tuple(software_seen),
        hardware_environments_observed=tuple(hardware_seen),
    )
