"""The benchmark measurement vocabulary (Benchmark & Measurement Framework, 2026-07-13).

Every figure the framework reports is a `Metric`, never a bare number — a bare float cannot say
*how* it was obtained, and this codebase's discipline (`ROADMAP.md` Guiding Principle 5, `Article
18` "silence must be distinguishable from failure") is that provenance is part of the value, not
an afterthought. Four statuses, and only four:

- `MEASURED` — read directly off a telemetry field that records exactly this fact
  (`ProviderObservationAttempted.observation_count`, `RuntimeTelemetryEvent.seconds`, ...).
- `DERIVED` — computed from measured fields by an explicit, reproducible formula (an average, a
  ratio, a count grouped a different way). Reproducible from the same stream, never a guess.
- `ESTIMATED` — computed from a proxy that stands in for the real quantity because the real
  quantity isn't directly recorded (e.g. `mime_type` as a document-type proxy). Always carries a
  `note` explaining the proxy.
- `UNAVAILABLE` — no telemetry captures this at all today. `value` is always `None`; `note`
  explains what would need to be recorded to make it measurable. Never fabricated, never silently
  dropped from the report.

This mirrors `ComparisonConfidence.magnitude` being structurally `None` for
`UNCORROBORATED_SINGLE_SOURCE` rather than a misleading low number: absence of data is a distinct,
representable state, not zero.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class MetricStatus(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class Metric(BaseModel, Generic[T]):
    """One reported figure, self-describing its own provenance."""

    model_config = ConfigDict(frozen=True)

    status: MetricStatus
    value: T | None
    unit: str | None = None
    note: str | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        if self.status == MetricStatus.UNAVAILABLE:
            return f"unavailable{f' ({self.note})' if self.note else ''}"
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.value}{unit}"


def measured(value: T, *, unit: str | None = None, note: str | None = None) -> Metric[T]:
    return Metric(status=MetricStatus.MEASURED, value=value, unit=unit, note=note)


def derived(value: T, *, note: str, unit: str | None = None) -> Metric[T]:
    return Metric(status=MetricStatus.DERIVED, value=value, unit=unit, note=note)


def estimated(value: T, *, note: str, unit: str | None = None) -> Metric[T]:
    return Metric(status=MetricStatus.ESTIMATED, value=value, unit=unit, note=note)


def unavailable(*, note: str, unit: str | None = None) -> Metric[None]:
    return Metric(status=MetricStatus.UNAVAILABLE, value=None, unit=unit, note=note)
