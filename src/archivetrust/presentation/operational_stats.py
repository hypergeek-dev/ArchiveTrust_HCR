"""Compatibility import for the F1 read-model aggregate.

`TelemetryAggregate` now lives under `presentation.read_model.core`. The old module remains as a
re-export while ViewModels and tests migrate one at a time.
"""

from archivetrust.presentation.read_model.core import CoreAggregate, TelemetryAggregate

__all__ = ["CoreAggregate", "TelemetryAggregate"]
