"""CSV/JSON export (Benchmark & Measurement Framework, 2026-07-13) — the programmatic surface
"allow future dashboards" and "allow CSV/JSON export" ask for. Both formats are derived from the
same flattened metric view `run.flatten_metrics` already builds for run comparison, so export and
comparison can never disagree about what a metric's path or value is.
"""

from __future__ import annotations

import csv
import io
import json

from archivetrust.benchmark.run import BenchmarkRun, flatten_metrics


def to_json(run: BenchmarkRun) -> str:
    """The full run, structured — every group, every field, every `Metric`'s status/value/unit/
    note intact. This is the source of truth for a dashboard; `to_csv` is a flattened convenience
    view of the same data."""
    return run.model_dump_json(indent=2)


def to_csv(run: BenchmarkRun) -> str:
    """One row per metric leaf: `metric, value, status, unit, note`. Values that are themselves
    dicts (a confidence distribution, a document-type breakdown) are serialized as compact JSON in
    the `value` column rather than expanded into further rows — CSV has no native nested shape,
    and inventing one would be a format decision this module shouldn't silently make."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "value", "status", "unit", "note"])
    for path, (value, status, unit, note) in sorted(flatten_metrics(run).items()):
        rendered = json.dumps(value) if isinstance(value, (dict, list)) else value
        writer.writerow([path, rendered, status, unit or "", note or ""])
    return buffer.getvalue()
