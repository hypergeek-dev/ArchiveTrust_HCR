"""Reference-free technical-reliability screening signals.

This package exists because of one hard constraint stated in
`docs/experiments/technical-reliability-screening/design-audit.md` §3.1: of the 26 metric
definitions in `htr/evaluation/definitions.py`, **24 require ground truth and are therefore
unusable** on this corpus, which has none and for which none is planned. What is left
(`EXECUTION_TIME_MS`, `GPU_MEMORY_MB`) is not enough to screen a pipeline on.

So this package computes the missing family: signals that need **no reference text at all**.

**Nothing here is an accuracy measure and nothing here may be turned into one.** Every function
below can be computed without knowing what the image says. A method can score perfectly on all of
them and still be transcribing nonsense; a method can be flagged by all of them and still be
correct. They answer "did this pipeline survive real archival pages, and does its output have the
*shape* of a transcription", never "is the output right".

## The run harness

Alongside the signals, this package owns the machinery that *executes* the screening over the real
60-page sample -- resumably, because the run costs hours and a user must be able to stop it:

* `run_configuration` -- the resolved, content-hashed configuration a run is pinned to;
* `task_identity`     -- the stable eight-field identity of one unit of work;
* `run_state`         -- run discovery, telemetry replay, the derived progress cache, the
                         refuse-to-silently-continue checks;
* `runner`            -- execution, safe stop, and skip-what-is-already-done;
* `run_report`        -- the report, rebuilt from the durable event log alone.

`scripts/reliability_test.py` is the Run / Stop / Resume terminal in front of them.

These are imported as submodules rather than re-exported here: `runner` reaches the provider
adapters, and a package-level import would put a GPU-adjacent import chain behind
`import archivetrust.htr.screening`, which the signal functions above must never require.
"""

from archivetrust.htr.screening.reliability_signals import (
    HEURISTIC_CATALOG,
    RELIABILITY_HEURISTICS_VERSION,
    HeuristicDefinition,
    OutputSignals,
    ReliabilityThresholds,
    RepeatedOutputSummary,
    evaluate_output,
    length_outlier_bounds,
    percentile,
    repeated_output_summary,
    timing_summary,
)

__all__ = [
    "HEURISTIC_CATALOG",
    "RELIABILITY_HEURISTICS_VERSION",
    "HeuristicDefinition",
    "OutputSignals",
    "ReliabilityThresholds",
    "RepeatedOutputSummary",
    "evaluate_output",
    "length_outlier_bounds",
    "percentile",
    "repeated_output_summary",
    "timing_summary",
]
