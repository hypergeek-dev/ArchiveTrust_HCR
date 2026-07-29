"""HTR evaluation engine (docs/htr-migration-plan.md Stage 9): recognition, historical-document,
segmentation, reliability/failure, and operational metric computation, producing
`htr.experiment.models.MetricResult`/`FailureRecord` instances.

Distinct from `archivetrust.evaluation` (kept where it is, unmodified in role): that package owns
the blinded double-annotation ground-truth *store* and annotation *workflow*
(`evaluation/ground_truth.py`, `evaluation/workflow.py`, `evaluation/evaluate.py`). This package
is the metric-*computation* layer that consumes ground truth (however it was collected) plus one
method's output and produces the versioned `MetricResult`/`FailureRecord` records
`htr/experiment/models.py` already defines. It reuses `evaluation/metrics.py`'s `levenshtein`/
`normalize_text`/`compare_text` directly rather than reimplementing them -- see `recognition.py`'s
module docstring for exactly what is reused vs. added.
"""

from __future__ import annotations
