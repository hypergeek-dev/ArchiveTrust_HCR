"""Independent evaluation-reference storage and scoring.

References here are evaluation evidence, not a voting provider: nothing in this package is
registered with the comparison engine, contributes to canonical decisions, or changes confidence.
The former provider-shaped ground-truth experiment was removed by Production Closure.
"""

from archivetrust.evaluation.ground_truth import (
    AdjudicationStatus,
    FileGroundTruthStore,
    GroundTruthAnnotation,
    GroundTruthValidationError,
)
from archivetrust.evaluation.metrics import TextComparison, compare_text

__all__ = [
    "AdjudicationStatus",
    "FileGroundTruthStore",
    "GroundTruthAnnotation",
    "GroundTruthValidationError",
    "TextComparison",
    "compare_text",
]
