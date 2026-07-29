"""Adaptive review sampling (Milestone 11): separate review intents, replaceable sampling
strategies, and an adaptive queue that merges them -- all additive to the existing operational
review path (`review/triage.py`, `review/service.py`), which this package never modifies."""

from archivetrust.review.sampling.coordinator import AdaptiveReviewCoordinator
from archivetrust.review.sampling.discovery import (
    by_classification_and_type,
    by_provider_participation_count,
    by_single_contributing_provider,
    discover_patterns,
)
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.log import (
    FileSamplingLogSink,
    InMemorySamplingLogSink,
    SamplingDecision,
    SamplingLogSink,
)
from archivetrust.review.sampling.progress import compute_calibration_progress_map
from archivetrust.review.sampling.queue import AdaptiveQueueEntry, build_adaptive_queue
from archivetrust.review.sampling.strategies import (
    RandomSamplingStrategy,
    RarePatternSamplingStrategy,
    SampledItem,
    SamplingStrategy,
    StratifiedSamplingStrategy,
    UncertaintySamplingStrategy,
    WeightedSamplingStrategy,
    strategy_by_name,
)

__all__ = [
    "AdaptiveQueueEntry",
    "AdaptiveReviewCoordinator",
    "FileSamplingLogSink",
    "InMemorySamplingLogSink",
    "RandomSamplingStrategy",
    "by_classification_and_type",
    "by_provider_participation_count",
    "by_single_contributing_provider",
    "discover_patterns",
    "RarePatternSamplingStrategy",
    "ReviewIntent",
    "SampledItem",
    "SamplingDecision",
    "SamplingLogSink",
    "SamplingStrategy",
    "StratifiedSamplingStrategy",
    "UncertaintySamplingStrategy",
    "WeightedSamplingStrategy",
    "build_adaptive_queue",
    "compute_calibration_progress_map",
    "strategy_by_name",
]
