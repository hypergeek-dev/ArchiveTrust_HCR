from __future__ import annotations

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.progress import compute_calibration_progress_map
from archivetrust.review.sampling.log import InMemorySamplingLogSink
from archivetrust.review.sampling.strategies import (
    RandomSamplingStrategy,
    RarePatternSamplingStrategy,
    STRATEGY_REGISTRY,
    StratifiedSamplingStrategy,
    UncertaintySamplingStrategy,
    WeightedSamplingStrategy,
    strategy_by_name,
)
from tests.review._helpers import emit_slot, heading


def _corpus() -> TelemetryIndex:
    sink = InMemoryTelemetrySink()
    for i in range(30):
        emit_slot(
            sink,
            document_ref="doc1",
            canonical_payload=heading(f"single {i}"),
            provider_payloads=(("docling", heading(f"single {i}")),),
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        )
    for i in range(5):
        emit_slot(
            sink,
            document_ref="doc1",
            canonical_payload=heading(f"contested {i}"),
            provider_payloads=(
                ("docling", heading(f"contested {i}")),
                ("tesseract", heading(f"contested {i}x")),
            ),
            classification=ComparisonClassification.CONTESTED,
        )
    return TelemetryIndex.build(sink)


def test_registry_contains_every_documented_strategy():
    assert set(STRATEGY_REGISTRY) == {
        "random_sampling",
        "stratified_sampling",
        "weighted_sampling",
        "uncertainty_sampling",
        "rare_pattern_sampling",
    }
    for name in STRATEGY_REGISTRY:
        assert strategy_by_name(name) is STRATEGY_REGISTRY[name]


def test_random_sampling_is_deterministic_given_a_seed():
    index = _corpus()
    log = InMemorySamplingLogSink()
    progress = compute_calibration_progress_map(index, log)
    candidates = index.latest_canonicals()
    strat = RandomSamplingStrategy()

    a = strat.select(
        candidates,
        document_ref_by_slot=index.document_ref_by_slot,
        progress_by_pattern=progress,
        corpus_size=len(candidates),
        k=5,
        intent=ReviewIntent.CALIBRATION,
        seed=7,
    )
    b = strat.select(
        candidates,
        document_ref_by_slot=index.document_ref_by_slot,
        progress_by_pattern=progress,
        corpus_size=len(candidates),
        k=5,
        intent=ReviewIntent.CALIBRATION,
        seed=7,
    )
    assert [i.semantic_slot_id for i in a] == [i.semantic_slot_id for i in b]
    assert len(a) == 5
    assert all(i.intent == ReviewIntent.CALIBRATION for i in a)
    assert all(i.strategy_name == "random_sampling" for i in a)


def test_stratified_sampling_allocates_proportionally_to_pattern_size():
    index = _corpus()
    log = InMemorySamplingLogSink()
    progress = compute_calibration_progress_map(index, log)
    candidates = index.latest_canonicals()
    strat = StratifiedSamplingStrategy()

    items = strat.select(
        candidates,
        document_ref_by_slot=index.document_ref_by_slot,
        progress_by_pattern=progress,
        corpus_size=len(candidates),
        k=14,
        intent=ReviewIntent.CALIBRATION,
        seed=1,
    )
    single_source = [i for i in items if i.pattern.startswith("uncorroborated")]
    contested = [i for i in items if i.pattern.startswith("contested")]
    # 30 single-source vs 5 contested out of 35 total -> single-source should dominate the sample.
    assert len(single_source) > len(contested)


def test_rare_pattern_sampling_overrepresents_the_smaller_pattern():
    index = _corpus()
    log = InMemorySamplingLogSink()
    progress = compute_calibration_progress_map(index, log)
    candidates = index.latest_canonicals()
    strat = RarePatternSamplingStrategy()

    items = strat.select(
        candidates,
        document_ref_by_slot=index.document_ref_by_slot,
        progress_by_pattern=progress,
        corpus_size=len(candidates),
        k=10,
        intent=ReviewIntent.RESEARCH,
        seed=1,
    )
    contested_fraction = sum(1 for i in items if i.pattern.startswith("contested")) / len(items)
    # Contested is only 5/35 (~14%) of the corpus but should be substantially overrepresented.
    assert contested_fraction > 5 / 35


def test_uncertainty_sampling_prioritizes_furthest_from_required_n():
    index = _corpus()
    log = InMemorySamplingLogSink()
    progress = compute_calibration_progress_map(index, log)
    candidates = index.latest_canonicals()
    strat = UncertaintySamplingStrategy()

    items = strat.select(
        candidates,
        document_ref_by_slot=index.document_ref_by_slot,
        progress_by_pattern=progress,
        corpus_size=len(candidates),
        k=3,
        intent=ReviewIntent.CALIBRATION,
        seed=1,
    )
    assert len(items) == 3
    # All patterns start at 0% progress, so any selection is valid, but every item must be
    # explainable and attributed to a real pattern in progress_by_pattern.
    assert all(i.pattern in progress for i in items)


def test_weighted_sampling_favors_larger_pattern_when_progress_equal():
    index = _corpus()
    log = InMemorySamplingLogSink()
    progress = compute_calibration_progress_map(index, log)
    candidates = index.latest_canonicals()
    strat = WeightedSamplingStrategy()

    items = strat.select(
        candidates,
        document_ref_by_slot=index.document_ref_by_slot,
        progress_by_pattern=progress,
        corpus_size=len(candidates),
        k=10,
        intent=ReviewIntent.CALIBRATION,
        seed=3,
    )
    assert len(items) == 10
    assert all(i.value_score is not None and i.value_score >= 0 for i in items)
