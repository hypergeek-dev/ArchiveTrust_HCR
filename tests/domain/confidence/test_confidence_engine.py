from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.engine import apply_confidence_engine, compute_canonical_confidence
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload


def _policy() -> ConfidencePolicy:
    return ConfidencePolicy(confidence_policy_version=1)


def _obs(provider: str, text: str, confidence: float | None = None) -> Observation:
    ev = Evidence.create(
        provider=provider, provider_version="1.0", raw_output=text, processing_stage=ProcessingStage.OCR,
        provider_confidence=confidence,
    )
    return Observation.from_evidence(
        provider_id=provider, provider_version="1.0", payload=HeadingPayload(text=text, level=1), evidence=(ev,)
    )


def _canonical(comparison_confidence: ComparisonConfidence, *contributors: Observation) -> CanonicalObservation:
    return CanonicalObservation.reconcile(
        semantic_slot_id="slot-1",
        payload=contributors[0].payload,
        contributing_observations=contributors,
        comparison_confidence=comparison_confidence,
        clustering_basis="test",
        reconciliation_basis="test",
        reconciliation_sequence=0,
    )


def test_corroborated_canonical_confidence_equals_comparison_magnitude():
    docling = _obs("docling", "Chapter 1", confidence=0.99)
    tesseract = _obs("tesseract", "Chapter 1", confidence=0.01)
    comparison = ComparisonConfidence(classification=ComparisonClassification.CORROBORATED, magnitude=0.85, basis="agree")
    canonical = _canonical(comparison, docling, tesseract)

    result = compute_canonical_confidence(canonical, (docling, tesseract), _policy())
    assert result.value == 0.85  # driven only by Comparison Confidence, not by either provider's own confidence


def test_contested_canonical_confidence_is_penalized_below_raw_magnitude():
    docling = _obs("docling", "1897")
    tesseract = _obs("tesseract", "1867")
    comparison = ComparisonConfidence(classification=ComparisonClassification.CONTESTED, magnitude=0.9, basis="disagree")
    canonical = _canonical(comparison, docling, tesseract)

    result = compute_canonical_confidence(canonical, (docling, tesseract), _policy())
    assert result.value < 0.9
    assert result.value == round(0.9 * _policy().contested_penalty_factor, 6)


def test_single_source_uses_only_that_providers_own_confidence():
    docling = _obs("docling", "Chapter 1", confidence=0.8)
    comparison = ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
    )
    canonical = _canonical(comparison, docling)

    result = compute_canonical_confidence(canonical, (docling,), _policy())
    assert result.value == round(_policy().single_source_base_confidence * 0.8, 6)


def test_single_source_without_provider_confidence_uses_base_only():
    docling = _obs("docling", "Chapter 1", confidence=None)
    comparison = ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
    )
    canonical = _canonical(comparison, docling)

    result = compute_canonical_confidence(canonical, (docling,), _policy())
    assert result.value == _policy().single_source_base_confidence


def test_no_cross_provider_blending_same_comparison_confidence_same_result_regardless_of_provider_confidence_values():
    # C8 / ROADMAP S5.3.1: two CORROBORATED slots with identical Comparison Confidence must
    # produce identical Canonical Confidence, no matter how wildly the two providers' own
    # Provider Confidence values differ -- because Canonical Confidence for CORROBORATED never
    # reads Provider Confidence at all.
    comparison = ComparisonConfidence(classification=ComparisonClassification.CORROBORATED, magnitude=0.7, basis="agree")

    low_confidence_pair = (_obs("docling", "x", confidence=0.1), _obs("tesseract", "x", confidence=0.15))
    high_confidence_pair = (_obs("docling", "x", confidence=0.95), _obs("tesseract", "x", confidence=0.9))

    result_low = compute_canonical_confidence(_canonical(comparison, *low_confidence_pair), low_confidence_pair, _policy())
    result_high = compute_canonical_confidence(_canonical(comparison, *high_confidence_pair), high_confidence_pair, _policy())
    assert result_low.value == result_high.value


def test_apply_confidence_engine_completes_every_node_never_mutates_input():
    docling = _obs("docling", "Chapter 1")
    comparison = ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE, magnitude=None, basis="n/a"
    )
    canonical = _canonical(comparison, docling)
    assert canonical.canonical_confidence is None

    graph = ReconciledObservationGraph(reconciliation_sequence=0, canonical_observations=(canonical,))
    completed_graph = apply_confidence_engine(graph, {docling.observation_id: docling}, _policy())

    assert canonical.canonical_confidence is None  # input untouched (C9)
    completed = completed_graph.canonical_observations[0]
    assert completed.canonical_confidence is not None
    assert completed.canonical_observation_id == canonical.canonical_observation_id


def test_three_confidence_levels_are_independently_inspectable():
    docling = _obs("docling", "Chapter 1", confidence=0.8)
    tesseract = _obs("tesseract", "Chapter 1", confidence=0.6)
    comparison = ComparisonConfidence(classification=ComparisonClassification.CORROBORATED, magnitude=0.9, basis="agree")
    canonical = _canonical(comparison, docling, tesseract)
    graph = ReconciledObservationGraph(reconciliation_sequence=0, canonical_observations=(canonical,))
    completed = apply_confidence_engine(
        graph, {docling.observation_id: docling, tesseract.observation_id: tesseract}, _policy()
    ).canonical_observations[0]

    # Provider Confidence: reachable per-provider by walking contributing_observations.
    provider_confidences = {docling.provider_id: docling.provider_confidence.value, tesseract.provider_id: tesseract.provider_confidence.value}
    assert provider_confidences == {"docling": 0.8, "tesseract": 0.6}
    # Comparison Confidence: on the Canonical Observation, untouched by Milestone 5.
    assert completed.comparison_confidence.magnitude == 0.9
    # Canonical Confidence: the Milestone 5 output, a distinct value.
    assert completed.canonical_confidence.value == 0.9
    assert completed.canonical_confidence is not completed.comparison_confidence
