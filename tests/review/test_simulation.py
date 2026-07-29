from __future__ import annotations

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.domain.calibration.progress import CalibrationMaturity
from archivetrust.domain.calibration.decision_policy import DecisionPolicy
from archivetrust.review.sampling.log import InMemorySamplingLogSink
from archivetrust.review.simulation import simulate_decision_policy, simulate_policy
from archivetrust.review.triage import TriagePolicy
from tests.review._helpers import emit_slot, heading


def _sink() -> InMemoryTelemetrySink:
    sink = InMemoryTelemetrySink()
    for i in range(20):
        emit_slot(
            sink,
            document_ref="doc1",
            canonical_payload=heading(f"single {i}"),
            provider_payloads=(("docling", heading(f"single {i}")),),
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        )
    return sink


def test_baseline_vs_candidate_queue_size_with_identical_policies():
    sink = _sink()
    index = TelemetryIndex.build(sink)
    log = InMemorySamplingLogSink()

    result = simulate_policy(index, log)

    assert result.baseline_operational_queue_size == result.candidate_operational_queue_size
    assert result.operational_queue_delta == 0
    assert result.reviews_saved == 0


def test_stricter_candidate_policy_reduces_queue_and_reports_reviews_saved():
    sink = _sink()
    index = TelemetryIndex.build(sink)
    log = InMemorySamplingLogSink()

    # Baseline v1-equivalent: queue every single-source type.
    baseline = TriagePolicy(single_source_review_types=None)
    # Candidate: never queue single-source at all.
    candidate = TriagePolicy(queue_single_source=False)

    result = simulate_policy(
        index, log, baseline_triage_policy=baseline, candidate_triage_policy=candidate
    )

    assert result.candidate_operational_queue_size < result.baseline_operational_queue_size
    assert result.reviews_saved > 0
    assert result.operational_queue_delta < 0


def test_calibration_projection_moves_pattern_toward_maturity():
    sink = _sink()
    index = TelemetryIndex.build(sink)
    log = InMemorySamplingLogSink()

    result = simulate_policy(index, log, candidate_calibration_reviews=200)

    assert result.pattern_projections
    projection = result.pattern_projections[0]
    assert projection.current_maturity == CalibrationMaturity.NO_EVIDENCE
    assert projection.projected_reviews > projection.current_reviews
    # 200 reviews against a 20-slot pattern should be more than enough to reach maturity.
    assert projection.projected_maturity != CalibrationMaturity.NO_EVIDENCE


def test_no_projection_requested_yields_empty_projections():
    sink = _sink()
    index = TelemetryIndex.build(sink)
    log = InMemorySamplingLogSink()

    result = simulate_policy(index, log, candidate_calibration_reviews=0)

    assert result.pattern_projections == ()
    assert result.candidate_calibration_reviews_simulated == 0


def test_decision_policy_simulation_is_conservative_without_calibration_evidence():
    sink = _sink()
    index = TelemetryIndex.build(sink)
    log = InMemorySamplingLogSink()

    (projection,) = simulate_decision_policy(index, log)

    assert projection.canonical_slots_evaluated == 20
    assert projection.automatic_acceptances == 0
    assert projection.human_reviews == 20
    assert projection.estimated_false_acceptance_rate is None


def test_decision_policy_simulation_sweeps_candidate_thresholds():
    sink = _sink()
    index = TelemetryIndex.build(sink)
    log = InMemorySamplingLogSink()
    policies = (
        DecisionPolicy(decision_policy_version=1, max_acceptable_error_rate=0.01),
        DecisionPolicy(decision_policy_version=2, max_acceptable_error_rate=0.10),
    )

    projections = simulate_decision_policy(index, log, policies=policies)

    assert [p.decision_policy_version for p in projections] == [1, 2]
    assert all(p.canonical_slots_evaluated == 20 for p in projections)
