from __future__ import annotations

from archivetrust.domain.comparison.matching import MatchCardinality, ProviderContribution
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.reporting import (
    RiskType,
    build_agreement_report,
    build_evidence_report,
    compute_risk_score,
    compute_trust_score,
)
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload
from archivetrust.domain.ontology.types import ObservationType


def _policy() -> ReconciliationPolicy:
    return ReconciliationPolicy(policy_version=1)


def test_single_source_agreement_report_flags_uncorroborated_never_a_low_number():
    contributions = (
        ProviderContribution(
            provider_id="docling", provider_version="1.0", cardinality=MatchCardinality.ONE_TO_ONE, observation_ids=("o1",)
        ),
    )
    report = build_agreement_report(
        cluster_id="cluster-1",
        observation_type=ObservationType.HEADING,
        clustering_basis="test",
        contributions=contributions,
        value_comparison_confidence=None,
        policy=_policy(),
    )
    assert report.classification == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE
    assert any(item.risk_type == RiskType.SINGLE_SOURCE_UNCORROBORATED for item in report.disagreements)

    trust = compute_trust_score(report, _policy())
    risk = compute_risk_score(report, _policy())
    assert trust.magnitude == 0.0
    assert risk.magnitude > 0.0
    # Trust and Risk are independent axes, not 1 - the other (S11).
    assert risk.magnitude != 1.0 - trust.magnitude or trust.magnitude == 0.0


def test_contested_value_produces_value_conflict_risk_item():
    contributions = (
        ProviderContribution(
            provider_id="docling", provider_version="1.0", cardinality=MatchCardinality.ONE_TO_ONE, observation_ids=("o1",)
        ),
        ProviderContribution(
            provider_id="tesseract", provider_version="1.0", cardinality=MatchCardinality.ONE_TO_ONE, observation_ids=("o2",)
        ),
    )
    contested = ComparisonConfidence(classification=ComparisonClassification.CONTESTED, magnitude=0.3, basis="disagree")
    report = build_agreement_report(
        cluster_id="cluster-1",
        observation_type=ObservationType.HEADING,
        clustering_basis="test",
        contributions=contributions,
        value_comparison_confidence=contested,
        policy=_policy(),
    )
    assert report.classification == ComparisonClassification.CONTESTED
    assert any(item.risk_type == RiskType.VALUE_CONFLICT for item in report.disagreements)


def test_evidence_report_presents_provider_confidence_unblended():
    ev1 = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="Ch 1", processing_stage=ProcessingStage.OCR, provider_confidence=0.9
    )
    ev2 = Evidence.create(
        provider="tesseract", provider_version="1.0", raw_output="Ch I", processing_stage=ProcessingStage.OCR, provider_confidence=0.6
    )
    obs1 = Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Ch 1", level=1), evidence=(ev1,))
    obs2 = Observation.from_evidence(provider_id="tesseract", provider_version="1.0", payload=HeadingPayload(text="Ch I", level=1), evidence=(ev2,))

    report = build_evidence_report(
        cluster_id="cluster-1",
        contributing_observations=(obs1, obs2),
        evidence_by_id={ev1.evidence_id: ev1, ev2.evidence_id: ev2},
    )
    confidences = {(e.provider_id, e.provider_confidence) for e in report.supporting}
    assert confidences == {("docling", 0.9), ("tesseract", 0.6)}
