"""Phase F -- Agreement Report, Evidence Report, Comparison Confidence, Trust Score, Risk Score
(MILESTONE4_COMPARISON_ENGINE.md S7-S11).

Trust and Risk are **two distinct structured outputs over one shared Agreement Report state**, per
S11's resolution of ROADMAP S15 Q6 -- never `1 - confidence` (that would re-conflate Uncertainty
and Disagreement, Constitution Article 11). `RiskType` is deliberately extensible with an `OTHER`
catch-all (S15: "an unenumerated disagreement type silently missing ... mitigated by the
extensible risk_type enum + a catch-all").
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.comparison.matching import MatchCardinality, ProviderContribution
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.types import ObservationType


class RiskType(str, Enum):
    VALUE_CONFLICT = "value_conflict"
    STRUCTURAL_CONFLICT = "structural_conflict"
    SEGMENTATION_CONFLICT = "segmentation_conflict"
    MISSING_EXPECTED = "missing_expected"
    SINGLE_SOURCE_UNCORROBORATED = "single_source_uncorroborated"
    LOW_PRECISION_GEOMETRY = "low_precision_geometry"
    VLM_ONLY_UNVERIFIED = "vlm_only_unverified"
    CYCLE_BROKEN = "cycle_broken"
    OTHER = "other"


class RiskItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    risk_type: RiskType
    involved_observation_ids: tuple[str, ...]
    involved_providers: tuple[str, ...]
    locus: str
    severity: float
    description: str


class ProviderConfidenceEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    observation_id: str
    evidence_id: str
    provider_confidence: float | None


class EvidenceReport(BaseModel):
    """The Article-4 traceability ledger for one cluster (S8). `provider_confidence` is always
    presented per-provider, side by side, never averaged (C8)."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    supporting: tuple[ProviderConfidenceEntry, ...]
    conflicting: tuple[ProviderConfidenceEntry, ...] = ()
    shared_contribution_observation_ids: tuple[str, ...] = ()


class AgreementReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    cluster_id: str
    observation_type: ObservationType
    clustering_basis: str
    participants: tuple[ProviderContribution, ...]
    classification: ComparisonClassification
    independent_corroborating_count: int
    disagreements: tuple[RiskItem, ...]


class TrustScore(BaseModel):
    """A positive corroboration signal -- how much independent, capability-weighted agreement
    supports the accepted value. Never `1 - RiskScore.magnitude` (S11)."""

    model_config = ConfigDict(frozen=True)

    magnitude: float
    basis: str


class RiskScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    magnitude: float
    risk_items: tuple[RiskItem, ...]


def build_agreement_report(
    *,
    cluster_id: str,
    observation_type: ObservationType,
    clustering_basis: str,
    contributions: tuple[ProviderContribution, ...],
    value_comparison_confidence: ComparisonConfidence | None,
    policy: ReconciliationPolicy,
) -> AgreementReport:
    independent = tuple(
        c for c in contributions if c.cardinality in (MatchCardinality.ONE_TO_ONE, MatchCardinality.MANY_TO_ONE)
    )
    if value_comparison_confidence is not None:
        classification = value_comparison_confidence.classification
    elif len(independent) <= 1:
        classification = ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE
    else:
        classification = ComparisonClassification.CORROBORATED

    disagreements: list[RiskItem] = []
    if classification == ComparisonClassification.CONTESTED:
        disagreements.append(
            RiskItem(
                risk_type=RiskType.VALUE_CONFLICT,
                involved_observation_ids=tuple(
                    oid for c in independent for oid in c.observation_ids
                ),
                involved_providers=tuple(c.provider_id for c in independent),
                locus=cluster_id,
                severity=policy.round_score(1.0 - (value_comparison_confidence.magnitude or 0.0))
                if value_comparison_confidence
                else 0.5,
                description="Contributing sources disagree on the accepted value beyond the "
                "Policy's corroboration threshold.",
            )
        )
    if len(independent) <= 1:
        disagreements.append(
            RiskItem(
                risk_type=RiskType.SINGLE_SOURCE_UNCORROBORATED,
                involved_observation_ids=tuple(oid for c in independent for oid in c.observation_ids),
                involved_providers=tuple(c.provider_id for c in independent),
                locus=cluster_id,
                severity=0.5,
                description="Only one independent provider contributed to this slot; nothing "
                "corroborates it, and nothing contradicts it either.",
            )
        )
    missing = tuple(c for c in contributions if c.cardinality == MatchCardinality.MISSING)
    if missing:
        disagreements.append(
            RiskItem(
                risk_type=RiskType.MISSING_EXPECTED,
                involved_observation_ids=(),
                involved_providers=tuple(c.provider_id for c in missing),
                locus=cluster_id,
                severity=policy.round_score(len(missing) / max(1, len(contributions))),
                description="A capable provider was invoked but produced no observation of this "
                "type for this slot.",
            )
        )

    return AgreementReport(
        cluster_id=cluster_id,
        observation_type=observation_type,
        clustering_basis=clustering_basis,
        participants=contributions,
        classification=classification,
        independent_corroborating_count=len(independent),
        disagreements=tuple(disagreements),
    )


def build_evidence_report(
    *,
    cluster_id: str,
    contributing_observations: tuple[Observation, ...],
    evidence_by_id: dict[str, Evidence],
    shared_contribution_observation_ids: tuple[str, ...] = (),
) -> EvidenceReport:
    entries = []
    for observation in contributing_observations:
        for evidence_id in observation.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            entries.append(
                ProviderConfidenceEntry(
                    provider_id=observation.provider_id,
                    provider_version=observation.provider_version,
                    observation_id=observation.observation_id,
                    evidence_id=evidence_id,
                    provider_confidence=evidence.provider_confidence,
                )
            )
    return EvidenceReport(
        cluster_id=cluster_id,
        supporting=tuple(entries),
        shared_contribution_observation_ids=shared_contribution_observation_ids,
    )


def compute_trust_score(agreement_report: AgreementReport, policy: ReconciliationPolicy) -> TrustScore:
    independent = agreement_report.independent_corroborating_count
    total_participants = max(1, len(agreement_report.participants))
    magnitude = policy.round_score(independent / total_participants) if independent >= 2 else 0.0
    basis = (
        f"{independent} independent corroborating source(s) out of {total_participants} "
        f"participant(s) (Policy v{policy.policy_version})"
    )
    return TrustScore(magnitude=magnitude, basis=basis)


def compute_risk_score(agreement_report: AgreementReport, policy: ReconciliationPolicy) -> RiskScore:
    if not agreement_report.disagreements:
        return RiskScore(magnitude=0.0, risk_items=())
    magnitude = policy.round_score(
        min(1.0, sum(item.severity for item in agreement_report.disagreements) / len(agreement_report.disagreements))
    )
    return RiskScore(magnitude=magnitude, risk_items=agreement_report.disagreements)
