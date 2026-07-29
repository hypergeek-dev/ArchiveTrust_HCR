"""Calibration-gated decision policy (Master Execution Program B4 / Milestone 11 Phase 5).

This module is deliberately downstream of both the Confidence Engine and calibration progress:
it does not change confidence, review triage, telemetry, or reconciliation. It answers one pure
question for a candidate production policy: is this canonical observation calibrated enough to
auto-accept, or must it stay in human review?
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from archivetrust.domain.calibration.progress import CalibrationMaturity, CalibrationProgress
from archivetrust.domain.confidence.models import CanonicalConfidence


class DecisionAction(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    HUMAN_REVIEW = "human_review"


class DecisionReason(str, Enum):
    ACCEPTABLE_CALIBRATED_RISK = "acceptable_calibrated_risk"
    G2_CERTIFICATION_REQUIRED = "g2_certification_required"
    MISSING_CANONICAL_CONFIDENCE = "missing_canonical_confidence"
    NO_CALIBRATION_EVIDENCE = "no_calibration_evidence"
    BELOW_MINIMUM_MATURITY = "below_minimum_maturity"
    ERROR_RATE_TOO_HIGH = "error_rate_too_high"


class DecisionPolicy(BaseModel):
    """Versioned policy, following the same frozen/versioned convention as confidence policies."""

    model_config = ConfigDict(frozen=True)

    decision_policy_version: int = 1
    max_acceptable_error_rate: float = 0.05
    min_calibration_maturity: CalibrationMaturity = CalibrationMaturity.MODERATE_CONFIDENCE

    @field_validator("decision_policy_version")
    @classmethod
    def _positive_version(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("decision_policy_version must be positive")
        return value

    @field_validator("max_acceptable_error_rate")
    @classmethod
    def _rate_in_unit_interval(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("max_acceptable_error_rate must be between 0 and 1")
        return value


class G2Certification(BaseModel):
    """Evidence that a specific decision-policy version has been certified for production use."""

    model_config = ConfigDict(frozen=True)

    certification_id: str
    campaign_id: str
    decision_policy_version: int
    ready_for_g2: bool
    held_out_false_acceptance_rate: float

    @field_validator("decision_policy_version")
    @classmethod
    def _positive_policy_version(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("decision_policy_version must be positive")
        return value

    @field_validator("held_out_false_acceptance_rate")
    @classmethod
    def _false_acceptance_rate_in_unit_interval(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("held_out_false_acceptance_rate must be between 0 and 1")
        return value


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: DecisionAction
    reason: DecisionReason
    decision_policy_version: int
    calibration_pattern: str
    calibration_maturity: CalibrationMaturity
    canonical_confidence_value: float | None
    estimated_error_rate: float | None
    estimated_error_rate_ci: tuple[float, float] | None
    """95% Wilson interval transformed from correctness to error probability."""
    g2_certification_id: str | None = None


_MATURITY_RANK = {
    CalibrationMaturity.NO_EVIDENCE: 0,
    CalibrationMaturity.EARLY_EVIDENCE: 1,
    CalibrationMaturity.EMERGING_CALIBRATION: 2,
    CalibrationMaturity.MODERATE_CONFIDENCE: 3,
    CalibrationMaturity.STATISTICALLY_MATURE: 4,
}


def decide(
    canonical_confidence: CanonicalConfidence | None,
    calibration_progress: CalibrationProgress,
    policy: DecisionPolicy | None = None,
    *,
    production: bool = False,
    g2_certification: G2Certification | None = None,
) -> Decision:
    """Return AUTO_ACCEPT only when confidence exists and calibration evidence is mature enough.

    The risk gate uses the upper bound of the transformed Wilson error interval when available,
    not only the point estimate. That makes auto-acceptance conservative while preserving the
    reported point estimate for analysis. `production=True` adds the B4/G2 adoption gate: the
    policy may only auto-accept when a matching G2 certification is supplied.
    """
    policy = policy or DecisionPolicy()
    confidence_value = canonical_confidence.value if canonical_confidence is not None else None
    error_rate = _estimated_error_rate(calibration_progress)
    error_ci = _estimated_error_ci(calibration_progress)

    if production and not _g2_allows_production(policy, g2_certification):
        return _decision(
            DecisionAction.HUMAN_REVIEW,
            DecisionReason.G2_CERTIFICATION_REQUIRED,
            policy,
            calibration_progress,
            confidence_value,
            error_rate,
            error_ci,
            g2_certification,
        )
    if canonical_confidence is None:
        return _decision(
            DecisionAction.HUMAN_REVIEW,
            DecisionReason.MISSING_CANONICAL_CONFIDENCE,
            policy,
            calibration_progress,
            confidence_value,
            error_rate,
            error_ci,
            g2_certification,
        )
    if error_rate is None:
        return _decision(
            DecisionAction.HUMAN_REVIEW,
            DecisionReason.NO_CALIBRATION_EVIDENCE,
            policy,
            calibration_progress,
            confidence_value,
            error_rate,
            error_ci,
            g2_certification,
        )
    if _MATURITY_RANK[calibration_progress.maturity] < _MATURITY_RANK[policy.min_calibration_maturity]:
        return _decision(
            DecisionAction.HUMAN_REVIEW,
            DecisionReason.BELOW_MINIMUM_MATURITY,
            policy,
            calibration_progress,
            confidence_value,
            error_rate,
            error_ci,
            g2_certification,
        )

    gated_error_rate = error_ci[1] if error_ci is not None else error_rate
    if gated_error_rate > policy.max_acceptable_error_rate:
        return _decision(
            DecisionAction.HUMAN_REVIEW,
            DecisionReason.ERROR_RATE_TOO_HIGH,
            policy,
            calibration_progress,
            confidence_value,
            error_rate,
            error_ci,
            g2_certification,
        )

    return _decision(
        DecisionAction.AUTO_ACCEPT,
        DecisionReason.ACCEPTABLE_CALIBRATED_RISK,
        policy,
        calibration_progress,
        confidence_value,
        error_rate,
        error_ci,
        g2_certification,
    )


def _estimated_error_rate(progress: CalibrationProgress) -> float | None:
    if progress.empirical_correctness is None:
        return None
    return round(1 - progress.empirical_correctness, 4)


def _estimated_error_ci(progress: CalibrationProgress) -> tuple[float, float] | None:
    if progress.wilson_95_ci is None:
        return None
    correctness_low, correctness_high = progress.wilson_95_ci
    return (round(1 - correctness_high, 4), round(1 - correctness_low, 4))


def _g2_allows_production(policy: DecisionPolicy, certification: G2Certification | None) -> bool:
    return (
        certification is not None
        and certification.ready_for_g2
        and certification.decision_policy_version == policy.decision_policy_version
        and certification.held_out_false_acceptance_rate <= policy.max_acceptable_error_rate
    )


def _decision(
    action: DecisionAction,
    reason: DecisionReason,
    policy: DecisionPolicy,
    progress: CalibrationProgress,
    confidence_value: float | None,
    error_rate: float | None,
    error_ci: tuple[float, float] | None,
    g2_certification: G2Certification | None = None,
) -> Decision:
    return Decision(
        action=action,
        reason=reason,
        decision_policy_version=policy.decision_policy_version,
        calibration_pattern=progress.pattern,
        calibration_maturity=progress.maturity,
        canonical_confidence_value=confidence_value,
        estimated_error_rate=error_rate,
        estimated_error_rate_ci=error_ci,
        g2_certification_id=g2_certification.certification_id if g2_certification is not None else None,
    )
