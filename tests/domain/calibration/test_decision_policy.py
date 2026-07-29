from __future__ import annotations

import pytest

from archivetrust.domain.calibration.decision_policy import (
    DecisionAction,
    DecisionPolicy,
    DecisionReason,
    G2Certification,
    decide,
)
from archivetrust.domain.calibration.pattern import CalibrationPattern
from archivetrust.domain.calibration.progress import (
    CalibrationMaturity,
    compute_calibration_progress,
)
from archivetrust.domain.confidence.models import CanonicalConfidence


def _progress(*, reviews: int, corrections: int):
    return compute_calibration_progress(
        CalibrationPattern(name="contested/heading", population=100, queued_for_review=0),
        reviews_completed=reviews,
        corrections_recorded=corrections,
    )


def test_policy_auto_accepts_only_when_calibrated_risk_bound_is_within_policy() -> None:
    progress = _progress(reviews=100, corrections=0)
    policy = DecisionPolicy(max_acceptable_error_rate=0.05)

    decision = decide(CanonicalConfidence(value=0.92, derivation="test"), progress, policy)

    assert decision.action is DecisionAction.AUTO_ACCEPT
    assert decision.reason is DecisionReason.ACCEPTABLE_CALIBRATED_RISK
    assert decision.estimated_error_rate == 0.0
    assert decision.estimated_error_rate_ci is not None
    assert decision.estimated_error_rate_ci[1] <= policy.max_acceptable_error_rate


def test_policy_routes_missing_confidence_to_human_review() -> None:
    decision = decide(None, _progress(reviews=100, corrections=0))

    assert decision.action is DecisionAction.HUMAN_REVIEW
    assert decision.reason is DecisionReason.MISSING_CANONICAL_CONFIDENCE


def test_policy_routes_patterns_without_calibration_evidence_to_human_review() -> None:
    decision = decide(CanonicalConfidence(value=0.9, derivation="test"), _progress(reviews=0, corrections=0))

    assert decision.action is DecisionAction.HUMAN_REVIEW
    assert decision.reason is DecisionReason.NO_CALIBRATION_EVIDENCE


def test_policy_routes_immature_calibration_to_human_review() -> None:
    policy = DecisionPolicy(min_calibration_maturity=CalibrationMaturity.STATISTICALLY_MATURE)
    decision = decide(CanonicalConfidence(value=0.9, derivation="test"), _progress(reviews=10, corrections=2), policy)

    assert decision.action is DecisionAction.HUMAN_REVIEW
    assert decision.reason is DecisionReason.BELOW_MINIMUM_MATURITY


def test_policy_routes_high_error_rate_to_human_review() -> None:
    policy = DecisionPolicy(max_acceptable_error_rate=0.05)
    decision = decide(CanonicalConfidence(value=0.9, derivation="test"), _progress(reviews=100, corrections=20), policy)

    assert decision.action is DecisionAction.HUMAN_REVIEW
    assert decision.reason is DecisionReason.ERROR_RATE_TOO_HIGH


def test_policy_validation_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        DecisionPolicy(max_acceptable_error_rate=1.5)
    with pytest.raises(ValueError, match="positive"):
        DecisionPolicy(decision_policy_version=0)


def test_production_mode_requires_g2_certification() -> None:
    progress = _progress(reviews=100, corrections=0)
    decision = decide(
        CanonicalConfidence(value=0.92, derivation="test"),
        progress,
        DecisionPolicy(max_acceptable_error_rate=0.05),
        production=True,
    )

    assert decision.action is DecisionAction.HUMAN_REVIEW
    assert decision.reason is DecisionReason.G2_CERTIFICATION_REQUIRED
    assert decision.g2_certification_id is None


def test_production_mode_rejects_mismatched_g2_certification() -> None:
    progress = _progress(reviews=100, corrections=0)
    policy = DecisionPolicy(decision_policy_version=2, max_acceptable_error_rate=0.05)
    certification = G2Certification(
        certification_id="g2-1",
        campaign_id="campaign-1",
        decision_policy_version=1,
        ready_for_g2=True,
        held_out_false_acceptance_rate=0.01,
    )

    decision = decide(
        CanonicalConfidence(value=0.92, derivation="test"),
        progress,
        policy,
        production=True,
        g2_certification=certification,
    )

    assert decision.action is DecisionAction.HUMAN_REVIEW
    assert decision.reason is DecisionReason.G2_CERTIFICATION_REQUIRED
    assert decision.g2_certification_id == "g2-1"


def test_production_mode_rejects_g2_certification_above_policy_risk() -> None:
    progress = _progress(reviews=100, corrections=0)
    policy = DecisionPolicy(max_acceptable_error_rate=0.05)
    certification = G2Certification(
        certification_id="g2-1",
        campaign_id="campaign-1",
        decision_policy_version=policy.decision_policy_version,
        ready_for_g2=True,
        held_out_false_acceptance_rate=0.08,
    )

    decision = decide(
        CanonicalConfidence(value=0.92, derivation="test"),
        progress,
        policy,
        production=True,
        g2_certification=certification,
    )

    assert decision.action is DecisionAction.HUMAN_REVIEW
    assert decision.reason is DecisionReason.G2_CERTIFICATION_REQUIRED


def test_production_mode_can_auto_accept_with_matching_g2_certification() -> None:
    progress = _progress(reviews=100, corrections=0)
    policy = DecisionPolicy(max_acceptable_error_rate=0.05)
    certification = G2Certification(
        certification_id="g2-1",
        campaign_id="campaign-1",
        decision_policy_version=policy.decision_policy_version,
        ready_for_g2=True,
        held_out_false_acceptance_rate=0.01,
    )

    decision = decide(
        CanonicalConfidence(value=0.92, derivation="test"),
        progress,
        policy,
        production=True,
        g2_certification=certification,
    )

    assert decision.action is DecisionAction.AUTO_ACCEPT
    assert decision.reason is DecisionReason.ACCEPTABLE_CALIBRATED_RISK
    assert decision.g2_certification_id == "g2-1"
