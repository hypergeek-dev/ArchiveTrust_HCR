"""Confidence calibration: sample-size statistics, per-pattern progress, information-gain
scoring, and blind-spot coverage analysis (Milestone 11).

Pattern *discovery* (which needs `TelemetryIndex`/`review.triage`) is not re-exported here -- see
`review/sampling/discovery.py` -- to keep this package's own dependency-direction test
(`tests/domain/test_dependency_direction.py`) honest: the domain layer imports nothing from
`learning` or `review`.
"""

from archivetrust.domain.calibration.agreement import (
    AdjudicationItem,
    AdjudicationReason,
    AgreementMetric,
    AgreementReport,
    ReviewerLabel,
    agreement_report,
    cohens_kappa,
    fleiss_kappa,
    items_requiring_adjudication,
    percent_agreement,
)
from archivetrust.domain.calibration.coverage import BlindSpot, BlindSpotReason, find_blind_spots
from archivetrust.domain.calibration.decision_policy import (
    Decision,
    DecisionAction,
    DecisionPolicy,
    DecisionReason,
    G2Certification,
    decide,
)
from archivetrust.domain.calibration.information_gain import (
    expected_information_gain,
    expected_variance_reduction,
)
from archivetrust.domain.calibration.pattern import CalibrationPattern, classification_type_key
from archivetrust.domain.calibration.progress import (
    MARGINS,
    CalibrationMaturity,
    CalibrationProgress,
    compute_calibration_progress,
)
from archivetrust.domain.calibration.statistics import Z_95, required_sample_size, wilson_ci

__all__ = [
    "AdjudicationItem",
    "AdjudicationReason",
    "AgreementMetric",
    "AgreementReport",
    "MARGINS",
    "BlindSpot",
    "BlindSpotReason",
    "CalibrationMaturity",
    "CalibrationPattern",
    "CalibrationProgress",
    "Decision",
    "DecisionAction",
    "DecisionPolicy",
    "DecisionReason",
    "G2Certification",
    "ReviewerLabel",
    "Z_95",
    "agreement_report",
    "classification_type_key",
    "compute_calibration_progress",
    "cohens_kappa",
    "decide",
    "expected_information_gain",
    "expected_variance_reduction",
    "fleiss_kappa",
    "find_blind_spots",
    "items_requiring_adjudication",
    "percent_agreement",
    "required_sample_size",
    "wilson_ci",
]
