"""Policy simulation (Milestone 11, Phase 9): estimate the effect of a candidate `TriagePolicy`
and a candidate calibration sampling budget, entirely from already-persisted telemetry.

No provider is invoked and no document is reprocessed -- every number here is either a direct
recount of `review_reason_for` over the corpus's existing canonical slots (for the operational
queue-size comparison) or a projection built from `domain/calibration/` statistics applied to a
hypothetical future review count (for calibration convergence). Production behavior is never
touched: this module only reads a `TelemetryIndex` and a `SamplingLogSink`, it writes nothing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.calibration.decision_policy import (
    DecisionAction,
    DecisionPolicy,
    decide,
)
from archivetrust.domain.calibration.pattern import CalibrationPattern, classification_type_key
from archivetrust.domain.calibration.progress import CalibrationMaturity, compute_calibration_progress
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.review.sampling.log import SamplingLogSink
from archivetrust.review.sampling.progress import compute_calibration_progress_map
from archivetrust.review.triage import TriagePolicy, review_reason_for


class PatternProjection(BaseModel):
    model_config = ConfigDict(frozen=True)

    pattern: str
    population: int
    current_reviews: int
    current_maturity: CalibrationMaturity
    projected_additional_reviews: int
    projected_reviews: int
    projected_maturity: CalibrationMaturity
    """Assumes the pattern's currently observed correctness rate continues to hold (or, for a
    pattern with zero current evidence, the same worst-case `p=0.5` `compute_calibration_progress`
    itself falls back to) -- an explicit, stated assumption, never a fabricated future outcome."""


class PolicySimulationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline_triage_policy_version: int
    candidate_triage_policy_version: int
    baseline_operational_queue_size: int
    candidate_operational_queue_size: int
    operational_queue_delta: int
    """`candidate - baseline`; negative means the candidate policy queues fewer items."""
    reviews_saved: int
    """`max(baseline - candidate, 0)` -- zero if the candidate queues the same or more."""
    candidate_calibration_reviews_simulated: int
    pattern_projections: tuple[PatternProjection, ...]


class DecisionPolicyProjection(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision_policy_version: int
    max_acceptable_error_rate: float
    min_calibration_maturity: CalibrationMaturity
    canonical_slots_evaluated: int
    automatic_acceptances: int
    human_reviews: int
    automatic_acceptance_rate: float
    expected_false_acceptances: float
    estimated_false_acceptance_rate: float | None
    """Estimated from calibration evidence only; held-out adjudicated labels are still required
    before reporting a production false-acceptance rate."""


def simulate_policy(
    index: TelemetryIndex,
    sampling_log: SamplingLogSink,
    *,
    baseline_triage_policy: TriagePolicy | None = None,
    candidate_triage_policy: TriagePolicy | None = None,
    candidate_calibration_reviews: int = 0,
) -> PolicySimulationResult:
    baseline = baseline_triage_policy or TriagePolicy()
    candidate = candidate_triage_policy or TriagePolicy()
    canonicals = index.latest_canonicals()

    baseline_queue = sum(1 for c in canonicals if review_reason_for(c, baseline) is not None)
    candidate_queue = sum(1 for c in canonicals if review_reason_for(c, candidate) is not None)

    progress_now = compute_calibration_progress_map(index, sampling_log)
    total_population = sum(p.population for p in progress_now.values())

    projections: list[PatternProjection] = []
    if candidate_calibration_reviews > 0 and total_population:
        for name, progress in progress_now.items():
            share = progress.population / total_population
            allocated = round(share * candidate_calibration_reviews)
            if allocated <= 0:
                continue
            rate = (
                progress.corrections_recorded / progress.reviews_completed
                if progress.reviews_completed > 0
                else 0.5
            )
            projected_reviews = progress.reviews_completed + allocated
            projected_corrections = round(rate * projected_reviews)
            projected = compute_calibration_progress(
                CalibrationPattern(name=name, population=progress.population, queued_for_review=0),
                reviews_completed=projected_reviews,
                corrections_recorded=min(projected_corrections, projected_reviews),
            )
            projections.append(
                PatternProjection(
                    pattern=name,
                    population=progress.population,
                    current_reviews=progress.reviews_completed,
                    current_maturity=progress.maturity,
                    projected_additional_reviews=allocated,
                    projected_reviews=projected_reviews,
                    projected_maturity=projected.maturity,
                )
            )

    return PolicySimulationResult(
        baseline_triage_policy_version=baseline.version,
        candidate_triage_policy_version=candidate.version,
        baseline_operational_queue_size=baseline_queue,
        candidate_operational_queue_size=candidate_queue,
        operational_queue_delta=candidate_queue - baseline_queue,
        reviews_saved=max(baseline_queue - candidate_queue, 0),
        candidate_calibration_reviews_simulated=candidate_calibration_reviews,
        pattern_projections=tuple(projections),
    )


def simulate_decision_policy(
    index: TelemetryIndex,
    sampling_log: SamplingLogSink,
    *,
    policies: tuple[DecisionPolicy, ...] | None = None,
) -> tuple[DecisionPolicyProjection, ...]:
    """Project B4 decision policies over the current replayed corpus without changing behavior."""
    candidate_policies = policies or (DecisionPolicy(),)
    progress_map = compute_calibration_progress_map(index, sampling_log)
    canonicals = index.latest_canonicals()

    projections: list[DecisionPolicyProjection] = []
    for policy in candidate_policies:
        automatic_acceptances = 0
        expected_false_acceptances = 0.0
        evaluated = 0
        for canonical in canonicals:
            progress = progress_map.get(classification_type_key(canonical))
            if progress is None:
                continue
            evaluated += 1
            decision = decide(canonical.canonical_confidence, progress, policy)
            if decision.action == DecisionAction.AUTO_ACCEPT:
                automatic_acceptances += 1
                expected_false_acceptances += decision.estimated_error_rate or 0.0

        auto_rate = round(automatic_acceptances / evaluated, 4) if evaluated else 0.0
        estimated_far = (
            round(expected_false_acceptances / automatic_acceptances, 4)
            if automatic_acceptances
            else None
        )
        projections.append(
            DecisionPolicyProjection(
                decision_policy_version=policy.decision_policy_version,
                max_acceptable_error_rate=policy.max_acceptable_error_rate,
                min_calibration_maturity=policy.min_calibration_maturity,
                canonical_slots_evaluated=evaluated,
                automatic_acceptances=automatic_acceptances,
                human_reviews=evaluated - automatic_acceptances,
                automatic_acceptance_rate=auto_rate,
                expected_false_acceptances=round(expected_false_acceptances, 4),
                estimated_false_acceptance_rate=estimated_far,
            )
        )
    return tuple(projections)
