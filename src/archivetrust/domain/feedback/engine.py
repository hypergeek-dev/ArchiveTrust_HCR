"""Applies a `HumanCorrection` to a `CanonicalObservation` (Constitution Article 15: supersession,
never erasure -- a correction always produces a *new* version via `CanonicalObservation.supersede`,
never an in-place edit).

**Scope, stated honestly (mirrors ROADMAP.md's own Risk-table entry: "Milestone 6 ships the data
model + capture contract only"):**
- `ACCEPT`, `REJECT`, `EDIT`, `FLAG_FOR_REVIEW` are fully implemented: each produces a superseding
  `CanonicalObservation` with an updated `canonical_confidence` reflecting the human verdict.
- `EDIT` only replaces payload content for payload types carrying a `.text` field (the same
  `hasattr(payload, "text")` test `domain/comparison/engine.py` uses) -- `raw_corrected_output` is
  a plain string (ROADMAP.md S8's own field list), not a structured payload, so types without a
  single text field to overwrite keep their original payload unchanged.
- `MERGE`/`SPLIT` are recorded (the correction itself, and a superseding version noting the
  proposal) but their structural execution -- rewriting `ReconciledObservationGraph` edges across
  *multiple* Canonical Observations -- is not implemented. That is graph surgery spanning more
  than one object, which is Review-UI/Milestone-7+ tooling territory, not a single-object
  supersession this function can express.

**Comparison Confidence is never touched here** -- it reflects cross-provider agreement, a fact
about the automated pipeline's state (Milestone 4's territory), not about human review. Only
`canonical_confidence` (Milestone 5's territory, "all evidence considered" per S5.3 -- human
feedback is evidence) changes.
"""

from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import CanonicalConfidence
from archivetrust.domain.feedback.models import CorrectionAction, HumanCorrection
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.ontology.base import Observation

_CONFIDENCE_BY_ACTION_NAME = {
    CorrectionAction.ACCEPT: "accepted_confidence",
    CorrectionAction.EDIT: "edited_confidence",
    CorrectionAction.REJECT: "rejected_confidence",
    CorrectionAction.FLAG_FOR_REVIEW: "flagged_confidence",
    CorrectionAction.MERGE: "flagged_confidence",
    CorrectionAction.SPLIT: "flagged_confidence",
    CorrectionAction.DIFFERENT_THINGS: "flagged_confidence",
}


def _confidence_value_for(action: CorrectionAction, policy: FeedbackPolicy) -> float:
    return getattr(policy, _CONFIDENCE_BY_ACTION_NAME[action])


def apply_human_correction(
    original: CanonicalObservation,
    correction: HumanCorrection,
    contributing_observations: tuple[Observation, ...],
    policy: FeedbackPolicy,
) -> CanonicalObservation:
    if correction.target_canonical_observation_id != original.canonical_observation_id:
        raise ValueError(
            "HumanCorrection.target_canonical_observation_id does not match the "
            "CanonicalObservation it is being applied to"
        )

    payload = original.payload
    if correction.action == CorrectionAction.EDIT:
        if correction.raw_corrected_output is None:
            raise ValueError("EDIT correction requires raw_corrected_output")
        if hasattr(payload, "text"):
            payload = payload.model_copy(update={"text": correction.raw_corrected_output})
        # Non-text payload types: original payload retained (see module docstring).

    confidence_value = policy.round_score(_confidence_value_for(correction.action, policy))
    derivation = (
        f"Human correction {correction.correction_id}: action={correction.action.value}, "
        f"category={correction.category.value} (Feedback Policy v{policy.feedback_policy_version})"
    )
    if correction.action in (CorrectionAction.MERGE, CorrectionAction.SPLIT):
        derivation += (
            " -- structural proposal recorded; graph restructuring across multiple Canonical "
            "Observations is not executed by this Milestone 6 implementation."
        )
    canonical_confidence = CanonicalConfidence(value=confidence_value, derivation=derivation)

    return original.supersede(
        payload=payload,
        contributing_observations=contributing_observations,
        comparison_confidence=original.comparison_confidence,
        clustering_basis=original.clustering_basis,
        reconciliation_basis=(
            f"human correction applied: {correction.action.value} ({correction.category.value})"
        ),
        canonical_confidence=canonical_confidence,
        human_correction_ref=correction.correction_id,
        rationale=correction.rationale,
    )
