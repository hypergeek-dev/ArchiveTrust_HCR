"""Versioned confidence outcomes for each `CorrectionAction` (mirrors `ConfidencePolicy`'s
versioning discipline, `domain/confidence/policy.py`). A human correction is itself a form of
evidence Milestone 5's Confidence Engine doesn't see -- these are placeholder, not empirically
validated, values (same posture as `ReconciliationPolicy`'s table-lattice tolerance, explicitly
provisional pending real review-workflow experience).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FeedbackPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    feedback_policy_version: int

    accepted_confidence: float = 0.95
    """A human explicitly confirmed the value is correct."""

    edited_confidence: float = 0.9
    """A human provided a corrected value -- trusted, though marginally less than an explicit
    accept of an unchanged value, since the correction itself could still be imperfect."""

    rejected_confidence: float = 0.1
    """A human flagged the value as wrong with no replacement given -- low confidence, not zero
    (the slot still exists; rejection is a strong negative signal, not proof of nonexistence)."""

    flagged_confidence: float = 0.3
    """A human deferred judgment -- lower than the pre-correction state should ever have been
    trusted at, since a reviewer thought it worth a second look."""

    rounding_ndigits: int = 6

    def round_score(self, value: float) -> float:
        return round(value, self.rounding_ndigits)
