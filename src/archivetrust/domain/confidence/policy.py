"""The Confidence Engine's versioned policy (ROADMAP.md S5.3.1, Constitution Article 13).

**Uncalibrated by default, on purpose.** S5.3.1 is explicit: "absent an explicit calibration
mechanism, the Confidence Engine's default posture is uncalibrated -- it must rely on Comparison
Confidence for cross-provider trust signal and treat raw Provider Confidence as informative only
within its own provider's values, never blended across providers as if equivalent." This policy
has no field that could combine two different providers' Provider Confidence values -- there is
structurally nothing here *to* blend, which is a stronger guarantee than a rule an implementer
could accidentally violate later.

Versioned exactly like `ReconciliationPolicy` (`domain/comparison/policy.py`), for the same reason
S5.3.1 requires of any future calibration mechanism: "It must be independently versioned, and that
version must be recorded so replay can identify which calibration produced a given Canonical
Confidence."
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfidencePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    confidence_policy_version: int

    contested_penalty_factor: float = 0.5
    """A CONTESTED slot's Comparison Confidence magnitude means "strong but conflicting signal,"
    never "trustworthy" (MILESTONE4_COMPARISON_ENGINE.md S9) -- this factor discounts it before
    it becomes Canonical Confidence, so a high-magnitude-but-contested value never outranks a
    lower-magnitude-but-uncontested one."""

    single_source_base_confidence: float = 0.4
    """Base Canonical Confidence for an UNCORROBORATED_SINGLE_SOURCE slot before any within-
    provider Provider Confidence adjustment. Deliberately below 0.5 -- uncorroborated is not
    "probably right," it is "we don't know," and the default posture should not imply otherwise."""

    rounding_ndigits: int = 6

    def round_score(self, value: float) -> float:
        return round(value, self.rounding_ndigits)
