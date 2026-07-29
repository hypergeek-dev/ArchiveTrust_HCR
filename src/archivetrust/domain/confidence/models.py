"""The three-level Confidence model (ROADMAP.md S5.3, Constitution Article 13).

Provider Confidence, Comparison Confidence, and Canonical Confidence are distinct types on
purpose -- a bare float is never an acceptable confidence value anywhere in the domain layer,
because it cannot say which of the three questions it answers. No function in this codebase may
accept an unqualified ``confidence: float`` parameter.

Cross-provider comparability (S5.3.1): ProviderConfidence is only ever meaningful together with
the ``provider``/``provider_version`` that produced it (structurally guaranteed here by requiring
both fields) and must never be combined or averaged across different providers as if on a shared
scale -- that responsibility belongs to a future, explicitly versioned calibration mechanism, not
to this type.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class ProviderConfidence(BaseModel):
    """A provider's own self-reported confidence in one piece of Evidence.

    Attached to Evidence, never to an Observation or Canonical Observation directly (S5.3 table).
    Always scoped to the (provider, provider_version) that produced it, so it can never be
    silently treated as comparable to another provider's self-report (S5.3.1).
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    provider_version: str
    value: float

    @model_validator(mode="after")
    def _validate_range(self) -> "ProviderConfidence":
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("ProviderConfidence.value must be within [0.0, 1.0]")
        return self


class ComparisonClassification(str, Enum):
    """The Trust Model's Uncertainty/Disagreement distinction (MILESTONE0_REVIEW.md S4,
    Constitution Article 11). A single low number can never distinguish "we could not corroborate
    this" from "we corroborated this and it scored low" -- this enum is what makes that
    distinction structurally impossible to collapse.
    """

    CORROBORATED = "corroborated"
    UNCORROBORATED_SINGLE_SOURCE = "uncorroborated_single_source"
    CONTESTED = "contested"


class ComparisonConfidence(BaseModel):
    """How strongly independent Observations agree (S5.3 table; shape per
    MILESTONE4_COMPARISON_ENGINE.md S9, defined here because Canonical Observation (Milestone 1)
    references it).

    ``magnitude`` is ``None`` exactly when ``classification`` is
    ``UNCORROBORATED_SINGLE_SOURCE`` -- an explicit not-applicable marker, never a low number
    (Constitution Article 11, MILESTONE1_DOMAIN_MODEL.md S1.2).
    """

    model_config = ConfigDict(frozen=True)

    classification: ComparisonClassification
    magnitude: float | None
    basis: str

    @model_validator(mode="after")
    def _validate_magnitude(self) -> "ComparisonConfidence":
        is_single_source = self.classification == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE
        if is_single_source and self.magnitude is not None:
            raise ValueError(
                "ComparisonConfidence.magnitude must be None for "
                "UNCORROBORATED_SINGLE_SOURCE -- it is a not-applicable marker, never a number"
            )
        if not is_single_source:
            if self.magnitude is None:
                raise ValueError(
                    "ComparisonConfidence.magnitude is required unless classification is "
                    "UNCORROBORATED_SINGLE_SOURCE"
                )
            if not (0.0 <= self.magnitude <= 1.0):
                raise ValueError("ComparisonConfidence.magnitude must be within [0.0, 1.0]")
        return self


class CanonicalConfidence(BaseModel):
    """How trustworthy a final canonical field is, all evidence considered (S5.3 table).

    Produced by the Confidence Engine (Milestone 5) from Comparison Confidence plus referenced
    Provider Confidence. Milestone 1 defines only the shape; Milestone 5 defines the derivation
    algorithm. ``derivation`` records a human-readable account of how this value was reached, so
    that "trustworthy, and why" (not just a number) survives to the canonical layer.
    """

    model_config = ConfigDict(frozen=True)

    value: float
    derivation: str

    @model_validator(mode="after")
    def _validate_range(self) -> "CanonicalConfidence":
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("CanonicalConfidence.value must be within [0.0, 1.0]")
        return self
