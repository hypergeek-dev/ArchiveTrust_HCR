from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.confidence.models import (
    CanonicalConfidence,
    ComparisonClassification,
    ComparisonConfidence,
    ProviderConfidence,
)


def test_provider_confidence_requires_provider_and_version():
    pc = ProviderConfidence(provider="docling", provider_version="1.0", value=0.8)
    assert pc.provider == "docling"


def test_provider_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ProviderConfidence(provider="docling", provider_version="1.0", value=1.5)


def test_uncorroborated_single_source_must_not_carry_a_magnitude():
    with pytest.raises(ValidationError):
        ComparisonConfidence(
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
            magnitude=0.1,
            basis="bogus",
        )


def test_uncorroborated_single_source_is_a_real_marker_not_a_low_number():
    cc = ComparisonConfidence(
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        magnitude=None,
        basis="only one provider attempted this slot",
    )
    assert cc.magnitude is None


def test_corroborated_requires_a_magnitude():
    with pytest.raises(ValidationError):
        ComparisonConfidence(
            classification=ComparisonClassification.CORROBORATED, magnitude=None, basis="x"
        )


def test_contested_and_corroborated_are_distinguishable_from_uncorroborated():
    contested = ComparisonConfidence(
        classification=ComparisonClassification.CONTESTED, magnitude=0.3, basis="providers disagree"
    )
    corroborated = ComparisonConfidence(
        classification=ComparisonClassification.CORROBORATED, magnitude=0.95, basis="all agree"
    )
    assert contested.classification != corroborated.classification
    assert {contested.classification, corroborated.classification}.isdisjoint(
        {ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE}
    )


def test_canonical_confidence_carries_a_derivation():
    cc = CanonicalConfidence(value=0.7, derivation="derived from comparison confidence 0.7")
    assert cc.derivation


def test_no_confidence_type_accepts_bare_float_without_level_context():
    # Each of the three levels requires structural context beyond a float: ProviderConfidence
    # needs (provider, provider_version); ComparisonConfidence needs (classification, basis);
    # CanonicalConfidence needs (derivation). None can be constructed from a float alone.
    for confidence_cls in (ProviderConfidence, ComparisonConfidence, CanonicalConfidence):
        with pytest.raises((ValidationError, TypeError)):
            confidence_cls(0.5)  # type: ignore[call-arg]
