from __future__ import annotations

from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.text_reconciliation import (
    ReconciliationBasisCode,
    TextCandidate,
    TextClassification,
    levenshtein_distance,
    reconcile_text,
)


def _policy() -> ReconciliationPolicy:
    return ReconciliationPolicy(policy_version=1)


def test_levenshtein_distance_basic_cases():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("kitten", "sitting") == 3


def test_single_candidate_is_uncorroborated_not_a_low_number():
    result = reconcile_text(
        (TextCandidate(observation_id="o1", provider_id="docling", text="Chapter 1"),), _policy()
    )
    assert result.classification == TextClassification.UNCORROBORATED_SINGLE_SOURCE
    assert result.magnitude is None
    assert result.accepted_text == "Chapter 1"
    assert result.reconciliation_basis_code == ReconciliationBasisCode.TEXT_SINGLE_SOURCE_ONLY_PROVIDER


def test_multiple_candidates_from_one_provider_is_still_single_source_basis_code():
    # Constitution Article 2/C1/C8: within-provider agreement is not independent corroboration --
    # a distinct basis_code from the only-one-candidate case, even though both are single-source.
    result = reconcile_text(
        (
            TextCandidate(observation_id="o1", provider_id="docling", text="Chapter 1"),
            TextCandidate(observation_id="o2", provider_id="docling", text="Chapter 1"),
        ),
        _policy(),
    )
    assert result.classification == TextClassification.UNCORROBORATED_SINGLE_SOURCE
    assert result.reconciliation_basis_code == ReconciliationBasisCode.TEXT_SINGLE_SOURCE_SAME_PROVIDER


def test_exact_agreement_after_normalization_is_corroborated():
    result = reconcile_text(
        (
            TextCandidate(observation_id="o1", provider_id="docling", text="Chapter 1"),
            TextCandidate(observation_id="o2", provider_id="tesseract", text="Chapter  1"),
        ),
        _policy(),
    )
    assert result.classification == TextClassification.CORROBORATED
    assert result.magnitude == 1.0
    assert result.reconciliation_basis_code == ReconciliationBasisCode.TEXT_EXACT_EQUALITY_CORROBORATED


def test_minor_character_disagreement_is_corroborated_with_consensus_text():
    result = reconcile_text(
        (
            TextCandidate(observation_id="o1", provider_id="docling", text="Chapter 1"),
            TextCandidate(observation_id="o2", provider_id="tesseract", text="Chapter l"),
        ),
        _policy(),
    )
    assert result.classification == TextClassification.CORROBORATED
    assert result.accepted_text  # a real consensus string, not blank
    assert result.reconciliation_basis_code == ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS


def test_major_disagreement_is_contested_not_silently_picked():
    result = reconcile_text(
        (
            TextCandidate(observation_id="o1", provider_id="docling", text="1897"),
            TextCandidate(observation_id="o2", provider_id="tesseract", text="1867"),
        ),
        _policy(),
    )
    assert result.classification == TextClassification.CONTESTED
    assert len(result.disagreements) == 1
    # Beyond the corroboration threshold, this is a distinct basis_code from the corroborated
    # majority-vote case above (Product Audit 2026-07-16 S7/S10 item 7; Master Execution Program
    # S4.1-3): a character-vote splice across genuinely divergent candidates produces text no
    # provider ever wrote, so CONTESTED never reaches `_majority_vote_consensus`.
    assert result.reconciliation_basis_code == ReconciliationBasisCode.TEXT_CONTESTED_REFERENCE_PICK


def test_contested_accepted_text_is_a_real_candidate_never_a_synthesized_splice():
    # The regression this fix exists for: two Swedish fragments divergent enough to be CONTESTED
    # must never be spliced character-by-character into something neither provider produced (the
    # "Fegeoana k ak. Saoea 1 g upp..." class of packet from the 2026-07-16 audit).
    candidates = (
        TextCandidate(
            observation_id="o1",
            provider_id="docling",
            text="vegetarisk mat. Skola 1 har ett eget kok",
        ),
        TextCandidate(
            observation_id="o2",
            provider_id="tesseract",
            text="Fragorna om skolmaten tog upp hur matsedeln",
        ),
    )
    result = reconcile_text(candidates, _policy())
    assert result.classification == TextClassification.CONTESTED
    assert result.reconciliation_basis_code == ReconciliationBasisCode.TEXT_CONTESTED_REFERENCE_PICK
    # accepted_text is exactly one candidate's real text -- never a blended/spliced string.
    assert result.accepted_text in {c.text for c in candidates}
    assert "no reconciled value" in result.reconciliation_basis
    assert "human review required" in result.reconciliation_basis


def test_corroborated_still_uses_majority_vote_consensus():
    # Within threshold, the consensus mechanism is unchanged -- only CONTESTED lost it.
    result = reconcile_text(
        (
            TextCandidate(observation_id="o1", provider_id="docling", text="Chapter 1"),
            TextCandidate(observation_id="o2", provider_id="tesseract", text="Chapter l"),
        ),
        _policy(),
    )
    assert result.classification == TextClassification.CORROBORATED
    assert result.reconciliation_basis_code == ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS


def test_reconciliation_never_prefers_a_provider_by_identity():
    # Same content, providers swapped -- the accepted text must be identical either way (C1).
    a = reconcile_text(
        (
            TextCandidate(observation_id="o1", provider_id="docling", text="Land Registry"),
            TextCandidate(observation_id="o2", provider_id="tesseract", text="Land Registri"),
        ),
        _policy(),
    )
    b = reconcile_text(
        (
            TextCandidate(observation_id="o1", provider_id="tesseract", text="Land Registri"),
            TextCandidate(observation_id="o2", provider_id="docling", text="Land Registry"),
        ),
        _policy(),
    )
    assert a.accepted_text == b.accepted_text


def test_reconciliation_is_deterministic_across_repeated_runs():
    candidates = (
        TextCandidate(observation_id="o1", provider_id="docling", text="Recorded 3 March 1921"),
        TextCandidate(observation_id="o2", provider_id="tesseract", text="Recorded 3 March l92l"),
        TextCandidate(observation_id="o3", provider_id="qwen2.5-vl", text="Recorded 3 March 1921"),
    )
    first = reconcile_text(candidates, _policy())
    second = reconcile_text(candidates, _policy())
    assert first == second
