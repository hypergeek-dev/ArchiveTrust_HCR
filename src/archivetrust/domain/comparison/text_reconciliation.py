"""Phase D -- Text Reconciliation (MILESTONE4_COMPARISON_ENGINE.md S5).

A deterministic ladder: normalization -> exact equality -> edit distance -> token/sequence
alignment -> deterministic consensus selection. No semantic/embedding similarity anywhere in this
module, by design (S5's four reasons, restated in the module-level rationale below) -- this is a
hard architectural line, not a missing feature.

**Semantic similarity is deliberately excluded, permanently, not just deferred**: two providers
reading "1897" and "1867" are semantically near-identical but factually contradictory; embeddings
are non-deterministic across model versions (C7); and an embedding distance has no evidentiary
basis (C2). Its only legitimate future home is as an *observation-producing* signal, never inside
this deterministic core (S5, S16 R5).
"""

from __future__ import annotations

import difflib
import unicodedata
from collections import Counter
from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.comparison.policy import ReconciliationPolicy


class TextClassification(str, Enum):
    CORROBORATED = "corroborated"
    UNCORROBORATED_SINGLE_SOURCE = "uncorroborated_single_source"
    CONTESTED = "contested"


class ReconciliationBasisCode(str, Enum):
    """Versioned, closed vocabulary for `CanonicalObservation.reconciliation_basis`'s decision
    paths (Constitution Article 26) -- covering `domain.comparison.engine`'s full reconciliation
    surface, not only this module's four text-reconciliation paths. Located here, not in
    `engine.py` (`ROADMAP_TELEMETRY_STANDARD.md` Phase 4's stated co-location), only to avoid a
    circular import: `engine.py` already imports from this module, never the reverse. Every member
    corresponds to a real, currently-reachable code path, verified before being added
    (`ARCHITECTURE_TELEMETRY_STANDARD.md` S1.1's evidentiary discipline, applied here too).
    """

    TEXT_SINGLE_SOURCE_ONLY_PROVIDER = "TEXT_SINGLE_SOURCE_ONLY_PROVIDER"
    """Exactly one candidate contributed to this slot -- `reconcile_text`'s single-candidate path."""
    TEXT_SINGLE_SOURCE_SAME_PROVIDER = "TEXT_SINGLE_SOURCE_SAME_PROVIDER"
    """Multiple candidates, but all from the same provider -- within-provider agreement is not
    independent corroboration (Constitution Article 2/C1/C8)."""
    TEXT_EXACT_EQUALITY_CORROBORATED = "TEXT_EXACT_EQUALITY_CORROBORATED"
    """Exact equality after normalization across independent providers."""
    TEXT_MAJORITY_VOTE_CONSENSUS = "TEXT_MAJORITY_VOTE_CONSENSUS"
    """Deterministic character-majority-vote consensus across independent providers, within the
    corroboration threshold -- `CORROBORATED` only. Never used for `CONTESTED` (see
    `TEXT_CONTESTED_REFERENCE_PICK`): synthesizing a character-vote splice across candidates that
    diverge beyond the threshold produces text no provider ever wrote (Product Audit 2026-07-16
    S7, S10 item 7; Master Execution Program S4.1-3) -- a defect, not a feature, since it was being
    presented to reviewers as the document's "current canonical value.\""""
    TEXT_CONTESTED_REFERENCE_PICK = "TEXT_CONTESTED_REFERENCE_PICK"
    """Candidates diverge beyond the corroboration threshold (`CONTESTED`). No value is synthesized
    across them -- `accepted_text` is one candidate's real, unmodified text (the same
    longest-then-lexicographic reference `disagreements` is built against), never a character-vote
    splice. This is the smallest fix consistent with Article 4 (every canonical field needs
    evidence-backed content, so the slot cannot go valueless while real candidate text exists) and
    Article 11 (disagreement must never be hidden behind a confident-looking value): the field
    still carries real, traceable text, but `classification=CONTESTED` and this basis code mark it
    as a reference pick pending human review, not an agreed reading."""
    TABLE_RECONCILIATION = "TABLE_RECONCILIATION"
    """`domain.comparison.engine`'s table-cluster path (`table_reconciliation.reconcile_table`'s
    neutral-separator-lattice basis) -- a Table's own top-level reconciliation, distinct from its
    per-cell text reconciliation (each cell independently gets one of the `TEXT_*` codes above)."""
    NON_TEXT_DETERMINISTIC_PICK = "NON_TEXT_DETERMINISTIC_PICK"
    """`domain.comparison.engine`'s fallback for Observation types with no content-level
    reconciliation defined (not text-bearing, not a Table): a single deterministic pick by
    `observation_id`, existence-only agreement scoring."""


class TextCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation_id: str
    provider_id: str
    text: str


class SpanDisagreement(BaseModel):
    """One aligned region where the reference candidate and another candidate disagree --
    localizes conflict rather than collapsing a whole field's score (S5 step 4)."""

    model_config = ConfigDict(frozen=True)

    against_observation_id: str
    reference_span: str
    candidate_span: str


class TextReconciliationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    classification: TextClassification
    accepted_text: str
    reconciliation_basis: str
    reconciliation_basis_code: ReconciliationBasisCode
    """Structured counterpart to `reconciliation_basis` (Constitution Article 26) -- in addition to,
    never instead of, the free-text explanation."""
    magnitude: float | None
    disagreements: tuple[SpanDisagreement, ...] = ()


def normalize(text: str, *, aggressive: bool) -> str:
    """Unicode NFC + whitespace collapse (S5 step 1). Case-folding/diacritic stripping only when
    `aggressive` is explicitly enabled -- off by default, since "for archival documents a
    diacritic *is* content."
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = " ".join(normalized.split())
    if aggressive:
        normalized = normalized.casefold()
        normalized = "".join(
            ch for ch in unicodedata.normalize("NFKD", normalized) if not unicodedata.combining(ch)
        )
    return normalized


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (0 if char_a == char_b else 1)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein_distance(a, b) / max(len(a), len(b))


def _majority_vote_consensus(normalized_texts: tuple[str, ...]) -> str:
    """Character-level majority vote per aligned position (S5 step 5), aligned against a
    content-chosen (never provider-chosen, C1) reference sequence: the longest text, ties broken
    by lexicographically smallest content. Insertions unique to one non-reference candidate are
    dropped as unsupported -- a documented simplification of full multi-sequence alignment.
    """
    reference = sorted(normalized_texts, key=lambda t: (-len(t), t))[0]
    votes: list[Counter[str]] = [Counter({ch: 1}) for ch in reference]

    for candidate in normalized_texts:
        if candidate == reference:
            continue
        matcher = difflib.SequenceMatcher(None, reference, candidate)
        for tag, ref_start, ref_end, cand_start, cand_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag in ("replace", "delete"):
                cand_chars = candidate[cand_start:cand_end]
                for offset, ref_index in enumerate(range(ref_start, ref_end)):
                    if offset < len(cand_chars):
                        votes[ref_index][cand_chars[offset]] += 1
            # "insert": a candidate-only insertion has no reference position to vote on; dropped.

    consensus_chars = []
    for position_votes in votes:
        best = sorted(position_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        consensus_chars.append(best)
    return "".join(consensus_chars)


def reconcile_text(
    candidates: tuple[TextCandidate, ...], policy: ReconciliationPolicy
) -> TextReconciliationResult:
    if not candidates:
        raise ValueError("reconcile_text requires at least one candidate")

    normalized_map = {
        c.observation_id: normalize(c.text, aggressive=policy.aggressive_text_normalization)
        for c in candidates
    }

    if len(candidates) == 1:
        only = candidates[0]
        return TextReconciliationResult(
            classification=TextClassification.UNCORROBORATED_SINGLE_SOURCE,
            accepted_text=only.text,
            reconciliation_basis=(
                f"single-source acceptance (only {only.provider_id} contributed to this slot)"
            ),
            reconciliation_basis_code=ReconciliationBasisCode.TEXT_SINGLE_SOURCE_ONLY_PROVIDER,
            magnitude=None,
        )

    # Corroboration requires *distinct providers* (C1/C8): several Observations from the same
    # provider agreeing with each other is self-agreement, never independent corroboration --
    # a provider cannot corroborate itself, no matter how many Observations it contributed.
    distinct_providers = {c.provider_id for c in candidates}
    if len(distinct_providers) == 1:
        sole_provider = next(iter(distinct_providers))
        deterministic_pick = sorted(candidates, key=lambda c: c.observation_id)[0]
        return TextReconciliationResult(
            classification=TextClassification.UNCORROBORATED_SINGLE_SOURCE,
            accepted_text=deterministic_pick.text,
            reconciliation_basis=(
                f"single-source acceptance: {len(candidates)} candidates, but all from "
                f"{sole_provider} -- within-provider agreement is not independent corroboration"
            ),
            reconciliation_basis_code=ReconciliationBasisCode.TEXT_SINGLE_SOURCE_SAME_PROVIDER,
            magnitude=None,
        )

    normalized_texts = tuple(normalized_map[c.observation_id] for c in candidates)
    if len(set(normalized_texts)) == 1:
        return TextReconciliationResult(
            classification=TextClassification.CORROBORATED,
            accepted_text=candidates[0].text,
            reconciliation_basis=(
                f"exact equality after normalization across {len(distinct_providers)} independent "
                f"providers ({len(candidates)} observations)"
            ),
            reconciliation_basis_code=ReconciliationBasisCode.TEXT_EXACT_EQUALITY_CORROBORATED,
            magnitude=1.0,
        )

    # Pairwise normalized edit distance -- the graded agreement signal (S5 step 3). Only pairs
    # from *different* providers count: cross-provider (dis)agreement is the corroboration signal;
    # two Observations from one provider disagreeing is provider-internal noise, not contest
    # between independent sources (the same distinct-provider rule as the single-source branch).
    pairwise_distances = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if candidates[i].provider_id == candidates[j].provider_id:
                continue
            distance = normalized_edit_distance(normalized_texts[i], normalized_texts[j])
            pairwise_distances.append(distance)
    max_distance = max(pairwise_distances)
    magnitude = policy.round_score(1.0 - max_distance)

    reference = sorted(candidates, key=lambda c: (-len(normalized_map[c.observation_id]), normalized_map[c.observation_id]))[0]
    disagreements = tuple(
        SpanDisagreement(
            against_observation_id=c.observation_id,
            reference_span=normalized_map[reference.observation_id],
            candidate_span=normalized_map[c.observation_id],
        )
        for c in candidates
        if c.observation_id != reference.observation_id
        and normalized_map[c.observation_id] != normalized_map[reference.observation_id]
    )

    classification = (
        TextClassification.CORROBORATED
        if max_distance <= policy.text_corroboration_edit_distance_threshold
        else TextClassification.CONTESTED
    )

    if classification == TextClassification.CORROBORATED:
        # Within threshold: a character-vote consensus is a legitimate reconciled reading -- every
        # candidate is close enough that per-position majority vote reflects real agreement.
        accepted_text = _majority_vote_consensus(normalized_texts)
        basis = (
            f"deterministic character-majority-vote consensus over {len(candidates)} observations "
            f"from {len(distinct_providers)} independent providers "
            f"(max pairwise normalized edit distance {round(max_distance, policy.rounding_ndigits)}, "
            f"threshold {policy.text_corroboration_edit_distance_threshold}, Policy v{policy.policy_version})"
        )
        basis_code = ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS
    else:
        # Beyond threshold: candidates diverge too far for a per-position vote to mean anything --
        # synthesizing one anyway produces text no provider ever wrote (the fix this branch exists
        # for). `accepted_text` is the same real, unmodified reference candidate `disagreements` is
        # built against -- never invented, never blended (Article 4: still evidence-backed content;
        # Article 11: the classification/basis_code, not a confident-looking value, is what marks
        # this contested).
        accepted_text = reference.text
        basis = (
            f"no reconciled value: {len(candidates)} observations from {len(distinct_providers)} "
            "independent providers diverge beyond the corroboration threshold "
            f"(max pairwise normalized edit distance {round(max_distance, policy.rounding_ndigits)}, "
            f"threshold {policy.text_corroboration_edit_distance_threshold}, Policy v{policy.policy_version}) "
            f"-- showing {reference.provider_id}'s reading as reference; human review required to "
            "select the correct candidate"
        )
        basis_code = ReconciliationBasisCode.TEXT_CONTESTED_REFERENCE_PICK

    return TextReconciliationResult(
        classification=classification,
        accepted_text=accepted_text,
        reconciliation_basis=basis,
        reconciliation_basis_code=basis_code,
        magnitude=magnitude,
        disagreements=disagreements,
    )
