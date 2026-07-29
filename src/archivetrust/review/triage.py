"""Triage — which uncertainties reach a reviewer, in what order, at which tier
(HUMAN_REVIEW_SPECIFICATION.md §8.1).

Triage reads only Comparison/Confidence outputs already present on each Canonical Observation
(via a replayed `JournalState`) — it never re-runs a provider (LP-2, read-only). It is deterministic
and reproducible from stored telemetry: the same journal state always yields the same queue
(HR-6, `ROADMAP_V2.md` GP 7).

Corroborated, high-confidence slots are *not* queued — reviewer attention is spent where
uncertainty is, not uniformly (UX-INV-5). The signal that selects a slot is also what lets the
interface explain "why am I reviewing this?" (`ReviewReason`, §6) and pick the disclosure tier.

**Scope note (honest).** The full `RiskScore` / `RiskItem` set that `MILESTONE4_COMPARISON_ENGINE.md`
§10–§11 defines is not yet carried in the telemetry stream (a documented Milestone 4 gap —
IMPLEMENTATION_STATUS.md, M4 telemetry item). Triage therefore selects on the signals that *are*
replayable today: `ComparisonClassification` and `CanonicalConfidence`. When risk items become
part of the stream, they extend this selection without changing its shape.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.application.journal import JournalState
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.review.packet import DisclosureTier, ReviewReason

TRIAGE_POLICY_VERSION = 2

DEFAULT_SINGLE_SOURCE_REVIEW_TYPES: frozenset[ObservationType] = frozenset(
    {
        ObservationType.HEADING,
        ObservationType.TABLE,
        ObservationType.METADATA,
        ObservationType.ARCHIVE_BOUNDARY,
    }
)
"""The high-value observation types (§8.1's own examples: headings, tables, metadata/dates, archive
boundaries) whose UNCORROBORATED_SINGLE_SOURCE slots are still queued under Triage Policy v2.
Policy v1 queued *every* single-source slot unconditionally — operationally unrealistic in a
deployment where one provider dominates (Operational Hardening milestone: the first validation run
put 86–100% of all canonical facts into review). v1's behavior remains representable:
`single_source_review_types=None` restores it explicitly.

`ObservationType.HANDWRITTEN_NOTE` was removed from this set in docs/htr-migration-plan.md Stage 5
(EXECUTED) along with the type itself -- superseded by line-level `TEXT_LINE`/`RAW_TRANSCRIPTION`
observations, whose own review-worthiness signal (per-line confidence) is a Phase-6+ concern, not
this triage set's."""

# Observation types whose disagreements are structural, warranting the Tier 3 apparatus
# (table / reading-order visualization, §10.3). `ObservationType.TABLE_CELL`/`RELATIONSHIP` were
# removed in docs/htr-migration-plan.md Stage 5 (EXECUTED) along with the types themselves.
_STRUCTURAL_TYPES: frozenset[ObservationType] = frozenset(
    {
        ObservationType.SECTION,
        ObservationType.TABLE,
        ObservationType.ARCHIVE_BOUNDARY,
    }
)

# Ordering key per reason — lower sorts first (more urgent), per §8.1's urgency ranking.
_REASON_URGENCY: dict[ReviewReason, int] = {
    ReviewReason.SOURCES_DISAGREE: 0,
    ReviewReason.LOW_CONFIDENCE: 1,
    ReviewReason.SINGLE_SOURCE: 2,
}


class TriagePolicy(BaseModel):
    """Versioned triage thresholds (LP-8; mirrors the Trust Engine's Reconciliation/Confidence
    Policy versioning). Values are deliberate placeholders pending corpus validation, recorded so
    a later, better-calibrated version supersedes rather than silently overwrites them.
    """

    model_config = ConfigDict(frozen=True)

    version: int = TRIAGE_POLICY_VERSION
    low_confidence_threshold: float = 0.6
    """A CORROBORATED slot with Canonical Confidence below this is queued as LOW_CONFIDENCE."""
    queue_single_source: bool = True
    """Whether UNCORROBORATED_SINGLE_SOURCE slots are queued at all (§8.1: high-value types).
    Kept as an explicit, versioned switch rather than a hidden constant."""
    single_source_review_types: frozenset[ObservationType] | None = DEFAULT_SINGLE_SOURCE_REVIEW_TYPES
    """Which observation types' single-source slots are queued (Policy v2). `None` means "every
    type" — exactly Policy v1's unconditional behavior, restorable explicitly rather than lost."""


class TriageItem(BaseModel):
    """One selected uncertainty: which slot, why, and at what disclosure tier."""

    model_config = ConfigDict(frozen=True)

    semantic_slot_id: str
    canonical_observation_id: str
    observation_type: ObservationType
    reason: ReviewReason
    disclosure_tier: DisclosureTier
    policy_version: int


def _reason_for(
    canonical: CanonicalObservation, policy: TriagePolicy
) -> ReviewReason | None:
    # Phase 23 (Review Queue Consistency): a slot a human has already decided on is resolved, not
    # merely re-scored -- `comparison_confidence.classification` is carried over unchanged by
    # `apply_human_correction` (it describes the *original* provider disagreement, which doesn't
    # stop being true), so without this check a CONTESTED slot stayed SOURCES_DISAGREE and was
    # re-queued forever, even immediately after being reviewed. `human_correction_ref` is set by
    # every decision except Skip (`apply_human_correction`, `review/service.py::submit_decision`)
    # -- Skip is deliberately excluded (§10.7: "a skip is never an acceptance"), so a skipped slot
    # correctly remains queued.
    if canonical.human_correction_ref is not None:
        return None
    classification = canonical.comparison_confidence.classification
    if classification == ComparisonClassification.CONTESTED:
        return ReviewReason.SOURCES_DISAGREE
    if classification == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE:
        if not policy.queue_single_source:
            return None
        if (
            policy.single_source_review_types is not None
            and canonical.observation_type not in policy.single_source_review_types
        ):
            return None
        return ReviewReason.SINGLE_SOURCE
    # CORROBORATED: queue only if trust is weak.
    confidence = canonical.canonical_confidence
    if confidence is not None and confidence.value < policy.low_confidence_threshold:
        return ReviewReason.LOW_CONFIDENCE
    return None


def review_reason_for(
    canonical: CanonicalObservation, policy: TriagePolicy | None = None
) -> ReviewReason | None:
    """Public form of the selection rule: why (if at all) this Canonical Observation would be
    queued under `policy`. The one shared answer to "does this count as awaiting review?" — the
    Processing Center's tiles and the actual review queue must never disagree about it.
    """
    return _reason_for(canonical, policy or TriagePolicy())


class TriageClassification(str, Enum):
    """Constitution Article 33: a deterministic, query-time-only classification -- never persisted
    as an event, computed fresh from the same signals `_reason_for` already reads. Distinguishes
    "genuinely nothing to review" from "would be reviewed under a more permissive policy" -- the
    Adversarial Audit's finding that `PARAGRAPH` is silently excluded from
    `DEFAULT_SINGLE_SOURCE_REVIEW_TYPES` was previously indistinguishable from "this slot is fine,"
    since both collapse to `review_reason_for(...) is None`. Additive: does not change
    `review_reason_for`/`_reason_for`'s existing return contract, which several benchmark/analytics/
    presentation call sites already depend on (`benchmark/aggregate.py`, `presentation/
    operations_viewmodel.py`, `review/simulation.py`, `review/sampling/discovery.py`) -- this is a
    new, richer query alongside them, not a replacement.
    """

    QUEUED = "queued"
    NOT_ELIGIBLE = "not_eligible"
    """Genuinely nothing to review under any reasonable policy -- e.g. well-corroborated, high
    Canonical Confidence."""
    WITHHELD_BY_POLICY = "withheld_by_policy"
    """Would be queued under a more permissive `TriagePolicy` (e.g. `queue_single_source=True` with
    a broader `single_source_review_types`), but excluded by the policy actually in force."""


def triage_classification_for(
    canonical: CanonicalObservation, policy: TriagePolicy | None = None
) -> TriageClassification:
    """Answers "why wasn't this reviewed" precisely (Constitution Article 33, Article 30's revised
    scope) -- computed fresh from the persisted `TriagePolicy` version and the canonical
    observation's own recorded state, never from a persisted event. Re-running this any number of
    times over unchanged inputs always yields the same classification and emits no telemetry.
    """
    policy = policy or TriagePolicy()
    if _reason_for(canonical, policy) is not None:
        return TriageClassification.QUEUED
    classification = canonical.comparison_confidence.classification
    if classification == ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE:
        if not policy.queue_single_source:
            return TriageClassification.WITHHELD_BY_POLICY
        if (
            policy.single_source_review_types is not None
            and canonical.observation_type not in policy.single_source_review_types
        ):
            return TriageClassification.WITHHELD_BY_POLICY
    return TriageClassification.NOT_ELIGIBLE


def _tier_for(canonical: CanonicalObservation, reason: ReviewReason) -> DisclosureTier:
    if canonical.observation_type in _STRUCTURAL_TYPES:
        return DisclosureTier.TIER_3_COMPLEX
    if reason == ReviewReason.SOURCES_DISAGREE:
        return DisclosureTier.TIER_2_MODERATE
    return DisclosureTier.TIER_1_SIMPLE


def triage_review_queue(
    journal_state: JournalState, policy: TriagePolicy | None = None
) -> tuple[TriageItem, ...]:
    """The ordered review queue for one document's replayed state.

    Sorted by reason urgency (§8.1), ties broken by `semantic_slot_id` for determinism.
    """
    policy = policy or TriagePolicy()
    items: list[TriageItem] = []
    for slot in journal_state.known_semantic_slots():
        history = journal_state.canonical_observation_history(slot)
        canonical = history[-1]  # latest version reflects any prior correction
        reason = _reason_for(canonical, policy)
        if reason is None:
            continue
        items.append(
            TriageItem(
                semantic_slot_id=slot,
                canonical_observation_id=canonical.canonical_observation_id,
                observation_type=canonical.observation_type,
                reason=reason,
                disclosure_tier=_tier_for(canonical, reason),
                policy_version=policy.version,
            )
        )
    items.sort(key=lambda i: (_REASON_URGENCY[i.reason], i.semantic_slot_id))
    return tuple(items)
