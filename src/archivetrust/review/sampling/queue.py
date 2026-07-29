"""The adaptive review queue (Milestone 11, Phase 6): merges operational triage with
calibration/research sampling into one ordered, explainable queue.

Operational entries are never recomputed here -- they are wrapped, verbatim, from
`review/triage.py::triage_review_queue()` output the caller already has (one call per document,
exactly as `ReviewService.open_document` already does today). This is what keeps the milestone's
own validation requirement true by construction: "operational review remains unchanged unless
explicitly configured otherwise" -- with no calibration/research items supplied, the adaptive
queue is a reordering-free wrapper around the existing operational queue, item for item.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.ontology.types import ObservationType
from archivetrust.review.packet import DisclosureTier, ReviewReason
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.strategies import SampledItem
from archivetrust.review.triage import TriageItem

_INTENT_ORDER: dict[ReviewIntent, int] = {
    ReviewIntent.OPERATIONAL: 0,
    ReviewIntent.CALIBRATION: 1,
    ReviewIntent.RESEARCH: 2,
}


class AdaptiveQueueEntry(BaseModel):
    """One queue entry, tagged with why it's there -- the answer a reviewer can always be shown
    (Phase 6)."""

    model_config = ConfigDict(frozen=True)

    document_ref: str
    semantic_slot_id: str
    canonical_observation_id: str
    observation_type: ObservationType
    intent: ReviewIntent
    explanation: str
    reason: ReviewReason | None = None
    """Set for OPERATIONAL entries (why triage queued it); `None` for sampled entries, which carry
    their reasoning in `explanation` and `value_score` instead."""
    disclosure_tier: DisclosureTier | None = None
    policy_version: int | None = None
    strategy_name: str | None = None
    value_score: float | None = None


def _operational_entry(document_ref: str, item: TriageItem) -> AdaptiveQueueEntry:
    return AdaptiveQueueEntry(
        document_ref=document_ref,
        semantic_slot_id=item.semantic_slot_id,
        canonical_observation_id=item.canonical_observation_id,
        observation_type=item.observation_type,
        intent=ReviewIntent.OPERATIONAL,
        explanation=f"Operational triage: {item.reason.value} (policy v{item.policy_version}).",
        reason=item.reason,
        disclosure_tier=item.disclosure_tier,
        policy_version=item.policy_version,
    )


def _sampled_entry(item: SampledItem) -> AdaptiveQueueEntry:
    return AdaptiveQueueEntry(
        document_ref=item.document_ref,
        semantic_slot_id=item.semantic_slot_id,
        canonical_observation_id=item.canonical_observation_id,
        observation_type=item.observation_type,
        intent=item.intent,
        explanation=item.explanation,
        strategy_name=item.strategy_name,
        value_score=item.value_score,
    )


def build_adaptive_queue(
    *,
    operational: Sequence[tuple[str, TriageItem]] = (),
    sampled: Sequence[SampledItem] = (),
    order_by: Literal["intent", "value"] = "intent",
) -> tuple[AdaptiveQueueEntry, ...]:
    """Merge operational triage items (as `(document_ref, TriageItem)` pairs, one call per
    document already made by the caller) with calibration/research `SampledItem`s.

    `order_by="intent"` (default) preserves operational items first, in their original relative
    order, then calibration, then research -- the queue's shape with no sampling configured is
    exactly the operational queue's shape. `order_by="value"` orders by `value_score` (sampled
    items only carry one; operational items sort last within ties, since triage itself has no
    numeric value score).
    """
    entries = [_operational_entry(doc_ref, item) for doc_ref, item in operational]
    entries += [_sampled_entry(item) for item in sampled]

    if order_by == "intent":
        entries.sort(key=lambda e: _INTENT_ORDER[e.intent])
    elif order_by == "value":
        entries.sort(key=lambda e: -(e.value_score if e.value_score is not None else -1.0))
    return tuple(entries)
