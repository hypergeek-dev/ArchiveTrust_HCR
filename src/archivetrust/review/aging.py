"""Review aging (Constitution Article 30, `ARCHITECTURE_TELEMETRY_STANDARD.md` Article 33).

Whether a currently-eligible review slot has gone unresolved past a configured horizon is a
*derived* fact — computed from the deterministic triage projection (`review.triage
.triage_review_queue`), the recorded `ReviewOutcomeRecorded` population, and the eligible slot's
own already-recorded `CanonicalDecisionCreated` timestamp. It is never its own persisted event
(Article 33): re-running this check any number of times over an unchanged telemetry stream must
never itself add telemetry, and repeatedly checking the same document must always yield the same
answer for the same wall-clock "now".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict

from archivetrust.application.journal import Journal, JournalState
from archivetrust.domain.telemetry.events import CanonicalDecisionCreated, ReviewOutcomeRecorded
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.review.triage import TriagePolicy, triage_review_queue


class AgedReviewSlot(BaseModel):
    """One currently-eligible review slot with no recorded outcome past `horizon`."""

    model_config = ConfigDict(frozen=True)

    semantic_slot_id: str
    canonical_observation_id: str
    eligible_since: str
    """UTC ISO-8601 -- the `recorded_at` of the `CanonicalDecisionCreated` event that produced the
    canonical version currently eligible for review. Never inferred; a slot whose eligibility
    timestamp cannot be established (historical data recorded before `recorded_at` existed) is
    excluded from aging results rather than guessed (Article 18's discipline, applied here)."""


def aged_review_slots(
    *,
    document_ref: str,
    sink: TelemetrySink,
    horizon: timedelta,
    policy: TriagePolicy | None = None,
    now: datetime | None = None,
) -> tuple[AgedReviewSlot, ...]:
    """Every currently-eligible slot (per the deterministic triage projection) with no recorded
    `ReviewOutcomeRecorded` and whose eligibility began more than `horizon` before `now`.

    Purely derived: reads `sink`, emits nothing (Constitution Article 33). Calling this twice in a
    row over an unchanged `sink` always returns the same result and appends zero events.
    """
    now = now or datetime.now(timezone.utc)
    events = tuple(sink.events_for_document(document_ref))
    journal_state = Journal().replay(iter(events))

    resolved_slot_ids = {
        event.semantic_slot_id for event in events if isinstance(event, ReviewOutcomeRecorded)
    }
    eligible_since_by_canonical_id: dict[str, str] = {
        event.canonical_observation.canonical_observation_id: event.recorded_at
        for event in events
        if isinstance(event, CanonicalDecisionCreated) and event.recorded_at is not None
    }

    aged: list[AgedReviewSlot] = []
    for item in triage_review_queue(journal_state, policy):
        if item.semantic_slot_id in resolved_slot_ids:
            continue
        eligible_since = eligible_since_by_canonical_id.get(item.canonical_observation_id)
        if eligible_since is None:
            # Never guessed (Article 18): no recorded_at means this predates that field, or the
            # producing event genuinely isn't in this stream -- excluded, not assumed aged.
            continue
        if now - datetime.fromisoformat(eligible_since) > horizon:
            aged.append(
                AgedReviewSlot(
                    semantic_slot_id=item.semantic_slot_id,
                    canonical_observation_id=item.canonical_observation_id,
                    eligible_since=eligible_since,
                )
            )
    return tuple(aged)
