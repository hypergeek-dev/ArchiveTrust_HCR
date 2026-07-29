"""Effort measurement derived from passive review telemetry (ROADMAP_V2.md S9).

A `ReviewSession` is the effort/outcome summary of one uncertainty's review, computed purely from
its `ReviewInteraction` stream -- never from anything the reviewer was asked to report. This is
the concrete instrument behind "human effort measurements" (S9) and "human effort, aggregated"
(S10, Quality Center): the durations and counts here are what the Quality Center rolls up over
time and what the Evolution Center reads as review-duration-outlier signal (S11).

Reproducibility (Guiding Principle 7): `from_interactions` is a pure function of the stored
interactions, so the same stored stream always yields the same session, wall-clock timestamps
notwithstanding.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.learning.review.interaction import ReviewInteraction, ReviewInteractionKind


class ReviewOutcome(str, Enum):
    """Whether the reviewer took the fast path or had to edit (ROADMAP_V2.md S9.2, GP 9).

    `INCOMPLETE` distinguishes a review that opened but never resolved from one resolved by accept
    -- the same "never tried vs. tried and produced nothing" discipline the Trust Engine applies
    with `ProviderObservationAttempted` (ROADMAP.md S12), so an abandoned review is never silently
    counted as an acceptance.
    """

    ACCEPTED = "accepted"
    EDITED = "edited"
    INCOMPLETE = "incomplete"


class ReviewSession(BaseModel):
    """The effort/outcome summary of one uncertainty's review. All fields are derived, never
    supplied -- passive telemetry only (ROADMAP_V2.md S9.4)."""

    model_config = ConfigDict(frozen=True)

    review_id: str
    target_canonical_observation_id: str
    reviewer_ref: str
    outcome: ReviewOutcome

    review_duration: float | None
    """Wall-clock seconds from REVIEW_OPENED to REVIEW_COMPLETED. None if the review never
    completed (outcome INCOMPLETE) -- an absent duration is never reported as zero effort."""

    edit_duration: float | None
    """Total seconds spent inside edits (sum over EDIT_STARTED->EDIT_COMMITTED spans). None if the
    reviewer never edited -- distinct from 0.0, which would falsely claim an instantaneous edit."""

    edit_count: int
    undo_count: int
    zoom_count: int
    pan_count: int
    overlay_toggle_count: int
    navigation_count: int

    @classmethod
    def from_interactions(cls, interactions: tuple[ReviewInteraction, ...]) -> "ReviewSession":
        """Fold one review's interactions into an effort summary.

        Interactions are sorted by timestamp before folding, so a caller need not pre-order them;
        ties keep input order (Python's stable sort), which is the correct behavior for two events
        recorded at the same monotonic instant.
        """
        if not interactions:
            raise ValueError("ReviewSession.from_interactions requires at least one interaction")

        review_ids = {i.review_id for i in interactions}
        if len(review_ids) != 1:
            raise ValueError(
                "ReviewSession.from_interactions requires all interactions to share one review_id "
                f"(got {sorted(review_ids)}) -- a session summarizes exactly one uncertainty (S9.2)"
            )

        ordered = tuple(sorted(interactions, key=lambda i: i.timestamp))
        first = ordered[0]

        opened_at: float | None = None
        completed_at: float | None = None
        edit_started_at: float | None = None
        edit_duration_total = 0.0
        edited = False
        counts = {
            ReviewInteractionKind.EDIT_COMMITTED: 0,
            ReviewInteractionKind.EDIT_UNDONE: 0,
            ReviewInteractionKind.ZOOMED: 0,
            ReviewInteractionKind.PANNED: 0,
            ReviewInteractionKind.OVERLAY_TOGGLED: 0,
            ReviewInteractionKind.NAVIGATED: 0,
        }
        accepted = False

        for interaction in ordered:
            kind = interaction.kind
            if kind is ReviewInteractionKind.REVIEW_OPENED and opened_at is None:
                opened_at = interaction.timestamp
            elif kind is ReviewInteractionKind.REVIEW_COMPLETED:
                completed_at = interaction.timestamp
            elif kind is ReviewInteractionKind.VALUE_ACCEPTED:
                accepted = True
            elif kind is ReviewInteractionKind.EDIT_STARTED:
                edit_started_at = interaction.timestamp
                edited = True
            elif kind is ReviewInteractionKind.EDIT_COMMITTED:
                counts[kind] += 1
                edited = True
                if edit_started_at is not None:
                    edit_duration_total += interaction.timestamp - edit_started_at
                    edit_started_at = None
            elif kind in counts:
                counts[kind] += 1

        if completed_at is None:
            outcome = ReviewOutcome.INCOMPLETE
        elif edited:
            outcome = ReviewOutcome.EDITED
        elif accepted:
            outcome = ReviewOutcome.ACCEPTED
        else:
            # Completed with neither an accept nor an edit recorded: treat as incomplete rather
            # than inventing an outcome the telemetry doesn't support.
            outcome = ReviewOutcome.INCOMPLETE

        review_duration = (
            completed_at - opened_at if opened_at is not None and completed_at is not None else None
        )
        edit_duration = edit_duration_total if edited else None

        return cls(
            review_id=first.review_id,
            target_canonical_observation_id=first.target_canonical_observation_id,
            reviewer_ref=first.reviewer_ref,
            outcome=outcome,
            review_duration=review_duration,
            edit_duration=edit_duration,
            edit_count=counts[ReviewInteractionKind.EDIT_COMMITTED],
            undo_count=counts[ReviewInteractionKind.EDIT_UNDONE],
            zoom_count=counts[ReviewInteractionKind.ZOOMED],
            pan_count=counts[ReviewInteractionKind.PANNED],
            overlay_toggle_count=counts[ReviewInteractionKind.OVERLAY_TOGGLED],
            navigation_count=counts[ReviewInteractionKind.NAVIGATED],
        )


def sessions_from_interactions(
    interactions: tuple[ReviewInteraction, ...],
) -> tuple[ReviewSession, ...]:
    """Group a mixed stream of interactions by `review_id` and summarize each review.

    Grouping preserves first-seen order of `review_id`s so the result is deterministic for a given
    input stream (Guiding Principle 7).
    """
    by_review: dict[str, list[ReviewInteraction]] = {}
    for interaction in interactions:
        by_review.setdefault(interaction.review_id, []).append(interaction)
    return tuple(
        ReviewSession.from_interactions(tuple(group)) for group in by_review.values()
    )
