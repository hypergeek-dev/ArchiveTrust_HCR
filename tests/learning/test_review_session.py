from __future__ import annotations

import pytest

from archivetrust.domain.shared.ids import new_id
from archivetrust.learning.review.interaction import ReviewInteraction, ReviewInteractionKind
from archivetrust.learning.review.session import (
    ReviewOutcome,
    ReviewSession,
    sessions_from_interactions,
)


def _interaction(review_id: str, kind: ReviewInteractionKind, ts: float) -> ReviewInteraction:
    return ReviewInteraction(
        interaction_id=new_id("interaction"),
        review_id=review_id,
        target_canonical_observation_id="canonical_1",
        reviewer_ref="reviewer_a",
        kind=kind,
        timestamp=ts,
    )


def test_accept_path_yields_accepted_outcome_and_review_duration() -> None:
    review = "r1"
    interactions = (
        _interaction(review, ReviewInteractionKind.REVIEW_OPENED, 100.0),
        _interaction(review, ReviewInteractionKind.OVERLAY_TOGGLED, 101.0),
        _interaction(review, ReviewInteractionKind.VALUE_ACCEPTED, 103.0),
        _interaction(review, ReviewInteractionKind.REVIEW_COMPLETED, 103.5),
    )
    session = ReviewSession.from_interactions(interactions)

    assert session.outcome is ReviewOutcome.ACCEPTED
    assert session.review_duration == pytest.approx(3.5)
    assert session.edit_duration is None  # never edited -> None, not 0.0
    assert session.overlay_toggle_count == 1
    assert session.edit_count == 0


def test_edit_path_measures_edit_duration_and_counts() -> None:
    review = "r2"
    interactions = (
        _interaction(review, ReviewInteractionKind.REVIEW_OPENED, 0.0),
        _interaction(review, ReviewInteractionKind.EDIT_STARTED, 2.0),
        _interaction(review, ReviewInteractionKind.EDIT_COMMITTED, 5.0),
        _interaction(review, ReviewInteractionKind.EDIT_UNDONE, 6.0),
        _interaction(review, ReviewInteractionKind.ZOOMED, 6.5),
        _interaction(review, ReviewInteractionKind.REVIEW_COMPLETED, 8.0),
    )
    session = ReviewSession.from_interactions(interactions)

    assert session.outcome is ReviewOutcome.EDITED
    assert session.review_duration == pytest.approx(8.0)
    assert session.edit_duration == pytest.approx(3.0)
    assert session.edit_count == 1
    assert session.undo_count == 1
    assert session.zoom_count == 1


def test_incomplete_review_has_no_duration() -> None:
    review = "r3"
    interactions = (
        _interaction(review, ReviewInteractionKind.REVIEW_OPENED, 0.0),
        _interaction(review, ReviewInteractionKind.ZOOMED, 1.0),
    )
    session = ReviewSession.from_interactions(interactions)

    assert session.outcome is ReviewOutcome.INCOMPLETE
    assert session.review_duration is None


def test_from_interactions_is_order_independent() -> None:
    review = "r4"
    ordered = [
        _interaction(review, ReviewInteractionKind.REVIEW_OPENED, 0.0),
        _interaction(review, ReviewInteractionKind.VALUE_ACCEPTED, 1.0),
        _interaction(review, ReviewInteractionKind.REVIEW_COMPLETED, 2.0),
    ]
    shuffled = tuple(reversed(ordered))
    assert ReviewSession.from_interactions(tuple(ordered)).review_duration == pytest.approx(
        ReviewSession.from_interactions(shuffled).review_duration
    )


def test_mixed_reviews_are_grouped() -> None:
    interactions = (
        _interaction("a", ReviewInteractionKind.REVIEW_OPENED, 0.0),
        _interaction("b", ReviewInteractionKind.REVIEW_OPENED, 0.0),
        _interaction("a", ReviewInteractionKind.VALUE_ACCEPTED, 1.0),
        _interaction("a", ReviewInteractionKind.REVIEW_COMPLETED, 2.0),
        _interaction("b", ReviewInteractionKind.EDIT_STARTED, 1.0),
        _interaction("b", ReviewInteractionKind.EDIT_COMMITTED, 2.0),
        _interaction("b", ReviewInteractionKind.REVIEW_COMPLETED, 3.0),
    )
    sessions = {s.review_id: s for s in sessions_from_interactions(interactions)}
    assert sessions["a"].outcome is ReviewOutcome.ACCEPTED
    assert sessions["b"].outcome is ReviewOutcome.EDITED


def test_rejects_multiple_review_ids() -> None:
    with pytest.raises(ValueError, match="one review_id"):
        ReviewSession.from_interactions(
            (
                _interaction("a", ReviewInteractionKind.REVIEW_OPENED, 0.0),
                _interaction("b", ReviewInteractionKind.REVIEW_OPENED, 0.0),
            )
        )
