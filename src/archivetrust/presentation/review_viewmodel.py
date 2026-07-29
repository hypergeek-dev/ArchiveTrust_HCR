"""The Review Center ViewModel (HUMAN_REVIEW_SPECIFICATION.md §2, §9, §10, §11).

This is where *all* review behavior lives — the decision model, the review queue, timing, and every
passive-telemetry emission — expressed against the `ReviewService` contract. The Qt Review view
(Increment 3) binds to it and renders it; it makes no review decisions of its own (the
implementation-phase rule: the View only orchestrates; behavior stays here and below).

Timing and identity are injected (`clock`, `id_factory`) so the whole ViewModel — including the
review-duration and edit-duration measurements — is exercised deterministically headless, with no
GUI and no wall clock (HR-6). In production the defaults use `time.monotonic` (per
`ReviewInteraction.timestamp`'s contract) and random ids.

**Interaction kind vs. correction action are deliberately independent** (the two-streams design,
§3): accepting a *different* provider's reading is, to the reviewer, an "accept" (interaction
`VALUE_ACCEPTED`) but mechanically changes the canonical value, so it rides the Trust Engine's
`EDIT` correction path — because the frozen Milestone-6 `ACCEPT` confirms the *current* value
unchanged (`domain/feedback/engine.py`). The Learning Platform stream records what the reviewer
did; the Trust Engine stream records the canonical effect; they legitimately differ.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum

from archivetrust.domain.shared.ids import new_id
from archivetrust.learning.review.interaction import ReviewInteraction, ReviewInteractionKind
from archivetrust.presentation.observable import Observable
from archivetrust.review.packet import ReviewPacket
from archivetrust.review.service import ReviewAction, ReviewService
from archivetrust.review.triage import TriageClassification


class ReviewStatus(str, Enum):
    IDLE = "idle"
    """Nothing loaded yet."""
    REVIEWING = "reviewing"
    """A packet is presented and awaiting a decision."""
    COMPLETE = "complete"
    """The queue is exhausted — no more uncertainties for this document."""


class ReviewViewModel:
    """Drives review of one document's queue. `load()` builds the queue and opens the first
    uncertainty; each decision advances to the next (§8's lifecycle, one uncertainty at a time,
    HR-9).
    """

    def __init__(
        self,
        *,
        service: ReviewService,
        reviewer_ref: str,
        document_ref: str,
        archive_object_ref: str,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[str], str] = new_id,
    ) -> None:
        self._service = service
        self._reviewer_ref = reviewer_ref
        self._document_ref = document_ref
        self._archive_object_ref = archive_object_ref
        self._clock = clock
        self._id_factory = id_factory

        self._queue: tuple[ReviewPacket, ...] = ()
        self._index = 0
        self._review_id: str | None = None
        self._editing = False
        self._opened_at: float | None = None
        """`clock()` at REVIEW_OPENED — turned into `review_duration_seconds` on the decision
        (Priority 5), measured with the same injected clock the interaction stream uses."""

        self.current_packet: Observable[ReviewPacket | None] = Observable(None)
        self.status: Observable[ReviewStatus] = Observable(ReviewStatus.IDLE)
        self.triage_summary: Observable[dict[TriageClassification, int]] = Observable({})
        """Constitution Article 33: deterministic, query-time counts of slots *not* in the queue --
        `NOT_ELIGIBLE` (genuinely fine) vs. `WITHHELD_BY_POLICY` (would be queued under a more
        permissive `TriagePolicy`) -- never a persisted event, recomputed on every `load()`."""

    # -- queue position (read-only, for the View's progress indicator) -----------------------

    @property
    def queue_total(self) -> int:
        return len(self._queue)

    @property
    def queue_index(self) -> int:
        """1-based position of the current uncertainty, or 0 when idle/complete."""
        return self._index + 1 if self.status.value == ReviewStatus.REVIEWING else 0

    # -- lifecycle ---------------------------------------------------------------------------

    def load(self) -> None:
        """Build the review queue for this document and open the first uncertainty (§8.1 → §8.2)."""
        self._queue = self._service.open_document(
            document_ref=self._document_ref, archive_object_ref=self._archive_object_ref
        )
        self.triage_summary.value = self._service.triage_summary(self._document_ref)
        self._index = 0
        self._open_current()

    def _open_current(self) -> None:
        if self._index < len(self._queue):
            packet = self._queue[self._index]
            self._review_id = self._id_factory("review")
            self._editing = False
            self._opened_at = self._clock()
            self.current_packet.value = packet
            self.status.value = ReviewStatus.REVIEWING
            self._emit(ReviewInteractionKind.REVIEW_OPENED)  # starts the review-duration clock
        else:
            self._review_id = None
            self.current_packet.value = None
            self.status.value = ReviewStatus.COMPLETE

    # -- decisions (§10.5) -------------------------------------------------------------------

    def accept(self, candidate_observation_id: str) -> None:
        """Accept a presented candidate as canonical — the fast path (§9.2, HR-8).

        If the chosen candidate's value already is the canonical value, this is a pure confirmation
        (`ACCEPT`); if it differs, the reviewer has chosen another source's reading, which changes
        the value and therefore rides the `EDIT` path (see module docstring). Either way the
        reviewer's action is recorded as `VALUE_ACCEPTED`.
        """
        packet = self._require_packet()
        candidate = self._candidate(packet, candidate_observation_id)
        self._emit(ReviewInteractionKind.VALUE_ACCEPTED)
        if candidate.value is None or candidate.value == packet.current_value:
            self._service.submit_decision(
                packet=packet, action=ReviewAction.ACCEPT_PROVIDER, **self._attribution()
            )
        else:
            self._service.submit_decision(
                packet=packet,
                action=ReviewAction.MANUAL_EDIT,
                corrected_output=candidate.value,
                **self._attribution(),
            )
        self._complete_and_advance()

    def begin_edit(self) -> None:
        """The reviewer started typing a manual value — starts the edit-duration clock (§11)."""
        packet = self._require_packet()
        if not self._editing:
            self._editing = True
            self._emit(ReviewInteractionKind.EDIT_STARTED)

    def commit_edit(self, value: str) -> None:
        """Commit a manual edit — a value no source presented (the exceptional path, §4.1)."""
        packet = self._require_packet()
        if not self._editing:
            self.begin_edit()
        self._emit(ReviewInteractionKind.EDIT_COMMITTED)
        self._service.submit_decision(
            packet=packet,
            action=ReviewAction.MANUAL_EDIT,
            corrected_output=value,
            **self._attribution(),
        )
        self._complete_and_advance()

    def undo_edit(self) -> None:
        """The reviewer discarded an in-progress edit — a friction signal (§11)."""
        self._require_packet()
        if self._editing:
            self._editing = False
            self._emit(ReviewInteractionKind.EDIT_UNDONE)

    def reject(self, rationale: str | None = None) -> None:
        packet = self._require_packet()
        self._service.submit_decision(
            packet=packet, action=ReviewAction.REJECT, rationale=rationale, **self._attribution()
        )
        self._complete_and_advance()

    def mark_ambiguous(self, rationale: str | None = None) -> None:
        """The archive admits multiple valid readings — a first-class finding (§11.4)."""
        packet = self._require_packet()
        self._service.submit_decision(
            packet=packet, action=ReviewAction.MARK_AMBIGUOUS, rationale=rationale, **self._attribution()
        )
        self._complete_and_advance()

    def mark_different_things(self, rationale: str | None = None) -> None:
        """Both readings may be correct, but they refer to different archive things/scopes."""
        packet = self._require_packet()
        self._service.submit_decision(
            packet=packet,
            action=ReviewAction.DIFFERENT_THINGS,
            rationale=rationale,
            **self._attribution(),
        )
        self._complete_and_advance()

    def request_further_review(self, rationale: str | None = None) -> None:
        packet = self._require_packet()
        self._service.submit_decision(
            packet=packet, action=ReviewAction.REQUEST_FURTHER_REVIEW, rationale=rationale, **self._attribution()
        )
        self._complete_and_advance()

    def mark_illegible(self, rationale: str | None = None) -> None:
        packet = self._require_packet()
        self._service.submit_decision(
            packet=packet,
            action=ReviewAction.ILLEGIBLE,
            rationale=rationale,
            **self._attribution(),
        )
        self._complete_and_advance()

    def skip(self) -> None:
        """Defer without judging (§10.7). Produces no correction, and — critically — no
        REVIEW_COMPLETED: the review stays INCOMPLETE, never counted as an acceptance.
        """
        packet = self._require_packet()
        self._service.submit_decision(
            packet=packet, action=ReviewAction.SKIP, **self._attribution()
        )
        self._advance()  # note: no REVIEW_COMPLETED emitted (see docstring)

    # -- passive UI-observation hooks (§11: the View calls these as the reviewer acts) --------

    def observe_zoom(self) -> None:
        self._emit(ReviewInteractionKind.ZOOMED)

    def observe_pan(self) -> None:
        self._emit(ReviewInteractionKind.PANNED)

    def observe_overlay_toggle(self) -> None:
        self._emit(ReviewInteractionKind.OVERLAY_TOGGLED)

    def observe_navigation(self) -> None:
        self._emit(ReviewInteractionKind.NAVIGATED)

    # -- internals ---------------------------------------------------------------------------

    def _attribution(self) -> dict:
        """Reviewer pseudonym + elapsed review time for this uncertainty (Priority 5) — attached
        to every decision so the Trust Engine's correction record is attributable without the
        Learning Platform's separate (session-local, monotonic-clock) interaction stream."""
        duration = self._clock() - self._opened_at if self._opened_at is not None else None
        return {"reviewer_ref": self._reviewer_ref, "review_duration_seconds": duration}

    def _complete_and_advance(self) -> None:
        self._emit(ReviewInteractionKind.REVIEW_COMPLETED)  # stops the review-duration clock
        self._advance()

    def _advance(self) -> None:
        self._index += 1
        self._open_current()

    def _require_packet(self) -> ReviewPacket:
        packet = self.current_packet.value
        if packet is None:
            raise RuntimeError("No uncertainty is currently under review")
        return packet

    @staticmethod
    def _candidate(packet: ReviewPacket, observation_id: str):
        for candidate in packet.candidates:
            if candidate.observation_id == observation_id:
                return candidate
        raise KeyError(f"candidate {observation_id!r} is not part of this packet")

    def _emit(self, kind: ReviewInteractionKind) -> None:
        packet = self.current_packet.value
        assert packet is not None and self._review_id is not None
        self._service.record_interaction(
            ReviewInteraction(
                interaction_id=self._id_factory("interaction"),
                review_id=self._review_id,
                target_canonical_observation_id=packet.canonical_observation_id,
                reviewer_ref=self._reviewer_ref,
                kind=kind,
                timestamp=self._clock(),
            )
        )
