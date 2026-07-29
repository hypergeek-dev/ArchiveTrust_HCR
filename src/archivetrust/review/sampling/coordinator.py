"""`AdaptiveReviewCoordinator` (Milestone 11, Phases 1, 2, 6): composes an existing `ReviewService`
with calibration/research sampling. It never subclasses or edits `ReviewService` -- every method
here either wraps an unmodified `ReviewService`/`triage_review_queue()` call or adds new,
independent behavior (sampling, logging), so the operational path stays exactly what it was before
this package existed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from archivetrust.application.journal import Journal
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.telemetry.events import (
    ReviewOutcomeRecorded,
    ReviewPacketClosed,
    ReviewPacketClosureKind,
    ReviewPacketDispatched,
    ReviewPacketOpened,
)
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.learning.source import TelemetrySource
from archivetrust.review.assembler import assemble_packet
from archivetrust.review.closure import (
    close_review_packet,
    dispatch_review_packet,
    open_review_packet,
)
from archivetrust.review.packet import DisclosureTier, ReviewPacket, ReviewReason
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.log import SamplingDecision, SamplingLogSink
from archivetrust.review.sampling.progress import compute_calibration_progress_map
from archivetrust.review.sampling.queue import AdaptiveQueueEntry, build_adaptive_queue
from archivetrust.review.sampling.strategies import strategy_by_name
from archivetrust.review.service import ReviewAction, ReviewDecisionResult, ReviewService
from archivetrust.review.triage import TriageItem, TriagePolicy, triage_review_queue

_NATURAL_REASON_BY_CLASSIFICATION: dict[ComparisonClassification, ReviewReason] = {
    ComparisonClassification.CONTESTED: ReviewReason.SOURCES_DISAGREE,
    ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE: ReviewReason.SINGLE_SOURCE,
    ComparisonClassification.CORROBORATED: ReviewReason.LOW_CONFIDENCE,
}
"""Every slot's classification maps to one of the three uncertainty categories `ReviewPacket.
review_reason` can express, independent of whether the current `TriagePolicy` would have queued
it. Used only for calibration/research entries, which triage never assigned a `ReviewReason` to;
this is an honest statement of "which kind of uncertainty this is," not a claim about why policy
queued it (it wouldn't have)."""


class AdaptiveReviewCoordinator:
    def __init__(
        self,
        *,
        review_service: ReviewService,
        telemetry_source: TelemetrySource,
        sampling_log: SamplingLogSink,
        triage_policy: TriagePolicy | None = None,
        journal: Journal | None = None,
        telemetry_sink: TelemetrySink | None = None,
        reviewer_ref: str | None = None,
    ) -> None:
        self._service = review_service
        self._source = telemetry_source
        self._log = sampling_log
        self._triage_policy = triage_policy or TriagePolicy()
        self._journal = journal or Journal()
        self._sink = telemetry_sink
        """The Trust Engine sink the packet lifecycle (`ReviewPacketDispatched`/`ReviewPacketClosed`)
        is recorded to. The coordinator is the operational boundary those events' own docstrings
        name ("a caller emits it only at the operational boundary where a packet ... becomes work
        assigned to a reviewer") — triage and `assemble_packet` stay projection-only. `None` keeps
        the pre-closure behavior for callers that only project queues and never hand work to a
        reviewer (analysis scripts); every real review surface must pass the sink."""
        self._reviewer_ref = reviewer_ref

    def _replay(self, document_ref: str):
        return self._journal.replay(self._source.events_for_document(document_ref))

    def build_queue(
        self,
        *,
        intent: ReviewIntent = ReviewIntent.CALIBRATION,
        strategy_name: str = "random_sampling",
        k: int = 0,
        seed: int | None = None,
        order_by: str = "intent",
    ) -> tuple[AdaptiveQueueEntry, ...]:
        """The corpus-wide adaptive queue: every document's operational triage items (computed by
        the unmodified `triage_review_queue`), plus up to `k` calibration/research items chosen by
        the named strategy, if `k > 0`.
        """
        index = TelemetryIndex.build(self._source)
        operational: list[tuple[str, TriageItem]] = []
        for document_ref in sorted(set(index.document_ref_by_slot.values())):
            state = self._replay(document_ref)
            operational.extend(
                (document_ref, item) for item in triage_review_queue(state, self._triage_policy)
            )

        sampled: tuple = ()
        if k > 0 and intent != ReviewIntent.OPERATIONAL:
            progress_by_pattern = compute_calibration_progress_map(index, self._log)
            strategy = strategy_by_name(strategy_name)
            candidates = index.latest_canonicals()
            sampled = strategy.select(
                candidates,
                document_ref_by_slot=index.document_ref_by_slot,
                progress_by_pattern=progress_by_pattern,
                corpus_size=len(candidates),
                k=k,
                intent=intent,
                seed=seed,
            )
            self._log_decisions(sampled)

        return build_adaptive_queue(operational=operational, sampled=sampled, order_by=order_by)

    def _log_decisions(self, sampled) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for item in sampled:
            self._log.append(
                SamplingDecision(
                    semantic_slot_id=item.semantic_slot_id,
                    document_ref=item.document_ref,
                    intent=item.intent,
                    strategy_name=item.strategy_name,
                    pattern=item.pattern,
                    explanation=item.explanation,
                    sampled_at=now,
                )
            )

    def open_packet(self, entry: AdaptiveQueueEntry, *, archive_object_ref: str) -> ReviewPacket:
        """Assemble the packet for one adaptive-queue entry via the same `assemble_packet`
        building block `ReviewService.open_document` uses -- packet shape never differs by intent,
        only how the slot was selected does (spec: "presentation varies, the packet does not").
        """
        state = self._replay(entry.document_ref)
        if entry.reason is not None:
            reason = entry.reason
            tier = entry.disclosure_tier or DisclosureTier.TIER_1_SIMPLE
            policy_version = entry.policy_version or self._triage_policy.version
        else:
            canonical = state.canonical_observation(entry.canonical_observation_id)
            reason = _NATURAL_REASON_BY_CLASSIFICATION[canonical.comparison_confidence.classification]
            tier = DisclosureTier.TIER_1_SIMPLE
            policy_version = self._triage_policy.version

        triage_item = TriageItem(
            semantic_slot_id=entry.semantic_slot_id,
            canonical_observation_id=entry.canonical_observation_id,
            observation_type=entry.observation_type,
            reason=reason,
            disclosure_tier=tier,
            policy_version=policy_version,
        )
        packet = assemble_packet(
            state,
            triage_item,
            document_ref=entry.document_ref,
            archive_object_ref=archive_object_ref,
        )
        dispatch = self.open_dispatch_for(entry)
        if self._sink is not None and dispatch is None:
            # Idempotent by derivation from durable telemetry, never from in-memory state: a
            # packet re-opened in the same or a later session reuses its existing open dispatch
            # rather than dispatching again — the projection (`review_closure_status`) must never
            # show two open packets for one piece of assigned work.
            dispatch = dispatch_review_packet(
                sink=self._sink,
                packet=packet,
                reviewer_ref=self._reviewer_ref,
                review_intent=entry.intent.value,
            )
        if self._sink is not None and dispatch is not None:
            already_opened = any(
                isinstance(event, ReviewPacketOpened) and event.packet_id == dispatch.packet_id
                for event in self._source.events_for_document(entry.document_ref)
            )
            if not already_opened:
                open_review_packet(
                    sink=self._sink,
                    dispatch=dispatch,
                    reviewer_ref=self._reviewer_ref,
                )
        return packet

    def open_dispatch_for(self, entry: AdaptiveQueueEntry) -> ReviewPacketDispatched | None:
        """The not-yet-closed `ReviewPacketDispatched` for this entry's slot, derived from the
        durable stream (replay-safe across restarts), or `None`. Public so the Work Queue UI can
        distinguish "pending" from "opened/in progress" and verify closure after a decision.
        """
        dispatch: ReviewPacketDispatched | None = None
        closed_packet_ids: set[str] = set()
        for event in self._source.events_for_document(entry.document_ref):
            if (
                isinstance(event, ReviewPacketDispatched)
                and event.semantic_slot_id == entry.semantic_slot_id
                and event.canonical_observation_id == entry.canonical_observation_id
            ):
                dispatch = event  # the latest dispatch for the slot governs
            elif isinstance(event, ReviewPacketClosed):
                closed_packet_ids.add(event.packet_id)
        if dispatch is not None and dispatch.packet_id not in closed_packet_ids:
            return dispatch
        return None

    def closure_for(self, entry: AdaptiveQueueEntry) -> ReviewPacketClosed | None:
        """The most recent durable `ReviewPacketClosed` for this entry's slot, or `None` — how a
        caller (the Work Queue UI) verifies, from the stream itself rather than from a return
        value, that a submitted decision's packet really closed.
        """
        closure: ReviewPacketClosed | None = None
        for event in self._source.events_for_document(entry.document_ref):
            if (
                isinstance(event, ReviewPacketClosed)
                and event.semantic_slot_id == entry.semantic_slot_id
                and event.canonical_observation_id == entry.canonical_observation_id
            ):
                closure = event
        return closure

    def submit_decision(self, *, entry: AdaptiveQueueEntry, **kwargs) -> ReviewDecisionResult:
        """Passthrough to `ReviewService.submit_decision`; additionally correlates the resulting
        `correction_id` back to `entry`'s `SamplingDecision` for non-operational entries, so
        `review/sampling/progress.py` can later count it as calibration/research evidence.

        Closes the entry's dispatched review packet after the decision succeeds (F4). Ordering is
        the invariant: closure is emitted only after `ReviewService.submit_decision` returned, so a
        failed correction application (an exception) leaves the packet open — never reported
        closed. Closure is idempotent: a repeat submission for a slot whose packet is already
        closed emits no second closure. Historical decisions recorded before this wiring existed
        (outcomes with no dispatch) are left exactly as they are — legacy history, readable, never
        rewritten.
        """
        dispatch_before = self.open_dispatch_for(entry)
        kwargs.setdefault("reviewer_ref", self._reviewer_ref)
        if "review_duration_seconds" not in kwargs and dispatch_before is not None:
            opened = next(
                (
                    event
                    for event in reversed(tuple(self._source.events_for_document(entry.document_ref)))
                    if isinstance(event, ReviewPacketOpened)
                    and event.packet_id == dispatch_before.packet_id
                ),
                None,
            )
            if opened is not None and opened.recorded_at is not None:
                kwargs["review_duration_seconds"] = max(
                    0.0,
                    (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(opened.recorded_at)
                    ).total_seconds(),
                )
        try:
            result = self._service.submit_decision(**kwargs)
        except Exception as error:
            if self._sink is not None and dispatch_before is not None:
                close_review_packet(
                    sink=self._sink,
                    dispatch=dispatch_before,
                    closure_kind=ReviewPacketClosureKind.FAILED_DURING_APPLICATION,
                    reason=f"{type(error).__name__}: {error}",
                )
            raise
        if entry.intent != ReviewIntent.OPERATIONAL and result.correction_id is not None:
            self._log.record_correction(entry.semantic_slot_id, result.correction_id)
        if self._sink is not None:
            dispatch = self.open_dispatch_for(entry)
            outcome = next(
                (e for e in result.emitted_events if isinstance(e, ReviewOutcomeRecorded)), None
            )
            if dispatch is not None and outcome is not None:
                # A packet never closes without a recorded outcome: the closure carries the
                # outcome_id (and correction_id when one exists) so replay can reconstruct the
                # full dispatch → decision → closure chain from the stream alone.
                closure_kind = {
                    ReviewAction.SKIP: ReviewPacketClosureKind.DEFERRED,
                    ReviewAction.ACCEPT_PROVIDER: ReviewPacketClosureKind.ACCEPTED,
                    ReviewAction.MANUAL_EDIT: ReviewPacketClosureKind.CORRECTED,
                    ReviewAction.REJECT: ReviewPacketClosureKind.REJECTED,
                    ReviewAction.ILLEGIBLE: ReviewPacketClosureKind.ILLEGIBLE,
                    ReviewAction.DIFFERENT_THINGS: ReviewPacketClosureKind.DIFFERENT_THINGS,
                }.get(result.action, ReviewPacketClosureKind.RESOLVED)
                close_review_packet(
                    sink=self._sink,
                    dispatch=dispatch,
                    closure_kind=closure_kind,
                    reason=result.action.value,
                    outcome_id=outcome.outcome_id,
                    correction_id=result.correction_id,
                )
        return result
