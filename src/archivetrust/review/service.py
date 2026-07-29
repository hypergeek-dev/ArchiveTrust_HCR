"""The Review Service (HUMAN_REVIEW_SPECIFICATION.md §3, §8, §10; ROADMAP_V2.md §9).

The service the interface (ViewModels, then the Qt client) consumes. It is the concrete realization
of the boundary-spanning Human Review System (`archivetrust.review.__init__`): a single decision
fans out into the three effects of §3, and this service is where they are orchestrated —

  (A/B) Trust Engine: build a `HumanCorrection`, apply it via the frozen Milestone-6 contract
        (`domain.feedback`, supersession never erasure), and emit its telemetry;
  (C)   Learning Platform: record passive review interactions to their own separate stream.

It reads the Trust Engine telemetry stream to assemble packets (LP-2) and never edits a Canonical
Observation out of band — the write is always the Trust Engine's own contract, exercised by the
human's explicit decision (§3, HR-1).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.application.journal import Journal, JournalState
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.current_state import CurrentDocumentState, project_current_document
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.feedback.engine import apply_human_correction
from archivetrust.domain.feedback.models import (
    CorrectionAction,
    CorrectionCategory,
    DatasetCandidate,
    HumanCorrection,
)
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.feedback.telemetry import human_correction_events
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDocumentCreated,
    HumanCorrectionApplied,
    ReviewOutcome,
    ReviewOutcomeRecorded,
    TelemetryEvent,
    stamp_recorded_at,
)
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.learning.review.interaction import ReviewInteraction
from archivetrust.learning.review.sink import ReviewInteractionSink
from archivetrust.review.assembler import assemble_packet
from archivetrust.review.packet import ReviewPacket
from archivetrust.review.triage import (
    TriageClassification,
    TriagePolicy,
    triage_classification_for,
    triage_review_queue,
)
from archivetrust.learning.source import TelemetrySource


class ReviewAction(str, Enum):
    """The reviewer-facing decision vocabulary (HUMAN_REVIEW_SPECIFICATION.md §10.5), phrased as
    collaboration, mapped internally onto the frozen `CorrectionAction` taxonomy. `SKIP` produces
    no correction at all (§10.7) — a skip is never an acceptance.
    """

    ACCEPT_PROVIDER = "accept_provider"
    MANUAL_EDIT = "manual_edit"
    REJECT = "reject"
    MARK_AMBIGUOUS = "mark_ambiguous"
    DIFFERENT_THINGS = "different_things"
    ILLEGIBLE = "illegible"
    REQUEST_FURTHER_REVIEW = "request_further_review"
    SKIP = "skip"


# §10.5's mapping table, made executable. Mark Ambiguous and Request Further Review both currently
# collapse onto FLAG_FOR_REVIEW — a taxonomy gap recorded in §10.5, surfaced (not hidden) here.
_ACTION_TO_CORRECTION: dict[ReviewAction, CorrectionAction] = {
    ReviewAction.ACCEPT_PROVIDER: CorrectionAction.ACCEPT,
    ReviewAction.MANUAL_EDIT: CorrectionAction.EDIT,
    ReviewAction.REJECT: CorrectionAction.REJECT,
    ReviewAction.MARK_AMBIGUOUS: CorrectionAction.FLAG_FOR_REVIEW,
    ReviewAction.DIFFERENT_THINGS: CorrectionAction.DIFFERENT_THINGS,
    ReviewAction.ILLEGIBLE: CorrectionAction.FLAG_FOR_REVIEW,
    ReviewAction.REQUEST_FURTHER_REVIEW: CorrectionAction.FLAG_FOR_REVIEW,
}

# What an EDIT to each observation type most plausibly corrects (Priority 3's deterministic
# inference): fixing a table's content is a lattice/segmentation problem, fixing a section or
# relationship is structural, fixing metadata is metadata — everything text-bearing is a
# transcription fix. Inferred only when the caller supplies no explicit category; never a
# fabricated certainty (the inference basis is recorded in the correction's rationale by callers
# that care).
_EDIT_CATEGORY_BY_TYPE: dict[ObservationType, CorrectionCategory] = {
    ObservationType.TABLE: CorrectionCategory.SEGMENTATION_ERROR,
    ObservationType.SECTION: CorrectionCategory.STRUCTURAL_ERROR,
    ObservationType.ARCHIVE_BOUNDARY: CorrectionCategory.STRUCTURAL_ERROR,
    ObservationType.METADATA: CorrectionCategory.METADATA_ERROR,
    ObservationType.LAYOUT_REGION: CorrectionCategory.CLASSIFICATION_ERROR,
    ObservationType.IMAGE: CorrectionCategory.CLASSIFICATION_ERROR,
}


def infer_correction_category(
    action: ReviewAction, observation_type: ObservationType
) -> CorrectionCategory:
    """Deterministic category inference (Operational Hardening milestone, Priority 3) — applied
    only when the reviewer supplied no explicit category, so categorization costs no interaction.
    REJECT asserts the fact isn't in the archive (spurious); an EDIT's category follows what kind
    of thing was edited; ACCEPT/FLAG assert no specific error and stay honestly OTHER.
    """
    if action == ReviewAction.REJECT:
        return CorrectionCategory.SPURIOUS_CONTENT
    if action == ReviewAction.DIFFERENT_THINGS:
        return CorrectionCategory.DIFFERENT_THINGS
    if action == ReviewAction.MANUAL_EDIT:
        return _EDIT_CATEGORY_BY_TYPE.get(observation_type, CorrectionCategory.TRANSCRIPTION_ERROR)
    return CorrectionCategory.OTHER


class ReviewDecisionResult(BaseModel):
    """The outcome of `submit_decision`. `resulting_canonical_observation` and `correction_id` are
    `None` for a SKIP (no ground truth produced, §10.7). `emitted_events` are the Trust Engine
    telemetry events the decision produced, returned so callers/tests can confirm the correction is
    replayable (HR-4) without reaching into the sink.
    """

    model_config = ConfigDict(frozen=True)

    action: ReviewAction
    semantic_slot_id: str
    resulting_canonical_observation: CanonicalObservation | None
    resulting_canonical_document: CanonicalDocument | None = None
    correction_id: str | None
    emitted_events: tuple[TelemetryEvent, ...]


class ReviewService:
    """Orchestrates review for documents whose telemetry lives in one Trust Engine stream.

    `telemetry_source` and `telemetry_sink` are typically the same object (an
    `InMemoryTelemetrySink` satisfies both), but are accepted separately so a read-only source and
    a write sink can be different implementations if a deployment needs it. `interaction_sink` is
    the Learning Platform's *separate* passive-telemetry store (LP-3).
    """

    def __init__(
        self,
        *,
        telemetry_source: TelemetrySource,
        telemetry_sink: TelemetrySink,
        interaction_sink: ReviewInteractionSink,
        feedback_policy: FeedbackPolicy | None = None,
        triage_policy: TriagePolicy | None = None,
        journal: Journal | None = None,
    ) -> None:
        self._source = telemetry_source
        self._sink = telemetry_sink
        self._interactions = interaction_sink
        # FeedbackPolicy carries no implicit version by design (versioned artifact, LP-8); v1 is
        # the current default the pipeline also uses, overridable per deployment.
        self._feedback_policy = feedback_policy or FeedbackPolicy(feedback_policy_version=1)
        self._triage_policy = triage_policy or TriagePolicy()
        self._journal = journal or Journal()

    def _replay(self, document_ref: str) -> JournalState:
        return self._journal.replay(self._source.events_for_document(document_ref))

    # -- Review queue and packets -----------------------------------------------------------

    def open_document(
        self, *, document_ref: str, archive_object_ref: str
    ) -> tuple[ReviewPacket, ...]:
        """The ordered review queue for one document, each uncertainty assembled into a packet
        (§8.1 triage → §8.2 packet). Reproducible from stored telemetry (HR-6).
        """
        state = self._replay(document_ref)
        return tuple(
            assemble_packet(
                state, item, document_ref=document_ref, archive_object_ref=archive_object_ref
            )
            for item in triage_review_queue(state, self._triage_policy)
        )

    def triage_summary(self, document_ref: str) -> dict[TriageClassification, int]:
        """Deterministic classification counts across every known semantic slot in this document
        (Constitution Article 33) -- computed fresh from replayed telemetry each call, never
        persisted. Lets the Review Center answer "why wasn't this reviewed" (distinguishing
        `NOT_ELIGIBLE` from `WITHHELD_BY_POLICY`) without a new event kind.
        """
        state = self._replay(document_ref)
        counts = {classification: 0 for classification in TriageClassification}
        for slot in state.known_semantic_slots():
            canonical = state.canonical_observation_history(slot)[-1]
            counts[triage_classification_for(canonical, self._triage_policy)] += 1
        return counts

    # -- Passive telemetry (Learning Platform stream, §3 effect C) ---------------------------

    def record_interaction(self, interaction: ReviewInteraction) -> None:
        """Ingest one passive review interaction (§11). The Learning Platform's own temporal
        stream — never the Trust Engine's frozen event set (LP-3).
        """
        self._interactions.append(interaction)

    # -- Decision (§3 effects A/B) -----------------------------------------------------------

    def submit_decision(
        self,
        *,
        packet: ReviewPacket,
        action: ReviewAction,
        corrected_output: str | None = None,
        category: CorrectionCategory | None = None,
        rationale: str | None = None,
        reviewer_ref: str | None = None,
        review_duration_seconds: float | None = None,
    ) -> ReviewDecisionResult:
        """Apply a reviewer's explicit decision.

        For every action except SKIP this builds a `HumanCorrection`, applies it through the Trust
        Engine's Milestone-6 contract (supersession, never erasure), and emits the correction
        telemetry — the write is the Trust Engine's, exercised by the human decision (§3, HR-1).
        SKIP builds no `HumanCorrection` (§10.7's ground-truth distinction still holds), but is a
        real human decision, not silence (Constitution Article 30) -- every action, this one
        included, emits exactly one `ReviewOutcomeRecorded`. The caller still records passive
        telemetry for SKIP separately via `record_interaction`.
        """
        current = project_current_document(
            self._source.events_for_document(packet.document_ref)
        )
        self._validate_current_packet(packet, current)

        if action == ReviewAction.SKIP:
            outcome_event = stamp_recorded_at(
                ReviewOutcomeRecorded(
                    event_id=new_id("event"),
                    document_ref=packet.document_ref,
                    outcome_id=new_id("review_outcome"),
                    semantic_slot_id=packet.semantic_slot_id,
                    canonical_observation_id=packet.canonical_observation_id,
                    action=action.value,
                    outcome=ReviewOutcome.DEFERRED,
                    correction_id=None,
                    reviewer_ref=reviewer_ref,
                    review_duration_seconds=review_duration_seconds,
                )
            )
            self._sink.append(outcome_event)
            return ReviewDecisionResult(
                action=action,
                semantic_slot_id=packet.semantic_slot_id,
                resulting_canonical_observation=None,
                resulting_canonical_document=None,
                correction_id=None,
                emitted_events=(outcome_event,),
            )

        if action == ReviewAction.MANUAL_EDIT and corrected_output is None:
            raise ValueError("MANUAL_EDIT requires corrected_output (§10.5)")

        if current.latest_canonical_document is None:
            raise ValueError(
                "A correction cannot complete without an existing canonical document snapshot"
            )

        state = self._replay(packet.document_ref)
        original = state.canonical_observation(packet.canonical_observation_id)
        contributing = self._contributing_observations(state, original)

        correction_action = _ACTION_TO_CORRECTION[action]
        correction = HumanCorrection(
            correction_id=new_id("correction"),
            target_canonical_observation_id=original.canonical_observation_id,
            category=category or infer_correction_category(action, packet.observation_type),
            action=correction_action,
            raw_ai_output=packet.current_value or "",
            raw_corrected_output=corrected_output,
            rationale=rationale,
        )

        resulting = apply_human_correction(
            original, correction, contributing, self._feedback_policy
        )
        events = human_correction_events(
            correction=correction,
            original=original,
            resulting=resulting,
            document_ref=packet.document_ref,
            policy=self._feedback_policy,
            dataset_candidate=self._dataset_candidate(packet, correction, contributing),
            reviewer_ref=reviewer_ref,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            review_duration_seconds=review_duration_seconds,
        )
        document = self._reassemble_document(
            current=current,
            original=original,
            resulting=resulting,
            correction_id=correction.correction_id,
        )
        document_event = CanonicalDocumentCreated(
            event_id=new_id("event"),
            document_ref=packet.document_ref,
            canonical_document=document,
            alignment_algorithm_version=current.policy_versions.alignment_version,
            reconciliation_policy_version=current.policy_versions.reconciliation_version,
            capability_matrix_version=current.policy_versions.capability_matrix_version,
            confidence_policy_version=current.policy_versions.confidence_version,
            feedback_policy_version=self._feedback_policy.feedback_policy_version,
        )
        applied_index = next(
            index for index, event in enumerate(events) if isinstance(event, HumanCorrectionApplied)
        )
        events = events[: applied_index + 1] + (document_event,) + events[applied_index + 1 :]
        outcome_by_action = {
            ReviewAction.ACCEPT_PROVIDER: ReviewOutcome.ACCEPTED,
            ReviewAction.MANUAL_EDIT: ReviewOutcome.CORRECTED,
            ReviewAction.REJECT: ReviewOutcome.REJECTED,
            ReviewAction.ILLEGIBLE: ReviewOutcome.ILLEGIBLE,
            ReviewAction.DIFFERENT_THINGS: ReviewOutcome.DIFFERENT_THINGS,
        }
        events = events + (
            ReviewOutcomeRecorded(
                event_id=new_id("event"),
                document_ref=packet.document_ref,
                outcome_id=new_id("review_outcome"),
                semantic_slot_id=packet.semantic_slot_id,
                canonical_observation_id=resulting.canonical_observation_id,
                action=action.value,
                outcome=outcome_by_action.get(action, ReviewOutcome.RESOLVED),
                correction_id=correction.correction_id,
                reviewer_ref=reviewer_ref,
                review_duration_seconds=review_duration_seconds,
            ),
        )
        events = tuple(stamp_recorded_at(event) for event in events)
        for event in events:
            self._sink.append(event)

        return ReviewDecisionResult(
            action=action,
            semantic_slot_id=packet.semantic_slot_id,
            resulting_canonical_observation=resulting,
            resulting_canonical_document=document,
            correction_id=correction.correction_id,
            emitted_events=events,
        )

    @staticmethod
    def _validate_current_packet(packet: ReviewPacket, current: CurrentDocumentState) -> None:
        try:
            slot = current.slot(packet.semantic_slot_id)
        except KeyError as error:
            raise ValueError(
                f"Review packet references unknown semantic slot {packet.semantic_slot_id}"
            ) from error
        if slot.current.canonical_observation_id != packet.canonical_observation_id:
            raise ValueError(
                "Stale review packet: its canonical observation is no longer current"
            )
        if any(
            outcome.canonical_observation_id == packet.canonical_observation_id
            for outcome in slot.review_outcomes
        ):
            raise ValueError("Review packet already has a recorded outcome")

    @staticmethod
    def _reassemble_document(
        *,
        current: CurrentDocumentState,
        original: CanonicalObservation,
        resulting: CanonicalObservation,
        correction_id: str,
    ) -> CanonicalDocument:
        previous = current.latest_canonical_document
        if previous is None:  # guarded by submit_decision; retained for direct-call safety
            raise ValueError("Cannot reassemble a document without a prior snapshot")
        # ``effective_contained_observations`` has already advanced any historical root ids to
        # the current version for their semantic slot.  This decision's new version is not yet in
        # the source projection, so replace exactly its predecessor here.
        contained = tuple(
            resulting.canonical_observation_id
            if observation_id == original.canonical_observation_id
            else observation_id
            for observation_id in current.effective_contained_observations
        )
        return previous.model_copy(
            update={
                "document_snapshot_id": new_id("document_snapshot"),
                "contained_observations": contained,
                "ontology_version": current.policy_versions.ontology_version
                or previous.ontology_version,
                "document_version": previous.document_version + 1,
                "reassembly_trigger": f"human_correction:{correction_id}",
                "supersedes": previous.document_snapshot_id,
                "superseded_by": None,
            }
        )

    @staticmethod
    def _dataset_candidate(
        packet: ReviewPacket,
        correction: HumanCorrection,
        contributing: tuple[Observation, ...],
    ) -> DatasetCandidate:
        """Every completed human correction is a training/calibration candidate (Operational
        Hardening milestone, Priority 2) — curation (which candidates actually enter a dataset)
        remains the future Dataset Generation milestone's job (ROADMAP.md S14); this only marks and
        links, never filters. The description records the full traceability chain: correction →
        Canonical Observation → contributing Observations → Evidence → Archive Object.
        """
        observation_ids = tuple(o.observation_id for o in contributing)
        evidence_ids = tuple(eid for o in contributing for eid in o.evidence_ids)
        return DatasetCandidate(
            candidate_id=new_id("dataset_candidate"),
            correction_id=correction.correction_id,
            archive_object_ref=packet.archive_object_ref,
            description=(
                f"{correction.action.value}/{correction.category.value} on "
                f"{packet.observation_type.value} slot {packet.semantic_slot_id}; canonical "
                f"{packet.canonical_observation_id}; observations {list(observation_ids)}; "
                f"evidence {list(evidence_ids)}; archive object {packet.archive_object_ref}"
            ),
        )

    @staticmethod
    def _contributing_observations(
        state: JournalState, canonical: CanonicalObservation
    ) -> tuple[Observation, ...]:
        return tuple(
            state.observation(ref.observation_id)
            for ref in canonical.contributing_observations
        )

