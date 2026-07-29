"""Standardized Telemetry (ROADMAP.md S12, S5.10; Constitution Article 16, Article 17).

Telemetry describes changes in *knowledge* -- what became known, compared, or decided about a
document -- never which function ran or which process executed (Article 16: process-level logging
is a separate, non-canonical concern and has no home here). Every event type below is one of the
canonical set enumerated in ROADMAP.md S12; no process-oriented event ("AdapterStarted", etc.) may
be added to this module.

Each event carries the *full* domain object(s) it announces (not just an id), because replay
(Article 17) must reconstruct Evidence, both Observation Graphs, every comparison decision, and
the full confidence evolution from telemetry alone, without re-running any provider.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.comparison.clustering import ClusteringBasisCode
from archivetrust.domain.confidence.models import ComparisonConfidence
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.shared.ids import content_address, new_id
from archivetrust.domain.shared.versioning import CURRENT_SCHEMA_VERSION


class TelemetryEventKind(str, Enum):
    """The canonical knowledge-evolution event set, ROADMAP.md S12 -- exhaustive by design. A
    new event kind may only be added by amending that section, never ad hoc in code.
    """

    PROVIDER_OBSERVATION_ATTEMPTED = "ProviderObservationAttempted"
    EVIDENCE_REJECTED = "EvidenceRejected"
    EVIDENCE_CREATED = "EvidenceCreated"
    OBSERVATION_CREATED = "ObservationCreated"
    OBSERVATION_MAPPED = "ObservationMapped"
    ALIGNMENT_ATTEMPTED = "AlignmentAttempted"
    OBSERVATION_ALIGNED = "ObservationAligned"
    OBSERVATION_LEFT_UNALIGNED = "ObservationLeftUnaligned"
    OBSERVATION_COMPARED = "ObservationCompared"
    OBSERVATION_ACCEPTED = "ObservationAccepted"
    OBSERVATION_REJECTED = "ObservationRejected"
    OBSERVATION_MERGED = "ObservationMerged"
    AGREEMENT_CALCULATED = "AgreementCalculated"
    CANONICAL_DECISION_CREATED = "CanonicalDecisionCreated"
    CONFIDENCE_CHANGED = "ConfidenceChanged"
    KNOWLEDGE_MERGED = "KnowledgeMerged"
    KNOWLEDGE_DISCARDED = "KnowledgeDiscarded"
    CANONICAL_DOCUMENT_CREATED = "CanonicalDocumentCreated"
    HUMAN_CORRECTION_SUBMITTED = "HumanCorrectionSubmitted"
    HUMAN_CORRECTION_APPLIED = "HumanCorrectionApplied"
    DATASET_CANDIDATE_CREATED = "DatasetCandidateCreated"
    PROVENANCE_CONTEXT_ESTABLISHED = "ProvenanceContextEstablished"
    """Constitution Article 32, `ARCHITECTURE_TELEMETRY_STANDARD.md` S8 -- the one event kind in
    this set that is a context-establishment event, not a knowledge-evolution event (S8.2). Kept in
    this same closed set, per Article 32's "no parallel telemetry system" requirement, rather than a
    second event vocabulary."""
    CANDIDATE_EXCLUDED = "CandidateExcluded"
    """Constitution Article 27 -- records that a candidate Observation pair was structurally
    excluded from a comparison group by the alignment mechanism itself, symmetric with
    `AlignmentAttempted`'s record of inclusion. Governs only mechanisms actually present in `src/`
    (`ARCHITECTURE_TELEMETRY_STANDARD.md` S1.1); see `domain.alignment.service.ClusteringAlignmentService`."""
    CANDIDATE_EXCLUDED_BATCH = "CandidateExcludedBatch"
    """F3 encoding repair -- semantically equivalent to a sequence of `CandidateExcluded` events,
    but batched so ambiguous pools do not dominate telemetry volume."""
    REVIEW_OUTCOME_RECORDED = "ReviewOutcomeRecorded"
    """Constitution Article 30 (revised 2026-07-14 alongside Article 33) -- every terminal human
    review decision, `SKIP` included, emits exactly one of these. Never emitted by triage or packet
    assembly (`review/triage.py`, `review/packet.py`): those are deterministic projections over
    already-persisted telemetry and must never themselves emit telemetry (Article 33); only
    `review.service.ReviewService.submit_decision`, a real human decision, does."""
    REVIEW_PACKET_DISPATCHED = "ReviewPacketDispatched"
    """F4 review closure -- records that a projected review packet was actually handed to a
    reviewer. Queue projection alone remains side-effect free; this event is the explicit lifecycle
    boundary between an eligible uncertainty and a dispatched packet."""
    REVIEW_PACKET_CLOSED = "ReviewPacketClosed"
    """F4 review closure -- terminal queue-level closure for one dispatched packet. Complements,
    but does not replace, `ReviewOutcomeRecorded`: decisions describe what a reviewer did; closure
    describes whether the packet lifecycle ended."""
    REVIEW_PACKET_CREATED = "ReviewPacketCreated"
    REVIEW_PACKET_OPENED = "ReviewPacketOpened"
    SEGMENTATION_RUN_COMPLETED = "SegmentationRunCompleted"
    """docs/htr-migration-plan.md Stage 3 -- one `SegmentationAdapter` run over a Page completed
    (docs/htr-domain-design.md §7)."""
    METHOD_RUN_COMPLETED = "MethodRunCompleted"
    """Stage 3 -- one `htr.experiment.models.MethodRun` reached a terminal outcome."""
    REVIEW_SUBMISSION_RECORDED = "ReviewSubmissionRecorded"
    """Stage 3 -- one blind `review.htr_models.ReviewSubmission` was recorded."""
    ADJUDICATION_RECORDED = "AdjudicationRecorded"
    """Stage 3 -- one `review.htr_models.Adjudication` resolved a disagreeing `AgreementResult`."""
    CANONICAL_RESULT_CREATED = "CanonicalResultCreated"
    """Stage 3 -- one `domain.canonical.result.CanonicalResult` was created (renaming
    `CanonicalDocumentCreated` semantics where line-level, per docs/htr-domain-design.md §2)."""


class ReviewOutcome(str, Enum):
    """Constitution Article 30. Deliberately two values only -- `aged_out` is not a value of this
    event at all, since aging is a derived, query-time fact (an eligible slot with no recorded
    outcome past a configured horizon), never a third outcome a decision itself can produce
    (`ARCHITECTURE_TELEMETRY_STANDARD.md` §1.1/Article 33's discipline, applied here)."""

    RESOLVED = "resolved"
    """A real `HumanCorrection` was built and applied (every `ReviewAction` except `SKIP`)."""
    DEFERRED = "deferred"
    """`ReviewAction.SKIP` -- a human decision to defer, not silence, and not a correction."""
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    ILLEGIBLE = "illegible"
    DIFFERENT_THINGS = "different_things"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    FAILED_DURING_APPLICATION = "failed_during_application"


class ReviewPacketClosureKind(str, Enum):
    """Terminal lifecycle states for a dispatched review packet."""

    RESOLVED = "resolved"
    """Closed by a resolving human decision."""
    DEFERRED = "deferred"
    """Closed by an explicit defer/skip decision; the underlying slot may remain eligible."""
    EXPIRED = "expired"
    """Closed because the packet exceeded an operational age horizon."""
    WITHDRAWN = "withdrawn"
    """Closed because the packet was withdrawn from the queue by policy or operator action."""
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    ILLEGIBLE = "illegible"
    DIFFERENT_THINGS = "different_things"
    FAILED_DURING_APPLICATION = "failed_during_application"


class ConfidenceLevel(str, Enum):
    """Which of the three confidence levels a ConfidenceChanged event concerns (S5.3, Article 13:
    "Telemetry (ConfidenceChanged) must indicate which of the three levels changed and why.")
    """

    PROVIDER = "provider"
    COMPARISON = "comparison"
    CANONICAL = "canonical"


class TelemetryEvent(BaseModel):
    """Abstract base for all telemetry events. `document_ref` scopes every event to the Archive
    Object it concerns, so replay can be run per-document (S5.10) without scanning unrelated
    history.

    `reconciliation_policy_version`/`capability_matrix_version` are carried on every event (not
    only comparison-specific ones) so `Journal` doesn't need per-event-type special-casing to
    read them. They are the one concrete requirement `MILESTONE4_COMPARISON_ENGINE.md` S12/S18
    item 2 places on Milestone 2's schemas: "without the Policy/Matrix versions, a replay months
    later could not reproduce a threshold-dependent decision." Non-comparison events (Evidence/
    Observation creation, human feedback) leave both `None` -- they are not the outcome of any
    Policy- or Matrix-parameterized decision.

    `confidence_policy_version` is the analogous field for Milestone 5's Confidence Engine
    (ROADMAP.md S5.3.1: any confidence-derivation mechanism "must be independently versioned, and
    that version must be recorded so replay can identify which calibration produced a given
    Canonical Confidence"). Only `ConfidenceChanged` events populate it in practice.

    `feedback_policy_version` is the same idea for Milestone 6's `FeedbackPolicy` (which
    `CorrectionAction` maps to which post-correction confidence value) -- kept as its own field
    rather than overloading `confidence_policy_version`, since a single `ConfidenceChanged` event
    emitted from a human correction is legitimately versioned by *both* policies at once (which
    Confidence Policy governed the pre-correction value, which Feedback Policy governs the
    post-correction one).

    `alignment_algorithm_version` is the same idea for the Alignment Observability layer (ROADMAP.md
    S5.12, Article 24): the alignment/grouping events populate it so replay can identify which
    alignment strategy produced a given grouping, and so future strategies can be compared on the
    same corpus. Only the three alignment events populate it in practice; every other event leaves
    it `None`.
    """

    model_config = ConfigDict(frozen=True)

    kind: TelemetryEventKind

    event_id: str
    document_ref: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    recorded_at: str | None = None
    """UTC ISO-8601 wall-clock time the event was produced (Operational Hardening milestone,
    Priority 13) -- what makes per-document processing duration, throughput, and ETA computable
    directly from the telemetry stream rather than from session-local UI timers that vanish on
    restart. `None` for historical events recorded before this field existed; never backfilled."""
    reconciliation_policy_version: int | None = None
    capability_matrix_version: int | None = None
    confidence_policy_version: int | None = None
    feedback_policy_version: int | None = None
    alignment_algorithm_version: int | None = None


class ProviderInvocationOutcome(str, Enum):
    """What one provider invocation actually produced (Operational Hardening milestone) — the
    executable form of Article 18's triad. "Never invoked" needs no value here: it is exactly the
    absence of a `ProviderObservationAttempted` event for that provider/document.
    """

    PRODUCED_OBSERVATIONS = "produced_observations"
    NO_OBSERVATIONS = "no_observations"
    """The provider ran to completion and genuinely observed nothing — an empty page, an
    image-only region with no detectable text, a document with no extractable content. A fact
    about the archive, not a failure."""
    FAILED = "failed"
    """The provider could not complete (crash, unreadable input, missing dependency) —
    `failure_reason` carries the recorded cause."""


class ProviderFailureCategory(str, Enum):
    """Which of a small, fixed set of things went wrong (Observability milestone, 2026-07-13) —
    set only when `outcome` is `FAILED`. Never inferred from `failure_reason`'s free text; each
    value is set only where the pipeline has a structural reason to believe it, never a guess.
    """

    TIMEOUT = "timeout"
    """A timeout-signalling exception (e.g. an HTTP client's own timeout error) was caught."""
    EXCEPTION = "exception"
    """An exception was caught (by the pipeline, or earlier by the provider's own client) that
    was not specifically recognizable as a timeout."""
    PROVIDER_ERROR = "provider_error"
    """The provider's own adapter/client caught an error internally and returned it via
    `ProviderRunResult.failure_reason` without the pipeline itself needing to catch anything —
    the well-behaved path every adapter contract expects."""
    CANCELLED = "cancelled"
    """Reserved for a future per-invocation cancellation signal; no code path sets this today
    (queue-level cancellation currently only takes effect between documents, never mid-invocation)."""
    UNKNOWN = "unknown"
    """A defensive fallback for a `FAILED` outcome that matched none of the above — not expected
    in practice, never used to paper over a case that should have a real category."""


class ProviderObservationAttempted(TelemetryEvent):
    """Records that a provider was invoked for a given archive region/page, regardless of
    outcome. Without this, replay cannot distinguish "never invoked here" from "invoked and found
    nothing" from "invoked and failed" (Constitution Article 18).

    `outcome`/`observation_count`/`failure_reason` (Operational Hardening milestone) make that
    distinction explicit *data* rather than something replay must infer by joining against
    `ObservationCreated` events. Optional so historical events (which predate the fields) still
    parse; `None` means "recorded before outcomes were captured", never "unknown by design".

    `duration_ms`/`retry_count`/`cold_start`/`init_duration_ms`/`timeout`/`failure_category`/
    `pages_processed` (Observability milestone, 2026-07-13) are the same discipline applied to
    runtime/reliability facts: each is `None` exactly when it genuinely could not be established,
    never defaulted to zero or guessed. See `application/pipeline.py`'s `run_pipeline` for exactly
    which of these every live invocation populates and why.
    """

    kind: TelemetryEventKind = TelemetryEventKind.PROVIDER_OBSERVATION_ATTEMPTED

    provider_id: str
    provider_version: str
    invocation_id: str
    target_region: str | None = None
    outcome: ProviderInvocationOutcome | None = None
    observation_count: int | None = None
    failure_reason: str | None = None
    duration_ms: int | None = None
    """Wall-clock time `run_pipeline` spent inside this invocation's `adapter.observe()` call,
    measured by `time.monotonic()`. `None` for historical events recorded before this field
    existed, and for the page-render-failure path (`WorkspaceProcessingService`), which never
    actually invokes an adapter."""
    retry_count: int | None = None
    """How many times `run_pipeline` itself retried this invocation before recording an outcome —
    always `0` today, since `run_pipeline` invokes each adapter exactly once; no retry loop exists
    at this layer. Does **not** capture a provider/runtime's own internal retry behavior (e.g.
    `VLLMRuntime`'s HTTP retry loop), which is a separate, deeper fact not yet threaded to this
    field. `None` only when even the pipeline-level fact could not be established (the invocation
    raised an exception before `run_pipeline` could record a normal outcome)."""
    cold_start: bool | None = None
    """Whether this was the first invocation of this `provider_id` since the current process (or
    more precisely, the calling `WorkspaceProcessingService` instance) started. `None` when no
    such tracking was supplied to `run_pipeline` — never guessed."""
    init_duration_ms: int | None = None
    """Reserved for a future decomposition of initialization cost out of `duration_ms`; no code
    path separates the two today, so this is always `None`."""
    timeout: bool | None = None
    """Whether this invocation ended because of an actual timeout exception. `None` for every
    non-`FAILED` outcome (not applicable — nothing failed, so there is no timeout question to
    answer); `True`/`False` for a `FAILED` outcome, set only from a real caught exception's type,
    never inferred from `failure_reason` text."""
    failure_category: ProviderFailureCategory | None = None
    """One of `ProviderFailureCategory`, set only when `outcome` is `FAILED`. `None` for every
    non-`FAILED` outcome and for historical `FAILED` events recorded before this field existed."""
    pages_processed: int | None = None
    """How many archive-object pages this invocation covered: `1` for a `PAGE_IMAGE` adapter (one
    page per invocation, always true by construction); `None` for a `DOCUMENT` adapter (the whole
    document is one invocation, but its page count is not resolved for that adapter kind today —
    genuinely unavailable, never estimated) or when the page-render-failure path recorded no real
    invocation at all."""


class EvidenceRejected(TelemetryEvent):
    """Records a provider response that failed schema/contract validation, as a knowledge-
    evolution event -- not just an application-log line (S12). The rejected raw output is
    preserved here even though no Evidence record could validly be created from it, per
    Constitution Article 5 (preserving rejected/malformed raw output is what makes audit and
    upstream bug reports possible later).
    """

    kind: TelemetryEventKind = TelemetryEventKind.EVIDENCE_REJECTED

    provider_id: str
    provider_version: str
    invocation_id: str
    processing_stage: ProcessingStage
    raw_output: str
    rejection_reason: str


class EvidenceCreated(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.EVIDENCE_CREATED

    invocation_id: str
    evidence: Evidence


class ObservationCreated(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_CREATED

    invocation_id: str
    observation: Observation


class ObservationMapped(TelemetryEvent):
    """Records that raw Evidence was mapped into the Canonical Observation Ontology -- the
    ontology-mapping act itself, distinct from ObservationCreated's record of the resulting
    object (S5.1 pipeline diagram: Evidence -> Observation is annotated with both
    ObservationCreated and ObservationMapped).

    **Known gap, disclosed not silently left implicit:** this event kind is not constructed
    anywhere in `src/` today (confirmed 2026-07-14, Phase 5) -- every provider importer maps
    `native_label -> ObservationType` via its own `MappingTable` (`domain/ontology/mapping.py`,
    Constitution Article 28) but none yet emits `ObservationMapped` to announce it.
    `mapping_table_entry_id`/`mapping_table_version` are added now so the schema is ready the day
    this event kind is actually wired, per Article 28 -- not because they are reachable yet.
    """

    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_MAPPED

    observation_id: str
    source_evidence_ids: tuple[str, ...]
    ontology_version: int
    mapping_table_entry_id: str | None = None
    """Constitution Article 28 -- which `MappingTableEntry` (`domain/ontology/mapping.py`) governed
    this mapping. `None` for historical events and until a real call site populates it."""
    mapping_table_version: int | None = None
    """The mapping table's `table_version` at the time this mapping was made (Article 28's
    independent-versioning discipline, mirroring Article 19 one layer inward)."""


class AlignmentAttempted(TelemetryEvent):
    """Records one alignment/grouping attempt for a comparison group (ROADMAP.md S5.12, Article
    24): which Observations were *considered* (`candidate_observation_ids`) and which were
    *selected* into the group (`selected_observation_ids`), and why (`alignment_rationale`). Makes
    explicit the hypothesis -- previously an unrecorded assumption -- that the grouped Observations
    refer to the same semantic object. `comparison_group_id` equals the resulting Canonical
    Observation's `semantic_slot_id`.
    """

    kind: TelemetryEventKind = TelemetryEventKind.ALIGNMENT_ATTEMPTED

    alignment_attempt_id: str
    comparison_group_id: str
    algorithm_name: str
    candidate_observation_ids: tuple[str, ...]
    selected_observation_ids: tuple[str, ...]
    alignment_rationale: str


class ObservationAligned(TelemetryEvent):
    """Records that one Observation was grouped with at least one other Observation into a
    comparison group, referencing the `AlignmentAttempt` that placed it there (S5.12).
    """

    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_ALIGNED

    observation_id: str
    comparison_group_id: str
    alignment_attempt_id: str


class ObservationLeftUnaligned(TelemetryEvent):
    """Records that one Observation was considered but aligned with no other -- the sole member of
    its group -- so that "no suitable comparison group exists for this Observation" is an explicit
    recorded fact, never a silent absence (Constitution Article 18, Article 24; S5.12).
    """

    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_LEFT_UNALIGNED

    observation_id: str
    comparison_group_id: str
    alignment_attempt_id: str
    reason: str


class ObservationCompared(TelemetryEvent):
    """Records that a set of Observations were clustered as plausibly referring to the same
    semantic slot (S5.9's clustering phase), and why (`clustering_basis`) -- content-independent
    per Constitution Article 10.
    """

    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_COMPARED

    semantic_slot_id: str
    compared_observation_ids: tuple[str, ...]
    clustering_basis: str


class ObservationAccepted(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_ACCEPTED

    observation_id: str
    canonical_observation_id: str
    aspect: str | None = None


class ObservationRejected(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_REJECTED

    observation_id: str
    reason: str


class ObservationMerged(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.OBSERVATION_MERGED

    source_observation_ids: tuple[str, ...]
    resulting_canonical_observation_id: str


class AgreementCalculated(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.AGREEMENT_CALCULATED

    semantic_slot_id: str
    comparison_confidence: ComparisonConfidence


class CanonicalDecisionCreated(TelemetryEvent):
    """The Comparison->Canonical Observation transition (Constitution Article 4's "Canonical
    Decision"; ROADMAP.md S5.1's terminology-alignment note ties this event name to exactly this
    transition, including supersession -- a superseding CanonicalObservation is also announced via
    this event, not a separate one).
    """

    kind: TelemetryEventKind = TelemetryEventKind.CANONICAL_DECISION_CREATED

    canonical_observation: CanonicalObservation


class ConfidenceChanged(TelemetryEvent):
    """Must indicate which of the three confidence levels changed and why (S5.3, Article 13)."""

    kind: TelemetryEventKind = TelemetryEventKind.CONFIDENCE_CHANGED

    subject_id: str
    level: ConfidenceLevel
    previous_value: float | None
    new_value: float | None
    reason: str


class KnowledgeMerged(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.KNOWLEDGE_MERGED

    merged_ids: tuple[str, ...]
    resulting_id: str
    description: str


class KnowledgeDiscarded(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.KNOWLEDGE_DISCARDED

    discarded_ids: tuple[str, ...]
    reason: str


class CanonicalDocumentCreated(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.CANONICAL_DOCUMENT_CREATED

    canonical_document: CanonicalDocument


class HumanCorrectionSubmitted(TelemetryEvent):
    """`reviewer_ref`/`submitted_at`/`review_duration_seconds` (Operational Hardening milestone,
    Priority 5) attribute the correction for dataset QA — inter-reviewer agreement, fatigue
    filtering. `reviewer_ref` is a configurable pseudonym (the same `reviewer_ref` the app is
    launched with), never a verified identity; `submitted_at` is UTC ISO-8601 wall-clock (the
    interaction stream's `time.monotonic` timestamps are not comparable across sessions). All
    optional so historical events still parse.
    """

    kind: TelemetryEventKind = TelemetryEventKind.HUMAN_CORRECTION_SUBMITTED

    correction_id: str
    target_canonical_observation_id: str
    category: str
    action: str
    raw_ai_output: str
    raw_corrected_output: str | None = None
    rationale: str | None = None
    reviewer_ref: str | None = None
    submitted_at: str | None = None
    review_duration_seconds: float | None = None


class HumanCorrectionApplied(TelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.HUMAN_CORRECTION_APPLIED

    correction_id: str
    resulting_canonical_observation: CanonicalObservation


class DatasetCandidateCreated(TelemetryEvent):
    """`archive_object_ref` (Operational Hardening milestone, Priority 2) closes the traceability
    chain at the event level: candidate → correction → Canonical Observation → Observations →
    Evidence → this immutable Archive Object, without parsing `description`.
    """

    kind: TelemetryEventKind = TelemetryEventKind.DATASET_CANDIDATE_CREATED

    candidate_id: str
    correction_id: str
    description: str
    archive_object_ref: str | None = None


class ReviewOutcomeRecorded(TelemetryEvent):
    """Constitution Article 30 -- the terminal record of one human review decision. `action` is a
    plain string (mirroring `HumanCorrectionSubmitted.action`'s own precedent) -- `ReviewAction`
    lives in `review/service.py`, an application-layer module this domain-layer schema must not
    import (dependency direction: `review` depends on `domain.telemetry`, never the reverse).
    `correction_id` is `None` exactly for `ReviewOutcome.DEFERRED` (`SKIP` builds no
    `HumanCorrection`); populated for every `RESOLVED` outcome.
    """

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_OUTCOME_RECORDED

    outcome_id: str
    semantic_slot_id: str
    canonical_observation_id: str
    action: str
    outcome: ReviewOutcome
    correction_id: str | None = None
    reviewer_ref: str | None = None
    review_duration_seconds: float | None = None


class ReviewPacketCreated(TelemetryEvent):
    """Durable identity for a packet before it is dispatched."""

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_PACKET_CREATED

    packet_id: str
    semantic_slot_id: str
    canonical_observation_id: str
    review_intent: str | None = None


class ReviewPacketOpened(TelemetryEvent):
    """A dispatched packet was opened/claimed by an attributable reviewer."""

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_PACKET_OPENED

    packet_id: str
    semantic_slot_id: str
    canonical_observation_id: str
    reviewer_ref: str | None = None


class ReviewPacketDispatched(TelemetryEvent):
    """Queue-level record that one projected review packet was actually handed to a reviewer.

    This event is deliberately not emitted by deterministic triage or packet assembly. A caller
    emits it only at the operational boundary where a packet leaves the projection and becomes work
    assigned to a reviewer.
    """

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_PACKET_DISPATCHED

    packet_id: str
    semantic_slot_id: str
    canonical_observation_id: str
    reviewer_ref: str | None = None
    review_intent: str | None = None


class ReviewPacketClosed(TelemetryEvent):
    """Terminal lifecycle state for a dispatched review packet."""

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_PACKET_CLOSED

    packet_id: str
    semantic_slot_id: str
    canonical_observation_id: str
    closure_kind: ReviewPacketClosureKind
    reason: str | None = None
    outcome_id: str | None = None
    correction_id: str | None = None

    @model_validator(mode="after")
    def _requires_terminal_outcome(self) -> "ReviewPacketClosed":
        if self.outcome_id is None:
            raise ValueError("ReviewPacketClosed requires a terminal outcome_id")
        return self


class SegmentationRunCompleted(TelemetryEvent):
    """One `SegmentationAdapter` run over a Page completed (docs/htr-domain-design.md §7).
    References the produced Regions/TextLines/InputCrops by id, never embedded -- the same
    reference-by-id discipline `Observation.evidence_ids` already enforces (Constitution
    Article 7)."""

    kind: TelemetryEventKind = TelemetryEventKind.SEGMENTATION_RUN_COMPLETED

    segmentation_run_id: str
    page_id: str
    segmentation_adapter_name: str
    region_ids: tuple[str, ...]
    text_line_ids: tuple[str, ...]
    input_crop_ids: tuple[str, ...]


class MethodRunCompleted(TelemetryEvent):
    """One `htr.experiment.models.MethodRun` reached a terminal outcome (docs/htr-domain-design.md
    §1, §4). Mirrors `ProviderObservationAttempted`'s outcome discipline (Constitution
    Article 18) at HTR-method-run granularity."""

    kind: TelemetryEventKind = TelemetryEventKind.METHOD_RUN_COMPLETED

    method_run_id: str
    experiment_run_id: str
    method_id: str
    evidence_id: str
    outcome: str
    failure_reason: str | None = None


class ReviewSubmissionRecorded(TelemetryEvent):
    """One blind `review.htr_models.ReviewSubmission` was recorded (docs/htr-domain-design.md
    §1)."""

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_SUBMISSION_RECORDED

    submission_id: str
    assignment_id: str
    reviewer_ref: str


class AdjudicationRecorded(TelemetryEvent):
    """One `review.htr_models.Adjudication` resolved a disagreeing `AgreementResult`
    (docs/htr-domain-design.md §1)."""

    kind: TelemetryEventKind = TelemetryEventKind.ADJUDICATION_RECORDED

    adjudication_id: str
    agreement_result_id: str
    adjudicator_ref: str


class CanonicalResultCreated(TelemetryEvent):
    """One `domain.canonical.result.CanonicalResult` was created (docs/htr-domain-design.md §2:
    "renaming CanonicalDocumentCreated semantics where line-level"). References the result by id
    rather than embedding it, unlike `CanonicalDocumentCreated`/`CanonicalDecisionCreated` (which
    embed the full object) -- a `CanonicalResult` can be read back from its own store by id, and
    keeping this event lean avoids duplicating potentially many per-line spans into telemetry."""

    kind: TelemetryEventKind = TelemetryEventKind.CANONICAL_RESULT_CREATED

    canonical_result_id: str
    page_id: str
    strategy: str
    strategy_version: int


class ProvenanceContextEstablished(TelemetryEvent):
    """The root Operational Context event for one processing run (Constitution Article 32,
    `ARCHITECTURE_TELEMETRY_STANDARD.md` S8). **Not** a knowledge-evolution event (S8.2) -- it
    establishes the fixed environment every later event for this `document_ref` is implicitly
    scoped to, resolved *positionally* by `latest_provenance_context` below (S8.6), never by a
    `context_ref` field repeated on every other event kind (Article 7's no-duplicated-
    representation discipline).

    Populates `reconciliation_policy_version`/`capability_matrix_version`/
    `confidence_policy_version` unconditionally -- the first event kind required to, so a document
    with, say, zero `ConfidenceChanged` events still leaves a record of which policy was loaded for
    its run. `feedback_policy_version` and `alignment_algorithm_version` are left `None` here: unlike
    the three policies above, neither is known at processing time -- `FeedbackPolicy` is resolved
    only later, at human-correction time, by a different subsystem entirely (mirroring
    `domain/feedback/telemetry.py`'s own existing precedent that `feedback_policy_version` is
    populated only where a correction actually occurred), and no alignment algorithm has run yet
    when this event is created. This is a narrow, code-evidenced refinement of Article 32/S8.3's
    "populate every named policy version unconditionally": refined to mean every version *knowable
    at this stage*, discovered during this event kind's first implementation pass (2026-07-14).
    """

    kind: TelemetryEventKind = TelemetryEventKind.PROVENANCE_CONTEXT_ESTABLISHED

    context_id: str
    ontology_version: int
    git_commit: str | None
    """`None` only when genuinely unresolvable (e.g. a packaged install with no `.git` directory)
    -- never a guessed or placeholder value (Article 18's silence-vs-failure discipline, applied to
    this fact)."""
    configuration_hash: str
    configuration_snapshot_reference: str
    """Inline snapshot content today, not an external pointer -- no blob store exists elsewhere in
    this codebase (`Evidence.raw_output` is stored inline for the identical reason, ROADMAP.md S15
    Q2's resolution). Becomes a real external reference only if/when such a store is introduced."""
    machine_identifier: str | None
    workspace_identifier: str

    @model_validator(mode="after")
    def _validate_context_id(self) -> "ProvenanceContextEstablished":
        expected = ProvenanceContextEstablished.compute_context_id(
            document_ref=self.document_ref,
            git_commit=self.git_commit,
            configuration_hash=self.configuration_hash,
            ontology_version=self.ontology_version,
            reconciliation_policy_version=self.reconciliation_policy_version,
            capability_matrix_version=self.capability_matrix_version,
            confidence_policy_version=self.confidence_policy_version,
            feedback_policy_version=self.feedback_policy_version,
            alignment_algorithm_version=self.alignment_algorithm_version,
            machine_identifier=self.machine_identifier,
            workspace_identifier=self.workspace_identifier,
        )
        if self.context_id != expected:
            raise ValueError(
                f"context_id {self.context_id!r} does not match its content address "
                f"{expected!r} -- construct ProvenanceContextEstablished via .create(), never by "
                "hand-setting context_id"
            )
        return self

    @staticmethod
    def compute_context_id(
        *,
        document_ref: str,
        git_commit: str | None,
        configuration_hash: str,
        ontology_version: int,
        reconciliation_policy_version: int | None,
        capability_matrix_version: int | None,
        confidence_policy_version: int | None,
        feedback_policy_version: int | None,
        alignment_algorithm_version: int | None,
        machine_identifier: str | None,
        workspace_identifier: str,
    ) -> str:
        """S8.4's identity rule: every field that constitutes what the run's environment actually
        was, excluding pure metadata (`event_id`, `recorded_at`) which carries no identity."""
        return content_address(
            document_ref,
            git_commit or "",
            configuration_hash,
            str(ontology_version),
            str(reconciliation_policy_version) if reconciliation_policy_version is not None else "",
            str(capability_matrix_version) if capability_matrix_version is not None else "",
            str(confidence_policy_version) if confidence_policy_version is not None else "",
            str(feedback_policy_version) if feedback_policy_version is not None else "",
            str(alignment_algorithm_version) if alignment_algorithm_version is not None else "",
            machine_identifier or "",
            workspace_identifier,
            prefix="context",
        )

    @classmethod
    def create(
        cls,
        *,
        document_ref: str,
        ontology_version: int,
        git_commit: str | None,
        configuration_hash: str,
        configuration_snapshot_reference: str,
        machine_identifier: str | None,
        workspace_identifier: str,
        reconciliation_policy_version: int,
        capability_matrix_version: int,
        confidence_policy_version: int,
        feedback_policy_version: int | None = None,
        alignment_algorithm_version: int | None = None,
    ) -> "ProvenanceContextEstablished":
        """The only supported construction path -- computes the content-addressed `context_id`
        (S8.4). `reconciliation_policy_version`/`capability_matrix_version`/
        `confidence_policy_version` are mandatory (knowable at processing time, S8.3);
        `feedback_policy_version`/`alignment_algorithm_version` remain optional, `None` unless a
        caller genuinely has them (see class docstring)."""
        context_id = cls.compute_context_id(
            document_ref=document_ref,
            git_commit=git_commit,
            configuration_hash=configuration_hash,
            ontology_version=ontology_version,
            reconciliation_policy_version=reconciliation_policy_version,
            capability_matrix_version=capability_matrix_version,
            confidence_policy_version=confidence_policy_version,
            feedback_policy_version=feedback_policy_version,
            alignment_algorithm_version=alignment_algorithm_version,
            machine_identifier=machine_identifier,
            workspace_identifier=workspace_identifier,
        )
        return cls(
            event_id=new_id("event"),
            document_ref=document_ref,
            context_id=context_id,
            ontology_version=ontology_version,
            git_commit=git_commit,
            configuration_hash=configuration_hash,
            configuration_snapshot_reference=configuration_snapshot_reference,
            machine_identifier=machine_identifier,
            workspace_identifier=workspace_identifier,
            reconciliation_policy_version=reconciliation_policy_version,
            capability_matrix_version=capability_matrix_version,
            confidence_policy_version=confidence_policy_version,
            feedback_policy_version=feedback_policy_version,
            alignment_algorithm_version=alignment_algorithm_version,
        )


class CandidateExcluded(TelemetryEvent):
    """One structurally-excluded candidate pair (Constitution Article 27), emitted by
    `domain.alignment.telemetry.alignment_events` from `AlignmentResult.excluded_pairs`. Symmetric
    with `AlignmentAttempted`'s inclusion record: this is what makes "this candidate pair was
    considered together but never merged, and here is exactly why" a queryable fact instead of a
    silent absence of any resulting cluster containing both.
    """

    kind: TelemetryEventKind = TelemetryEventKind.CANDIDATE_EXCLUDED

    candidate_observation_id: str
    compared_against_observation_id: str
    excluding_mechanism: str
    basis_code: ClusteringBasisCode
    """Constitution Article 26 -- a stable, enumerable reason code, never free text. Only clustering
    exclusion mechanisms exist today (Article 27), so this is strictly typed to
    `ClusteringBasisCode`; a future exclusion mechanism with its own vocabulary is a new field or a
    new event kind, decided when it actually exists (S1.1), never speculatively widened now."""
    structural: bool


class CandidateExclusionRecord(BaseModel):
    """One structurally-excluded candidate pair inside `CandidateExcludedBatch`."""

    model_config = ConfigDict(frozen=True)

    candidate_observation_id: str
    compared_against_observation_id: str
    excluding_mechanism: str
    basis_code: ClusteringBasisCode
    structural: bool


class CandidateExcludedBatch(TelemetryEvent):
    """Batched encoding of structurally-excluded candidate pairs.

    Semantically equivalent to a sequence of `CandidateExcluded` events, but compact enough for
    high-volume ambiguous pools. `Journal.replay` expands it into the established query shape.
    """

    kind: TelemetryEventKind = TelemetryEventKind.CANDIDATE_EXCLUDED_BATCH

    exclusions: tuple[CandidateExclusionRecord, ...] = Field(max_length=1000)


EVENT_TYPE_BY_KIND: dict[TelemetryEventKind, type[TelemetryEvent]] = {
    event_cls.model_fields["kind"].default: event_cls
    for event_cls in (
        ProviderObservationAttempted,
        EvidenceRejected,
        EvidenceCreated,
        ObservationCreated,
        ObservationMapped,
        AlignmentAttempted,
        ObservationAligned,
        ObservationLeftUnaligned,
        ObservationCompared,
        ObservationAccepted,
        ObservationRejected,
        ObservationMerged,
        AgreementCalculated,
        CanonicalDecisionCreated,
        ConfidenceChanged,
        KnowledgeMerged,
        KnowledgeDiscarded,
        CanonicalDocumentCreated,
        HumanCorrectionSubmitted,
        HumanCorrectionApplied,
        DatasetCandidateCreated,
        ProvenanceContextEstablished,
        CandidateExcluded,
        CandidateExcludedBatch,
        ReviewOutcomeRecorded,
        ReviewPacketCreated,
        ReviewPacketDispatched,
        ReviewPacketOpened,
        ReviewPacketClosed,
        SegmentationRunCompleted,
        MethodRunCompleted,
        ReviewSubmissionRecorded,
        AdjudicationRecorded,
        CanonicalResultCreated,
    )
}


def latest_provenance_context(
    events: Iterable[TelemetryEvent],
) -> ProvenanceContextEstablished | None:
    """S8.6's positional lineage rule: the Operational Context of an event is the last
    `ProvenanceContextEstablished` event preceding it in append order for the same `document_ref`
    -- never a per-event `context_ref` field (Article 7). Callers pass the prefix of
    `TelemetrySink.events_for_document(document_ref)` up to and including the event of interest;
    this returns whichever `ProvenanceContextEstablished` appeared last in that prefix, or `None`
    if none precedes it -- historical data recorded before this event kind existed, never inferred
    (`ARCHITECTURE_TELEMETRY_STANDARD.md` S8.6).
    """
    context: ProvenanceContextEstablished | None = None
    for event in events:
        if isinstance(event, ProvenanceContextEstablished):
            context = event
    return context


def stamp_recorded_at(event: TelemetryEvent) -> TelemetryEvent:
    """Sets `recorded_at` to the current UTC wall-clock time if not already set (Operational
    Hardening milestone, Priority 13) -- called once, at the single point each event reaches a
    sink (`WorkspaceProcessingService.process`, `ReviewService.submit_decision`), rather than at
    every one of the many domain-layer telemetry-construction call sites. `TelemetryEvent` is
    frozen, so this returns a new instance; idempotent (a pre-stamped event is returned unchanged).
    """
    if event.recorded_at is not None:
        return event
    return event.model_copy(update={"recorded_at": datetime.now(timezone.utc).isoformat()})


def parse_event(data: dict[str, Any]) -> TelemetryEvent:
    """Reconstructs the correct event subclass from a plain dict (e.g. a deserialized JSON line),
    keyed by `kind` -- the same polymorphism problem, and the same solution, as
    Observation/CanonicalObservation's payload coercion (domain/ontology/base.py).
    """
    kind = TelemetryEventKind(data["kind"])
    event_cls = EVENT_TYPE_BY_KIND[kind]
    return event_cls.model_validate(data)
