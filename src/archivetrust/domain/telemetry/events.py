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
from archivetrust.domain.canonical.result import CanonicalResult
from archivetrust.domain.comparison.clustering import ClusteringBasisCode
from archivetrust.domain.confidence.models import ComparisonConfidence
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.shared.ids import content_address, new_id
from archivetrust.domain.shared.versioning import CURRENT_SCHEMA_VERSION
from archivetrust.htr.corpus.models import (
    Collection,
    Dataset,
    DatasetVersion,
    InputCrop,
    Page,
    Region,
    ResearchProject,
    TextLine,
)
from archivetrust.htr.experiment.models import (
    Experiment,
    ExperimentRun,
    ExperimentVersion,
    FailureRecord,
    MethodRun,
    MetricDefinition,
    MetricResult,
    ReproducibilityManifest,
)

# `htr.corpus.models`/`htr.experiment.models` are pure frozen Pydantic domain types whose only
# imports are `domain.evidence.models` and `domain.shared.ids` (verified: their packages'
# `__init__` files pull nothing else, and `htr/__init__.py` is docstring-only), so importing them
# here introduces no cycle and violates neither guard in
# `tests/domain/test_dependency_direction.py`. Events embed the *full* domain object they announce
# -- this module's own docstring requires it, "because replay must reconstruct [state] from
# telemetry alone".
#
# Three HTR-adjacent record types are deliberately NOT imported and travel as a validated
# `record` dict instead, reconstructed by the application-layer replay in
# `application/htr_journal.py`:
#   * `providers.transkribus.external_import.ExternalImport` -- `tests/domain/
#     test_dependency_direction.py::test_domain_layer_never_imports_providers` forbids it outright.
#   * `review.htr_models.*` and `evaluation.ground_truth.TranscriptionConvention` -- importing
#     these would invert the dependency direction `ReviewOutcomeRecorded`'s docstring states above
#     ("`review` depends on `domain.telemetry`, never the reverse") and create a package-level
#     cycle, since `review/service.py` imports this module.
# See `docs/architecture/htr-telemetry.md` §5 for why this split is drawn exactly here.


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

    # -- HTR research persistence (docs/architecture/htr-event-model.md §3) --------------------
    # The five kinds above were added by the prior transformation with no producer anywhere in
    # `src/` (docs/htr-telemetry-knowledge-gap-analysis.md §1). They now have real producers in
    # `htr/persistence/durable_store.py`, alongside the kinds below. See
    # `docs/architecture/htr-telemetry.md` for the built implementation.

    RESEARCH_PROJECT_CREATED = "ResearchProjectCreated"
    DATASET_CREATED = "DatasetCreated"
    DATASET_VERSION_CREATED = "DatasetVersionCreated"
    COLLECTION_CREATED = "CollectionCreated"
    """Beyond the event-model doc §3's enumerated list, and deliberately so: §3 lists
    `DocumentRegistered` but omits the `Collection` that owns the documents, while
    `htr/research_store.py` has carried a `Collection` bucket since Stage 11 and
    `docs/htr-domain-design.md` §1 places `Collection` between `DatasetVersion` and `Document`.
    Replay could not reconstruct the corpus tree without it. See
    `docs/architecture/htr-telemetry.md` §6 for the four documented additions to §3's list."""
    DOCUMENT_REGISTERED = "DocumentRegistered"
    PAGE_REGISTERED = "PageRegistered"
    REGION_DETECTED = "RegionDetected"
    TEXT_LINE_DETECTED = "TextLineDetected"
    INPUT_CROP_CREATED = "InputCropCreated"
    TRANSCRIPTION_CONVENTION_REGISTERED = "TranscriptionConventionRegistered"
    EXPERIMENT_CREATED = "ExperimentCreated"
    EXPERIMENT_VERSION_CREATED = "ExperimentVersionCreated"
    EXPERIMENT_RUN_STARTED = "ExperimentRunStarted"
    EXPERIMENT_RUN_COMPLETED = "ExperimentRunCompleted"
    EXPERIMENT_RUN_FAILED = "ExperimentRunFailed"
    METHOD_RUN_STARTED = "MethodRunStarted"
    METHOD_RUN_FAILED = "MethodRunFailed"
    RAW_METHOD_RESULT_RECORDED = "RawMethodResultRecorded"
    PARSED_METHOD_RESULT_RECORDED = "ParsedMethodResultRecorded"
    NORMALIZED_METHOD_RESULT_RECORDED = "NormalizedMethodResultRecorded"
    REVIEWED_RESULT_RECORDED = "ReviewedResultRecorded"
    """Beyond §3's enumerated list. §3 names the three *machine* transcript stages
    (raw/parsed/normalized) but `htr/research_store.py::MethodRunTranscript` -- which the Stage 11
    comparison ViewModel already reads -- carries a fourth, `reviewed_text`, whose own docstring
    states it "belongs to a different lineage than the first three: it comes from a human
    ReviewSubmission/Adjudication, not from the method". Folding it into
    `ReviewSubmissionRecorded` would conflate the blind-review *target* lineage with the
    *method-run* lineage that docstring explicitly separates, so it is its own kind."""
    METRIC_DEFINITION_REGISTERED = "MetricDefinitionRegistered"
    """Beyond §3's enumerated list: §3 lists `MetricCalculated` (the result) but not the versioned
    `MetricDefinition` it was computed against, which `htr/research_store.py` stores and which
    `docs/htr-domain-design.md` §3 requires be independently versioned for replay to know which
    calibration produced a value."""
    METRIC_CALCULATED = "MetricCalculated"
    RELIABILITY_ISSUE_CLASSIFIED = "ReliabilityIssueClassified"
    REVIEW_ASSIGNED = "ReviewAssigned"
    AGREEMENT_CALCULATED_HTR = "AgreementCalculatedHtr"
    """Reviewer-pair *textual* agreement, deliberately distinct from the pre-existing
    `AGREEMENT_CALCULATED` (cross-provider *structural* agreement) -- event-model doc §3's
    "reusing the name for a different meaning would violate the do-not-rename rule"."""
    GROUND_TRUTH_TEXT_RECORDED = "GroundTruthTextRecorded"
    """Beyond §3's enumerated list: `htr/research_store.py` carries a resolved
    `text_line_id -> reference transcription` mapping that drives every metric display, and
    without an event for it a replayed store shows metrics with no reference text to explain
    them. Records only the resolved string, never a competing `GroundTruthItem` entity --
    `evaluation/ground_truth.py` still owns that workflow."""
    REPRODUCIBILITY_MANIFEST_RECORDED = "ReproducibilityManifestRecorded"
    EXTERNAL_RESULT_IMPORTED = "ExternalResultImported"
    RESEARCH_OBSERVATION_CREATED = "ResearchObservationCreated"
    CANDIDATE_FINDING_CREATED = "CandidateFindingCreated"
    FINDING_REVIEWED = "FindingReviewed"
    FINDING_STATUS_CHANGED = "FindingStatusChanged"
    RESEARCH_REPORT_GENERATED = "ResearchReportGenerated"


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

    `correlation_id`/`causation_id` (docs/architecture/htr-event-model.md §4) are the explicit
    provenance-DAG mechanism: `correlation_id` is one value shared by every event belonging to the
    same logical unit of work, `causation_id` is the `event_id` of the event that *directly caused*
    this one. They live on this shared base rather than on HTR subclasses alone because §4 specifies
    they apply to all events, old and new. Both default to `None`, so this is purely additive: every
    pre-existing construction call site keeps working unmodified, and `None` honestly means "this
    event was recorded before, or outside, any correlated unit of work" -- never a guessed or
    backfilled linkage. Only the HTR producers in `htr/persistence/durable_store.py` populate them
    today; the pre-existing OCR-era producers deliberately still leave both `None` rather than
    having a correlation invented for them retroactively.
    """

    model_config = ConfigDict(frozen=True)

    kind: TelemetryEventKind

    event_id: str
    document_ref: str
    correlation_id: str | None = None
    causation_id: str | None = None
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


HTR_RESEARCH_SCOPE = "htr:research"
"""The `document_ref` an HTR research event carries when it genuinely concerns no single Archive
Object (a project, a dataset, an experiment, a metric).

`document_ref` is non-optional on `TelemetryEvent` and is documented there as scoping an event "to
the Archive Object it concerns". Most HTR research entities have no such scope: an `Experiment`
spans a whole `DatasetVersion`. Rather than either loosen the base field or invent a plausible-
looking document reference, HTR events carry this explicit sentinel and do their real scoping
through `HtrTelemetryEvent`'s purpose-built `project_id`/`experiment_id`/`experiment_run_id`/
`subject_id` fields (event-model doc §3's mandated field list). Where a genuine Archive Object
*does* exist -- `PageRegistered` -- the real `archive_object_ref` is used instead, so page-scoped
HTR history stays per-document replayable.
"""


class HtrActorType(str, Enum):
    """Who or what produced an HTR research event (event-model doc §3's `actor_type`)."""

    SYSTEM = "system"
    """Deterministic orchestration -- registration, projection, evaluation bookkeeping."""
    HUMAN = "human"
    """An attributable person: a reviewer, an adjudicator, a curator."""
    METHOD = "method"
    """An HTR recognition method (SATRN/Florence-2/Transkribus) or a segmentation adapter."""


class HtrTelemetryEvent(TelemetryEvent):
    """Shared base for the HTR research event kinds (event-model doc §1's layer 3,
    "`HtrTelemetryEvent` subtypes ... appended to `FileTelemetrySink`").

    Carries §3's mandated scope/actor field set once, here, rather than repeating a dozen optional
    fields across ~35 subclasses. Every field is optional with a default: an HTR event populates
    exactly the scopes that genuinely apply to it and leaves the rest `None` -- the same
    "`None` means could-not-be-established, never guessed" discipline
    `ProviderObservationAttempted` already applies to its runtime facts (Constitution Article 18).

    `correlation_id`/`causation_id` are *not* declared here: they live on `TelemetryEvent` itself,
    since §4 specifies they apply to all events, old and new.
    """

    project_id: str | None = None
    dataset_id: str | None = None
    dataset_version_id: str | None = None
    experiment_id: str | None = None
    experiment_version_id: str | None = None
    experiment_run_id: str | None = None
    method_run_id: str | None = None
    subject_id: str | None = None
    """The document/page/region/line/crop id this event is *about*, when applicable -- distinct
    from the scope ids above, which say which unit of work it belongs to."""
    actor_type: HtrActorType = HtrActorType.SYSTEM
    actor_id: str | None = None
    source_component: str | None = None
    application_commit: str | None = None


class SegmentationRunCompleted(HtrTelemetryEvent):
    """One `SegmentationAdapter` run over a Page completed (docs/htr-domain-design.md §7).
    References the produced Regions/TextLines/InputCrops by id, never embedded -- the same
    reference-by-id discipline `Observation.evidence_ids` already enforces (Constitution
    Article 7). The referenced entities are announced individually, with their full objects, by
    `RegionDetected`/`TextLineDetected`/`InputCropCreated`, so replay reconstructs them from those
    rather than from this summary."""

    kind: TelemetryEventKind = TelemetryEventKind.SEGMENTATION_RUN_COMPLETED

    segmentation_run_id: str
    page_id: str
    segmentation_adapter_name: str
    region_ids: tuple[str, ...]
    text_line_ids: tuple[str, ...]
    input_crop_ids: tuple[str, ...]


class MethodRunCompleted(HtrTelemetryEvent):
    """One `htr.experiment.models.MethodRun` reached a terminal outcome (docs/htr-domain-design.md
    §1, §4). Mirrors `ProviderObservationAttempted`'s outcome discipline (Constitution
    Article 18) at HTR-method-run granularity.

    Deliberately lean: the full `MethodRun` object is carried by `MethodRunStarted`, and this event
    is the terminal marker whose `causation_id` points back at it -- exactly the pair event-model
    doc §4 uses as its worked example ("a `MethodRunCompleted` event's `causation_id` is the
    `MethodRunStarted` event's `event_id`"). Replay reconstructs the `MethodRun` from
    `MethodRunStarted`; this event is what makes "it reached a terminal outcome" a recorded fact
    rather than an inference from the absence of a failure.

    `method_run_id` narrows `HtrTelemetryEvent`'s optional scope field to required here: a terminal
    outcome with no run to attach it to is meaningless, not merely unscoped."""

    kind: TelemetryEventKind = TelemetryEventKind.METHOD_RUN_COMPLETED

    method_run_id: str
    experiment_run_id: str
    method_id: str
    evidence_id: str
    outcome: str
    failure_reason: str | None = None


class ReviewSubmissionRecorded(HtrTelemetryEvent):
    """One blind `review.htr_models.ReviewSubmission` was recorded (docs/htr-domain-design.md
    §1).

    `record` carries the full `ReviewSubmission.model_dump(mode="json")` so replay can rebuild the
    submission field-for-field. It is a dict rather than the typed model because importing
    `review.htr_models` here would invert this module's stated dependency direction -- see the
    import-block comment at the top of this file. Optional and defaulted so the pre-existing
    id-only construction shape (`tests/domain/telemetry/test_htr_events.py`) still validates;
    `None` means "recorded before the full record was carried", never "no submission"."""

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_SUBMISSION_RECORDED

    submission_id: str
    assignment_id: str
    reviewer_ref: str
    record: dict[str, Any] | None = None


class AdjudicationRecorded(HtrTelemetryEvent):
    """One `review.htr_models.Adjudication` resolved a disagreeing `AgreementResult`
    (docs/htr-domain-design.md §1). `record` follows `ReviewSubmissionRecorded.record`'s reasoning
    exactly."""

    kind: TelemetryEventKind = TelemetryEventKind.ADJUDICATION_RECORDED

    adjudication_id: str
    agreement_result_id: str
    adjudicator_ref: str
    record: dict[str, Any] | None = None


class CanonicalResultCreated(HtrTelemetryEvent):
    """One `domain.canonical.result.CanonicalResult` was created (docs/htr-domain-design.md §2:
    "renaming CanonicalDocumentCreated semantics where line-level").

    Originally id-only, on the stated reasoning that "a `CanonicalResult` can be read back from its
    own store by id". That reasoning no longer holds: as of this pass the telemetry stream *is* the
    store (docs/architecture/htr-telemetry.md), so an id-only event would make `CanonicalResult`
    the one HTR entity replay could not reconstruct. `canonical_result` therefore carries the full
    object, matching `CanonicalDocumentCreated`/`CanonicalDecisionCreated`'s embedding. Optional and
    defaulted, so the pre-existing id-only construction shape still validates -- `None` means
    "recorded before the object was carried", never "no result"."""

    kind: TelemetryEventKind = TelemetryEventKind.CANONICAL_RESULT_CREATED

    canonical_result_id: str
    page_id: str
    strategy: str
    strategy_version: int
    canonical_result: CanonicalResult | None = None


# -- HTR research entity events (docs/architecture/htr-event-model.md §3) -----------------------
#
# Each embeds the full frozen domain object it announces, per this module's docstring: replay must
# reconstruct state "from telemetry alone, without re-running any provider". The producers are in
# `htr/persistence/durable_store.py`; the replay branches are in `application/htr_journal.py`.


class ResearchProjectCreated(HtrTelemetryEvent):
    """A `ResearchProject` was registered -- the root of the corpus tree."""

    kind: TelemetryEventKind = TelemetryEventKind.RESEARCH_PROJECT_CREATED

    project: ResearchProject


class DatasetCreated(HtrTelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.DATASET_CREATED

    dataset: Dataset


class DatasetVersionCreated(HtrTelemetryEvent):
    """An immutable `DatasetVersion` snapshot (docs/htr-domain-design.md §3). A membership change is
    always a new version with `supersedes` set, never an edit -- so the event stream carries the
    whole version chain and replay resolves supersession the same way `FileGroundTruthStore.latest()`
    does."""

    kind: TelemetryEventKind = TelemetryEventKind.DATASET_VERSION_CREATED

    dataset_version: DatasetVersion


class CollectionCreated(HtrTelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.COLLECTION_CREATED

    collection: Collection


class DocumentRegistered(HtrTelemetryEvent):
    """One Document -- i.e. one existing `acquisition.archive_object.ArchiveObject`, referenced by
    id -- became a member of a `Collection`.

    Carries no Document *object*: `htr/corpus/models.py`'s module docstring is explicit that no
    `Document` model exists ("ArchiveObject *is* the document concept at this layer"), so this event
    records the membership fact and nothing more. Emitted once per `archive_object_ref` on the
    `Collection` that names it, caused by that `CollectionCreated` event."""

    kind: TelemetryEventKind = TelemetryEventKind.DOCUMENT_REGISTERED

    archive_object_ref: str
    collection_id: str


class PageRegistered(HtrTelemetryEvent):
    """A `Page` of a Document was registered. The one HTR event kind whose `document_ref` is a real
    `ArchiveObject` id rather than `HTR_RESEARCH_SCOPE` -- a page genuinely belongs to exactly one
    Archive Object, so per-document replay stays meaningful here."""

    kind: TelemetryEventKind = TelemetryEventKind.PAGE_REGISTERED

    page: Page


class RegionDetected(HtrTelemetryEvent):
    """A segmentation stage detected a `Region` on a `Page` (docs/htr-domain-design.md §7).
    Independent of any recognition method."""

    kind: TelemetryEventKind = TelemetryEventKind.REGION_DETECTED

    region: Region


class TextLineDetected(HtrTelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.TEXT_LINE_DETECTED

    text_line: TextLine


class InputCropCreated(HtrTelemetryEvent):
    """A content-addressed `InputCrop` was cropped for one `TextLine`.

    `InputCrop.hash` is computed by `InputCrop.compute_hash`/`InputCrop.create` and validated by the
    model's own `_validate_hash` -- this event embeds the already-hashed object and introduces no
    second hashing scheme (event-model doc §2: "`content_address(...)` ... no second hashing scheme
    introduced"). Image *bytes* are never carried here; `InputCrop.storage_path` points at them,
    exactly as the model already specifies."""

    kind: TelemetryEventKind = TelemetryEventKind.INPUT_CROP_CREATED

    input_crop: InputCrop


class TranscriptionConventionRegistered(HtrTelemetryEvent):
    """A versioned `evaluation.ground_truth.TranscriptionConvention` was registered. `record`
    carries its full dump; see `ReviewSubmissionRecorded.record` for why it is a dict."""

    kind: TelemetryEventKind = TelemetryEventKind.TRANSCRIPTION_CONVENTION_REGISTERED

    convention_id: str
    convention_version: int
    record: dict[str, Any] | None = None


class ExperimentCreated(HtrTelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.EXPERIMENT_CREATED

    experiment: Experiment


class ExperimentVersionCreated(HtrTelemetryEvent):
    """One versioned configuration of an `Experiment` (docs/htr-domain-design.md §3). Frozen once an
    `ExperimentRun` references it; a later edit is a new version with `supersedes` set."""

    kind: TelemetryEventKind = TelemetryEventKind.EXPERIMENT_VERSION_CREATED

    experiment_version: ExperimentVersion


class ExperimentRunStarted(HtrTelemetryEvent):
    """An `ExperimentRun` began. This event's `event_id` is the causal root of the whole run, and
    the run's own `experiment_run_id` is the `correlation_id` every later event in the run shares
    (event-model doc §4)."""

    kind: TelemetryEventKind = TelemetryEventKind.EXPERIMENT_RUN_STARTED

    experiment_run: ExperimentRun


class ExperimentRunCompleted(HtrTelemetryEvent):
    """An `ExperimentRun` reached a terminal, successful outcome. Carries the terminal `ExperimentRun`
    record (with `completed_at` populated) so replay reconstructs the completed run rather than the
    started one."""

    kind: TelemetryEventKind = TelemetryEventKind.EXPERIMENT_RUN_COMPLETED

    experiment_run: ExperimentRun


class ExperimentRunFailed(HtrTelemetryEvent):
    """An `ExperimentRun` could not complete. Recorded, never inferred from a missing
    `ExperimentRunCompleted` (Constitution Article 18's silence-vs-failure discipline)."""

    kind: TelemetryEventKind = TelemetryEventKind.EXPERIMENT_RUN_FAILED

    experiment_run_id: str
    reason: str


class MethodRunStarted(HtrTelemetryEvent):
    """One method x input execution began, carrying the full `MethodRun` record.

    A `MethodRun` is a frozen record constructed once, already carrying its `outcome` and
    `completed_at` (see `htr/experiment/models.py`), so this event is what durably *establishes* the
    run for replay, and `MethodRunCompleted`/`MethodRunFailed` are the terminal markers that chain
    from it by `causation_id`. Two events per run is deliberate, not redundant: event-model doc §4
    names exactly this pair as its causation example."""

    kind: TelemetryEventKind = TelemetryEventKind.METHOD_RUN_STARTED

    method_run: MethodRun


class MethodRunFailed(HtrTelemetryEvent):
    """A `MethodRun` could not complete, carrying the full `FailureRecord`
    (docs/htr-domain-design.md §1: "FailureRecord (if applicable -- preserved, never excluded)")."""

    kind: TelemetryEventKind = TelemetryEventKind.METHOD_RUN_FAILED

    failure_record: FailureRecord


class RawMethodResultRecorded(HtrTelemetryEvent):
    """The raw, unparsed text one `MethodRun` produced -- stage 1 of the four
    `MethodRunTranscript` stages, which never collapse into one another.

    `evidence_id` references the `Evidence` record the raw payload was captured into; the payload is
    not duplicated here (event-model doc §2: "the event does not duplicate the raw payload")."""

    kind: TelemetryEventKind = TelemetryEventKind.RAW_METHOD_RESULT_RECORDED

    method_run_id: str
    text: str | None = None
    evidence_id: str | None = None


class ParsedMethodResultRecorded(HtrTelemetryEvent):
    """Stage 2 of `MethodRunTranscript`. Caused by the `RawMethodResultRecorded` it parsed."""

    kind: TelemetryEventKind = TelemetryEventKind.PARSED_METHOD_RESULT_RECORDED

    method_run_id: str
    text: str | None = None


class NormalizedMethodResultRecorded(HtrTelemetryEvent):
    """Stage 3 of `MethodRunTranscript`. Caused by the `ParsedMethodResultRecorded` it normalized."""

    kind: TelemetryEventKind = TelemetryEventKind.NORMALIZED_METHOD_RESULT_RECORDED

    method_run_id: str
    text: str | None = None


class ReviewedResultRecorded(HtrTelemetryEvent):
    """Stage 4 of `MethodRunTranscript` -- the human-reviewed text for one `MethodRun`'s line.

    A different lineage from stages 1-3: it comes from a human `ReviewSubmission`/`Adjudication`,
    not from the method. Emitted only when a human has actually reviewed the run's line, so
    "nobody has reviewed this" (no event) and "a reviewer agreed with the machine" (an event whose
    `text` equals the normalized text) stay distinguishable -- the exact distinction
    `MethodRunTranscript.reviewed_text`'s docstring requires be preserved."""

    kind: TelemetryEventKind = TelemetryEventKind.REVIEWED_RESULT_RECORDED

    method_run_id: str
    text: str | None = None
    reviewer_ref: str | None = None
    actor_type: HtrActorType = HtrActorType.HUMAN


class MetricDefinitionRegistered(HtrTelemetryEvent):
    kind: TelemetryEventKind = TelemetryEventKind.METRIC_DEFINITION_REGISTERED

    metric_definition: MetricDefinition


class MetricCalculated(HtrTelemetryEvent):
    """One `MetricDefinition` was computed for one `MethodRun`.

    The computation itself stays a pure function in `htr/evaluation/*` (gap analysis §7: "keep them
    pure; persistence is the caller's job"). This event is the caller's persistence of its return
    value, and its `causation_id` points at the transcript-stage event whose text was scored."""

    kind: TelemetryEventKind = TelemetryEventKind.METRIC_CALCULATED

    metric_result: MetricResult


class ReliabilityIssueClassified(HtrTelemetryEvent):
    """A reliability/failure classification was recorded for one `MethodRun`
    (`htr/evaluation/failures.py::classify_reliability`'s output, persisted by its caller)."""

    kind: TelemetryEventKind = TelemetryEventKind.RELIABILITY_ISSUE_CLASSIFIED

    method_run_id: str
    classification: str
    detail: str | None = None


class GroundTruthTextRecorded(HtrTelemetryEvent):
    """The resolved reference transcription for one `TextLine`, from whatever authority produced it
    (a closed blind dual review, an imported gold standard).

    Records only the resolved string -- `evaluation/ground_truth.py::FileGroundTruthStore` still owns
    the `GroundTruthAnnotation` workflow and its own durable JSONL stream. This event exists so a
    replayed research store can show *why* a metric has the value it does, and deliberately does not
    duplicate that store's records."""

    kind: TelemetryEventKind = TelemetryEventKind.GROUND_TRUTH_TEXT_RECORDED

    text_line_id: str
    text: str


class ReviewAssigned(HtrTelemetryEvent):
    """One reviewer was assigned to independently transcribe one target (blind double annotation).
    `record` carries the full `ReviewAssignment` dump; see `ReviewSubmissionRecorded.record`."""

    kind: TelemetryEventKind = TelemetryEventKind.REVIEW_ASSIGNED

    assignment_id: str
    target_ref: str
    record: dict[str, Any] | None = None
    actor_type: HtrActorType = HtrActorType.HUMAN


class AgreementCalculatedHtr(HtrTelemetryEvent):
    """Reviewer-pair *textual* agreement for one target, computed from two blind submissions.

    Distinct kind from the pre-existing `AgreementCalculated` (cross-provider *structural*
    agreement over a semantic slot) because the two measure different things -- reusing one name for
    both would violate the "do not rename when semantics differ" rule (event-model doc §3)."""

    kind: TelemetryEventKind = TelemetryEventKind.AGREEMENT_CALCULATED_HTR

    agreement_result_id: str
    target_ref: str
    agrees: bool
    similarity_score: float | None = None
    record: dict[str, Any] | None = None


class ReproducibilityManifestRecorded(HtrTelemetryEvent):
    """The one `ReproducibilityManifest` for an `ExperimentRun` -- everything needed to reproduce
    it. Mirrors `ProvenanceContextEstablished`'s "fixed environment" role at experiment-run rather
    than per-document granularity."""

    kind: TelemetryEventKind = TelemetryEventKind.REPRODUCIBILITY_MANIFEST_RECORDED

    manifest: ReproducibilityManifest


class ExternalResultImported(HtrTelemetryEvent):
    """A manually-imported external result (e.g. a Transkribus export) was wrapped as a
    `MethodRun`'s provenance record.

    `record` carries the full `providers.transkribus.external_import.ExternalImport` dump rather
    than the typed object: `tests/domain/test_dependency_direction.py` forbids this domain module
    from importing `providers/*` at all. `application/htr_journal.py` reconstructs the typed model
    on replay."""

    kind: TelemetryEventKind = TelemetryEventKind.EXTERNAL_RESULT_IMPORTED

    external_import_id: str
    record: dict[str, Any] | None = None


# -- Research-knowledge lifecycle: schema only, no producer yet ---------------------------------
#
# These five kinds complete event-model doc §3's required list, whose layers 10-12 are the
# `ResearchObservation`/`ResearchFinding` knowledge lifecycle. That lifecycle -- the entities, the
# extraction step, the promotion gates -- is explicitly a *later* phase and is deliberately not
# built here. What is landed now is the closed event vocabulary, so the enum does not have to be
# reopened later.
#
# **Disclosed, not silently implicit**: no code in `src/` constructs any of the five today, exactly
# as `ObservationMapped` above has disclosed for its own kind since 2026-07-14, and as the five
# HTR kinds at the top of this section did before this pass gave them producers. Each references
# its subject by id and carries no entity object, precisely so that the later phase which defines
# `ResearchObservation`/`ResearchFinding` is free to model them without having to migrate a
# speculative schema guessed at here.


class ResearchObservationCreated(HtrTelemetryEvent):
    """Layer 10. A `ResearchObservation` was extracted from durable records by an explicit
    extraction step -- never auto-derived from a telemetry event (event-model doc §1's hard rule)."""

    kind: TelemetryEventKind = TelemetryEventKind.RESEARCH_OBSERVATION_CREATED

    observation_ref: str
    summary: str
    evidence_refs: tuple[str, ...] = ()


class CandidateFindingCreated(HtrTelemetryEvent):
    """Layer 11. A `ResearchFinding` was constructed at `Candidate` status by an explicit step --
    a `ResearchObservation` never auto-promotes."""

    kind: TelemetryEventKind = TelemetryEventKind.CANDIDATE_FINDING_CREATED

    finding_ref: str
    statement: str
    observation_refs: tuple[str, ...] = ()


class FindingReviewed(HtrTelemetryEvent):
    """Layer 12. A human actor reviewed a candidate finding. No finding advances past `Candidate`
    without one of these recording an attributable reviewer."""

    kind: TelemetryEventKind = TelemetryEventKind.FINDING_REVIEWED

    finding_ref: str
    reviewer_ref: str
    verdict: str
    actor_type: HtrActorType = HtrActorType.HUMAN


class FindingStatusChanged(HtrTelemetryEvent):
    """Layer 12. A status transition on a `ResearchFinding` -- the only mutation a finding permits."""

    kind: TelemetryEventKind = TelemetryEventKind.FINDING_STATUS_CHANGED

    finding_ref: str
    new_status: str
    previous_status: str | None = None
    reason: str | None = None


class ResearchReportGenerated(HtrTelemetryEvent):
    """A `research.reports.models.ResearchReport` was generated over one or more `ExperimentRun`s.

    Still no producer today, but the *reason* changed on 2026-07-30 and is recorded here rather than
    left stale. The previous reason -- "`build_research_report` reads in-process dataclasses rather
    than durable records (gap analysis §8)" -- no longer holds: it now sources every field from
    durable records via
    `htr/experiment/baseline_execution.py::build_research_report_from_store`. What remains deferred is
    *announcing* a generated report as an event, which belongs to the research-knowledge lifecycle
    this kind is grouped with (`docs/architecture/htr-telemetry.md` §6's five schema-only kinds), not
    to report generation itself. A report is a derived projection over the runs it covers, and
    Article 33's discipline is that projections do not themselves emit telemetry -- so this event is
    for the later phase that records a report as a *published research artifact*, not for every
    regeneration of one."""

    kind: TelemetryEventKind = TelemetryEventKind.RESEARCH_REPORT_GENERATED

    report_ref: str
    experiment_run_refs: tuple[str, ...] = ()


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
        # HTR research persistence (docs/architecture/htr-event-model.md §3)
        ResearchProjectCreated,
        DatasetCreated,
        DatasetVersionCreated,
        CollectionCreated,
        DocumentRegistered,
        PageRegistered,
        RegionDetected,
        TextLineDetected,
        InputCropCreated,
        TranscriptionConventionRegistered,
        ExperimentCreated,
        ExperimentVersionCreated,
        ExperimentRunStarted,
        ExperimentRunCompleted,
        ExperimentRunFailed,
        MethodRunStarted,
        MethodRunFailed,
        RawMethodResultRecorded,
        ParsedMethodResultRecorded,
        NormalizedMethodResultRecorded,
        ReviewedResultRecorded,
        MetricDefinitionRegistered,
        MetricCalculated,
        ReliabilityIssueClassified,
        GroundTruthTextRecorded,
        ReviewAssigned,
        AgreementCalculatedHtr,
        ReproducibilityManifestRecorded,
        ExternalResultImported,
        ResearchObservationCreated,
        CandidateFindingCreated,
        FindingReviewed,
        FindingStatusChanged,
        ResearchReportGenerated,
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
