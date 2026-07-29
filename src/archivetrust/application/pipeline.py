"""The end-to-end ingestion pipeline (ROADMAP.md S5.8, Milestone 7): adapters -> evidence ->
observations -> comparison -> confidence -> canonical document -> telemetry.

Orchestrates the domain-layer pieces built in Milestones 1-6 into one callable. This module is
`application`, not `domain` (S5.8's module boundaries) -- it sequences pure domain functions but
owns no domain logic of its own; every actual decision (clustering, reconciliation, confidence
derivation) still happens inside the domain-layer engines this module simply calls in order.

**Sequencing note, restated from `domain/confidence/telemetry.py`:** the Confidence Engine must
run *before* `CanonicalDecisionCreated` is emitted, so that event is only ever emitted once, already
confidence-complete. This module enforces that ordering by construction (there is no code path
here that emits comparison telemetry before confidence has been applied).
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.alignment.telemetry import alignment_events
from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.engine import ComparisonEngineResult, run_comparison_engine
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.telemetry import comparison_result_to_events
from archivetrust.domain.confidence.engine import apply_confidence_engine
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.confidence.telemetry import confidence_changed_events
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDocumentCreated,
    EvidenceCreated,
    EvidenceRejected,
    ObservationCreated,
    ObservationMapped,
    ProviderFailureCategory,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
    TelemetryEvent,
)
from archivetrust.providers.base import ProviderAdapter


class AdapterInvocation(BaseModel):
    """One provider adapter run against one document. `source` is intentionally `Any`-shaped
    (adapters accept whatever provider-native input shape they need, S5.8's own provider-adapter
    boundary) -- this type only pins down enough for the pipeline to route results correctly.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    adapter: ProviderAdapter
    source: object
    invocation_id: str
    pages_processed: int | None = None
    """How many pages this one invocation covers, when known ahead of time (Observability
    milestone, 2026-07-13) — `1` for a per-page `PAGE_IMAGE` invocation; `None` for a `DOCUMENT`
    invocation (whole-document adapters like Docling), whose page count is not resolved here to
    avoid requiring every `DOCUMENT`-kind source to be a probeable PDF. Copied verbatim onto the
    resulting `ProviderObservationAttempted.pages_processed` — never estimated by the pipeline."""


class PipelineRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    reconciled_graph: ReconciledObservationGraph | None
    canonical_document: CanonicalDocument | None
    """Both `None` when no provider produced any Observation (Operational Hardening milestone) — an
    empty/unreadable document asserts no facts, so there is no Canonical Document to assemble
    (Article 12: a CanonicalDocument's only legal source is a non-empty graph). The per-invocation
    `ProviderObservationAttempted.outcome` events record *why* nothing was observed; a run over a
    genuinely empty document is a completed run, never an exception."""
    events: tuple[TelemetryEvent, ...]


def _categorize_exception(exc: Exception) -> ProviderFailureCategory:
    """Structural, never textual: a timeout is recognized by the exception's *type* (the builtin
    `TimeoutError`, or any exception type named after it — `httpx.TimeoutException` and its
    `ConnectTimeout`/`ReadTimeout`/`WriteTimeout`/`PoolTimeout` subclasses all end in "Timeout",
    without this module needing an unconditional `httpx` import for what is an optional runtime
    extra). Never pattern-matches `str(exc)` — that would be inferring a cause from a human-
    readable message, exactly what this milestone's "do not infer" requirement forbids.
    """
    if isinstance(exc, TimeoutError):
        return ProviderFailureCategory.TIMEOUT
    if any(cls.__name__.endswith("Timeout") for cls in type(exc).__mro__):
        return ProviderFailureCategory.TIMEOUT
    return ProviderFailureCategory.EXCEPTION


def run_comparison_and_assembly(
    *,
    document_ref: str,
    archive_object_ref: str,
    provider_graphs: tuple[ProviderObservationGraph, ...],
    evidence_by_id: dict[str, Evidence],
    reconciliation_policy: ReconciliationPolicy,
    capability_matrix: CapabilityMatrix,
    confidence_policy: ConfidencePolicy,
    reconciliation_sequence: int = 0,
    reassembly_trigger: str = "initial_assembly",
    logical_document_id: str | None = None,
    supersedes_document: str | None = None,
) -> PipelineRunResult:
    """Everything downstream of Evidence/Observations: Comparison -> Confidence -> Canonical
    Document -> telemetry (S5.8). Factored out of `run_pipeline` so this stage can be re-run on its
    own against Evidence/Observations obtained some other way than a live provider invocation --
    concretely, `application.regenerate.regenerate_review_queue` re-runs exactly this against
    Evidence/Observations replayed from persisted telemetry, so a Comparison Engine fix (e.g. the
    2026-07-14 clustering correction) can be verified, and a review queue rebuilt, without
    re-invoking a single provider. Behavior is unchanged for `run_pipeline`'s own callers -- this
    is the same code that used to live inline there.
    """
    comparison_result = run_comparison_engine(
        provider_graphs, evidence_by_id, capability_matrix, reconciliation_policy, reconciliation_sequence
    )
    observations_by_id: dict[str, Observation] = {
        observation.observation_id: observation
        for graph in provider_graphs
        for observation in graph.observations
    }
    completed_graph = apply_confidence_engine(comparison_result.reconciled_graph, observations_by_id, confidence_policy)
    completed_result = ComparisonEngineResult(
        reconciled_graph=completed_graph,
        bundles=comparison_result.bundles,
        alignment=comparison_result.alignment,
    )

    events: list[TelemetryEvent] = []
    # Alignment Observability (ROADMAP.md S5.12): the grouping hypothesis is recorded first, since it
    # precedes and underlies every comparison decision below it. Every Observation's Aligned/
    # Unaligned outcome is on the record before any Canonical Observation is emitted.
    events.extend(alignment_events(comparison_result.alignment, document_ref=document_ref))
    events.extend(
        comparison_result_to_events(
            completed_result, document_ref=document_ref, policy=reconciliation_policy, capability_matrix=capability_matrix
        )
    )
    events.extend(
        confidence_changed_events(completed_graph.canonical_observations, document_ref=document_ref, policy=confidence_policy)
    )

    canonical_document = CanonicalDocument.assemble(
        reconciled_graph=completed_graph,
        archive_object_ref=archive_object_ref,
        reassembly_trigger=reassembly_trigger,
        logical_document_id=logical_document_id,
        supersedes=supersedes_document,
    )
    events.append(
        CanonicalDocumentCreated(event_id=_event_id(), document_ref=document_ref, canonical_document=canonical_document)
    )

    return PipelineRunResult(reconciled_graph=completed_graph, canonical_document=canonical_document, events=tuple(events))


def run_pipeline(
    *,
    document_ref: str,
    archive_object_ref: str,
    invocations: tuple[AdapterInvocation, ...],
    reconciliation_policy: ReconciliationPolicy,
    capability_matrix: CapabilityMatrix,
    confidence_policy: ConfidencePolicy,
    reconciliation_sequence: int = 0,
    reassembly_trigger: str = "initial_assembly",
    logical_document_id: str | None = None,
    supersedes_document: str | None = None,
    warm_providers: set[str] | None = None,
) -> PipelineRunResult:
    """`warm_providers` (Observability milestone, 2026-07-13): an optional, caller-owned set of
    `provider_id`s already invoked since process start, mutated in place. When supplied, every
    invocation's `cold_start` is a real fact ("has this `provider_id` been seen in this set
    before?"); when omitted (e.g. an isolated call with no memory of prior invocations, as most
    tests make), `cold_start` stays honestly `None` rather than defaulting to a guess.
    """
    events: list[TelemetryEvent] = []
    evidence_by_id: dict[str, Evidence] = {}
    provider_graphs: list[ProviderObservationGraph] = []

    for invocation in invocations:
        cold_start = (
            invocation.adapter.provider_id not in warm_providers if warm_providers is not None else None
        )
        started_at = time.monotonic()
        try:
            result = invocation.adapter.observe(
                document_ref=document_ref, invocation_id=invocation.invocation_id, source=invocation.source
            )
        except Exception as exc:  # noqa: BLE001 -- recorded as a FAILED attempt, never a crashed run
            # `ProviderAdapter.observe`'s own contract says it must never raise for a malformed
            # response; an adapter that lets an exception escape anyway (today: the runtime-backed
            # VLM adapters have no internal try/except) is treated exactly like one that caught its
            # own failure and returned it via `failure_reason` — recorded and the run continues to
            # the next invocation, the same as every other provider failure already does. This is
            # the one deliberate behavior change this milestone makes (previously: an uncaught
            # exception here aborted the whole document, losing even earlier-successful
            # invocations' telemetry); it brings runtime-backed adapters in line with how
            # Docling/Tesseract already behave, rather than special-casing which adapter family
            # gets to fail gracefully.
            duration_ms = int((time.monotonic() - started_at) * 1000)
            if warm_providers is not None:
                warm_providers.add(invocation.adapter.provider_id)
            provider_version = getattr(invocation.adapter, "_provider_version", "unknown")
            category = _categorize_exception(exc)
            events.append(
                ProviderObservationAttempted(
                    event_id=_event_id(),
                    document_ref=document_ref,
                    provider_id=invocation.adapter.provider_id,
                    provider_version=provider_version,
                    invocation_id=invocation.invocation_id,
                    outcome=ProviderInvocationOutcome.FAILED,
                    observation_count=0,
                    failure_reason=str(exc),
                    duration_ms=duration_ms,
                    retry_count=None,
                    cold_start=cold_start,
                    timeout=category == ProviderFailureCategory.TIMEOUT,
                    failure_category=category,
                    pages_processed=invocation.pages_processed,
                )
            )
            continue
        duration_ms = int((time.monotonic() - started_at) * 1000)
        if warm_providers is not None:
            warm_providers.add(invocation.adapter.provider_id)

        if result.failure_reason is not None:
            outcome = ProviderInvocationOutcome.FAILED
            failure_category: ProviderFailureCategory | None = ProviderFailureCategory.PROVIDER_ERROR
            timeout = False
        elif result.observations:
            outcome = ProviderInvocationOutcome.PRODUCED_OBSERVATIONS
            failure_category = None
            timeout = None
        else:
            outcome = ProviderInvocationOutcome.NO_OBSERVATIONS
            failure_category = None
            timeout = None
        events.append(
            ProviderObservationAttempted(
                event_id=_event_id(),
                document_ref=document_ref,
                provider_id=result.attempt.provider_id,
                provider_version=result.attempt.provider_version,
                invocation_id=result.attempt.invocation_id,
                target_region=result.attempt.target_region,
                outcome=outcome,
                observation_count=len(result.observations),
                failure_reason=result.failure_reason,
                duration_ms=duration_ms,
                retry_count=0,
                cold_start=cold_start,
                timeout=timeout,
                failure_category=failure_category,
                pages_processed=invocation.pages_processed,
            )
        )
        for evidence in result.evidence:
            evidence_by_id[evidence.evidence_id] = evidence
            events.append(
                EvidenceCreated(
                    event_id=_event_id(), document_ref=document_ref,
                    invocation_id=invocation.invocation_id, evidence=evidence,
                )
            )
        for observation in result.observations:
            events.append(
                ObservationCreated(
                    event_id=_event_id(), document_ref=document_ref,
                    invocation_id=invocation.invocation_id, observation=observation,
                )
            )
            # Constitution Article 28: which MappingTableEntry produced this Observation's
            # payload, if any -- stamped onto the contributing Evidence's supporting_metadata by
            # the decoder that constructed it (`providers/decoding/base.py::MappedPayload`).
            # `None` for Observations that aren't produced via a label-mapping lookup at all --
            # never guessed.
            mapping_entry_id: str | None = None
            mapping_table_version: int | None = None
            for evidence_id in observation.evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is not None and "mapping_table_entry_id" in evidence.supporting_metadata:
                    mapping_entry_id = evidence.supporting_metadata["mapping_table_entry_id"]
                    mapping_table_version = evidence.supporting_metadata.get("mapping_table_version")
                    break
            events.append(
                ObservationMapped(
                    event_id=_event_id(), document_ref=document_ref,
                    observation_id=observation.observation_id,
                    source_evidence_ids=observation.evidence_ids,
                    ontology_version=observation.ontology_version,
                    mapping_table_entry_id=mapping_entry_id,
                    mapping_table_version=mapping_table_version,
                )
            )
        for rejection in result.rejections:
            events.append(
                EvidenceRejected(
                    event_id=_event_id(), document_ref=document_ref,
                    provider_id=result.attempt.provider_id, provider_version=result.attempt.provider_version,
                    invocation_id=invocation.invocation_id, processing_stage=rejection.processing_stage,
                    raw_output=rejection.raw_output, rejection_reason=rejection.rejection_reason,
                )
            )
        if result.observations:
            provider_graphs.append(
                ProviderObservationGraph(
                    provider_id=result.attempt.provider_id,
                    provider_version=result.attempt.provider_version,
                    invocation_id=invocation.invocation_id,
                    observations=result.observations,
                )
            )

    if not invocations:
        raise ValueError("run_pipeline requires at least one provider invocation")

    if not provider_graphs:
        # No provider observed anything (Operational Hardening milestone). This is a completed,
        # recorded fact about the document — empty, image-only-with-no-text, or every provider
        # failed (each invocation's outcome/failure_reason above says which) — never an exception.
        # No facts were observed, so there is no comparison to run and no Canonical Document to
        # assemble; the attempt telemetry alone is this run's honest record (Article 18).
        return PipelineRunResult(
            reconciled_graph=None, canonical_document=None, events=tuple(events)
        )

    result = run_comparison_and_assembly(
        document_ref=document_ref,
        archive_object_ref=archive_object_ref,
        provider_graphs=tuple(provider_graphs),
        evidence_by_id=evidence_by_id,
        reconciliation_policy=reconciliation_policy,
        capability_matrix=capability_matrix,
        confidence_policy=confidence_policy,
        reconciliation_sequence=reconciliation_sequence,
        reassembly_trigger=reassembly_trigger,
        logical_document_id=logical_document_id,
        supersedes_document=supersedes_document,
    )
    events.extend(result.events)

    return PipelineRunResult(
        reconciled_graph=result.reconciled_graph, canonical_document=result.canonical_document, events=tuple(events)
    )


def _event_id() -> str:
    return new_id("event")
