"""Review-queue regeneration from persisted telemetry (Review Packet Integrity Audit,
2026-07-14).

Answers "can a Comparison Engine fix be verified, and a review queue rebuilt, without
reprocessing a single document?" -- yes, because Evidence and Observations are already durable,
replayable facts (`EvidenceCreated`/`ObservationCreated`, S5.10), entirely independent of whatever
Comparison/Confidence decision a *previous* engine version drew from them. This module discards
only the previously recorded Canonical layer and re-derives it via
`application.pipeline.run_comparison_and_assembly` -- the same comparison/confidence code
`run_pipeline` uses for a live run, just fed Evidence/Observations sourced from replay instead of
a fresh provider invocation. No provider is ever invoked here.

Lives in `review`, not `application` (ROADMAP_V2.md LP-1, `tests/review/test_separation.py`): it
is the boundary-spanning package permitted to depend on the Trust Engine (`application.journal`,
`application.pipeline`) while assembling `ReviewPacket`s, exactly like `review.service` already
does -- the Trust Engine itself must never import `review`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from archivetrust.application.journal import Journal
from archivetrust.application.pipeline import PipelineRunResult, run_comparison_and_assembly
from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.telemetry.events import (
    EvidenceCreated,
    EvidenceRejected,
    ObservationCreated,
    ProviderObservationAttempted,
    TelemetryEvent,
)
from archivetrust.review.assembler import assemble_packet
from archivetrust.review.packet import ReviewPacket
from archivetrust.review.triage import TriagePolicy, triage_review_queue

# Everything upstream of Comparison: what a provider actually produced, untouched by any
# Comparison Engine version. Replaying only these discards a document's entire prior Canonical
# layer (CanonicalDecisionCreated, AgreementCalculated, ConfidenceChanged, AlignmentAttempted,
# ObservationAligned/LeftUnaligned, CanonicalDocumentCreated, HumanCorrectionApplied) without
# touching the Evidence/Observations those decisions were drawn from.
_PRE_COMPARISON_EVENT_TYPES = (
    ProviderObservationAttempted,
    EvidenceRejected,
    EvidenceCreated,
    ObservationCreated,
)


class RegenerationError(ValueError):
    """Raised when a document's persisted telemetry has no Observations to regenerate from --
    distinct from a bug in this module, so a caller regenerating an entire corpus can tell "this
    document never observed anything" (a legitimate, already-recorded fact, Article 18) from an
    actual failure.
    """


def regenerate_review_queue(
    original_events: Iterable[TelemetryEvent],
    *,
    document_ref: str,
    archive_object_ref: str,
    reconciliation_policy: ReconciliationPolicy,
    capability_matrix: CapabilityMatrix,
    confidence_policy: ConfidencePolicy,
    triage_policy: TriagePolicy | None = None,
    journal: Journal | None = None,
) -> tuple[ReviewPacket, ...]:
    """Rebuilds one document's review queue by re-running Comparison + Confidence against its
    persisted Evidence and Observations, discarding whatever Canonical layer a previous engine
    version produced -- never re-invoking a provider.

    `original_events` is that document's full recorded telemetry (e.g.
    `telemetry_source.events_for_document(document_ref)`); only its pre-Comparison events are
    used as input. The regenerated Canonical layer is replayed through the same `Journal.replay`
    every other `JournalState` goes through, so `triage_review_queue`/`assemble_packet` need no
    special-casing to consume it.
    """
    journal = journal or Journal()
    result, pre_comparison_events = regenerate_canonical_layer(
        original_events,
        document_ref=document_ref,
        archive_object_ref=archive_object_ref,
        reconciliation_policy=reconciliation_policy,
        capability_matrix=capability_matrix,
        confidence_policy=confidence_policy,
        journal=journal,
    )

    regenerated_state = journal.replay(pre_comparison_events + result.events)
    return tuple(
        assemble_packet(
            regenerated_state, item, document_ref=document_ref, archive_object_ref=archive_object_ref
        )
        for item in triage_review_queue(regenerated_state, triage_policy)
    )


def regenerate_canonical_layer(
    original_events: Iterable[TelemetryEvent],
    *,
    document_ref: str,
    archive_object_ref: str,
    reconciliation_policy: ReconciliationPolicy,
    capability_matrix: CapabilityMatrix,
    confidence_policy: ConfidencePolicy,
    journal: Journal | None = None,
) -> tuple[PipelineRunResult, tuple[TelemetryEvent, ...]]:
    """Provider-free canonical recomputation plus the exact persisted input events used.

    The result is not appended here.  Operator tooling can inspect/diff it in dry-run mode before
    constructing explicit superseding decisions and a document snapshot.
    """

    journal = journal or Journal()
    pre_comparison_events = tuple(
        event for event in original_events if isinstance(event, _PRE_COMPARISON_EVENT_TYPES)
    )
    source_state = journal.replay(pre_comparison_events)

    evidence_by_id = {e.evidence_id: e for e in source_state.all_evidence()}
    # Grouped by (provider_id, provider_version), not by the original invocation_id: the
    # Comparison Engine only ever dedupes `participating_providers` by that pair
    # (`domain/comparison/engine.py`), never reads `ProviderObservationGraph.invocation_id` itself,
    # so which of a provider's several invocations an Observation came from is immaterial to
    # re-running Comparison -- and grouping this way sidesteps `ProviderObservationGraph`'s
    # non-empty invariant, which a per-invocation query could hit for an invocation telemetry
    # recorded as attempted but that (per the replayed Observations themselves) produced nothing.
    observations_by_provider: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for observation in source_state.all_observations():
        observations_by_provider[(observation.provider_id, observation.provider_version)].append(observation)

    provider_graphs = tuple(
        ProviderObservationGraph(
            provider_id=provider_id,
            provider_version=provider_version,
            invocation_id=f"regenerated:{provider_id}:{provider_version}",
            observations=tuple(observations),
        )
        for (provider_id, provider_version), observations in sorted(observations_by_provider.items())
    )
    if not provider_graphs:
        raise RegenerationError(
            f"{document_ref!r} has no Observations in its persisted telemetry -- "
            "nothing to regenerate a Comparison pass from"
        )

    result = run_comparison_and_assembly(
        document_ref=document_ref,
        archive_object_ref=archive_object_ref,
        provider_graphs=provider_graphs,
        evidence_by_id=evidence_by_id,
        reconciliation_policy=reconciliation_policy,
        capability_matrix=capability_matrix,
        confidence_policy=confidence_policy,
    )

    return result, pre_comparison_events
