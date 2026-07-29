"""Bridges a `ComparisonEngineResult` into telemetry events (MILESTONE4_COMPARISON_ENGINE.md S12).

**Scope, stated honestly:** this emits `CanonicalDecisionCreated` (one per Canonical Observation)
and `AgreementCalculated` (one per cluster's Comparison Confidence), both carrying the
Reconciliation Policy and Capability Matrix versions in effect -- the two events S12's table
identifies as carrying the engine's actual output. It does **not** yet emit `ObservationCompared`
(per-cluster clustering detail beyond what `clustering_basis` already carries on the Canonical
Observation itself), `ObservationAccepted`/`ObservationRejected` (per-contributor accept/reject
detail), or `KnowledgeDiscarded` (cycle-broken-edge detail, available on
`StructuralReconciliationResult.dropped_for_cycle` but not threaded through here). These are real,
named gaps against S12's full table, not fabricated as complete -- see IMPLEMENTATION_STATUS.md.
"""

from __future__ import annotations

from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.engine import ComparisonEngineResult
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import AgreementCalculated, CanonicalDecisionCreated, TelemetryEvent


def comparison_result_to_events(
    result: ComparisonEngineResult,
    *,
    document_ref: str,
    policy: ReconciliationPolicy,
    capability_matrix: CapabilityMatrix,
) -> tuple[TelemetryEvent, ...]:
    events: list[TelemetryEvent] = []
    for canonical in result.reconciled_graph.canonical_observations:
        events.append(
            CanonicalDecisionCreated(
                event_id=new_id("event"),
                document_ref=document_ref,
                canonical_observation=canonical,
                reconciliation_policy_version=policy.policy_version,
                capability_matrix_version=capability_matrix.matrix_version,
            )
        )
        events.append(
            AgreementCalculated(
                event_id=new_id("event"),
                document_ref=document_ref,
                semantic_slot_id=canonical.semantic_slot_id,
                comparison_confidence=canonical.comparison_confidence,
                reconciliation_policy_version=policy.policy_version,
                capability_matrix_version=capability_matrix.matrix_version,
            )
        )
    return tuple(events)
