"""Safe operator regeneration of stale persisted canonical decisions.

Only recorded Evidence and Observations are reused.  No provider adapter is imported or invoked,
and history is advanced exclusively by appending superseding canonical facts and a document
snapshot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.application.current_state import CurrentStateService
from archivetrust.application.progress import FileProcessingProgressSink
from archivetrust.application.recovery import incomplete_run_ids, reconcile_incomplete_runs
from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.text_reconciliation import ReconciliationBasisCode
from archivetrust.domain.comparison.text_reconciliation import TextCandidate, reconcile_text
from archivetrust.domain.confidence.engine import compute_canonical_confidence
from archivetrust.domain.confidence.models import ComparisonClassification, ComparisonConfidence
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.current_state import CurrentDocumentState
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    TelemetryEvent,
    stamp_recorded_at,
)
from archivetrust.infrastructure.storage.integrity import (
    default_hash_chain_sidecar_path,
    verify_hash_chain_sidecar,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.review.regenerate import regenerate_canonical_layer
from archivetrust.workspace.store import WorkspaceStore


class SlotRegeneration(BaseModel):
    model_config = ConfigDict(frozen=True)

    semantic_slot_id: str
    source_canonical_observation_id: str
    resulting_canonical_observation_id: str
    old_reconciliation_version: int | None
    new_reconciliation_version: int
    value_changed: bool
    classification_changed: bool
    became_contested: bool
    human_review_required: bool


class DocumentRegeneration(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    status: str
    reason: str
    source_event_count: int
    appended_event_count: int
    prior_document_snapshot_id: str | None
    resulting_document_snapshot_id: str | None
    slots: tuple[SlotRegeneration, ...] = ()
    error: str | None = None


class RegenerationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_schema: str = "archivetrust.current_state_regeneration.v1"
    workspace_id: str
    dry_run: bool
    started_at: str
    finished_at: str
    backup_path: str | None
    documents: tuple[DocumentRegeneration, ...]
    failures: int
    integrity_ok: bool | None
    integrity_reason: str | None


def regenerate_document(
    *,
    state: CurrentDocumentState,
    source,
    sink,
    reconciliation_policy: ReconciliationPolicy,
    capability_matrix,
    confidence_policy: ConfidencePolicy,
    dry_run: bool,
) -> DocumentRegeneration:
    """Plan and optionally append one idempotent document migration."""

    source_events = tuple(source.events_for_document(state.document_ref))
    stale = tuple(
        slot
        for slot in state.slots
        if slot.current.comparison_confidence.classification
        is ComparisonClassification.CONTESTED
        and slot.current.reconciliation_basis_code
        == ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS.value
    )
    if not stale and not state.requires_reassembly:
        return DocumentRegeneration(
            document_ref=state.document_ref,
            status="unchanged",
            reason="no stale contested decision or incomplete document reassembly",
            source_event_count=len(source_events),
            appended_event_count=0,
            prior_document_snapshot_id=(
                state.latest_canonical_document.document_snapshot_id
                if state.latest_canonical_document is not None
                else None
            ),
            resulting_document_snapshot_id=None,
        )
    if state.latest_canonical_document is None:
        raise ValueError("cannot migrate a canonical decision without a prior document snapshot")

    result, _ = regenerate_canonical_layer(
        source_events,
        document_ref=state.document_ref,
        archive_object_ref=state.archive_object_ref,
        reconciliation_policy=reconciliation_policy,
        capability_matrix=capability_matrix,
        confidence_policy=confidence_policy,
    )
    regenerated_by_lineage: dict[tuple[str, frozenset[str]], list] = {}
    for candidate in result.reconciled_graph.canonical_observations:
        key = (
            candidate.observation_type.value,
            frozenset(ref.observation_id for ref in candidate.contributing_observations),
        )
        regenerated_by_lineage.setdefault(key, []).append(candidate)

    decision_events: list[TelemetryEvent] = []
    replacements: dict[str, str] = {}
    slot_results: list[SlotRegeneration] = []
    for slot in stale:
        old = slot.current
        key = (
            old.observation_type.value,
            frozenset(ref.observation_id for ref in old.contributing_observations),
        )
        candidates = regenerated_by_lineage.get(key, ())
        if len(candidates) > 1:
            raise ValueError(
                f"slot {slot.semantic_slot_id} has {len(candidates)} regenerated lineage matches"
            )
        if candidates:
            candidate = candidates[0]
        else:
            # A newer alignment policy may split the historical cluster, leaving no one-to-one
            # regenerated graph node.  The unsafe persisted fact was specifically the synthesized
            # text for this historical semantic slot, so re-run the current text policy directly
            # over that slot's recorded contributing Observations.  This is deterministic,
            # provider-free, and preserves the honest historical slot identity while replacing
            # the fabricated splice with a real candidate reading.
            text_result = reconcile_text(
                tuple(
                    TextCandidate(
                        observation_id=observation.observation_id,
                        provider_id=observation.provider_id,
                        text=getattr(observation.payload, "text", "") or "",
                    )
                    for observation in slot.contributing_observations
                ),
                reconciliation_policy,
            )
            candidate = old.model_copy(
                update={
                    "payload": old.payload.model_copy(update={"text": text_result.accepted_text}),
                    "comparison_confidence": ComparisonConfidence(
                        classification=ComparisonClassification(text_result.classification.value),
                        magnitude=text_result.magnitude,
                        basis=text_result.reconciliation_basis,
                    ),
                    "reconciliation_basis": text_result.reconciliation_basis,
                    "reconciliation_basis_code": text_result.reconciliation_basis_code.value,
                }
            )
            candidate = candidate.model_copy(
                update={
                    "canonical_confidence": compute_canonical_confidence(
                        candidate, slot.contributing_observations, confidence_policy
                    )
                }
            )
        superseding = candidate.model_copy(
            update={
                "canonical_observation_id": new_id("canonical_observation"),
                "semantic_slot_id": old.semantic_slot_id,
                "reconciliation_sequence": old.reconciliation_sequence + 1,
                "parent_observations": old.parent_observations,
                "child_observations": old.child_observations,
                "related_observations": old.related_observations,
                "supersedes": old.canonical_observation_id,
                "superseded_by": None,
                "human_correction_ref": None,
                "rationale": "production closure regeneration of stale contested synthesis",
            }
        )
        replacements[old.canonical_observation_id] = superseding.canonical_observation_id
        decision_events.append(
            CanonicalDecisionCreated(
                event_id=new_id("event"),
                document_ref=state.document_ref,
                canonical_observation=superseding,
                reconciliation_policy_version=reconciliation_policy.policy_version,
                capability_matrix_version=capability_matrix.matrix_version,
                confidence_policy_version=confidence_policy.confidence_policy_version,
            )
        )
        old_value = getattr(old.payload, "text", None)
        new_value = getattr(superseding.payload, "text", None)
        new_classification = superseding.comparison_confidence.classification
        slot_results.append(
            SlotRegeneration(
                semantic_slot_id=old.semantic_slot_id,
                source_canonical_observation_id=old.canonical_observation_id,
                resulting_canonical_observation_id=superseding.canonical_observation_id,
                old_reconciliation_version=state.policy_versions.reconciliation_version,
                new_reconciliation_version=reconciliation_policy.policy_version,
                value_changed=old_value != new_value,
                classification_changed=(
                    old.comparison_confidence.classification is not new_classification
                ),
                became_contested=new_classification is ComparisonClassification.CONTESTED,
                human_review_required=new_classification is ComparisonClassification.CONTESTED,
            )
        )

    previous = state.latest_canonical_document
    contained = tuple(
        replacements.get(observation_id, observation_id)
        for observation_id in state.effective_contained_observations
    )
    document = previous.model_copy(
        update={
            "document_snapshot_id": new_id("document_snapshot"),
            "contained_observations": contained,
            "ontology_version": result.canonical_document.ontology_version,
            "document_version": previous.document_version + 1,
            "reassembly_trigger": "production_closure_stale_decision_regeneration",
            "supersedes": previous.document_snapshot_id,
            "superseded_by": None,
        }
    )
    document_event = CanonicalDocumentCreated(
        event_id=new_id("event"),
        document_ref=state.document_ref,
        canonical_document=document,
        reconciliation_policy_version=reconciliation_policy.policy_version,
        capability_matrix_version=capability_matrix.matrix_version,
        confidence_policy_version=confidence_policy.confidence_policy_version,
    )
    events = tuple(stamp_recorded_at(event) for event in (*decision_events, document_event))
    if not dry_run:
        for event in events:
            sink.append(event)
    return DocumentRegeneration(
        document_ref=state.document_ref,
        status="planned" if dry_run else "migrated",
        reason=(
            "stale contested synthesis superseded under current approved policy"
            if stale
            else "completed a previously missing correction-aware document reassembly"
        ),
        source_event_count=len(source_events),
        appended_event_count=len(events),
        prior_document_snapshot_id=previous.document_snapshot_id,
        resulting_document_snapshot_id=document.document_snapshot_id,
        slots=tuple(slot_results),
    )


def _workspace(store: WorkspaceStore, identity: str):
    return next(
        (workspace for workspace in store.list() if workspace.id == identity or workspace.name == identity),
        None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="archivetrust-admin regenerate-current-state")
    parser.add_argument("command", choices=("regenerate-current-state", "reconcile-runs"))
    parser.add_argument("workspace")
    parser.add_argument("--deployment-root", default="archivetrust_data")
    parser.add_argument("--document", default=None)
    parser.add_argument("--apply", action="store_true", help="append the planned supersessions")
    parser.add_argument("--backup", default=None, help="required destination directory with --apply")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    store = WorkspaceStore(Path(args.deployment_root) / "workspaces")
    workspace = _workspace(store, args.workspace)
    if workspace is None:
        raise SystemExit(f"workspace {args.workspace!r} not found")
    layout = store.layout_for(workspace.id)
    if args.command == "reconcile-runs":
        progress = FileProcessingProgressSink(layout.telemetry_dir / "processing.jsonl")
        open_runs = incomplete_run_ids(progress)
        reconciled = reconcile_incomplete_runs(progress) if args.apply else ()
        payload = {
            "report_schema": "archivetrust.run_reconciliation.v1",
            "workspace_id": workspace.id,
            "dry_run": not args.apply,
            "open_run_ids": open_runs,
            "reconciled_run_ids": reconciled,
            "remaining_open_run_ids": incomplete_run_ids(progress),
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    event_path = layout.telemetry_dir / "events.jsonl"
    if args.apply and not args.backup:
        raise SystemExit("--backup is required with --apply")

    backup_path: Path | None = None
    if args.apply:
        backup_path = Path(args.backup).resolve()
        workspace_root = layout.root.resolve()
        if backup_path == workspace_root or workspace_root in backup_path.parents:
            raise SystemExit("backup destination must be outside the source workspace")
        if backup_path.exists():
            raise SystemExit(f"backup destination already exists: {backup_path}")
        shutil.copytree(workspace_root, backup_path)

    sink = FileTelemetrySink(event_path)
    current = CurrentStateService(sink)
    completed: set[str] = set()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None and checkpoint_path.exists():
        completed = set(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("completed", ()))

    started = datetime.now(timezone.utc).isoformat()
    rows: list[DocumentRegeneration] = []
    states = current.documents()
    if args.document:
        states = tuple(state for state in states if state.document_ref == args.document)
        if not states:
            raise SystemExit(f"document {args.document!r} not found")
    for state in states:
        if state.document_ref in completed:
            continue
        try:
            row = regenerate_document(
                state=state,
                source=sink,
                sink=sink,
                reconciliation_policy=ReconciliationPolicy(policy_version=1),
                capability_matrix=production_capability_matrix(),
                confidence_policy=ConfidencePolicy(confidence_policy_version=1),
                dry_run=not args.apply,
            )
        except Exception as error:  # report per-document failure; never silently continue
            row = DocumentRegeneration(
                document_ref=state.document_ref,
                status="failed",
                reason="regeneration failed",
                source_event_count=len(tuple(sink.events_for_document(state.document_ref))),
                appended_event_count=0,
                prior_document_snapshot_id=(
                    state.latest_canonical_document.document_snapshot_id
                    if state.latest_canonical_document is not None
                    else None
                ),
                resulting_document_snapshot_id=None,
                error=f"{type(error).__name__}: {error}",
            )
        rows.append(row)
        if row.status != "failed":
            completed.add(state.document_ref)
            if checkpoint_path is not None and args.apply:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps({"workspace_id": workspace.id, "completed": sorted(completed)}, indent=2),
                    encoding="utf-8",
                )

    integrity_ok: bool | None = None
    integrity_reason: str | None = "not_checked_in_dry_run"
    if args.apply:
        integrity = verify_hash_chain_sidecar(event_path, default_hash_chain_sidecar_path(event_path))
        integrity_ok = integrity.ok
        integrity_reason = integrity.reason
    report = RegenerationReport(
        workspace_id=workspace.id,
        dry_run=not args.apply,
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        backup_path=str(backup_path) if backup_path is not None else None,
        documents=tuple(rows),
        failures=sum(row.status == "failed" for row in rows),
        integrity_ok=integrity_ok,
        integrity_reason=integrity_reason,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 1 if report.failures or integrity_ok is False else 0


if __name__ == "__main__":
    sys.exit(main())
