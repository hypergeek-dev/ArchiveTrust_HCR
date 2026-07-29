from __future__ import annotations

import json

from archivetrust.admin.regenerate import main, regenerate_document
from archivetrust.application.current_state import CurrentStateService
from archivetrust.application.journal import Journal
from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.text_reconciliation import ReconciliationBasisCode
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from archivetrust.workspace.store import WorkspaceStore

from tests.review._helpers import emit_document_snapshot, emit_slot, heading


def _stale_sink() -> tuple[InMemoryTelemetrySink, object]:
    sink = InMemoryTelemetrySink()
    stale = emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("synthetic splice"),
        provider_payloads=(
            ("source_a", heading("Alpha heading")),
            ("source_b", heading("Omega heading")),
        ),
        classification=ComparisonClassification.CONTESTED,
        reconciliation_basis_code=ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS.value,
    )
    emit_document_snapshot(sink, document_ref="doc1", archive_object_ref="archive1")
    return sink, stale


def _run(sink, *, dry_run: bool):
    state = CurrentStateService(sink).document("doc1")
    return regenerate_document(
        state=state,
        source=sink,
        sink=sink,
        reconciliation_policy=ReconciliationPolicy(policy_version=1),
        capability_matrix=production_capability_matrix(),
        confidence_policy=ConfidencePolicy(confidence_policy_version=1),
        dry_run=dry_run,
    )


def test_regeneration_is_dry_run_safe_append_only_and_idempotent() -> None:
    sink, stale = _stale_sink()
    before = tuple(sink.all_events())

    planned = _run(sink, dry_run=True)
    assert planned.status == "planned"
    assert planned.appended_event_count == 2
    assert tuple(sink.all_events()) == before

    migrated = _run(sink, dry_run=False)
    assert migrated.status == "migrated"
    assert len(migrated.slots) == 1
    assert migrated.slots[0].source_canonical_observation_id == stale.canonical_observation_id
    assert migrated.slots[0].human_review_required is True

    current = CurrentStateService(sink).document("doc1")
    latest = current.slot(stale.semantic_slot_id).current
    assert latest.supersedes == stale.canonical_observation_id
    assert latest.reconciliation_basis_code == ReconciliationBasisCode.TEXT_CONTESTED_REFERENCE_PICK.value
    assert latest.payload.text in {"Alpha heading", "Omega heading"}
    assert current.requires_reassembly is False
    assert current.latest_canonical_document.supersedes == migrated.prior_document_snapshot_id
    assert current.effective_contained_observations == (latest.canonical_observation_id,)

    replayed = Journal().replay(sink.events_for_document("doc1"))
    history = replayed.canonical_observation_history(stale.semantic_slot_id)
    assert history[0] == stale
    assert history[-1] == latest

    rerun = _run(sink, dry_run=False)
    assert rerun.status == "unchanged"
    assert rerun.appended_event_count == 0


def test_admin_command_requires_backup_and_writes_checkpoint_and_report(tmp_path) -> None:
    deployment = tmp_path / "deployment"
    store = WorkspaceStore(deployment / "workspaces")
    workspace = store.create("Migration target")
    layout = store.layout_for(workspace.id)
    sink = FileTelemetrySink(layout.telemetry_dir / "events.jsonl")
    emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("synthetic splice"),
        provider_payloads=(
            ("source_a", heading("Alpha heading")),
            ("source_b", heading("Omega heading")),
        ),
        classification=ComparisonClassification.CONTESTED,
        reconciliation_basis_code=ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS.value,
    )
    emit_document_snapshot(sink, document_ref="doc1", archive_object_ref="archive1")
    before = (layout.telemetry_dir / "events.jsonl").read_bytes()

    dry_report = tmp_path / "dry-report.json"
    assert main(
        [
            "regenerate-current-state",
            workspace.id,
            "--deployment-root",
            str(deployment),
            "--report",
            str(dry_report),
        ]
    ) == 0
    assert (layout.telemetry_dir / "events.jsonl").read_bytes() == before
    assert json.loads(dry_report.read_text(encoding="utf-8"))["dry_run"] is True

    backup = tmp_path / "backup"
    checkpoint = tmp_path / "checkpoint.json"
    report = tmp_path / "apply-report.json"
    assert main(
        [
            "regenerate-current-state",
            workspace.id,
            "--deployment-root",
            str(deployment),
            "--apply",
            "--backup",
            str(backup),
            "--checkpoint",
            str(checkpoint),
            "--report",
            str(report),
        ]
    ) == 0
    assert (backup / "telemetry" / "events.jsonl").read_bytes() == before
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["completed"] == ["doc1"]
    applied = json.loads(report.read_text(encoding="utf-8"))
    assert applied["dry_run"] is False
    assert applied["integrity_ok"] is True
    assert applied["documents"][0]["status"] == "migrated"
