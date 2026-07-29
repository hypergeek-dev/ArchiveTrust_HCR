from __future__ import annotations

import json

from archivetrust.admin.regenerate import main
from archivetrust.application.progress import (
    FileProcessingProgressSink,
    ProcessingProgressEvent,
    ProcessingProgressKind,
    utc_now_iso,
)
from archivetrust.workspace.store import WorkspaceStore


def test_admin_run_reconciliation_dry_run_and_apply_are_explicit(tmp_path):
    deployment = tmp_path / "deployment"
    store = WorkspaceStore(deployment / "workspaces")
    workspace = store.create("Recovery")
    layout = store.layout_for(workspace.id)
    progress = FileProcessingProgressSink(layout.telemetry_dir / "processing.jsonl")
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.RUN_STARTED,
            run_id="open-run",
            recorded_at=utc_now_iso(),
            pending_count=1,
        )
    )

    dry_report = tmp_path / "dry.json"
    assert main(
        [
            "reconcile-runs",
            workspace.id,
            "--deployment-root",
            str(deployment),
            "--report",
            str(dry_report),
        ]
    ) == 0
    assert json.loads(dry_report.read_text(encoding="utf-8"))["remaining_open_run_ids"] == [
        "open-run"
    ]

    apply_report = tmp_path / "apply.json"
    assert main(
        [
            "reconcile-runs",
            workspace.id,
            "--deployment-root",
            str(deployment),
            "--report",
            str(apply_report),
            "--apply",
        ]
    ) == 0
    payload = json.loads(apply_report.read_text(encoding="utf-8"))
    assert payload["reconciled_run_ids"] == ["open-run"]
    assert payload["remaining_open_run_ids"] == []
