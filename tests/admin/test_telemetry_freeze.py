from __future__ import annotations

import hashlib
import json

from archivetrust.admin.cli import main
from archivetrust.admin.telemetry_freeze import (
    freeze_workspace_telemetry,
    load_active_telemetry_epoch,
)
from archivetrust.workspace.store import WorkspaceStore


def test_freeze_workspace_telemetry_records_cursors_without_rewriting_streams(tmp_path) -> None:
    deployment = tmp_path / "deployment"
    store = WorkspaceStore(deployment / "workspaces")
    workspace = store.create("Stable")
    layout = store.layout_for(workspace.id)
    events = layout.telemetry_dir / "events.jsonl"
    processing = layout.telemetry_dir / "processing.jsonl"
    event_payload = b'{"kind":"Example","document_ref":"doc-1","recorded_at":"2026-07-16T12:00:00+00:00"}\n'
    processing_payload = b'{"kind":"ProcessingRunStarted","run_id":"run-1","recorded_at":"2026-07-16T12:01:00+00:00"}\n'
    events.write_bytes(event_payload)
    processing.write_bytes(processing_payload)

    report = freeze_workspace_telemetry(
        layout,
        workspace_id=workspace.id,
        label="recent-stable",
        stable_since="2026-07-16T00:00:00+00:00",
        stable_basis="test stable basis",
    )

    assert events.read_bytes() == event_payload
    assert processing.read_bytes() == processing_payload
    event_stream = next(stream for stream in report.streams if stream.stream_name == "events.jsonl")
    assert event_stream.cursor_byte_offset == len(event_payload)
    assert event_stream.cursor_line_count == 1
    assert event_stream.sha256 == hashlib.sha256(event_payload).hexdigest()
    assert event_stream.last_recorded_at == "2026-07-16T12:00:00+00:00"
    assert report.label == "recent-stable"
    assert report.stable_basis == "test stable basis"
    assert (layout.reports_dir / "telemetry_freezes" / f"{report.freeze_id}.json").exists()

    epoch = load_active_telemetry_epoch(layout)
    assert epoch is not None
    assert epoch.history_freeze_id == report.freeze_id
    assert epoch.stream_cursors["events.jsonl"].byte_offset == len(event_payload)


def test_admin_freeze_telemetry_history_command_writes_report_and_active_epoch(tmp_path) -> None:
    deployment = tmp_path / "deployment"
    store = WorkspaceStore(deployment / "workspaces")
    workspace = store.create("Stable")
    layout = store.layout_for(workspace.id)
    (layout.telemetry_dir / "events.jsonl").write_text(
        '{"kind":"Example","document_ref":"doc-1","recorded_at":"2026-07-16T12:00:00+00:00"}\n',
        encoding="utf-8",
    )
    report_path = tmp_path / "freeze-report.json"

    assert main(
        [
            "freeze-telemetry-history",
            workspace.name,
            "--deployment-root",
            str(deployment),
            "--label",
            "stable-reset",
            "--basis",
            "test basis",
            "--report",
            str(report_path),
        ]
    ) == 0

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_id"] == "archivetrust.telemetry_history_freeze.v1"
    assert payload["workspace_id"] == workspace.id
    assert payload["label"] == "stable-reset"
    assert (layout.config_dir / "active_telemetry_epoch.json").exists()
