from __future__ import annotations

import time

from archivetrust.acquisition.events import AcquisitionCompleted, AcquisitionTelemetrySink
from archivetrust.acquisition.folder_watch import FolderWatchConfig, FolderWatchSource
from archivetrust.acquisition.manager import AcquisitionManager
from archivetrust.acquisition.manual_import import ManualImportSource
from archivetrust.application.progress import (
    FileProcessingProgressSink,
    ProcessingProgressEvent,
    ProcessingProgressKind,
    utc_now_iso,
)
from archivetrust.workspace.layout import WorkspaceLayout


def test_run_scan_cycle_registers_discovered_files_and_fills_pending(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    sink = AcquisitionTelemetrySink()
    manager = AcquisitionManager("ws1", layout, sink)

    watched = tmp_path / "watched"
    source = FolderWatchSource(FolderWatchConfig(path=watched, stability_check_interval_seconds=0.02))
    manager.add_source(source)
    (watched / "beslut.pdf").write_bytes(b"a decision")
    time.sleep(0.05)

    registered = manager.run_scan_cycle()
    assert len(registered) == 1
    assert len(manager.pending) == 1
    assert any(isinstance(e, AcquisitionCompleted) for e in sink.all_events())


def test_disabled_source_is_not_scanned(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    sink = AcquisitionTelemetrySink()
    manager = AcquisitionManager("ws1", layout, sink)

    watched = tmp_path / "watched"
    source = FolderWatchSource(FolderWatchConfig(path=watched, stability_check_interval_seconds=0.02))
    manager.add_source(source)
    manager.set_enabled(source.source_id, False)
    (watched / "beslut.pdf").write_bytes(b"a decision")
    time.sleep(0.05)

    assert manager.run_scan_cycle() == ()


def test_reopened_workspace_requeues_registered_but_never_completed_documents(tmp_path) -> None:
    """Production Incident Recovery, 2026-07-14: a Workspace closed (or crashed) mid-run must not
    silently drop never-processed Archive Objects on reopen just because they are already
    registered (dedup) -- `pending` should be recomputed from durable acquisition + processing
    state, not restart empty."""
    from archivetrust.acquisition.events import FileAcquisitionTelemetrySink

    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    acquisition_sink_path = tmp_path / "acquisition.jsonl"
    progress_path = tmp_path / "processing.jsonl"

    acquisition_sink = FileAcquisitionTelemetrySink(acquisition_sink_path)
    manager = AcquisitionManager("ws1", layout, acquisition_sink)
    source = ManualImportSource()
    (tmp_path / "done.pdf").write_bytes(b"done")
    (tmp_path / "crashed.pdf").write_bytes(b"crashed")
    (tmp_path / "never_started.pdf").write_bytes(b"never started")
    registered = manager.import_files(
        source,
        (tmp_path / "done.pdf", tmp_path / "crashed.pdf", tmp_path / "never_started.pdf"),
    )
    done_obj, crashed_obj, _never_started_obj = registered

    progress = FileProcessingProgressSink(progress_path)
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_STARTED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            archive_object_id=done_obj.id,
        )
    )
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_COMPLETED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            archive_object_id=done_obj.id,
            had_failure=False,
        )
    )
    # crashed_obj was started but the process died before a completion record was written.
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_STARTED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            archive_object_id=crashed_obj.id,
        )
    )

    # Simulate reopening the Workspace: fresh manager and progress sink instances over the same
    # durable files, exactly as `composition.py` builds them on `open_workspace`.
    reopened_sink = FileAcquisitionTelemetrySink(acquisition_sink_path)
    reopened_progress = FileProcessingProgressSink(progress_path)
    reopened_manager = AcquisitionManager("ws1", layout, reopened_sink, reopened_progress)

    pending_ids = {obj.id for obj in reopened_manager.pending}
    assert done_obj.id not in pending_ids
    assert crashed_obj.id in pending_ids
    assert len(pending_ids) == 2


def test_reopened_workspace_requeues_historical_failed_completion(tmp_path) -> None:
    from archivetrust.acquisition.events import FileAcquisitionTelemetrySink

    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    acquisition_path = tmp_path / "acquisition.jsonl"
    progress_path = tmp_path / "processing.jsonl"
    sink = FileAcquisitionTelemetrySink(acquisition_path)
    manager = AcquisitionManager("ws1", layout, sink)
    source = ManualImportSource()
    source_path = tmp_path / "failed.pdf"
    source_path.write_bytes(b"failed")
    (failed,) = manager.import_files(source, (source_path,))
    progress = FileProcessingProgressSink(progress_path)
    progress.record(
        ProcessingProgressEvent(
            kind=ProcessingProgressKind.DOCUMENT_COMPLETED,
            run_id="run1",
            recorded_at=utc_now_iso(),
            archive_object_id=failed.id,
            original_filename=failed.original_filename,
            had_failure=True,
        )
    )

    reopened = AcquisitionManager(
        "ws1",
        layout,
        FileAcquisitionTelemetrySink(acquisition_path),
        FileProcessingProgressSink(progress_path),
    )

    assert [item.id for item in reopened.pending] == [failed.id]


def test_source_status_reports_configured_sources(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    manager = AcquisitionManager("ws1", layout, AcquisitionTelemetrySink())
    source = FolderWatchSource(FolderWatchConfig(path=tmp_path / "watched"))
    manager.add_source(source)

    statuses = manager.source_status()
    assert len(statuses) == 1
    assert statuses[0].source_id == source.source_id
    assert statuses[0].enabled is True
