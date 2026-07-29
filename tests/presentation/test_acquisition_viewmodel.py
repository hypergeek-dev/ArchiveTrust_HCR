from __future__ import annotations

from archivetrust.acquisition.events import AcquisitionTelemetrySink
from archivetrust.acquisition.manager import AcquisitionManager
from archivetrust.presentation.acquisition_viewmodel import AcquisitionManagerViewModel
from archivetrust.workspace.layout import WorkspaceLayout


def _vm(tmp_path) -> AcquisitionManagerViewModel:
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    manager = AcquisitionManager("ws1", layout, AcquisitionTelemetrySink())
    return AcquisitionManagerViewModel(manager=manager, layout=layout, workspace_id="ws1")


def test_add_manual_import_appears_in_configured_sources(tmp_path) -> None:
    vm = _vm(tmp_path)
    vm.add_manual_import()
    rows = vm.configured_sources()
    assert len(rows) == 1
    assert rows[0].kind == "manual_import"
    assert rows[0].implemented is True


def test_add_folder_watch_appears_in_configured_sources(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    vm = _vm(tmp_path)
    vm.add_folder_watch(str(tmp_path / "incoming"))
    rows = vm.configured_sources()
    assert rows[0].kind == "folder_watch"


def test_not_yet_connected_kinds_excludes_implemented_ones(tmp_path) -> None:
    vm = _vm(tmp_path)
    kinds = {kind for kind, _label in vm.not_yet_connected_kinds()}
    assert "manual_import" not in kinds
    assert "folder_watch" not in kinds
    assert "sharepoint" in kinds
    assert "email_inbox" in kinds


def test_trigger_scan_returns_registered_count(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw
    import time

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    vm = _vm(tmp_path)
    watched = tmp_path / "watched"
    vm.add_folder_watch(str(watched))
    watched.mkdir(exist_ok=True)
    (watched / "record.pdf").write_bytes(b"content")
    time.sleep(0.05)

    assert vm.trigger_scan() == 1
