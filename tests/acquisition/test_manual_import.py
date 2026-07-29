from __future__ import annotations

from archivetrust.acquisition.events import AcquisitionTelemetrySink
from archivetrust.acquisition.manual_import import ManualImportSource
from archivetrust.workspace.layout import WorkspaceLayout


def test_import_files_registers_each_file(tmp_path) -> None:
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    a = incoming / "a.pdf"
    a.write_bytes(b"file a")
    b = incoming / "b.pdf"
    b.write_bytes(b"file b")

    source = ManualImportSource()
    sink = AcquisitionTelemetrySink()
    registered = source.import_files(
        (a, b), workspace_id="ws1", layout=layout, known_hashes=set(), sink=sink
    )
    assert len(registered) == 2
    assert source.health().documents_imported == 2


def test_import_folder_recursive(tmp_path) -> None:
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    incoming = tmp_path / "incoming"
    (incoming / "sub").mkdir(parents=True)
    (incoming / "top.pdf").write_bytes(b"top level")
    (incoming / "sub" / "nested.pdf").write_bytes(b"nested")

    source = ManualImportSource()
    sink = AcquisitionTelemetrySink()
    registered = source.import_folder(
        incoming, recursive=True, workspace_id="ws1", layout=layout, known_hashes=set(), sink=sink
    )
    assert len(registered) == 2


def test_scan_never_discovers_on_its_own() -> None:
    # Manual Import is always operator-triggered (ROADMAP.md §5.13.2) -- scan() never returns
    # candidates on its own, unlike Folder Watch.
    assert tuple(ManualImportSource().scan(frozenset())) == ()
