from __future__ import annotations

from datetime import datetime, timezone

from archivetrust.acquisition.events import AcquisitionTelemetrySink, ArchiveObjectRegistered
from archivetrust.acquisition.registration import register_archive_object
from archivetrust.acquisition.source import DiscoveredFile
from archivetrust.workspace.layout import WorkspaceLayout


def _discovered(path) -> DiscoveredFile:
    return DiscoveredFile(
        path=path, original_filename=path.name, byte_size=path.stat().st_size,
        modified_at=datetime.now(timezone.utc),
    )


def test_registration_copies_file_and_never_touches_original(tmp_path) -> None:
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    source_file = tmp_path / "incoming" / "protokoll.pdf"
    source_file.parent.mkdir()
    source_file.write_bytes(b"%PDF-1.4 original bytes")
    original_bytes = source_file.read_bytes()

    sink = AcquisitionTelemetrySink()
    known: set[str] = set()
    archive_object = register_archive_object(
        _discovered(source_file), workspace_id="ws1", layout=layout, known_hashes=known,
        source_id="manual-import", sink=sink,
    )

    assert archive_object is not None
    assert source_file.read_bytes() == original_bytes  # original untouched
    copied = layout.archive_dir / archive_object.storage_path
    assert copied.exists()
    assert copied.read_bytes() == original_bytes
    assert any(isinstance(e, ArchiveObjectRegistered) for e in sink.all_events())


def test_duplicate_content_is_not_registered_twice(tmp_path) -> None:
    layout = WorkspaceLayout(root=tmp_path / "ws").ensure()
    first_file = tmp_path / "a.pdf"
    first_file.write_bytes(b"identical content")
    second_file = tmp_path / "b.pdf"
    second_file.write_bytes(b"identical content")

    sink = AcquisitionTelemetrySink()
    known: set[str] = set()
    first = register_archive_object(
        _discovered(first_file), workspace_id="ws1", layout=layout, known_hashes=known,
        source_id="manual-import", sink=sink,
    )
    second = register_archive_object(
        _discovered(second_file), workspace_id="ws1", layout=layout, known_hashes=known,
        source_id="manual-import", sink=sink,
    )

    assert first is not None
    assert second is None  # duplicate content hash, not registered again
    registered_events = [e for e in sink.all_events() if isinstance(e, ArchiveObjectRegistered)]
    assert len(registered_events) == 1
