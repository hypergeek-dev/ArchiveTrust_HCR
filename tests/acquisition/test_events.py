from __future__ import annotations

import json

from archivetrust.acquisition.archive_object import ArchiveObject
from archivetrust.acquisition.events import (
    AcquisitionEventKind,
    ArchiveObjectRegistered,
    WorkspaceCreated,
    parse_acquisition_event,
)


def test_round_trips_through_json() -> None:
    event = WorkspaceCreated(event_id="e1", workspace_id="ws1", name="Lund Municipality")
    parsed = parse_acquisition_event(json.loads(event.model_dump_json()))
    assert parsed == event


def test_archive_object_registered_round_trips_the_full_object() -> None:
    archive_object = ArchiveObject(
        id="ao1", workspace_id="ws1", source_id="manual-import", original_filename="a.pdf",
        content_hash="deadbeef", byte_size=10, mime_type="application/pdf", storage_path="deadbeef_a.pdf",
    )
    event = ArchiveObjectRegistered(event_id="e2", workspace_id="ws1", archive_object=archive_object)
    parsed = parse_acquisition_event(json.loads(event.model_dump_json()))
    assert isinstance(parsed, ArchiveObjectRegistered)
    assert parsed.archive_object == archive_object


def test_every_prompt_named_event_kind_is_present() -> None:
    expected = {
        "WorkspaceCreated", "WorkspaceOpened", "AcquisitionStarted", "ArchiveObjectDiscovered",
        "ArchiveObjectRegistered", "AcquisitionCompleted", "AcquisitionFailed",
    }
    assert {kind.value for kind in AcquisitionEventKind} == expected
