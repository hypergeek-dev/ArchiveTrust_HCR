"""Freeze a workspace telemetry prefix and start a new measurement epoch.

The freeze is deliberately non-destructive: telemetry JSONL files are never moved, truncated, or
rewritten.  The command records stable byte/line cursors plus content hashes, then writes an active
epoch pointer that later tools can use to count only post-freeze activity.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from archivetrust.workspace.layout import WorkspaceLayout

_FREEZE_SCHEMA = "archivetrust.telemetry_history_freeze.v1"
_EPOCH_SCHEMA = "archivetrust.active_telemetry_epoch.v1"
_DEFAULT_STABLE_SINCE = "2026-07-16T00:00:00+00:00"
_DEFAULT_STABLE_BASIS = (
    "Release baseline 2026-07-16: stable runnable profile documented as Docling + Tesseract."
)


class TelemetryStreamFreeze(BaseModel):
    model_config = ConfigDict(frozen=True)

    stream_name: str
    relative_path: str
    exists: bool
    size_bytes: int
    line_count: int
    sha256: str | None
    cursor_byte_offset: int
    cursor_line_count: int
    last_recorded_at: str | None = None


class TelemetryStreamCursor(BaseModel):
    model_config = ConfigDict(frozen=True)

    relative_path: str
    byte_offset: int
    line_count: int
    sha256_at_freeze: str | None


class ActiveTelemetryEpoch(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = _EPOCH_SCHEMA
    workspace_id: str
    epoch_id: str
    started_at: str
    history_freeze_id: str
    history_freeze_label: str
    stable_since: str
    stable_basis: str
    stream_cursors: dict[str, TelemetryStreamCursor]


class TelemetryHistoryFreeze(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = _FREEZE_SCHEMA
    freeze_id: str
    workspace_id: str
    label: str
    stable_since: str
    stable_basis: str
    created_at: str
    manifest_path: str
    active_epoch_path: str | None
    streams: tuple[TelemetryStreamFreeze, ...]


def freeze_workspace_telemetry(
    layout: WorkspaceLayout,
    *,
    workspace_id: str,
    label: str = "recent-stable",
    stable_since: str = _DEFAULT_STABLE_SINCE,
    stable_basis: str = _DEFAULT_STABLE_BASIS,
    activate: bool = True,
) -> TelemetryHistoryFreeze:
    """Record an immutable historical boundary for the workspace telemetry directory."""

    created_at = datetime.now(timezone.utc).isoformat()
    freeze_id = f"{_timestamp_id(created_at)}_{_slug(label)}"
    manifest_dir = layout.reports_dir / "telemetry_freezes"
    manifest_path = manifest_dir / f"{freeze_id}.json"
    active_epoch_path = layout.config_dir / "active_telemetry_epoch.json"
    streams = tuple(_stream_freeze(layout.root, path) for path in _telemetry_files(layout))
    report = TelemetryHistoryFreeze(
        freeze_id=freeze_id,
        workspace_id=workspace_id,
        label=label,
        stable_since=stable_since,
        stable_basis=stable_basis,
        created_at=created_at,
        manifest_path=str(manifest_path),
        active_epoch_path=str(active_epoch_path) if activate else None,
        streams=streams,
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if activate:
        epoch = ActiveTelemetryEpoch(
            workspace_id=workspace_id,
            epoch_id=freeze_id,
            started_at=created_at,
            history_freeze_id=freeze_id,
            history_freeze_label=label,
            stable_since=stable_since,
            stable_basis=stable_basis,
            stream_cursors={
                stream.stream_name: TelemetryStreamCursor(
                    relative_path=stream.relative_path,
                    byte_offset=stream.cursor_byte_offset,
                    line_count=stream.cursor_line_count,
                    sha256_at_freeze=stream.sha256,
                )
                for stream in streams
                if stream.exists
            },
        )
        active_epoch_path.parent.mkdir(parents=True, exist_ok=True)
        active_epoch_path.write_text(epoch.model_dump_json(indent=2), encoding="utf-8")
    return report


def load_active_telemetry_epoch(layout: WorkspaceLayout) -> ActiveTelemetryEpoch | None:
    path = layout.config_dir / "active_telemetry_epoch.json"
    if not path.exists():
        return None
    return ActiveTelemetryEpoch.model_validate_json(path.read_text(encoding="utf-8"))


def _telemetry_files(layout: WorkspaceLayout) -> tuple[Path, ...]:
    telemetry_dir = layout.telemetry_dir
    preferred = [
        telemetry_dir / "events.jsonl",
        telemetry_dir / "acquisition.jsonl",
        telemetry_dir / "processing.jsonl",
        telemetry_dir / "sampling.jsonl",
        telemetry_dir / "events.jsonl.chain.jsonl",
        telemetry_dir / "events.jsonl.chain.json",
    ]
    discovered = sorted(
        path
        for path in telemetry_dir.glob("*")
        if path.is_file() and path.suffix in {".jsonl", ".json"}
    )
    ordered = []
    seen: set[Path] = set()
    for path in (*preferred, *discovered):
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return tuple(ordered)


def _stream_freeze(root: Path, path: Path) -> TelemetryStreamFreeze:
    relative = path.relative_to(root).as_posix()
    if not path.exists():
        return TelemetryStreamFreeze(
            stream_name=path.name,
            relative_path=relative,
            exists=False,
            size_bytes=0,
            line_count=0,
            sha256=None,
            cursor_byte_offset=0,
            cursor_line_count=0,
        )
    digest = hashlib.sha256()
    line_count = 0
    size = 0
    ends_with_newline = True
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            line_count += chunk.count(b"\n")
            ends_with_newline = chunk.endswith(b"\n")
    if size and not ends_with_newline:
        line_count += 1
    return TelemetryStreamFreeze(
        stream_name=path.name,
        relative_path=relative,
        exists=True,
        size_bytes=size,
        line_count=line_count,
        sha256=digest.hexdigest(),
        cursor_byte_offset=size,
        cursor_line_count=line_count,
        last_recorded_at=_last_recorded_at(path),
    )


def _last_recorded_at(path: Path) -> str | None:
    if path.suffix != ".jsonl":
        return None
    last: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload: Any = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("recorded_at"), str):
                last = payload["recorded_at"]
    return last


def _timestamp_id(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "telemetry-freeze"
