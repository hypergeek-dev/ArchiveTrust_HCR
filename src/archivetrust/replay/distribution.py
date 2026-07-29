"""Provider-free replay archive loading and summary/export helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.application.export.json_export import export_workspace_json, write_workspace_json
from archivetrust.application.current_state import CurrentStateService
from archivetrust.application.journal import Journal
from archivetrust.domain.telemetry.events import (
    CanonicalDocumentCreated,
    ReviewOutcomeRecorded,
    TelemetryEvent,
)
from archivetrust.infrastructure.storage.integrity import (
    default_hash_chain_manifest_path,
    default_hash_chain_sidecar_path,
    verify_hash_chain_manifest,
    verify_hash_chain_sidecar,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink


class ReplayArchiveError(ValueError):
    """Raised when a replay archive path cannot be resolved to domain telemetry."""


class ReplayIntegrity(BaseModel):
    model_config = ConfigDict(frozen=True)

    checked: bool
    ok: bool | None
    reason: str | None = None
    manifest_path: str | None = None


class ReplaySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    archive_path: str
    event_path: str
    event_count: int
    document_count: int
    evidence_count: int
    rejected_evidence_count: int
    observation_count: int
    canonical_observation_count: int
    canonical_document_count: int
    review_outcome_count: int
    integrity: ReplayIntegrity


@dataclass(frozen=True)
class _WorkspaceStub:
    id: str
    name: str


class ReplayArchive:
    """Read-only handle over a domain telemetry archive."""

    def __init__(self, archive_path: Path, event_path: Path, sink: FileTelemetrySink) -> None:
        self.archive_path = archive_path
        self.event_path = event_path
        self._sink = sink

    @classmethod
    def open(cls, path: str | Path) -> "ReplayArchive":
        archive_path = Path(path)
        event_path = _resolve_event_path(archive_path)
        return cls(
            archive_path=archive_path,
            event_path=event_path,
            sink=FileTelemetrySink(event_path),
        )

    def events(self) -> tuple[TelemetryEvent, ...]:
        return tuple(self._sink.all_events())

    def events_by_document(self) -> dict[str, tuple[TelemetryEvent, ...]]:
        grouped: dict[str, list[TelemetryEvent]] = defaultdict(list)
        for event in self.events():
            grouped[event.document_ref].append(event)
        return {document_ref: tuple(events) for document_ref, events in grouped.items()}

    def integrity(self) -> ReplayIntegrity:
        # Prefer the live append-only sidecar (WS5, `*.chain.jsonl`); fall back to the legacy
        # one-document manifest (`*.chain.json`) still carried by older replay archives.
        sidecar_path = default_hash_chain_sidecar_path(self.event_path)
        if sidecar_path.exists():
            result = verify_hash_chain_sidecar(self.event_path, sidecar_path)
            return ReplayIntegrity(
                checked=True,
                ok=result.ok,
                reason=result.reason,
                manifest_path=str(sidecar_path),
            )
        manifest_path = default_hash_chain_manifest_path(self.event_path)
        if not manifest_path.exists():
            return ReplayIntegrity(checked=False, ok=None, reason="manifest_missing")
        result = verify_hash_chain_manifest(self.event_path, manifest_path)
        return ReplayIntegrity(
            checked=True,
            ok=result.ok,
            reason=result.reason,
            manifest_path=str(manifest_path),
        )

    def summary(self) -> ReplaySummary:
        events = self.events()
        current_states = CurrentStateService(self._sink).documents(
            integrity_status=("passed" if self.integrity().ok else "failed")
        )
        replayed_states = {
            state.document_ref: Journal().replay(
                self._sink.events_for_document(state.document_ref)
            )
            for state in current_states
        }
        return ReplaySummary(
            archive_path=str(self.archive_path),
            event_path=str(self.event_path),
            event_count=len(events),
            document_count=len(current_states),
            evidence_count=sum(len(state.all_evidence()) for state in replayed_states.values()),
            rejected_evidence_count=sum(len(state.evidence_rejections()) for state in replayed_states.values()),
            observation_count=sum(len(state.all_observations()) for state in replayed_states.values()),
            canonical_observation_count=sum(
                len(state.slots) for state in current_states
            ),
            canonical_document_count=sum(
                1 for event in events if isinstance(event, CanonicalDocumentCreated)
            ),
            review_outcome_count=sum(1 for event in events if isinstance(event, ReviewOutcomeRecorded)),
            integrity=self.integrity(),
        )

    def export_json(self, *, workspace_id: str, workspace_name: str) -> str:
        workspace = _WorkspaceStub(id=workspace_id, name=workspace_name)
        return export_workspace_json(workspace=workspace, telemetry_source=self._sink)

    def write_export_json(
        self,
        output_path: str | Path,
        *,
        workspace_id: str,
        workspace_name: str,
        integrity_sidecar: bool = False,
    ) -> Path:
        workspace = _WorkspaceStub(id=workspace_id, name=workspace_name)
        return write_workspace_json(
            output_path,
            workspace=workspace,
            telemetry_source=self._sink,
            integrity_sidecar=integrity_sidecar,
        )


def summarize_archive(path: str | Path) -> ReplaySummary:
    return ReplayArchive.open(path).summary()


def export_archive_json(
    path: str | Path,
    output_path: str | Path,
    *,
    workspace_id: str = "replay-archive",
    workspace_name: str = "Replay archive",
    integrity_sidecar: bool = False,
) -> Path:
    return ReplayArchive.open(path).write_export_json(
        output_path,
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        integrity_sidecar=integrity_sidecar,
    )


def _resolve_event_path(path: Path) -> Path:
    if path.is_file():
        return path

    if not path.exists():
        raise ReplayArchiveError(f"Replay archive path does not exist: {path}")

    candidates = (
        path / "telemetry" / "events.jsonl",
        path / "events.jsonl",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise ReplayArchiveError(
        f"Replay archive must be a JSONL file or contain telemetry/events.jsonl or events.jsonl: {path}"
    )
