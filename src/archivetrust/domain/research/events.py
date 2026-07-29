"""Research Telemetry event vocabulary (Constitution Article 34,
`ARCHITECTURE_TELEMETRY_STANDARD.md` §9).

`ResearchEvent` is deliberately **not** a subclass of `domain.telemetry.events.TelemetryEvent` --
the two hierarchies must never be type-confused, since a Research Telemetry event describes
self-knowledge about the system's own reasoning across an arbitrary `corpus_ref`, never one Archive
Object's knowledge evolution (which is exactly what `TelemetryEvent.document_ref` exists to scope).
This module was added 2026-07-14 after a Phase 9 implementation attempt found this standard's own
§7 had, by oversight, proposed folding these three event kinds into the closed, document-scoped
`TelemetryEventKind` set -- corrected per the identical precedent ROADMAP.md §12.1 already
established for Acquisition telemetry (a separate stream, never diluting §12's closure discipline).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

CURRENT_RESEARCH_SCHEMA_VERSION = 1
"""Independent of the document-scoped stream's `schema_version` counter (Article 19's discipline,
applied a third time -- Research Telemetry's own serialization shape versions independently)."""


class ResearchEventKind(str, Enum):
    """The Research Telemetry event set -- exhaustive by design, mirroring
    `TelemetryEventKind`'s own closure discipline for a second, independent vocabulary.
    """

    AUDIT_CONDUCTED = "AuditConducted"
    BENCHMARK_EXECUTED = "BenchmarkExecuted"
    REPLAY_EXECUTED = "ReplayExecuted"
    """Emitted only for a deliberate, investigatory replay (an audit's own reconstruction
    exercise) -- never for the routine, constant replay production code performs on every document
    open, which remains governed by Article 33 and must never emit anything (§9.5)."""
    TELEMETRY_INTEGRITY_CHECKED = "TelemetryIntegrityChecked"
    CALIBRATION_MEASURED = "CalibrationMeasured"
    """Reserved, not yet wired -- `domain/calibration/` emits no telemetry of any kind today
    (verified 2026-07-14). Named now so this vocabulary needs no second amendment later, mirroring
    the same discipline Article 28 already established for `ObservationMapped`'s once-unwired
    fields; no code path constructs this event kind yet."""


class ResearchEvent(BaseModel):
    """Abstract base for all Research Telemetry events. `corpus_ref` scopes every event to
    whatever corpus/sample it examined -- a Workspace id, a named fixture corpus, or a documented
    dataset identifier -- never a single `document_ref` (Article 34).
    """

    model_config = ConfigDict(frozen=True)

    kind: ResearchEventKind
    event_id: str
    corpus_ref: str
    commit_ref: str | None = None
    """The git commit the examined code was at, if known -- never guessed (Article 18's discipline,
    applied here): lets a later query detect a finding's commit predates a change to the file it
    examined (architecture document §11)."""
    recorded_at: str | None = None
    schema_version: int = CURRENT_RESEARCH_SCHEMA_VERSION


class AuditConducted(ResearchEvent):
    """One audit's registered finding (Constitution Article 29). `supersedes_audit_id` chains
    findings exactly as `CanonicalObservation.supersedes` chains canonical versions (Article 15) --
    a superseded finding is never deleted or edited, only ever pointed to by whatever superseded it.
    """

    kind: ResearchEventKind = ResearchEventKind.AUDIT_CONDUCTED

    audit_id: str
    title: str
    claim: str
    verdict: str
    population_size: int | None = None
    sample_basis: str | None = None
    supersedes_audit_id: str | None = None


class BenchmarkExecuted(ResearchEvent):
    kind: ResearchEventKind = ResearchEventKind.BENCHMARK_EXECUTED

    benchmark_id: str
    title: str
    document_count: int | None = None
    metrics_ref: str | None = None
    supersedes_benchmark_id: str | None = None


class ReplayExecuted(ResearchEvent):
    kind: ResearchEventKind = ResearchEventKind.REPLAY_EXECUTED

    replay_id: str
    events_replayed: int | None = None
    divergence_found: bool
    divergence_detail: str | None = None


class TelemetryIntegrityChecked(ResearchEvent):
    """Research Telemetry's own self-check (§5/§9.7's recursion, terminating here alongside the
    immutable Archive Object) -- event-count and schema-version distribution for a named,
    document-scoped corpus, checked by replay/audit tooling, never by a sink appending an event
    about its own append.
    """

    kind: ResearchEventKind = ResearchEventKind.TELEMETRY_INTEGRITY_CHECKED

    sink_ref: str
    event_count: int
    schema_version_distribution: dict[str, int]
    gaps_found: tuple[str, ...] = ()


EVENT_TYPE_BY_KIND: dict[ResearchEventKind, type[ResearchEvent]] = {
    event_cls.model_fields["kind"].default: event_cls
    for event_cls in (
        AuditConducted,
        BenchmarkExecuted,
        ReplayExecuted,
        TelemetryIntegrityChecked,
    )
}
"""`CalibrationMeasured` is deliberately absent -- reserved in `ResearchEventKind` but not yet
constructible, since no code path emits it (see that member's own docstring)."""


def parse_research_event(data: dict[str, Any]) -> ResearchEvent:
    """Reconstructs the correct event subclass from a plain dict, keyed by `kind` -- the same
    polymorphism mechanism `domain.telemetry.events.parse_event` uses for the document-scoped
    stream, applied here to a deliberately independent vocabulary.
    """
    kind = ResearchEventKind(data["kind"])
    event_cls = EVENT_TYPE_BY_KIND[kind]
    return event_cls.model_validate(data)
