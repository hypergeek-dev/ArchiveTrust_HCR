"""Telemetry integrity checking (Constitution Article 34, `ARCHITECTURE_TELEMETRY_STANDARD.md`
§9.7) -- Research Telemetry's own self-check, and where the recursion this standard requires
terminates (alongside the immutable Archive Object, upstream).

Reads a document-scoped `TelemetrySink`, never mutates it -- this is a pure, read-only check
against already-persisted telemetry, exactly like `review/triage.py`'s deterministic projections
(Article 33's discipline: a *check* is not itself new knowledge). The resulting
`TelemetryIntegrityChecked` event is Research Telemetry's own record that the check was performed;
this function does not append it anywhere -- the caller decides which `ResearchTelemetrySink` (if
any) to record it in.
"""

from __future__ import annotations

from collections import Counter

from archivetrust.domain.research.events import TelemetryIntegrityChecked
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.sink import TelemetrySink


def check_telemetry_integrity(
    sink: TelemetrySink,
    *,
    document_refs: tuple[str, ...],
    corpus_ref: str,
) -> TelemetryIntegrityChecked:
    """Event-count and schema-version distribution across a named corpus of `document_ref`s --
    the two facts §9.3 specifies, computed from `sink` alone, never guessed.
    """
    events = [event for ref in document_refs for event in sink.events_for_document(ref)]
    distribution = Counter(str(event.schema_version) for event in events)

    gaps_found: list[str] = []
    for ref in document_refs:
        if not any(event.document_ref == ref for event in events):
            gaps_found.append(f"no events recorded for document_ref={ref!r}")

    return TelemetryIntegrityChecked(
        event_id=new_id("event"),
        corpus_ref=corpus_ref,
        sink_ref=repr(sink),
        event_count=len(events),
        schema_version_distribution=dict(distribution),
        gaps_found=tuple(gaps_found),
    )
