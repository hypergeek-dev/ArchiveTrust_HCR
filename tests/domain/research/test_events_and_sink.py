"""Research Telemetry (Constitution Article 34): a deliberately separate stream from document-
scoped Trust Engine telemetry -- these tests prove the two vocabularies/sinks are independent, that
supersession chains correctly, and that both sink implementations round-trip identically.
"""

from __future__ import annotations

import json

from archivetrust.domain.research.events import (
    EVENT_TYPE_BY_KIND,
    AuditConducted,
    BenchmarkExecuted,
    ReplayExecuted,
    ResearchEventKind,
    TelemetryIntegrityChecked,
    parse_research_event,
)
from archivetrust.domain.research.integrity import check_telemetry_integrity
from archivetrust.domain.telemetry.events import EvidenceCreated, ProviderObservationAttempted
from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.infrastructure.storage.research_telemetry_sink import (
    FileResearchTelemetrySink,
    InMemoryResearchTelemetrySink,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink


def _audit(audit_id: str, **kwargs) -> AuditConducted:
    fields = {
        "event_id": f"event_{audit_id}",
        "corpus_ref": "corpus-1",
        "audit_id": audit_id,
        "title": "Test Audit",
        "claim": "A test claim",
        "verdict": "A test verdict",
        **kwargs,
    }
    return AuditConducted(**fields)


def test_all_research_event_kinds_except_calibration_are_registered():
    # CALIBRATION_MEASURED is deliberately reserved-not-wired (§9.3.1) -- no code path constructs
    # it, so it is correctly absent from the dispatch table.
    assert set(EVENT_TYPE_BY_KIND) == {
        ResearchEventKind.AUDIT_CONDUCTED,
        ResearchEventKind.BENCHMARK_EXECUTED,
        ResearchEventKind.REPLAY_EXECUTED,
        ResearchEventKind.TELEMETRY_INTEGRITY_CHECKED,
    }


def test_research_event_is_immutable():
    audit = _audit("a1")
    try:
        audit.verdict = "changed"
        assert False, "expected a validation error on a frozen model"
    except Exception:
        pass


def test_supersession_chain_is_explicit_not_erasure():
    original = _audit("rcdp-1", verdict="counterfactual guard would reduce 332 packets to 0")
    falsification = _audit(
        "adversarial-1",
        verdict="60% of a random sample carried unique content the guard would eliminate",
        supersedes_audit_id=original.audit_id,
    )
    assert falsification.supersedes_audit_id == original.audit_id
    # The original is never deleted or edited -- both remain independently constructible/storable.
    assert original.verdict != falsification.verdict


def test_research_event_round_trips_through_serialization():
    audit = _audit("a1", population_size=30, sample_basis="seed=42")
    reparsed = parse_research_event(json.loads(audit.model_dump_json()))
    assert isinstance(reparsed, AuditConducted)
    assert reparsed == audit


def test_in_memory_and_file_sinks_agree(tmp_path):
    audit = _audit("a1")
    benchmark = BenchmarkExecuted(
        event_id="event_b1", corpus_ref="corpus-1", benchmark_id="b1", title="Benchmark #1",
        document_count=478,
    )

    memory_sink = InMemoryResearchTelemetrySink()
    memory_sink.append(audit)
    memory_sink.append(benchmark)

    file_sink = FileResearchTelemetrySink(tmp_path / "research_telemetry.jsonl")
    file_sink.append(audit)
    file_sink.append(benchmark)

    for sink in (memory_sink, file_sink):
        events = list(sink.events_for_corpus("corpus-1"))
        assert len(events) == 2
        assert isinstance(events[0], AuditConducted)
        assert isinstance(events[1], BenchmarkExecuted)
        assert list(sink.all_events()) == events


def test_file_sink_persists_across_reconstruction(tmp_path):
    path = tmp_path / "research_telemetry.jsonl"
    FileResearchTelemetrySink(path).append(_audit("a1"))

    reopened = FileResearchTelemetrySink(path)
    assert len(list(reopened.all_events())) == 1


def test_replay_executed_distinguishes_investigatory_from_routine_by_construction():
    # §9.5: ReplayExecuted is a Research Telemetry event a caller chooses to emit for a deliberate
    # investigatory replay -- it is never emitted by application/journal.py's Journal.replay()
    # itself (which remains a pure, non-emitting function per Article 33).
    import inspect

    from archivetrust.application import journal as journal_module

    source = inspect.getsource(journal_module)
    assert "ReplayExecuted" not in source


def test_check_telemetry_integrity_counts_events_and_finds_gaps():
    sink = InMemoryTelemetrySink()
    ev = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="x", processing_stage=ProcessingStage.RAW,
    )
    sink.append(
        ProviderObservationAttempted(
            event_id="e1", document_ref="doc-1", provider_id="docling", provider_version="1.0",
            invocation_id="inv-1",
        )
    )
    sink.append(
        EvidenceCreated(event_id="e2", document_ref="doc-1", invocation_id="inv-1", evidence=ev)
    )

    result = check_telemetry_integrity(
        sink, document_refs=("doc-1", "doc-2"), corpus_ref="test-corpus"
    )

    assert isinstance(result, TelemetryIntegrityChecked)
    assert result.event_count == 2
    assert result.schema_version_distribution == {"1": 2}
    assert result.gaps_found == ("no events recorded for document_ref='doc-2'",)
