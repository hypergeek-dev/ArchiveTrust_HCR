"""Phase 10 (Constitution Articles 26-29, 34): `EvidenceExplorerViewModel` -- the Architect
Client's chain-of-custody surface -- surfaces `basis_code`, `mapping_table_entry_id`, structurally-
excluded candidate pairs, and a cross-reference to any Research Telemetry finding examining this
document's Workspace, unmasked (unlike Review Center, per Phase 7's own disclosed decision).
"""

from __future__ import annotations

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.research.events import AuditConducted
from archivetrust.domain.telemetry.events import ObservationMapped, ProvenanceContextEstablished
from archivetrust.infrastructure.storage.research_telemetry_sink import FileResearchTelemetrySink
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.presentation.evidence_explorer_viewmodel import EvidenceExplorerViewModel

from tests.review._helpers import emit_slot, heading

DOC = "doc-1"


def test_document_trace_surfaces_reconciliation_basis_code_and_mapping_table_entry_id():
    sink = InMemoryTelemetrySink()
    canonical = emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter I"))),
        classification=ComparisonClassification.CONTESTED,
    )
    docling_ref = canonical.contributing_observations[0]
    sink.append(
        ObservationMapped(
            event_id="event_mapped", document_ref=DOC,
            observation_id=docling_ref.observation_id, source_evidence_ids=(),
            ontology_version=1, mapping_table_entry_id="docling:section_header",
            mapping_table_version=1,
        )
    )

    vm = EvidenceExplorerViewModel(sink)
    trace = vm.document_trace(DOC)

    assert len(trace.canonical_traces) == 1
    # emit_slot's fixture doesn't set reconciliation_basis_code -- confirms an honest None, not a
    # guessed value, for data that predates it.
    assert trace.canonical_traces[0].reconciliation_basis_code is None
    contributor = next(
        c for c in trace.canonical_traces[0].contributors
        if c.observation_id == docling_ref.observation_id
    )
    assert contributor.mapping_table_entry_id == "docling:section_header"


def test_document_trace_surfaces_excluded_candidate_pairs():
    from archivetrust.domain.alignment.service import ClusteringAlignmentService
    from archivetrust.domain.alignment.telemetry import alignment_events
    from archivetrust.domain.comparison.clustering import ClusteringBasisCode
    from archivetrust.domain.comparison.policy import ReconciliationPolicy
    from tests.domain.alignment._helpers import evidence, observation, paragraph

    sink = InMemoryTelemetrySink()
    ev_a1 = evidence("docling", "Fragment A", None)
    ev_a2 = evidence("docling", "Fragment B", None)
    obs_a1 = observation("docling", paragraph("Fragment A"), ev_a1)
    obs_a2 = observation("docling", paragraph("Fragment B"), ev_a2)
    evidence_by_id = {ev_a1.evidence_id: ev_a1, ev_a2.evidence_id: ev_a2}
    result = ClusteringAlignmentService().align(
        (obs_a1, obs_a2), evidence_by_id, ReconciliationPolicy(policy_version=1)
    )
    for event in alignment_events(result, document_ref=DOC):
        sink.append(event)

    vm = EvidenceExplorerViewModel(sink)
    trace = vm.document_trace(DOC)

    assert len(trace.excluded_pairs) == 1
    assert trace.excluded_pairs[0].basis_code == ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER.value
    assert trace.excluded_pairs[0].structural is True


def test_document_trace_finds_no_examining_findings_when_registry_is_absent(tmp_path, monkeypatch):
    import archivetrust.presentation.evidence_explorer_viewmodel as module

    monkeypatch.setattr(module, "_RESEARCH_TELEMETRY_PATH", tmp_path / "does_not_exist.jsonl")
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    vm = EvidenceExplorerViewModel(sink)
    trace = vm.document_trace(DOC)
    assert trace.examining_findings == ()


def test_document_trace_cross_references_a_matching_research_telemetry_finding(tmp_path, monkeypatch):
    import archivetrust.presentation.evidence_explorer_viewmodel as module

    registry_path = tmp_path / "research_telemetry.jsonl"
    monkeypatch.setattr(module, "_RESEARCH_TELEMETRY_PATH", registry_path)
    FileResearchTelemetrySink(registry_path).append(
        AuditConducted(
            event_id="event_a1", corpus_ref="workspace:ws-42", audit_id="a1",
            title="Test Finding", claim="claim", verdict="verdict",
        )
    )

    sink = InMemoryTelemetrySink()
    sink.append(
        ProvenanceContextEstablished.create(
            document_ref=DOC, ontology_version=1, git_commit=None,
            configuration_hash="hash", configuration_snapshot_reference="[]",
            machine_identifier=None, workspace_identifier="ws-42",
            reconciliation_policy_version=1, capability_matrix_version=1,
            confidence_policy_version=1,
        )
    )
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )

    vm = EvidenceExplorerViewModel(sink)
    trace = vm.document_trace(DOC)
    assert trace.examining_findings == ("Test Finding",)
