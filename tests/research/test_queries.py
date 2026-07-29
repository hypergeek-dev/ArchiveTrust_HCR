"""Phase 11 (`ROADMAP_TELEMETRY_STANDARD.md`): the query library's document-scoped functions,
plus the six 2026-07-14 audits re-expressed as one query each against the checked-in Research
Telemetry registry -- permanent regression fixtures per the roadmap's explicit requirement. If a
future telemetry-schema change breaks one of these, that is a detected regression against a real
historical finding, not merely a unit test.
"""

from __future__ import annotations

import pytest

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.ontology.payloads import ParagraphPayload
from archivetrust.domain.telemetry.events import CandidateExcluded, KnowledgeDiscarded
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.research.queries import (
    find_audits_examining,
    is_finding_superseded,
    latest_finding_for,
    structurally_excluded_provider_pairs,
    supersession_chain,
    why_did_canonical_change,
    why_was_confidence_assigned,
    why_was_evidence_discarded,
    why_was_observation_ignored,
    why_was_provenance_retained,
    why_wasnt_review_generated,
    why_wasnt_this_clustered,
)
from archivetrust.review.triage import TriageClassification

from tests.review._helpers import emit_slot, heading
from scripts.register_research_telemetry import OUTPUT_PATH, main as register_main

DOC = "doc-1"


@pytest.fixture(scope="module", autouse=True)
def _registered_corpus():
    if not OUTPUT_PATH.exists():
        register_main()
    return OUTPUT_PATH


# -- Document-scoped questions ----------------------------------------------------------------


def test_why_was_observation_ignored_reports_unaligned_reason():
    from archivetrust.application.journal import Journal
    from archivetrust.domain.telemetry.events import ObservationLeftUnaligned

    sink = InMemoryTelemetrySink()
    sink.append(
        ObservationLeftUnaligned(
            event_id="e1", document_ref=DOC, observation_id="obs-1",
            comparison_group_id="group-1", alignment_attempt_id="attempt-1",
            reason="no plausible match",
        )
    )
    state = Journal().replay(sink.all_events())
    assert why_was_observation_ignored(state, "obs-1") == "no plausible match"


def test_why_was_observation_ignored_reports_none_when_not_ignored():
    from archivetrust.application.journal import Journal

    state = Journal().replay(InMemoryTelemetrySink().all_events())
    assert why_was_observation_ignored(state, "obs-absent") is None


def test_why_wasnt_review_generated_distinguishes_not_eligible_from_withheld():
    sink = InMemoryTelemetrySink()
    corroborated = emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter 1"))),
        classification=ComparisonClassification.CORROBORATED,
    )
    single_source = emit_slot(
        sink, document_ref=DOC, canonical_payload=ParagraphPayload(text="Body text"),
        provider_payloads=(("docling", ParagraphPayload(text="Body text")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    assert why_wasnt_review_generated(corroborated) == TriageClassification.NOT_ELIGIBLE
    assert why_wasnt_review_generated(single_source) == TriageClassification.WITHHELD_BY_POLICY


def test_why_was_evidence_discarded_reports_knowledge_discarded_reason():
    events = [
        KnowledgeDiscarded(
            event_id="e1", document_ref=DOC, discarded_ids=("ev-1",),
            reason="superseded by a corrected re-extraction",
        )
    ]
    assert why_was_evidence_discarded(events, "ev-1") == "superseded by a corrected re-extraction"
    assert why_was_evidence_discarded(events, "ev-absent") is None


def test_structurally_excluded_provider_pairs_surfaces_only_structural_exclusions():
    from archivetrust.application.journal import Journal
    from archivetrust.domain.comparison.clustering import ClusteringBasisCode

    sink = InMemoryTelemetrySink()
    sink.append(
        CandidateExcluded(
            event_id="e1", document_ref=DOC, candidate_observation_id="a",
            compared_against_observation_id="b", excluding_mechanism="capability_matrix",
            basis_code=ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER, structural=True,
        )
    )
    sink.append(
        CandidateExcluded(
            event_id="e2", document_ref=DOC, candidate_observation_id="c",
            compared_against_observation_id="d", excluding_mechanism="threshold",
            basis_code=ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER, structural=False,
        )
    )
    state = Journal().replay(sink.all_events())
    result = structurally_excluded_provider_pairs(state)
    assert len(result) == 1
    assert result[0].candidate_observation_id == "a"


def test_why_did_canonical_change_and_provenance_retained_reflect_correction():
    sink = InMemoryTelemetrySink()
    canonical = emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter 1"))),
        classification=ComparisonClassification.CORROBORATED,
    )
    from archivetrust.application.journal import Journal

    state = Journal().replay(sink.all_events())
    history = why_did_canonical_change(state, canonical.semantic_slot_id)
    assert history[-1].canonical_observation_id == canonical.canonical_observation_id
    provenance = why_was_provenance_retained(state, canonical.canonical_observation_id)
    assert len(provenance) > 0


def test_why_was_confidence_assigned_reports_none_for_unknown_subject():
    from archivetrust.application.journal import Journal

    state = Journal().replay(InMemoryTelemetrySink().all_events())
    assert why_was_confidence_assigned(state, "unknown-subject") == ()


def test_why_wasnt_this_clustered_reports_none_when_no_attempt_recorded():
    from archivetrust.application.journal import Journal

    state = Journal().replay(InMemoryTelemetrySink().all_events())
    assert why_wasnt_this_clustered(state, "slot-absent") is None


# -- Corpus-scoped questions: the six 2026-07-14 audits, one query each ------------------------


def test_review_packet_integrity_audit_reproduced_by_query(_registered_corpus):
    matches = find_audits_examining(_registered_corpus, "Nonsensical review items")
    assert len(matches) == 1
    finding = matches[0]
    assert finding.audit_id == "review-packet-integrity-2026-07-14"
    assert "84-93%" in finding.verdict


def test_semantic_alignment_audit_reproduced_by_query(_registered_corpus):
    matches = find_audits_examining(_registered_corpus, "522 review packets")
    assert len(matches) == 1
    finding = matches[0]
    assert finding.audit_id == "semantic-alignment-2026-07-14"
    assert finding.population_size == 522
    assert "67%" in finding.verdict


def test_human_reviewability_audit_reproduced_by_query(_registered_corpus):
    matches = find_audits_examining(_registered_corpus, "actually reviewable")
    assert len(matches) == 1
    finding = matches[0]
    assert finding.audit_id == "human-reviewability-2026-07-14"
    assert "4 of 6" in finding.verdict


def test_semantic_contract_audit_reproduced_by_query(_registered_corpus):
    finding = latest_finding_for(_registered_corpus, "meta-audit")
    assert finding is not None
    assert finding.audit_id == "semantic-contract-2026-07-14"
    assert "tesseract_layoutparser" in finding.verdict


def test_observation_typing_prevalence_audit_reproduced_by_query(_registered_corpus):
    matches = find_audits_examining(_registered_corpus, "full-corpus measurement")
    assert len(matches) == 1
    finding = matches[0]
    assert finding.audit_id == "observation-typing-prevalence-2026-07-14"
    assert "100% prevalence" in finding.verdict


def test_adversarial_audit_falsifies_rcdp_by_supersession_chain_query(_registered_corpus):
    chain = supersession_chain(_registered_corpus, "adversarial-architecture-2026-07-14")
    assert [event.audit_id for event in chain] == [
        "review-packet-integrity-2026-07-14",
        "semantic-alignment-2026-07-14",
        "human-reviewability-2026-07-14",
        "semantic-contract-2026-07-14",
        "observation-typing-prevalence-2026-07-14",
        "rcdp-remaining-clustering-defect-2026-07-14",
        "adversarial-architecture-2026-07-14",
    ]
    rcdp = is_finding_superseded(_registered_corpus, "rcdp-remaining-clustering-defect-2026-07-14")
    assert rcdp is not None
    assert rcdp.audit_id == "adversarial-architecture-2026-07-14"
    assert "FALSIFIED" in rcdp.verdict
    assert "60%" in rcdp.verdict
