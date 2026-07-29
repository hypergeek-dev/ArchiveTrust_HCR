"""Phase 7: `ReviewPacket`/`CandidateView` carry `reconciliation_basis_code` (Article 26/28) and
`mapping_table_entry_id` (Article 28) in the data model, and `ReviewService.triage_summary`
(Article 33) answers "why wasn't this reviewed" without a persisted event.
"""

from __future__ import annotations

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.ontology.payloads import TablePayload
from archivetrust.domain.telemetry.events import ObservationMapped
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.review.service import ReviewService
from archivetrust.review.triage import TriageClassification

from tests.review._helpers import emit_slot, heading

DOC = "doc_1"
ARCHIVE = "archive_object_1"


def _service(sink: InMemoryTelemetrySink) -> ReviewService:
    return ReviewService(
        telemetry_source=sink, telemetry_sink=sink, interaction_sink=InMemoryReviewInteractionSink()
    )


def test_assembled_packet_carries_reconciliation_basis_code():
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("X"),
        provider_payloads=(("docling", heading("X")), ("qwen", heading("Y"))),
        classification=ComparisonClassification.CONTESTED,
    )
    # emit_slot's fixture doesn't set reconciliation_basis_code (defaults None) -- confirms the
    # packet honestly reflects that, never guessing a code for data that predates it.
    (packet,) = _service(sink).open_document(document_ref=DOC, archive_object_ref=ARCHIVE)
    assert packet.reconciliation_basis_code is None


def test_assembled_candidate_carries_mapping_table_entry_id_when_recorded():
    sink = InMemoryTelemetrySink()
    canonical = emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("Chapter 1"),
        provider_payloads=(("docling", heading("Chapter 1")), ("qwen", heading("Chapter I"))),
        classification=ComparisonClassification.CONTESTED,
    )
    docling_ref, qwen_ref = canonical.contributing_observations
    # Directly record an ObservationMapped for one real contributing Observation id, proving
    # assemble_packet's per-candidate lookup reads it through; the other is left unrecorded,
    # confirming it stays None rather than being guessed (Article 18's discipline).
    sink.append(
        ObservationMapped(
            event_id="event_mapped",
            document_ref=DOC,
            observation_id=docling_ref.observation_id,
            source_evidence_ids=(),
            ontology_version=1,
            mapping_table_entry_id="docling:section_header",
            mapping_table_version=1,
        )
    )

    (packet,) = _service(sink).open_document(document_ref=DOC, archive_object_ref=ARCHIVE)

    by_id = {c.observation_id: c for c in packet.candidates}
    assert by_id[docling_ref.observation_id].mapping_table_entry_id == "docling:section_header"
    assert by_id[qwen_ref.observation_id].mapping_table_entry_id is None


def test_assembled_table_candidate_carries_neutral_structure_summary():
    sink = InMemoryTelemetrySink()
    canonical = emit_slot(
        sink,
        document_ref=DOC,
        canonical_payload=TablePayload(row_count=2, column_count=2, caption_text="Ledger"),
        provider_payloads=(
            ("docling", TablePayload(row_count=2, column_count=2, caption_text="Ledger")),
            ("qwen", TablePayload(row_count=3, column_count=2, caption_text="Ledger")),
        ),
        classification=ComparisonClassification.CONTESTED,
    )
    docling_ref, qwen_ref = canonical.contributing_observations

    (packet,) = _service(sink).open_document(document_ref=DOC, archive_object_ref=ARCHIVE)

    by_id = {c.observation_id: c for c in packet.candidates}
    assert by_id[docling_ref.observation_id].structure_summary == {
        "kind": "table",
        "rows": 2,
        "columns": 2,
        "caption": "Ledger",
    }
    assert by_id[qwen_ref.observation_id].structure_summary["rows"] == 3
    assert by_id[docling_ref.observation_id].value is None


def test_triage_summary_counts_every_known_slot_by_classification():
    sink = InMemoryTelemetrySink()
    # Contested: queued.
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("Contested"),
        provider_payloads=(("docling", heading("Contested")), ("qwen", heading("Other"))),
        classification=ComparisonClassification.CONTESTED,
    )
    # Corroborated, high confidence: not eligible.
    emit_slot(
        sink, document_ref=DOC, canonical_payload=heading("Fine"),
        provider_payloads=(("docling", heading("Fine")), ("qwen", heading("Fine"))),
        classification=ComparisonClassification.CORROBORATED, canonical_confidence=0.95,
    )

    summary = _service(sink).triage_summary(DOC)

    assert summary[TriageClassification.QUEUED] == 1
    assert summary[TriageClassification.NOT_ELIGIBLE] == 1
    assert summary[TriageClassification.WITHHELD_BY_POLICY] == 0
    assert sum(summary.values()) == 2
