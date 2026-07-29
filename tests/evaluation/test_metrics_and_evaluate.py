"""Release WS7: metric correctness on known small examples, and the evaluator's behavior over
real (fixture-emitted) telemetry — misses, canonical-vs-provider pairing, verdicts.
"""

from __future__ import annotations

import pytest

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.evaluation.evaluate import evaluate
from archivetrust.evaluation.ground_truth import (
    FileGroundTruthStore,
    GroundTruthAnnotation,
    VerificationStatus,
)
from archivetrust.evaluation.metrics import compare_text, levenshtein, normalize_text
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from tests.review._helpers import emit_slot, heading


def test_levenshtein_known_values():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "") == 3


def test_normalize_collapses_whitespace_but_keeps_case():
    assert normalize_text("  Beslut\n  från   KS ") == "Beslut från KS"
    assert normalize_text("BESLUT") != normalize_text("beslut")


def test_compare_text_known_values():
    result = compare_text("Beslut från KS", "Beslut fran KS")
    assert not result.exact_match
    assert result.character_error_rate == pytest.approx(1 / 14)
    assert result.word_error_rate == pytest.approx(1 / 3)

    perfect = compare_text("Innehåll", "Innehåll")
    assert perfect.exact_match
    assert perfect.character_error_rate == 0.0
    assert perfect.normalized_similarity == 1.0


def _annotation(annotation_id: str, document_ref: str, text: str):
    return GroundTruthAnnotation(
        annotation_id=annotation_id,
        archive_object_ref=document_ref,
        content_hash="hash",
        page=1,
        field="page1_heading",
        observation_type="heading",
        text=text,
        annotator="tester",
        method="fixture",
        source="test",
        created_at="2026-07-16T00:00:00+00:00",
        verification_status=VerificationStatus.VERIFIED,
        legal_basis="test campaign authorization",
        sampling_basis="complete fixture population",
    )


def test_evaluate_scores_providers_and_canonical_with_misses(tmp_path):
    sink = InMemoryTelemetrySink()
    # docling reads the heading correctly; tesseract garbles it; canonical selects docling's.
    emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("Innehållsförteckning"),
        provider_payloads=(
            ("docling", heading("Innehållsförteckning")),
            ("tesseract_layoutparser", heading("lnnehallsforteckning")),
        ),
        classification=ComparisonClassification.CONTESTED,
    )
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    store.append(_annotation("gt_1", "doc1", "Innehållsförteckning"))

    result = evaluate(store=store, telemetry_source=sink)

    by_system = {s.system: s for s in result.summaries}
    assert by_system["docling"].exact_matches == 1
    assert by_system["docling"].mean_cer_matched == 0.0
    assert by_system["tesseract_layoutparser"].exact_matches == 0
    assert by_system["tesseract_layoutparser"].mean_cer_matched > 0
    assert by_system["canonical"].exact_matches == 1
    assert result.central_result["verdict"] in (
        "sample_insufficient",  # a single field is honestly insufficient
    )
    assert result.central_result["paired_fields"] == 1


def test_evaluate_reports_missing_type_as_miss_not_empty_string(tmp_path):
    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    # Annotate a paragraph — no provider produced any paragraph observation.
    paragraph = _annotation("gt_1", "doc1", "Beslut från kommunstyrelsen.")
    paragraph = paragraph.model_copy(update={"field": "page1_first_paragraph", "observation_type": "paragraph"})
    store.append(paragraph)

    result = evaluate(store=store, telemetry_source=sink)

    docling = next(s for s in result.summaries if s.system == "docling")
    assert docling.misses == 1
    assert docling.mean_cer_matched is None
    assert docling.mean_cer_all == 1.0


def test_evaluate_excludes_illegible_and_disputed(tmp_path):
    from archivetrust.evaluation.ground_truth import AdjudicationStatus

    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    store.append(_annotation("gt_ok", "doc1", "A"))
    store.append(
        _annotation("gt_ill", "doc1", "x").model_copy(update={"text": None, "illegible": True})
    )
    store.append(
        _annotation("gt_disp", "doc1", "B").model_copy(
            update={"adjudication_status": AdjudicationStatus.DISPUTED}
        )
    )

    result = evaluate(store=store, telemetry_source=sink)
    assert result.annotations_total == 3
    assert result.annotations_evaluable == 1
    assert result.annotations_illegible == 1
    assert result.annotations_disputed == 1


def test_evaluate_refuses_unverified_and_ai_assisted_records(tmp_path):
    from archivetrust.evaluation.ground_truth import AnnotatorKind

    sink = InMemoryTelemetrySink()
    emit_slot(
        sink,
        document_ref="doc1",
        canonical_payload=heading("A"),
        provider_payloads=(("docling", heading("A")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    store.append(
        _annotation("unverified", "doc1", "A").model_copy(
            update={"verification_status": VerificationStatus.UNVERIFIED}
        )
    )
    store.append(
        _annotation("ai", "doc1", "A").model_copy(
            update={"annotator_kind": AnnotatorKind.AI_ASSISTED}
        )
    )

    result = evaluate(store=store, telemetry_source=sink)

    assert result.annotations_total == 2
    assert result.annotations_evaluable == 0
    assert result.annotations_unverified == 1
    assert result.annotations_ai_excluded == 1
    assert result.field_scores == ()
