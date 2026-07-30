from __future__ import annotations

from dataclasses import dataclass

from archivetrust.presentation.display_names import (
    LEGACY_METHOD_SUFFIX,
    basis_code_label,
    benchmark_status_label,
    canonicalization_strategy_label,
    capability_label,
    document_label,
    is_legacy_method,
    method_label,
    method_run_outcome_label,
    profile_label,
    provider_label,
    result_stage_label,
    runtime_label,
    short_ref,
    source_kind_label,
)
from archivetrust.runtime.provider_profiles import ProviderProfileName


@dataclass(frozen=True)
class ArchiveObjectStub:
    original_filename: str


def test_the_three_real_htr_methods_have_researcher_facing_names() -> None:
    assert method_label("satrn") == "SATRN (Riksarkivet)"
    assert method_label("florence2_htr") == "Florence-2 (vlm-htr line OCR)"
    assert method_label("transkribus_swedish_lion_1") == "Transkribus Swedish Lion I"
    assert profile_label(ProviderProfileName.MAXIMUM_QUALITY) == "Maximum quality"


def test_method_label_keys_match_the_adapters_own_method_id_constants() -> None:
    """A label keyed on an id no adapter declares would silently never be used."""
    from archivetrust.providers.florence2_htr.adapter import METHOD_ID as FLORENCE_ID
    from archivetrust.providers.satrn.adapter import METHOD_ID as SATRN_ID
    from archivetrust.providers.transkribus.adapter import METHOD_ID as TRANSKRIBUS_ID

    for method_id in (SATRN_ID, FLORENCE_ID, TRANSKRIBUS_ID):
        assert LEGACY_METHOD_SUFFIX not in method_label(method_id)
        assert method_label(method_id) != method_id  # a real label, not the raw id


def test_retired_ocr_providers_are_labeled_as_retired_not_erased() -> None:
    """docs/htr-domain-design.md section 8: legacy runs stay readable and clearly labeled, so
    stored telemetry naming a deleted provider must not render as a peer HTR method."""
    for legacy_id in ("docling", "tesseract_layoutparser", "paddleocr-vl", "surya"):
        assert is_legacy_method(legacy_id) is True
        assert method_label(legacy_id).endswith(LEGACY_METHOD_SUFFIX)

    assert method_label("tesseract_layoutparser") == (
        f"Tesseract + LayoutParser{LEGACY_METHOD_SUFFIX}"
    )
    assert is_legacy_method("satrn") is False


def test_provider_label_still_resolves_for_the_retained_provider_admin_surfaces() -> None:
    """`providers/base.py`'s ProviderAdapter was deliberately not deleted, so the legacy provider
    management pages still call this name; it must stay an exact alias of `method_label`."""
    assert provider_label("satrn") == method_label("satrn")
    assert provider_label("docling") == method_label("docling")


def test_unknown_identifiers_are_humanized_without_leaking_snake_case() -> None:
    assert method_label("new_reader_engine") == "New Reader Engine"
    assert source_kind_label("folder_watch") == "Watched folder"
    assert runtime_label(None) == "-"
    assert method_label("htr_pipeline") == "HTR Pipeline"  # HTR stays uppercase


def test_htr_vocabulary_labels_cover_the_stage_and_status_enums() -> None:
    assert result_stage_label("raw") == "Raw output"
    assert result_stage_label("normalized") == "Normalized"
    assert result_stage_label("canonical") == "Canonical"
    assert canonicalization_strategy_label("human_approved_selection") == (
        "Human-approved selection"
    )
    assert benchmark_status_label("requires_adjudication") == "Requires adjudication"
    assert method_run_outcome_label("no_output") == "Ran, produced no text"
    assert capability_label("external_upload_required") == "Requires external upload"


def test_every_enum_member_of_the_domain_enums_has_an_explicit_label() -> None:
    """A missing entry would fall through to `_humanize`, which produces a plausible-looking but
    unreviewed string -- catch that here rather than in a screenshot."""
    from archivetrust.domain.canonical.result import CanonicalizationStrategy
    from archivetrust.presentation.htr_comparison_viewmodel import STAGE_ORDER
    from archivetrust.review.blind_review.agreement import BenchmarkStatus

    for strategy in CanonicalizationStrategy:
        assert canonicalization_strategy_label(strategy) != strategy.value
    for status in BenchmarkStatus:
        assert benchmark_status_label(status) != status.value
    for stage in STAGE_ORDER:
        assert result_stage_label(stage) != stage


def test_every_knowledge_lifecycle_enum_member_has_an_explicit_label() -> None:
    """The fourteen `ObservationType`s, the eight `FindingStatus`es, and the four other
    knowledge-lifecycle enums. A missing entry would fall through to `_humanize`, whose output for
    `experiment_validity_boundary` ("Experiment Validity Boundary") reads as jargon a reader could
    gloss as a judgement on the method that ran -- which is precisely the confusion that member exists
    to prevent."""
    from archivetrust.htr.knowledge.models import (
        EvidenceReferenceKind,
        FindingConfidence,
        FindingStatus,
        HypothesisRelationship,
        ObservationConfidence,
        ObservationReviewStatus,
        ObservationType,
        ResearchQuestionStatus,
    )
    from archivetrust.presentation.display_names import (
        evidence_reference_kind_label,
        finding_status_label,
        hypothesis_relationship_label,
        knowledge_confidence_label,
        observation_review_status_label,
        observation_type_label,
        research_question_status_label,
        _EVIDENCE_REFERENCE_KIND_LABELS,
        _FINDING_STATUS_LABELS,
        _HYPOTHESIS_RELATIONSHIP_LABELS,
        _KNOWLEDGE_CONFIDENCE_LABELS,
        _OBSERVATION_REVIEW_STATUS_LABELS,
        _OBSERVATION_TYPE_LABELS,
        _RESEARCH_QUESTION_STATUS_LABELS,
    )

    # Exhaustive over each enum, keyed by value, so a new member without a label fails here.
    for enum_class, table in (
        (ObservationType, _OBSERVATION_TYPE_LABELS),
        (FindingStatus, _FINDING_STATUS_LABELS),
        (ObservationReviewStatus, _OBSERVATION_REVIEW_STATUS_LABELS),
        (ObservationConfidence, _KNOWLEDGE_CONFIDENCE_LABELS),
        (FindingConfidence, _KNOWLEDGE_CONFIDENCE_LABELS),
        (EvidenceReferenceKind, _EVIDENCE_REFERENCE_KIND_LABELS),
        (ResearchQuestionStatus, _RESEARCH_QUESTION_STATUS_LABELS),
        (HypothesisRelationship, _HYPOTHESIS_RELATIONSHIP_LABELS),
    ):
        assert {member.value for member in enum_class} <= set(table), enum_class.__name__

    assert len(_OBSERVATION_TYPE_LABELS) == 14
    assert len(_FINDING_STATUS_LABELS) == 8

    # The three labels that exist to prevent a specific misreading, asserted by their wording.
    assert (
        observation_type_label(ObservationType.EXPERIMENT_VALIDITY_BOUNDARY)
        == "What this comparison cannot measure"
    )
    assert (
        observation_type_label(ObservationType.CONFIDENCE_ANOMALY)
        == "Confidence did not match accuracy"
    )
    assert "not reproduced" in finding_status_label(FindingStatus.PROVISIONALLY_SUPPORTED)
    assert "reproduced in another run" in finding_status_label(FindingStatus.SUPPORTED).lower()
    assert (
        observation_review_status_label(ObservationReviewStatus.ACCEPTED)
        == "Confirmed against the records"
    )
    assert "not a telemetry record" in evidence_reference_kind_label(
        EvidenceReferenceKind.EXTERNAL_DOCUMENT
    )
    assert knowledge_confidence_label(FindingConfidence.MODERATE) == "Moderate"
    assert "nothing drafted yet" in research_question_status_label(ResearchQuestionStatus.OPEN)
    assert (
        hypothesis_relationship_label(HypothesisRelationship.NO_HYPOTHESIS_ASSERTED)
        == "The experiment asserted no hypothesis"
    )


def test_basis_code_labels_speak_of_methods_not_ocr_providers() -> None:
    assert basis_code_label("contested") == "Methods disagree"
    assert basis_code_label("corroborated") == "Multiple methods agree"


def test_document_labels_prefer_original_filename_then_short_reference() -> None:
    assert document_label("archive_object_abcd", archive_object=ArchiveObjectStub("scan-1904.pdf")) == "scan-1904.pdf"
    assert short_ref("archive_object_1234567890abcdef", prefix=15) == "archive_object_..."
    assert document_label("archive_object_1234567890abcdef") == "archive_ob..."
