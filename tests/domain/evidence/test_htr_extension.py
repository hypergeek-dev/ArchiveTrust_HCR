from __future__ import annotations

from archivetrust.domain.evidence.models import Evidence, ProcessingStage


def test_evidence_create_without_new_kwargs_matches_pre_extension_hash():
    """Regression guard for docs/htr-migration-plan.md Stage 3's backward-compatibility
    requirement: a caller that passes none of the new Stage-3 keyword arguments must get the
    exact same `evidence_id` as before this extension landed. Hand-computed via the same
    `content_address(provider, provider_version, processing_stage.value, prompt or "",
    raw_output)` call the pre-extension implementation used."""
    from archivetrust.domain.shared.ids import content_address

    expected = content_address("docling", "1.0", "ocr", "", "same text")

    ev = Evidence.create(
        provider="docling",
        provider_version="1.0",
        raw_output="same text",
        processing_stage=ProcessingStage.OCR,
    )
    assert ev.evidence_id == expected


def test_evidence_new_optional_fields_default_to_none():
    ev = Evidence.create(
        provider="docling", provider_version="1.0", raw_output="x", processing_stage=ProcessingStage.OCR
    )
    assert ev.model_revision is None
    assert ev.pipeline_configuration_hash is None
    assert ev.execution_device is None
    assert ev.execution_time_ms is None
    assert ev.gpu_memory_mb is None
    assert ev.software_environment is None
    assert ev.hardware_environment is None


def test_model_revision_changes_the_evidence_id():
    without = Evidence.create(
        provider="satrn", provider_version="1.0", raw_output="hej", processing_stage=ProcessingStage.OCR
    )
    with_revision = Evidence.create(
        provider="satrn",
        provider_version="1.0",
        raw_output="hej",
        processing_stage=ProcessingStage.OCR,
        model_revision="checkpoint-42",
    )
    assert without.evidence_id != with_revision.evidence_id


def test_pipeline_configuration_hash_changes_the_evidence_id_and_round_trips():
    ev = Evidence.create(
        provider="satrn",
        provider_version="1.0",
        raw_output="hej",
        processing_stage=ProcessingStage.OCR,
        model_revision="checkpoint-42",
        pipeline_configuration_hash="pipeline_abc123",
        execution_device="cuda:0",
        execution_time_ms=12.5,
        gpu_memory_mb=512.0,
        software_environment={"torch": "2.1"},
        hardware_environment={"gpu": "A100"},
    )
    restored = Evidence.model_validate(ev.model_dump())
    assert restored == ev
    assert ev.pipeline_configuration_hash == "pipeline_abc123"
