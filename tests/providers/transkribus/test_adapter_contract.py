"""TranskribusAdapter contract + behavior tests -- manual-import mode only. No network, no
subprocess, no GPU: `recognize()` only ever reads local fixture files from disk. These run in the
default `pytest tests -q` sweep (no real_model marker needed -- there is no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.domain.evidence.models import Evidence
from archivetrust.htr.experiment.models import FailureRecord
from archivetrust.providers.htr_adapter import HtrMethodAdapter, RecognitionInput
from archivetrust.providers.transkribus.adapter import (
    TranskribusAdapter,
    build_evidence,
    build_failure_record,
    build_observation_payloads,
    normalize_transcription,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "transkribus"


def _input_for(filename: str, *, export_format: str | None = None) -> RecognitionInput:
    configuration = {"export_file_path": str(FIXTURES / filename)}
    if export_format is not None:
        configuration["export_format"] = export_format
    return RecognitionInput(configuration=configuration)


# --- Protocol conformance -----------------------------------------------------------------


def test_transkribus_adapter_satisfies_htr_method_adapter_protocol():
    assert isinstance(TranskribusAdapter(), HtrMethodAdapter)


def test_capabilities_declare_every_flag_explicitly_no_silent_omission():
    capabilities = TranskribusAdapter().get_capabilities()
    dumped = capabilities.model_dump()
    for field in (
        "confidence_supported",
        "geometry_supported",
        "line_level_supported",
        "page_level_supported",
        "local_execution_supported",
        "external_upload_required",
    ):
        assert field in dumped
    # Manual-import-mode truths, not defaults (module docstring's capability-flag section):
    assert capabilities.local_execution_supported is False  # no model runs in this process
    assert capabilities.external_upload_required is False  # this adapter uploads nothing
    assert capabilities.confidence_supported is True
    assert capabilities.geometry_supported is True
    assert capabilities.line_level_supported is True
    assert capabilities.page_level_supported is True


def test_metadata_never_fabricates_a_model_version():
    metadata = TranskribusAdapter().get_metadata()
    assert metadata.method_id == "transkribus_swedish_lion_1"
    assert metadata.vendor == "READ-COOP (Transkribus)"
    # Honest about the absence of a pinned version at this (pre-file) point, not a fabricated hash.
    assert "unpinned" in metadata.model_revision or "no fixed checkpoint" in metadata.model_revision


# --- validate_environment / health_check: file checks, never GPU/network -------------------


def test_validate_environment_with_no_configured_directory_is_valid_but_says_why():
    validation = TranskribusAdapter().validate_environment()
    assert validation.valid is True
    assert any("export_file_path" in m for m in validation.messages)


def test_validate_environment_reports_missing_configured_import_directory():
    adapter = TranskribusAdapter(import_directory=str(FIXTURES / "does_not_exist_dir"))
    validation = adapter.validate_environment()
    assert validation.valid is False
    assert validation.messages


def test_validate_environment_accepts_a_real_existing_directory():
    adapter = TranskribusAdapter(import_directory=str(FIXTURES))
    validation = adapter.validate_environment()
    assert validation.valid is True


def test_health_check_probes_the_directory_not_a_network_endpoint():
    adapter = TranskribusAdapter(import_directory=str(FIXTURES))
    result = adapter.health_check()
    assert result.healthy is True


def test_health_check_unhealthy_for_missing_directory():
    adapter = TranskribusAdapter(import_directory=str(FIXTURES / "nope"))
    result = adapter.health_check()
    assert result.healthy is False


# --- recognize(): PAGE XML success path -----------------------------------------------------


def test_recognize_page_xml_returns_parsed_text_geometry_and_confidence():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_page.xml"))
    assert result.text == (
        "Anno 1712 den 3 Januarii holltes ting\nmedh allmogen aff Sochnen\nNB dombook"
    )
    assert result.confidence == pytest.approx((0.93 + 0.87 + 0.79) / 3)
    assert result.execution_time_ms is None  # never conflated with adapter parse time
    assert result.raw_response["adapter_parse_time_ms"] >= 0.0
    assert result.raw_response["vendor_reported_accuracy"] == pytest.approx(0.912)
    assert result.raw_response["export_format"] == "page_xml"
    assert len(result.raw_response["regions"]) == 2
    assert result.model_revision == "Swedish Lion I - v3"


def test_recognize_alto_xml_returns_parsed_text_and_word_confidence():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_alto.xml"))
    assert result.text == "Anno 1712 den\nmedh allmogen\nNB dombook"
    assert result.confidence is not None
    assert result.raw_response["export_format"] == "alto_xml"
    # ALTO carries no vendor-accuracy element -- must not be fabricated.
    assert result.raw_response["vendor_reported_accuracy"] is None


def test_recognize_plain_text_returns_joined_lines_with_no_confidence():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_plain.txt"))
    assert result.text == "Anno 1712 den 3 Januarii holltes ting\nmedh allmogen aff Sochnen\nNB dombook"
    assert result.confidence is None
    assert result.raw_response["export_format"] == "plain_text"


def test_recognize_respects_explicit_export_format_override():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_page.xml", export_format="page_xml"))
    assert result.raw_response["export_format"] == "page_xml"


# --- recognize(): failure paths -------------------------------------------------------------


def test_recognize_with_no_export_file_path_is_a_malformed_input_failure_not_a_crash():
    adapter = TranskribusAdapter()
    result = adapter.recognize(RecognitionInput())
    assert result.text is None
    assert result.raw_response["category"] == "malformed_input"


def test_recognize_missing_file_is_a_file_not_found_failure():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("does_not_exist.xml"))
    assert result.text is None
    assert result.raw_response["category"] == "file_not_found"


def test_recognize_malformed_page_xml_is_a_malformed_xml_failure():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("malformed_page.xml"))
    assert result.text is None
    assert result.raw_response["category"] == "malformed_xml"


def test_recognize_malformed_alto_xml_is_a_malformed_xml_failure():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("malformed_alto.xml"))
    assert result.text is None
    assert result.raw_response["category"] == "malformed_xml"


def test_recognize_empty_plain_text_file_is_an_empty_file_failure():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("empty_file.txt"))
    assert result.text is None
    assert result.raw_response["category"] == "empty_file"


def test_recognize_missing_page_element_is_a_missing_required_element_failure():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("missing_page_element.xml"))
    assert result.text is None
    assert result.raw_response["category"] == "missing_required_element"


# --- normalization, payloads, evidence, failure records -------------------------------------


def test_normalize_transcription_only_trims_whitespace_and_normalizes_unicode():
    assert normalize_transcription("  hej\n") == "hej"
    decomposed = chr(0x61) + chr(0x30A)
    precomposed = chr(0xE5)
    assert normalize_transcription(decomposed) == precomposed


def test_build_observation_payloads_returns_parsed_and_normalized_only_two_stages():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_page.xml"))
    payloads = build_observation_payloads(result)
    assert payloads is not None
    parsed_payload, normalized_payload = payloads
    assert parsed_payload.text == result.text
    assert normalized_payload.text == normalize_transcription(result.text)
    assert parsed_payload is not normalized_payload


def test_build_observation_payloads_returns_none_for_a_failed_result():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("does_not_exist.xml"))
    assert build_observation_payloads(result) is None


def test_build_evidence_raw_output_is_the_untouched_export_file_not_the_parsed_text():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_page.xml"))
    evidence = build_evidence(result)
    assert isinstance(evidence, Evidence)
    assert evidence.raw_output != result.text
    assert "<PcGts" in evidence.raw_output
    assert evidence.provider == "transkribus_swedish_lion_1"
    assert evidence.provider_confidence == result.confidence
    # No local execution -- explicit None, never a misleading default.
    assert evidence.execution_device is None
    assert evidence.execution_time_ms is None
    assert evidence.gpu_memory_mb is None
    assert evidence.software_environment is None
    assert evidence.hardware_environment is None
    assert evidence.supporting_metadata["vendor_reported_accuracy"] == pytest.approx(0.912)


def test_build_evidence_is_content_addressed():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_page.xml"))
    evidence = build_evidence(result)
    assert evidence.evidence_id == Evidence.compute_id(
        provider="transkribus_swedish_lion_1",
        provider_version=evidence.provider_version,
        raw_output=evidence.raw_output,
        processing_stage=evidence.processing_stage,
        prompt=None,
        model_revision=evidence.model_revision,
        pipeline_configuration_hash=None,
    )


def test_build_evidence_returns_none_for_a_failed_result():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("does_not_exist.xml"))
    assert build_evidence(result) is None


@pytest.mark.parametrize(
    "filename,expected_category",
    [
        ("does_not_exist.xml", "file_not_found"),
        ("malformed_page.xml", "malformed_xml"),
        ("malformed_alto.xml", "malformed_xml"),
        ("empty_file.txt", "empty_file"),
        ("missing_page_element.xml", "missing_required_element"),
    ],
)
def test_each_failure_category_becomes_a_failure_record_not_a_crash(filename, expected_category):
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for(filename))
    failure = build_failure_record(result, method_run_id="method_run_1")
    assert isinstance(failure, FailureRecord)
    assert failure.category == expected_category


def test_build_failure_record_returns_none_for_a_successful_result():
    adapter = TranskribusAdapter()
    result = adapter.recognize(_input_for("sample_page.xml"))
    assert build_failure_record(result, method_run_id="method_run_1") is None
