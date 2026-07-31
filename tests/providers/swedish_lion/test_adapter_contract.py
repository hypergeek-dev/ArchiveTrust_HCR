"""Fast Swedish Lion adapter tests -- protocol conformance + failure-path unit tests, all using a
fake `SwedishLionWorkerFacade` (mirrors `providers/florence2_htr/test_adapter_contract.py`'s and
`providers/satrn/test_adapter_contract.py`'s discipline). No model download, no GPU, no network
required -- these run in the default `pytest tests -q` sweep. Real-model tests live in
`test_real_inference.py`, marked `real_model`.
"""

from __future__ import annotations

import pytest

from archivetrust.domain.evidence.models import Evidence
from archivetrust.providers.htr_adapter import HtrMethodAdapter, RecognitionInput
from archivetrust.providers.swedish_lion.adapter import (
    SwedishLionAdapter,
    build_evidence,
    build_failure_record,
    build_observation_payloads,
    normalize_transcription,
)
from archivetrust.providers.swedish_lion.facade import SwedishLionWorkerResult


class _FakeFacade:
    """A fake `SwedishLionWorkerFacade` returning a pre-programmed result, so each failure
    category can be exercised without a real model download."""

    def __init__(self, *, result: dict | None = None) -> None:
        self._result = result
        self.calls: list[dict] = []

    def run_inference(
        self,
        *,
        image_path,
        device_request,
        model_id,
        model_revision,
        processor_model_id,
        processor_revision,
        num_beams,
    ):
        self.calls.append(
            {
                "image_path": image_path,
                "device_request": device_request,
                "model_id": model_id,
                "model_revision": model_revision,
                "processor_model_id": processor_model_id,
                "processor_revision": processor_revision,
                "num_beams": num_beams,
            }
        )
        return SwedishLionWorkerResult(self._result)


def _success_result(**overrides) -> dict:
    base = {
        "ok": True,
        "raw_decoded": "<s> hejdå vän </s>",
        "text": "hejdå vän",
        "elapsed_seconds": 0.31,
        "device_used": "cuda",
        "gpu_name": "NVIDIA GeForce RTX 3070",
        "peak_gpu_memory_mb": 3000.0,
        "software_environment": {"torch": "2.1.0", "transformers": "4.49.0"},
        "model_revision": "deadbeef",
    }
    base.update(overrides)
    return base


# --- Protocol conformance -----------------------------------------------------------------


def test_swedish_lion_adapter_satisfies_htr_method_adapter_protocol():
    adapter = SwedishLionAdapter(facade=_FakeFacade(result=_success_result()))
    assert isinstance(adapter, HtrMethodAdapter)


def test_capabilities_declare_every_flag_explicitly_no_silent_omission():
    adapter = SwedishLionAdapter(facade=_FakeFacade(result=_success_result()))
    capabilities = adapter.get_capabilities()
    dumped = capabilities.model_dump()
    for field in (
        "confidence_supported",
        "geometry_supported",
        "line_level_supported",
        "page_level_supported",
        "local_execution_supported",
        "external_upload_required",
    ):
        assert field in dumped, f"{field} missing from MethodCapabilities dump"
    assert capabilities.line_level_supported is True
    assert capabilities.page_level_supported is False
    assert capabilities.geometry_supported is False
    assert capabilities.external_upload_required is False
    assert capabilities.local_execution_supported is True
    # No confidence signal is requested/exposed by this adapter -- an honest False, not fabricated.
    assert capabilities.confidence_supported is False


def test_metadata_reports_pinned_fine_tuned_checkpoint_not_a_floating_tag():
    adapter = SwedishLionAdapter(facade=_FakeFacade(result=_success_result()))
    metadata = adapter.get_metadata()
    assert metadata.method_id == "swedish_lion"
    assert "Riksarkivet/trocr-base-handwritten-hist-swe-2" in metadata.model_revision
    assert "@" in metadata.model_revision
    revision_part = metadata.model_revision.split("@", 1)[1]
    assert revision_part not in ("main", "latest", "")
    assert len(revision_part) == 40  # a real git-style commit sha


# --- recognize() success path -------------------------------------------------------------


def test_recognize_returns_clean_text_not_special_token_text():
    facade = _FakeFacade(result=_success_result())
    adapter = SwedishLionAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert result.text == "hejdå vän"
    assert "<s>" not in result.text
    assert result.raw_response["raw_decoded"] == "<s> hejdå vän </s>"
    assert result.execution_time_ms == pytest.approx(310.0)
    assert result.model_revision == "deadbeef"
    assert result.confidence is None
    assert facade.calls[0]["image_path"] == "/tmp/line.jpg"
    assert facade.calls[0]["num_beams"] == 1


def test_recognize_prefers_page_image_ref_when_no_input_crop_id():
    facade = _FakeFacade(result=_success_result())
    adapter = SwedishLionAdapter(facade=facade)
    adapter.recognize(RecognitionInput(page_image_ref="/tmp/page.jpg"))
    assert facade.calls[0]["image_path"] == "/tmp/page.jpg"


def test_recognize_with_neither_crop_nor_page_ref_is_a_malformed_input_failure_not_a_crash():
    adapter = SwedishLionAdapter(facade=_FakeFacade(result=_success_result()))
    result = adapter.recognize(RecognitionInput())
    assert result.text is None
    assert result.raw_response["category"] == "malformed_input"


def test_recognize_configuration_can_override_device_and_num_beams():
    facade = _FakeFacade(result=_success_result())
    adapter = SwedishLionAdapter(facade=facade, device_request="auto")
    adapter.recognize(
        RecognitionInput(input_crop_id="/tmp/line.jpg", configuration={"device": "cpu", "num_beams": 3})
    )
    assert facade.calls[0]["device_request"] == "cpu"
    assert facade.calls[0]["num_beams"] == 3


# --- raw vs parsed vs normalized separation (three genuinely distinct stages) --------------


def test_normalize_transcription_only_trims_whitespace_and_normalizes_unicode():
    padded = chr(32) + chr(32) + "hej" + chr(10)
    assert normalize_transcription(padded) == "hej"
    decomposed = chr(0x61) + chr(0x30A)
    precomposed = chr(0xE5)
    assert decomposed != precomposed
    assert normalize_transcription(decomposed) == precomposed


def test_build_observation_payloads_produces_three_genuinely_separate_stored_values():
    facade = _FakeFacade(
        result=_success_result(raw_decoded="<s>  Hejdå  </s>", text="  Hejdå  ")
    )
    adapter = SwedishLionAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    payloads = build_observation_payloads(result)
    assert payloads is not None
    raw_payload, parsed_payload, normalized_payload = payloads
    assert raw_payload.text == "<s>  Hejdå  </s>"
    assert parsed_payload.text == "  Hejdå  "
    assert normalized_payload.text == "Hejdå"
    texts = {raw_payload.text, parsed_payload.text, normalized_payload.text}
    assert len(texts) == 3
    assert raw_payload is not parsed_payload
    assert parsed_payload is not normalized_payload


def test_build_observation_payloads_returns_none_for_a_failed_result():
    facade = _FakeFacade(result={"ok": False, "category": "empty_output", "message": "no text"})
    adapter = SwedishLionAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert build_observation_payloads(result) is None


# --- Evidence provenance population ---------------------------------------------------------


def test_build_evidence_populates_all_htr_provenance_fields_with_true_raw_output():
    facade = _FakeFacade(result=_success_result())
    adapter = SwedishLionAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    evidence = build_evidence(result)
    assert isinstance(evidence, Evidence)
    # Evidence.raw_output is the TRUE raw decoder text (special tokens intact), not the clean text.
    assert evidence.raw_output == "<s> hejdå vän </s>"
    assert evidence.provider == "swedish_lion"
    assert evidence.model_revision == "deadbeef"
    assert evidence.execution_device == "cuda"
    assert evidence.execution_time_ms == pytest.approx(310.0)
    assert evidence.gpu_memory_mb == pytest.approx(3000.0)
    assert evidence.software_environment == {"torch": "2.1.0", "transformers": "4.49.0"}
    assert evidence.hardware_environment == {"gpu_name": "NVIDIA GeForce RTX 3070"}
    assert evidence.provider_confidence is None
    assert evidence.processing_stage.value == "ocr"
    assert evidence.supporting_metadata["processor_model_id"] == "microsoft/trocr-base-handwritten"
    assert evidence.evidence_id == Evidence.compute_id(
        provider="swedish_lion",
        provider_version=evidence.provider_version,
        raw_output="<s> hejdå vän </s>",
        processing_stage=evidence.processing_stage,
        prompt=None,
        model_revision="deadbeef",
        pipeline_configuration_hash=None,
    )


def test_build_evidence_returns_none_for_a_failed_result():
    facade = _FakeFacade(result={"ok": False, "category": "empty_output", "message": "no text"})
    adapter = SwedishLionAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert build_evidence(result) is None


# --- Failure paths: recorded as FailureRecord, never silently swallowed --------------------


@pytest.mark.parametrize(
    "worker_response",
    [
        {"ok": False, "category": "model_load_failed", "message": "404 repo not found"},
        {"ok": False, "category": "cuda_oom", "message": "CUDA out of memory."},
        {"ok": False, "category": "malformed_input", "message": "cannot identify image file"},
        {"ok": False, "category": "empty_output", "message": "no predictions"},
    ],
)
def test_each_facade_failure_category_becomes_a_failure_record_not_a_crash(worker_response):
    facade = _FakeFacade(result=worker_response)
    adapter = SwedishLionAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert result.text is None
    failure = build_failure_record(result, method_run_id="method_run_1")
    assert failure is not None
    assert failure.category == worker_response["category"]
    assert failure.reason == worker_response["message"]


def test_build_failure_record_returns_none_for_a_successful_result():
    facade = _FakeFacade(result=_success_result())
    adapter = SwedishLionAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert build_failure_record(result, method_run_id="method_run_1") is None


def test_validate_environment_reports_invalid_when_dependencies_missing(monkeypatch):
    import archivetrust.providers.swedish_lion.adapter as adapter_module

    monkeypatch.setattr(
        adapter_module, "swedish_lion_dependencies_available", lambda: (False, "missing: torch")
    )
    adapter = SwedishLionAdapter(facade=_FakeFacade(result=_success_result()))
    validation = adapter.validate_environment()
    assert validation.valid is False
    assert validation.messages
    assert any("missing" in m.lower() for m in validation.messages)


def test_health_check_is_cheap_and_does_not_invoke_the_facade(monkeypatch):
    import archivetrust.providers.swedish_lion.adapter as adapter_module

    monkeypatch.setattr(
        adapter_module, "swedish_lion_dependencies_available", lambda: (True, "torch==2.1.0 importable")
    )
    facade = _FakeFacade(result=_success_result())
    adapter = SwedishLionAdapter(facade=facade)
    health = adapter.health_check()
    assert health.healthy is True
    assert facade.calls == []  # health_check must never run a real/fake inference call
