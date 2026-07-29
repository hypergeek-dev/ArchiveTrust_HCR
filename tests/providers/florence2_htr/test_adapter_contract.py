"""Fast Florence-2 adapter tests -- protocol conformance + failure-path unit tests, all using a
fake `Florence2WorkerFacade` (mirrors `providers/satrn/test_adapter_contract.py`'s discipline). No
model download, no GPU, no network required -- these run in the default `pytest tests -q` sweep.
Real-model tests live in `test_real_inference.py`, marked `real_model`.
"""

from __future__ import annotations

import math

import pytest

from archivetrust.domain.evidence.models import Evidence
from archivetrust.providers.florence2_htr.adapter import (
    Florence2Adapter,
    build_evidence,
    build_failure_record,
    build_observation_payloads,
    normalize_transcription,
)
from archivetrust.providers.florence2_htr.facade import (
    Florence2WorkerResult,
    sequence_log_prob_to_confidence_proxy,
)
from archivetrust.providers.htr_adapter import HtrMethodAdapter, RecognitionInput


class _FakeFacade:
    """A fake `Florence2WorkerFacade` returning a pre-programmed result, so each failure category
    can be exercised without a real model download."""

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
        task_prompt,
    ):
        self.calls.append(
            {
                "image_path": image_path,
                "device_request": device_request,
                "model_id": model_id,
                "model_revision": model_revision,
                "processor_model_id": processor_model_id,
                "processor_revision": processor_revision,
                "task_prompt": task_prompt,
            }
        )
        return Florence2WorkerResult(self._result)


def _success_result(**overrides) -> dict:
    base = {
        "ok": True,
        "raw_decoded": "</s><s>  hejdå vän  </s>",
        "parsed": {"<OCR>": "  hejdå vän  "},
        "parsed_text": "  hejdå vän  ",
        "sequence_log_prob": -0.5,
        "elapsed_seconds": 0.42,
        "device_used": "cuda",
        "gpu_name": "NVIDIA GeForce RTX 3070",
        "peak_gpu_memory_mb": 4000.0,
        "software_environment": {"torch": "2.1.0", "transformers": "4.49.0"},
        "model_revision": "deadbeef",
    }
    base.update(overrides)
    return base


# --- Protocol conformance -----------------------------------------------------------------


def test_florence2_adapter_satisfies_htr_method_adapter_protocol():
    adapter = Florence2Adapter(facade=_FakeFacade(result=_success_result()))
    assert isinstance(adapter, HtrMethodAdapter)


def test_capabilities_declare_every_flag_explicitly_no_silent_omission():
    adapter = Florence2Adapter(facade=_FakeFacade(result=_success_result()))
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
    # Florence-2-specific truths, not defaults: line-level (recognize() only, see module
    # docstring), no geometry from this task token, no external upload.
    assert capabilities.line_level_supported is True
    assert capabilities.page_level_supported is False
    assert capabilities.geometry_supported is False
    assert capabilities.external_upload_required is False
    assert capabilities.local_execution_supported is True
    # Confidence is a real proxy signal, not fabricated -- see facade.py.
    assert capabilities.confidence_supported is True


def test_metadata_reports_pinned_fine_tuned_checkpoint_not_a_floating_tag():
    adapter = Florence2Adapter(facade=_FakeFacade(result=_success_result()))
    metadata = adapter.get_metadata()
    assert metadata.method_id == "florence2_htr"
    assert "nazounoryuu/florence_base__mixed__line_bbox__ocr" in metadata.model_revision
    assert "@" in metadata.model_revision
    revision_part = metadata.model_revision.split("@", 1)[1]
    assert revision_part not in ("main", "latest", "")
    assert len(revision_part) == 40  # a real git-style commit sha


# --- recognize() success path -------------------------------------------------------------


def test_recognize_returns_parsed_text_not_raw_special_token_text():
    facade = _FakeFacade(result=_success_result())
    adapter = Florence2Adapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    # RecognitionResult.text is the *parsed* text -- special tokens (</s><s>...</s>) never leak
    # into it; that untouched string lives only in raw_response["raw_decoded"].
    assert result.text == "  hejdå vän  "
    assert "</s>" not in result.text
    assert result.raw_response["raw_decoded"] == "</s><s>  hejdå vän  </s>"
    assert result.execution_time_ms == pytest.approx(420.0)
    assert result.model_revision == "deadbeef"
    assert facade.calls[0]["image_path"] == "/tmp/line.jpg"
    assert facade.calls[0]["task_prompt"] == "<OCR>"


def test_recognize_confidence_is_exp_of_sequence_log_prob_proxy():
    facade = _FakeFacade(result=_success_result(sequence_log_prob=-0.5))
    adapter = Florence2Adapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert result.confidence == pytest.approx(math.exp(-0.5))
    assert 0.0 < result.confidence <= 1.0


def test_recognize_prefers_page_image_ref_when_no_input_crop_id():
    facade = _FakeFacade(result=_success_result())
    adapter = Florence2Adapter(facade=facade)
    adapter.recognize(RecognitionInput(page_image_ref="/tmp/page.jpg"))
    assert facade.calls[0]["image_path"] == "/tmp/page.jpg"


def test_recognize_with_neither_crop_nor_page_ref_is_a_malformed_input_failure_not_a_crash():
    adapter = Florence2Adapter(facade=_FakeFacade(result=_success_result()))
    result = adapter.recognize(RecognitionInput())
    assert result.text is None
    assert result.raw_response["category"] == "malformed_input"


def test_recognize_configuration_can_override_device_and_task_prompt():
    facade = _FakeFacade(result=_success_result())
    adapter = Florence2Adapter(facade=facade, device_request="auto")
    adapter.recognize(
        RecognitionInput(
            input_crop_id="/tmp/line.jpg", configuration={"device": "cpu", "task_prompt": "<OCR>"}
        )
    )
    assert facade.calls[0]["device_request"] == "cpu"


# --- confidence proxy helper ---------------------------------------------------------------


def test_confidence_proxy_helper_clips_to_unit_interval_and_handles_none():
    assert sequence_log_prob_to_confidence_proxy(None) is None
    assert sequence_log_prob_to_confidence_proxy(0.0) == pytest.approx(1.0)
    assert sequence_log_prob_to_confidence_proxy(-100.0) == pytest.approx(0.0, abs=1e-6)
    # A real, meaningfully-in-between value.
    assert 0.0 < sequence_log_prob_to_confidence_proxy(-1.5) < 1.0


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
        result=_success_result(
            raw_decoded="</s><s>  Hejdå  </s>", parsed_text="  Hejdå  ", parsed={"<OCR>": "  Hejdå  "}
        )
    )
    adapter = Florence2Adapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    payloads = build_observation_payloads(result)
    assert payloads is not None
    raw_payload, parsed_payload, normalized_payload = payloads
    assert raw_payload.text == "</s><s>  Hejdå  </s>"
    assert parsed_payload.text == "  Hejdå  "
    assert normalized_payload.text == "Hejdå"
    # Three genuinely different objects/values -- never one overwritten by another.
    texts = {raw_payload.text, parsed_payload.text, normalized_payload.text}
    assert len(texts) == 3
    assert raw_payload is not parsed_payload
    assert parsed_payload is not normalized_payload


# --- Evidence provenance population ---------------------------------------------------------


def test_build_evidence_populates_all_htr_provenance_fields_with_true_raw_output():
    facade = _FakeFacade(result=_success_result())
    adapter = Florence2Adapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    evidence = build_evidence(result)
    assert isinstance(evidence, Evidence)
    # Evidence.raw_output is the TRUE raw decoder text, not the parsed transcription.
    assert evidence.raw_output == "</s><s>  hejdå vän  </s>"
    assert evidence.provider == "florence2_htr"
    assert evidence.model_revision == "deadbeef"
    assert evidence.execution_device == "cuda"
    assert evidence.execution_time_ms == pytest.approx(420.0)
    assert evidence.gpu_memory_mb == pytest.approx(4000.0)
    assert evidence.software_environment == {"torch": "2.1.0", "transformers": "4.49.0"}
    assert evidence.hardware_environment == {"gpu_name": "NVIDIA GeForce RTX 3070"}
    assert evidence.provider_confidence == pytest.approx(math.exp(-0.5))
    assert evidence.processing_stage.value == "vlm_inference"
    assert evidence.supporting_metadata["processor_model_id"] == "microsoft/Florence-2-base-ft"
    assert evidence.supporting_metadata["vlm_htr_repo_revision"]
    # Content-addressed id is reproducible from the same content -- not hand-set.
    assert evidence.evidence_id == Evidence.compute_id(
        provider="florence2_htr",
        provider_version=evidence.provider_version,
        raw_output="</s><s>  hejdå vän  </s>",
        processing_stage=evidence.processing_stage,
        prompt="<OCR>",
        model_revision="deadbeef",
        pipeline_configuration_hash=None,
    )


def test_build_evidence_returns_none_for_a_failed_result():
    facade = _FakeFacade(result={"ok": False, "category": "empty_output", "message": "no text"})
    adapter = Florence2Adapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert build_evidence(result) is None


# --- Failure paths: recorded as FailureRecord, never silently swallowed --------------------


@pytest.mark.parametrize(
    "worker_response",
    [
        {"ok": False, "category": "model_load_failed", "message": "404 repo not found"},
        {"ok": False, "category": "cuda_oom", "message": "CUDA out of memory."},
        {"ok": False, "category": "malformed_input", "message": "cannot identify image file"},
        {"ok": False, "category": "malformed_output", "message": "post_process_generation failed"},
        {"ok": False, "category": "empty_output", "message": "no predictions"},
    ],
)
def test_each_facade_failure_category_becomes_a_failure_record_not_a_crash(worker_response):
    facade = _FakeFacade(result=worker_response)
    adapter = Florence2Adapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert result.text is None
    failure = build_failure_record(result, method_run_id="method_run_1")
    assert failure is not None
    assert failure.category == worker_response["category"]
    assert failure.reason == worker_response["message"]


def test_build_failure_record_returns_none_for_a_successful_result():
    facade = _FakeFacade(result=_success_result())
    adapter = Florence2Adapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert build_failure_record(result, method_run_id="method_run_1") is None


def test_validate_environment_reports_invalid_when_dependencies_missing(monkeypatch):
    import archivetrust.providers.florence2_htr.adapter as adapter_module

    monkeypatch.setattr(
        adapter_module, "florence2_dependencies_available", lambda: (False, "missing: timm")
    )
    adapter = Florence2Adapter(facade=_FakeFacade(result=_success_result()))
    validation = adapter.validate_environment()
    assert validation.valid is False
    assert validation.messages
    assert any("missing" in m.lower() for m in validation.messages)


def test_health_check_is_cheap_and_does_not_invoke_the_facade(monkeypatch):
    import archivetrust.providers.florence2_htr.adapter as adapter_module

    monkeypatch.setattr(
        adapter_module, "florence2_dependencies_available", lambda: (True, "torch==2.1.0 importable")
    )
    facade = _FakeFacade(result=_success_result())
    adapter = Florence2Adapter(facade=facade)
    health = adapter.health_check()
    assert health.healthy is True
    assert facade.calls == []  # health_check must never run a real/fake inference call
