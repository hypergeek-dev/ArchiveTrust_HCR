"""Fast SATRN adapter tests -- protocol conformance + failure-path unit tests, all using a fake
`SatrnWorkerFacade` (mirrors `runtime/transformers_runtime.py`'s test-seam discipline). No
subprocess, no GPU, no network, no `.venv-satrn` required -- these run in the default `pytest
tests -q` sweep. Real-model tests live in `test_real_inference.py`, marked `real_model`.
"""

from __future__ import annotations

import pytest

from archivetrust.domain.evidence.models import Evidence
from archivetrust.providers.htr_adapter import HtrMethodAdapter, RecognitionInput
from archivetrust.providers.satrn.adapter import (
    SatrnAdapter,
    build_evidence,
    build_failure_record,
    build_observation_payloads,
    normalize_transcription,
)
from archivetrust.providers.satrn.facade import SatrnWorkerResult, SatrnWorkerTimeout


class _FakeFacade:
    """A fake `SatrnWorkerFacade` returning a pre-programmed result or raising a pre-programmed
    exception, so each failure category can be exercised without a real subprocess."""

    def __init__(self, *, result: dict | None = None, raise_exc: Exception | None = None) -> None:
        self._result = result
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    def run_inference(self, *, image_path, device_request, model_id, revision):
        self.calls.append(
            {
                "image_path": image_path,
                "device_request": device_request,
                "model_id": model_id,
                "revision": revision,
            }
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        return SatrnWorkerResult(self._result)


def _success_result(**overrides) -> dict:
    base = {
        "ok": True,
        "text": "hejdå vän",
        "score": 0.71,
        "elapsed_seconds": 0.42,
        "device_used": "cuda",
        "model_revision": "deadbeef",
        "config_revision": "deadbeef",
        "gpu_name": "NVIDIA GeForce RTX 3070",
        "peak_gpu_memory_mb": 512.0,
        "software_environment": {"torch": "2.1.0+cu121"},
    }
    base.update(overrides)
    return base


# --- Protocol conformance -----------------------------------------------------------------


def test_satrn_adapter_satisfies_htr_method_adapter_protocol():
    adapter = SatrnAdapter(facade=_FakeFacade(result=_success_result()))
    assert isinstance(adapter, HtrMethodAdapter)


def test_capabilities_declare_every_flag_explicitly_no_silent_omission():
    adapter = SatrnAdapter(facade=_FakeFacade(result=_success_result()))
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
    # SATRN-specific truths, not defaults: line-level only, no geometry, no external upload.
    assert capabilities.line_level_supported is True
    assert capabilities.page_level_supported is False
    assert capabilities.geometry_supported is False
    assert capabilities.external_upload_required is False
    assert capabilities.local_execution_supported is True


def test_metadata_reports_pinned_revision_not_a_floating_tag():
    adapter = SatrnAdapter(facade=_FakeFacade(result=_success_result()))
    metadata = adapter.get_metadata()
    assert metadata.method_id == "satrn"
    assert metadata.model_revision not in ("main", "latest", "")
    assert len(metadata.model_revision) == 40  # a real git-style commit sha


# --- recognize() success path -------------------------------------------------------------


def test_recognize_returns_raw_unmodified_text_and_real_metadata():
    facade = _FakeFacade(result=_success_result(text="  Hejdå  \n"))
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    # Untouched -- whitespace is NOT stripped here, that is normalize_transcription's job only.
    assert result.text == "  Hejdå  \n"
    assert result.confidence == 0.71
    assert result.execution_time_ms == pytest.approx(420.0)
    assert result.model_revision == "deadbeef"
    assert facade.calls[0]["image_path"] == "/tmp/line.jpg"


def test_recognize_prefers_page_image_ref_when_no_input_crop_id():
    facade = _FakeFacade(result=_success_result())
    adapter = SatrnAdapter(facade=facade)
    adapter.recognize(RecognitionInput(page_image_ref="/tmp/page.jpg"))
    assert facade.calls[0]["image_path"] == "/tmp/page.jpg"


def test_recognize_with_neither_crop_nor_page_ref_is_a_malformed_input_failure_not_a_crash():
    adapter = SatrnAdapter(facade=_FakeFacade(result=_success_result()))
    result = adapter.recognize(RecognitionInput())
    assert result.text is None
    assert result.raw_response["category"] == "malformed_input"


# --- raw vs normalized separation -----------------------------------------------------------


def test_normalize_transcription_only_trims_whitespace_and_normalizes_unicode():
    padded = chr(32) + chr(32) + "hej" + chr(10)
    assert normalize_transcription(padded) == "hej"
    # NFC-composes a decomposed a + combining-ring-above (U+0061 U+030A) into the single
    # precomposed codepoint U+00E5, without touching any other letters.
    decomposed = chr(0x61) + chr(0x30A)
    precomposed = chr(0xE5)
    assert decomposed != precomposed  # sanity: the two forms differ before normalizing
    assert normalize_transcription(decomposed) == precomposed


def test_build_observation_payloads_keeps_raw_and_normalized_as_separate_stored_values():
    facade = _FakeFacade(result=_success_result(text="  Hejdå  "))
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    payloads = build_observation_payloads(result)
    assert payloads is not None
    raw_payload, normalized_payload = payloads
    assert raw_payload.text == "  Hejdå  "
    assert normalized_payload.text == "Hejdå"
    # Genuinely separate objects, not one field silently overwritten with the other.
    assert raw_payload.text != normalized_payload.text
    assert raw_payload is not normalized_payload


# --- Evidence provenance population ---------------------------------------------------------


def test_build_evidence_populates_all_htr_provenance_fields():
    facade = _FakeFacade(result=_success_result())
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    evidence = build_evidence(result)
    assert isinstance(evidence, Evidence)
    assert evidence.raw_output == "hejdå vän"
    assert evidence.provider == "satrn"
    assert evidence.model_revision == "deadbeef"
    assert evidence.execution_device == "cuda"
    assert evidence.execution_time_ms == pytest.approx(420.0)
    assert evidence.gpu_memory_mb == pytest.approx(512.0)
    assert evidence.software_environment == {"torch": "2.1.0+cu121"}
    assert evidence.hardware_environment == {"gpu_name": "NVIDIA GeForce RTX 3070"}
    assert evidence.provider_confidence == 0.71
    # Content-addressed id is reproducible from the same content -- not hand-set.
    assert evidence.evidence_id == Evidence.compute_id(
        provider="satrn",
        provider_version=evidence.provider_version,
        raw_output="hejdå vän",
        processing_stage=evidence.processing_stage,
        prompt=None,
        model_revision="deadbeef",
        pipeline_configuration_hash=None,
    )


def test_build_evidence_returns_none_for_a_failed_result():
    facade = _FakeFacade(result={"ok": False, "category": "empty_output", "message": "no text"})
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert build_evidence(result) is None


# --- Failure paths: recorded as FailureRecord, never silently swallowed --------------------


@pytest.mark.parametrize(
    "worker_response",
    [
        {"ok": False, "category": "model_load_failed", "message": "404 repo not found"},
        {"ok": False, "category": "cuda_oom", "message": "CUDA out of memory."},
        {"ok": False, "category": "malformed_input", "message": "cv2.error: bad image"},
        {"ok": False, "category": "empty_output", "message": "no predictions"},
    ],
)
def test_each_worker_failure_category_becomes_a_failure_record_not_a_crash(worker_response):
    facade = _FakeFacade(result=worker_response)
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert result.text is None
    failure = build_failure_record(result, method_run_id="method_run_1")
    assert failure is not None
    assert failure.category == worker_response["category"]
    assert failure.reason == worker_response["message"]


def test_timeout_from_the_facade_becomes_a_timeout_failure_record():
    facade = _FakeFacade(raise_exc=SatrnWorkerTimeout("did not finish within 300.0s"))
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert result.text is None
    failure = build_failure_record(result, method_run_id="method_run_1")
    assert failure.category == "timeout"


def test_missing_isolated_venv_becomes_a_model_load_failed_failure_record():
    facade = _FakeFacade(raise_exc=FileNotFoundError("Isolated SATRN interpreter not found"))
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert result.text is None
    failure = build_failure_record(result, method_run_id="method_run_1")
    assert failure.category == "model_load_failed"


def test_build_failure_record_returns_none_for_a_successful_result():
    facade = _FakeFacade(result=_success_result())
    adapter = SatrnAdapter(facade=facade)
    result = adapter.recognize(RecognitionInput(input_crop_id="/tmp/line.jpg"))
    assert build_failure_record(result, method_run_id="method_run_1") is None


def test_validate_environment_reports_invalid_when_isolated_venv_missing(monkeypatch):
    import archivetrust.providers.satrn.adapter as adapter_module

    monkeypatch.setattr(
        adapter_module, "satrn_python_available", lambda: (False, "not found at /nowhere")
    )
    adapter = SatrnAdapter(facade=_FakeFacade(result=_success_result()))
    validation = adapter.validate_environment()
    assert validation.valid is False
    assert validation.messages  # never a bare False with no explanation
    assert any("not found" in m for m in validation.messages)


def test_health_check_is_cheap_and_does_not_invoke_the_facade():
    facade = _FakeFacade(result=_success_result())
    adapter = SatrnAdapter(facade=facade)
    adapter.health_check()
    assert facade.calls == []  # health_check must never run a real/fake inference call
