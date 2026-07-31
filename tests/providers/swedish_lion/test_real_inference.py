"""Real-inference Swedish Lion adapter tests -- downloads the real fine-tuned checkpoint
`Riksarkivet/trocr-base-handwritten-hist-swe-2` (first run only, cached afterwards) and runs real
forward passes through plain `transformers` in this project's main venv. Requires:

- The `transformers` extra installed (`pip install -e ".[transformers]"`) -- no extra dependencies
  beyond that (unlike Florence-2, no `timm`/`einops`, no `trust_remote_code`).
- Network access to huggingface.co (first run only -- afterwards the HF cache is warm).

Marked `real_model` (registered in `pyproject.toml`), same convention as
`tests/providers/satrn/test_real_inference.py` and
`tests/providers/florence2_htr/test_real_inference.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.providers.htr_adapter import RecognitionInput
from archivetrust.providers.swedish_lion.adapter import (
    DEFAULT_MODEL_REVISION,
    SwedishLionAdapter,
    build_evidence,
    build_failure_record,
)
from archivetrust.providers.swedish_lion.facade import swedish_lion_dependencies_available

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "htr" / "trolldomskommissionen_sample_line.jpg"

pytestmark = pytest.mark.real_model

_deps_ok, _deps_message = swedish_lion_dependencies_available()
requires_swedish_lion_deps = pytest.mark.skipif(not _deps_ok, reason=_deps_message)


@requires_swedish_lion_deps
def test_real_recognition_on_the_fixture_line_image_produces_real_output():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    adapter = SwedishLionAdapter()
    result = adapter.recognize(RecognitionInput(input_crop_id=str(FIXTURE)))

    assert result.text is not None
    assert result.text.strip() != ""
    assert result.execution_time_ms is not None
    assert result.execution_time_ms > 0
    assert result.model_revision == DEFAULT_MODEL_REVISION
    # This adapter does not request/expose a confidence signal -- see module docstring.
    assert result.confidence is None
    # Raw response carries the genuinely raw (special-token-laden) decoder text, distinct from
    # the clean RecognitionResult.text.
    assert result.raw_response["raw_decoded"] != result.text

    evidence = build_evidence(result)
    assert evidence is not None
    assert evidence.raw_output == result.raw_response["raw_decoded"]
    assert evidence.raw_output != result.text
    assert evidence.model_revision == result.model_revision
    assert evidence.execution_device in ("cuda", "cpu")
    assert evidence.software_environment is not None
    assert "transformers" in evidence.software_environment
    assert evidence.processing_stage.value == "ocr"
    if evidence.execution_device == "cuda":
        assert evidence.gpu_memory_mb is not None
        assert evidence.gpu_memory_mb > 0
        assert evidence.hardware_environment is not None
        assert "gpu_name" in evidence.hardware_environment


@requires_swedish_lion_deps
def test_real_recognition_forced_onto_cpu_still_produces_output_and_reports_cpu_device():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    adapter = SwedishLionAdapter(device_request="cpu")
    result = adapter.recognize(RecognitionInput(input_crop_id=str(FIXTURE)))

    assert result.text is not None
    assert result.text.strip() != ""
    evidence = build_evidence(result)
    assert evidence is not None
    assert evidence.execution_device == "cpu"
    assert evidence.gpu_memory_mb is None
    assert evidence.hardware_environment is None


@requires_swedish_lion_deps
def test_real_malformed_image_produces_a_recorded_failure_not_a_crash(tmp_path):
    """Real failure path: a genuinely corrupt (non-image) file fed straight to the real facade."""
    corrupt_image = tmp_path / "corrupt.jpg"
    corrupt_image.write_bytes(b"this is not a jpeg file at all, just garbage bytes \x00\x01\x02")

    adapter = SwedishLionAdapter()
    result = adapter.recognize(RecognitionInput(input_crop_id=str(corrupt_image)))

    assert result.text is None
    assert result.raw_response.get("ok") is False
    assert result.raw_response.get("category") == "malformed_input"

    failure = build_failure_record(result, method_run_id="method_run_real_test")
    assert failure is not None
    assert failure.category == "malformed_input"
    assert failure.reason
