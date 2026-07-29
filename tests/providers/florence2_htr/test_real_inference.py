"""Real-inference Florence-2 adapter tests (`docs/htr-migration-plan.md` Stage 7 validation) --
downloads the real fine-tuned checkpoint `nazounoryuu/florence_base__mixed__line_bbox__ocr` (first
run only, cached afterwards) and runs real forward passes through plain `transformers` in this
project's main venv. Requires:

- The `transformers` extra installed (`pip install -e ".[transformers]"`), plus `timm`/`einops`
  (Florence-2's `trust_remote_code` modeling file's own runtime dependencies -- see
  `providers/florence2_htr/README.md`).
- `transformers` pinned inside this project's declared `>=4.40,<6.0` range to a version that can
  actually construct Florence-2's remote config (verified working: `4.49.0`; verified broken:
  `5.13.1` -- see `facade.py`'s module docstring).
- Network access to huggingface.co (first run only -- afterwards the HF cache is warm).

Marked `real_model` (registered in `pyproject.toml`), same convention as
`tests/providers/satrn/test_real_inference.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.providers.florence2_htr.adapter import (
    DEFAULT_MODEL_REVISION,
    Florence2Adapter,
    build_evidence,
    build_failure_record,
)
from archivetrust.providers.florence2_htr.facade import florence2_dependencies_available
from archivetrust.providers.htr_adapter import RecognitionInput

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "htr" / "trolldomskommissionen_sample_line.jpg"

pytestmark = pytest.mark.real_model

_deps_ok, _deps_message = florence2_dependencies_available()
requires_florence2_deps = pytest.mark.skipif(not _deps_ok, reason=_deps_message)


@requires_florence2_deps
def test_real_recognition_on_the_fixture_line_image_produces_real_output():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    adapter = Florence2Adapter()
    result = adapter.recognize(RecognitionInput(input_crop_id=str(FIXTURE)))

    # Meaningful assertions about REAL model output, not just "did not crash":
    assert result.text is not None
    assert result.text.strip() != ""
    assert result.execution_time_ms is not None
    assert result.execution_time_ms > 0
    # The exact fine-tuned-checkpoint revision this session pinned and verified by downloading it.
    assert result.model_revision == DEFAULT_MODEL_REVISION
    assert result.confidence is not None
    assert 0.0 < result.confidence <= 1.0
    # Raw response carries the genuinely raw (special-token-laden) decoder text, distinct from
    # the parsed RecognitionResult.text.
    assert result.raw_response["raw_decoded"] != result.text
    assert "<s>" in result.raw_response["raw_decoded"] or "</s>" in result.raw_response["raw_decoded"]

    evidence = build_evidence(result)
    assert evidence is not None
    assert evidence.raw_output == result.raw_response["raw_decoded"]
    assert evidence.raw_output != result.text  # raw vs. parsed genuinely differ
    assert evidence.model_revision == result.model_revision
    assert evidence.execution_device in ("cuda", "cpu")
    assert evidence.software_environment is not None
    assert "transformers" in evidence.software_environment
    assert evidence.processing_stage.value == "vlm_inference"
    if evidence.execution_device == "cuda":
        assert evidence.gpu_memory_mb is not None
        assert evidence.gpu_memory_mb > 0
        assert evidence.hardware_environment is not None
        assert "gpu_name" in evidence.hardware_environment


@requires_florence2_deps
def test_real_recognition_forced_onto_cpu_still_produces_output_and_reports_cpu_device():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    adapter = Florence2Adapter(device_request="cpu")
    result = adapter.recognize(RecognitionInput(input_crop_id=str(FIXTURE)))

    assert result.text is not None
    assert result.text.strip() != ""
    evidence = build_evidence(result)
    assert evidence is not None
    assert evidence.execution_device == "cpu"
    assert evidence.gpu_memory_mb is None
    assert evidence.hardware_environment is None


@requires_florence2_deps
def test_real_malformed_image_produces_a_recorded_failure_not_a_crash(tmp_path):
    """Real failure path: a genuinely corrupt (non-image) file fed straight to the real facade.
    Proves the malformed_input classification is exercised for real, not merely unit-tested
    against a fake facade -- mirrors SATRN's equivalent real-malformed-input test."""
    corrupt_image = tmp_path / "corrupt.jpg"
    corrupt_image.write_bytes(b"this is not a jpeg file at all, just garbage bytes \x00\x01\x02")

    adapter = Florence2Adapter()
    result = adapter.recognize(RecognitionInput(input_crop_id=str(corrupt_image)))

    assert result.text is None
    assert result.raw_response.get("ok") is False
    assert result.raw_response.get("category") == "malformed_input"

    failure = build_failure_record(result, method_run_id="method_run_real_test")
    assert failure is not None
    assert failure.category == "malformed_input"
    assert failure.reason  # a real, non-empty message
