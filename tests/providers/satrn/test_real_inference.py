"""Real-inference SATRN adapter tests (`docs/htr-migration-plan.md` Stage 6 validation) --
downloads the real `Riksarkivet/satrn_htr` checkpoint (first run only, cached afterwards) and
runs real forward passes through the isolated `.venv-satrn` interpreter. Requires:

- `.venv-satrn` set up per `src/archivetrust/providers/satrn/README.md` (or
  `ARCHIVETRUST_SATRN_PYTHON` pointing at an equivalent interpreter with `mmocr`/`mmengine`/
  `mmcv`/`mmdet`/`torch`/`huggingface_hub` installed).
- Network access to huggingface.co (first run only -- afterwards the HF cache is warm).

Marked `real_model` (registered in `pyproject.toml`) so a CI/dev loop that wants a fast default
sweep can run `pytest -m "not real_model"`; the task's own validation commands run these
explicitly (`pytest tests/providers/satrn -q -v`) and as part of the full default suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.providers.htr_adapter import RecognitionInput
from archivetrust.providers.satrn.adapter import SatrnAdapter, build_evidence
from archivetrust.providers.satrn.facade import satrn_python_available

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "htr" / "trolldomskommissionen_sample_line.jpg"
CORRUPT_FIXTURE_TEXT = b"this is not a jpeg file at all, just garbage bytes \x00\x01\x02"

pytestmark = pytest.mark.real_model

_satrn_python_ok, _satrn_python_message = satrn_python_available()
requires_satrn_venv = pytest.mark.skipif(not _satrn_python_ok, reason=_satrn_python_message)


@requires_satrn_venv
def test_real_recognition_on_the_fixture_line_image_produces_real_output():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    adapter = SatrnAdapter()
    result = adapter.recognize(RecognitionInput(input_crop_id=str(FIXTURE)))

    # Meaningful assertions about REAL model output, not just "did not crash":
    assert result.text is not None
    assert result.text.strip() != ""
    assert result.execution_time_ms is not None
    assert result.execution_time_ms > 0
    # The exact revision this session pinned and verified by downloading the real checkpoint.
    assert result.model_revision == "a40c7093232eaa47a83ce6469fc4abd033486bdc"
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0

    evidence = build_evidence(result)
    assert evidence is not None
    assert evidence.raw_output == result.text
    assert evidence.model_revision == result.model_revision
    assert evidence.execution_device in ("cuda", "cpu")
    assert evidence.software_environment is not None
    assert "torch" in evidence.software_environment
    if evidence.execution_device == "cuda":
        assert evidence.gpu_memory_mb is not None
        assert evidence.gpu_memory_mb > 0
        assert evidence.hardware_environment is not None
        assert "gpu_name" in evidence.hardware_environment


@requires_satrn_venv
def test_real_recognition_forced_onto_cpu_still_produces_output_and_reports_cpu_device():
    """Real CPU-fallback path -- proves `device="cpu"` genuinely bypasses the GPU, not just a
    label. Measured in this session: ~10.7s on CPU vs. ~1.3s on GPU for the same input."""
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    adapter = SatrnAdapter(device_request="cpu")
    result = adapter.recognize(RecognitionInput(input_crop_id=str(FIXTURE)))

    assert result.text is not None
    assert result.text.strip() != ""
    evidence = build_evidence(result)
    assert evidence is not None
    assert evidence.execution_device == "cpu"
    assert evidence.gpu_memory_mb is None
    assert evidence.hardware_environment is None


@requires_satrn_venv
def test_real_malformed_image_produces_a_recorded_failure_not_a_crash(tmp_path):
    """Real failure path: a genuinely corrupt (non-JPEG) file fed straight to the real worker
    subprocess. Proves the malformed_input classification is exercised for real, not merely
    unit-tested against a fake facade."""
    from archivetrust.providers.satrn.adapter import build_failure_record

    corrupt_image = tmp_path / "corrupt.jpg"
    corrupt_image.write_bytes(CORRUPT_FIXTURE_TEXT)

    adapter = SatrnAdapter()
    result = adapter.recognize(RecognitionInput(input_crop_id=str(corrupt_image)))

    assert result.text is None
    assert result.raw_response.get("ok") is False
    assert result.raw_response.get("category") == "malformed_input"

    failure = build_failure_record(result, method_run_id="method_run_real_test")
    assert failure is not None
    assert failure.category == "malformed_input"
    assert failure.reason  # a real, non-empty message from the real worker's exception
