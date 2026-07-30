"""Locks in the per-call isolation properties the SATRN repetition diagnostic verified empirically.

`docs/experiments/technical-reliability-screening/satrn-repetition-diagnostic.md` investigated why
SATRN returned one identical string for 6 of 15 byte-distinct crops, and excluded every
integration-defect hypothesis. Four properties of `facade.py` carried that exclusion:

1. each `run_inference` launches its **own** child process -- no worker is reused across crops, so
   no model state, KV cache or decoder hidden state can survive from one crop to the next;
2. the crop's **own path** is passed straight through as argv -- the facade creates no temporary
   file, so there is no name to collide and no stale content to re-read;
3. each call parses **only its own** child's stdout -- no shared buffer, no response queue, so an
   earlier response cannot be handed to a later call;
4. a failed or malformed response yields a failure record -- it never falls back to a previous
   successful result.

Those properties were true when measured, but nothing stopped a later change from breaking them: a
"keep a warm worker to avoid the ~9s model load" optimization (which `providers/satrn/README.md`
explicitly lists as not implemented, and which the smoke test's own runtime makes tempting) would
silently reintroduce exactly the cross-call-state bug class the diagnostic ruled out. These tests
fail if it does, without a GPU, a model download or the isolated venv.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from archivetrust.providers.satrn import facade as satrn_facade
from archivetrust.providers.satrn.facade import _SubprocessSatrnWorkerFacade


class _RecordingRun:
    """Stands in for `subprocess.run`, recording every launch and returning a distinct response per
    call so a reused or mismatched response is detectable."""

    def __init__(self, *, responses: list[dict] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._responses = responses

    def __call__(self, args, **kwargs):  # noqa: ANN001, ANN204
        self.calls.append(list(args))
        index = len(self.calls) - 1
        if self._responses is not None:
            payload = self._responses[index]
        else:
            payload = {
                "ok": True,
                "text": f"text-for-call-{index}",
                "score": 0.5 + index / 100,
                "elapsed_seconds": 1.0,
                "device_used": "cpu",
                "model_revision": "a40c7093232eaa47a83ce6469fc4abd033486bdc",
                "config_revision": "a40c7093232eaa47a83ce6469fc4abd033486bdc",
            }
        return subprocess.CompletedProcess(args, 0, json.dumps(payload) + "\n", "")


@pytest.fixture
def fake_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points `_find_satrn_python` at a real file so the existence check passes -- the interpreter
    is never actually executed, because `subprocess.run` is replaced."""
    interpreter = tmp_path / "python.exe"
    interpreter.write_text("", encoding="utf-8")
    monkeypatch.setenv("ARCHIVETRUST_SATRN_PYTHON", str(interpreter))
    return interpreter


def _run_three(facade: _SubprocessSatrnWorkerFacade, tmp_path: Path) -> list[Path]:
    crops = []
    for index in range(3):
        crop = tmp_path / f"crop_{index}.png"
        crop.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]) * 32)
        crops.append(crop)
        facade.run_inference(
            image_path=str(crop),
            device_request="cpu",
            model_id="Riksarkivet/satrn_htr",
            revision="a40c7093232eaa47a83ce6469fc4abd033486bdc",
        )
    return crops


def test_each_recognition_launches_its_own_worker_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_venv: Path
) -> None:
    """Property 1. Three crops must produce three separate launches -- not one launch reused."""
    recording = _RecordingRun()
    monkeypatch.setattr(satrn_facade.subprocess, "run", recording)
    _run_three(_SubprocessSatrnWorkerFacade(), tmp_path)
    assert len(recording.calls) == 3


def test_the_crop_path_is_passed_through_and_no_temporary_file_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_venv: Path
) -> None:
    """Properties 2. The argv image path must be the caller's own file, byte-for-byte, and the
    facade must not have written a copy anywhere."""
    recording = _RecordingRun()
    monkeypatch.setattr(satrn_facade.subprocess, "run", recording)
    before = set(tmp_path.rglob("*"))
    crops = _run_three(_SubprocessSatrnWorkerFacade(), tmp_path)

    passed = [call[2] for call in recording.calls]
    assert passed == [str(crop) for crop in crops]
    assert len(set(passed)) == 3, "two calls were handed the same image path"
    # The only new files are the crops the test itself wrote -- no temp copy, no scratch file.
    assert set(tmp_path.rglob("*")) - before == set(crops)


def test_each_call_returns_its_own_workers_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_venv: Path
) -> None:
    """Property 3. Distinct per-call responses must come back distinct and in order -- a shared
    read buffer or a response-matching bug would show up as a repeat here."""
    recording = _RecordingRun()
    monkeypatch.setattr(satrn_facade.subprocess, "run", recording)
    facade = _SubprocessSatrnWorkerFacade()
    results = []
    for index in range(3):
        crop = tmp_path / f"c{index}.png"
        crop.write_bytes(bytes([index]) * 16)
        results.append(
            facade.run_inference(
                image_path=str(crop),
                device_request="cpu",
                model_id="Riksarkivet/satrn_htr",
                revision=None,
            )
        )
    assert [r["text"] for r in results] == ["text-for-call-0", "text-for-call-1", "text-for-call-2"]
    assert len({r["score"] for r in results}) == 3


def test_a_failed_call_never_falls_back_to_the_previous_successful_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_venv: Path
) -> None:
    """Property 4. The failure must surface as a failure, not as a silently retained earlier text --
    the parsing-fallback hypothesis the diagnostic checked for."""
    responses = [
        {"ok": True, "text": "a real transcription", "score": 0.5, "elapsed_seconds": 1.0},
        {"ok": False, "category": "malformed_input", "message": "image not found"},
    ]
    recording = _RecordingRun(responses=responses)
    monkeypatch.setattr(satrn_facade.subprocess, "run", recording)
    facade = _SubprocessSatrnWorkerFacade()

    first = facade.run_inference(
        image_path=str(tmp_path / "a.png"), device_request="cpu", model_id="m", revision=None
    )
    second = facade.run_inference(
        image_path=str(tmp_path / "b.png"), device_request="cpu", model_id="m", revision=None
    )
    assert first["text"] == "a real transcription"
    assert second.get("ok") is False
    assert "text" not in second
    assert second["category"] == "malformed_input"


def test_worker_stdout_polluted_by_mmengine_logging_still_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_venv: Path
) -> None:
    """A real property the diagnostic measured and that `_worker.py`'s docstring does not describe:
    mmengine writes checkpoint-load and deprecation lines to **stdout**, so the worker's actual
    stdout is 5 lines, not the 1 its contract claims. The facade survives because it parses the
    last non-empty line. This test pins that behaviour so a future 'parse stdout as a whole' change
    cannot break it silently.
    """

    def polluted_run(args, **kwargs):  # noqa: ANN001, ANN202
        noise = (
            "Loads checkpoint by local backend from path: C:\\...\\model.pth\n"
            '07/30 20:18:37 - mmengine - WARNING - Failed to search registry with scope "mmocr"\n'
            '07/30 20:18:37 - mmengine - WARNING - "FileClient" will be deprecated in future.\n'
            '07/30 20:18:37 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend".\n'
        )
        payload = {"ok": True, "text": "staden den 27 dennes", "score": 0.5366043906658888}
        return subprocess.CompletedProcess(args, 0, noise + json.dumps(payload) + "\n", "")

    monkeypatch.setattr(satrn_facade.subprocess, "run", polluted_run)
    result = _SubprocessSatrnWorkerFacade().run_inference(
        image_path=str(tmp_path / "c.png"), device_request="cpu", model_id="m", revision=None
    )
    assert result["ok"] is True
    assert result["text"] == "staden den 27 dennes"
    assert result["score"] == 0.5366043906658888
