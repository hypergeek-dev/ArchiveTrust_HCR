from __future__ import annotations

import json
import sys
import time

import pytest

import archivetrust.htr.training.telemetry_sampler as ts


class _FakeCompletedProcess:
    def __init__(self, stdout: str):
        self.stdout = stdout


def _fake_nvidia_smi_run(argv, **kwargs):
    return _FakeCompletedProcess("NVIDIA GeForce RTX 3070, 42, 2456, 8192, 61\n")


@pytest.fixture(autouse=True)
def _fast_interval(monkeypatch):
    monkeypatch.setattr(ts, "DEFAULT_SAMPLE_INTERVAL_SECONDS", 0.02)


def test_gpu_sample_parses_a_real_shaped_nvidia_smi_line(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(ts.subprocess, "run", _fake_nvidia_smi_run)
    sample = ts._query_gpu_sample()
    assert sample.name == "NVIDIA GeForce RTX 3070"
    assert sample.utilization_pct == 42.0
    assert sample.memory_used_mb == 2456.0
    assert sample.memory_total_mb == 8192.0
    assert sample.temperature_c == 61.0


def test_gpu_sample_is_all_none_when_nvidia_smi_absent(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda name: None)
    sample = ts._query_gpu_sample()
    assert sample.name is None
    assert sample.utilization_pct is None
    assert sample.memory_used_mb is None


def test_gpu_sample_is_all_none_on_subprocess_failure(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _raise(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(ts.subprocess, "run", _raise)
    sample = ts._query_gpu_sample()
    assert sample.name is None


def test_system_sample_degrades_gracefully_when_psutil_absent(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "psutil", None)  # forces `import psutil` to raise ImportError
    sample = ts._query_system_sample(tmp_path)
    assert sample.ram_used_mb is None
    assert sample.ram_total_mb is None
    assert sample.cpu_pct is None
    # disk telemetry is stdlib-only and must still work even without psutil
    assert sample.disk_free_gb is not None
    assert sample.disk_free_gb > 0


def test_system_sample_reports_real_positive_disk_free_space(tmp_path):
    sample = ts._query_system_sample(tmp_path)
    assert sample.disk_free_gb is not None
    assert sample.disk_free_gb > 0


def test_sampler_writes_at_least_one_jsonl_line_and_a_status_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda name: None)  # no real GPU needed for this test
    run_state_dir = tmp_path / "run_state"
    output_dir = tmp_path / "epoch_output"
    sampler = ts.TelemetrySampler(run_state_dir=run_state_dir, epoch=3, output_dir=output_dir, interval_seconds=0.02)
    sampler.start()
    time.sleep(0.1)
    sampler.stop()

    jsonl_path = run_state_dir / "telemetry" / "gpu_samples.jsonl"
    status_path = run_state_dir / "telemetry" / "status.json"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    first_row = json.loads(lines[0])
    assert first_row["epoch"] == 3

    assert status_path.exists()


def test_status_json_reflects_container_alive_false_after_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda name: None)
    run_state_dir = tmp_path / "run_state"
    sampler = ts.TelemetrySampler(
        run_state_dir=run_state_dir, epoch=1, output_dir=tmp_path / "out", interval_seconds=0.02
    )
    sampler.start()
    time.sleep(0.05)
    sampler.stop()

    status = json.loads((run_state_dir / "telemetry" / "status.json").read_text(encoding="utf-8"))
    assert status["container_alive"] is False  # the final tick on stop() always marks this


def test_sampler_works_as_a_context_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda name: None)
    run_state_dir = tmp_path / "run_state"
    with ts.TelemetrySampler(run_state_dir=run_state_dir, epoch=2, output_dir=tmp_path / "out", interval_seconds=0.02):
        time.sleep(0.05)

    assert (run_state_dir / "telemetry" / "status.json").exists()


def test_malformed_nvidia_smi_output_does_not_crash(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(ts.subprocess, "run", lambda *a, **k: _FakeCompletedProcess("not,enough,fields\n"))
    sample = ts._query_gpu_sample()
    assert sample.name is None  # falls back to all-None rather than raising
