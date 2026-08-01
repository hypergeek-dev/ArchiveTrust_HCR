"""Tests for `memory_probe.py` (WP9). Never invokes real Docker or `nvidia-smi` -- both are
monkeypatched so the probe's *selection logic* (largest-first, first success wins, every attempt
recorded, safety margin computed from real total minus real peak) is verified deterministically,
fast, and without a GPU."""

from __future__ import annotations

import time

import pytest

import archivetrust.htr.training.memory_probe as mp

_real_sleep = time.sleep
"""Captured before any monkeypatching -- `mp.time` and this module's `time` are the *same* module
object (`import time` everywhere refers to one singleton), so patching `mp.time.sleep` mutates
`time.sleep` globally. Without capturing the original first, the patched lambda would recurse into
itself."""


class _FakeCompletedProcess:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def _fake_subprocess_run(argv, **kwargs):
    if "memory.total" in argv[1]:
        return _FakeCompletedProcess("8192\n")
    if "memory.used" in argv[1]:
        return _FakeCompletedProcess("2422\n")
    raise AssertionError(f"unexpected subprocess.run call: {argv}")


class _FakeContainerEpochRunner:
    """Stands in for `ContainerEpochRunner` -- scripted per-batch-size outcome, real tiny sleep so
    the background VRAM poller genuinely gets a chance to run at least once."""

    outcomes: dict[int, bool] = {}

    def __init__(self, *, batch_size, gradient_accumulation, precision, max_image_width, optimizer, learning_rate):
        self._batch_size = batch_size

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        _real_sleep(0.02)
        ok = self.outcomes.get(self._batch_size, False)
        from archivetrust.htr.training.training_session import EpochResult

        return EpochResult(
            ok=ok,
            duration_seconds=0.02,
            error_message=None if ok else f"CUDA OOM at batch_size={self._batch_size}",
        )


@pytest.fixture(autouse=True)
def _patch_environment(monkeypatch):
    monkeypatch.setattr(mp.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(mp.time, "sleep", lambda s: _real_sleep(0.002))
    monkeypatch.setattr(mp, "ContainerEpochRunner", _FakeContainerEpochRunner)
    _FakeContainerEpochRunner.outcomes = {}
    yield


def test_chooses_the_largest_batch_size_that_succeeds(tmp_path):
    _FakeContainerEpochRunner.outcomes = {16: True, 8: True, 4: True}
    report = mp.run_memory_probe(
        probe_train_list_path="train.txt", probe_validation_list_path="val.txt",
        parent_checkpoint_dir="parent", output_root=tmp_path,
        candidate_batch_sizes=(16, 8, 4, 2, 1),
    )
    assert report.chosen_batch_size == 16
    assert len(report.attempts) == 1  # first candidate already succeeded -- no need to try smaller


def test_records_every_attempt_including_failures(tmp_path):
    _FakeContainerEpochRunner.outcomes = {16: False, 8: False, 4: True}
    report = mp.run_memory_probe(
        probe_train_list_path="train.txt", probe_validation_list_path="val.txt",
        parent_checkpoint_dir="parent", output_root=tmp_path,
        candidate_batch_sizes=(16, 8, 4, 2, 1),
    )
    assert [a.batch_size for a in report.attempts] == [16, 8, 4]
    assert report.attempts[0].ok is False
    assert report.attempts[1].ok is False
    assert report.attempts[2].ok is True
    assert report.chosen_batch_size == 4


def test_never_chooses_a_batch_size_when_all_candidates_fail(tmp_path):
    _FakeContainerEpochRunner.outcomes = {}  # every batch size OOMs
    report = mp.run_memory_probe(
        probe_train_list_path="train.txt", probe_validation_list_path="val.txt",
        parent_checkpoint_dir="parent", output_root=tmp_path,
        candidate_batch_sizes=(16, 8, 4, 2, 1),
    )
    assert report.chosen_batch_size is None
    assert report.chosen_peak_vram_mb is None
    assert len(report.attempts) == 5


def test_batch_sizes_are_tried_largest_first(tmp_path):
    _FakeContainerEpochRunner.outcomes = {1: True}
    report = mp.run_memory_probe(
        probe_train_list_path="train.txt", probe_validation_list_path="val.txt",
        parent_checkpoint_dir="parent", output_root=tmp_path,
        candidate_batch_sizes=(16, 8, 4, 2, 1),
    )
    assert [a.batch_size for a in report.attempts] == [16, 8, 4, 2, 1]


def test_safety_margin_is_real_total_minus_real_peak(tmp_path):
    _FakeContainerEpochRunner.outcomes = {16: True}
    report = mp.run_memory_probe(
        probe_train_list_path="train.txt", probe_validation_list_path="val.txt",
        parent_checkpoint_dir="parent", output_root=tmp_path,
        candidate_batch_sizes=(16, 8, 4, 2, 1),
    )
    assert report.total_vram_mb == 8192.0
    assert report.chosen_peak_vram_mb == 2422.0
    assert report.safety_margin_mb == pytest.approx(8192.0 - 2422.0)


def test_effective_batch_size_accounts_for_gradient_accumulation(tmp_path):
    _FakeContainerEpochRunner.outcomes = {16: True}
    report = mp.run_memory_probe(
        probe_train_list_path="train.txt", probe_validation_list_path="val.txt",
        parent_checkpoint_dir="parent", output_root=tmp_path,
        candidate_batch_sizes=(16,), gradient_accumulation=4,
    )
    assert report.chosen_effective_batch_size == 64
