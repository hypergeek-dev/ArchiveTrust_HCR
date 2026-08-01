from __future__ import annotations

import json
import zipfile

import pytest

from archivetrust.htr.training.session_report import generate_session_reports
from archivetrust.htr.training.training_session import EpochResult, run_training_session


class _FakeEpochRunner:
    def __init__(self):
        self.calls = 0

    def run_epoch(self, *, existing_model_dir, output_dir, train_list_path, validation_list_path, epoch_seed):
        from pathlib import Path

        self.calls += 1
        ckpt_dir = Path(output_dir) / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
            zf.writestr("config.json", "{}")
        return EpochResult(
            ok=True, train_cer=0.5, val_cer=0.4, duration_seconds=1.0,
            checkpoint_dir=str(ckpt_dir), best_val_checkpoint_dir=str(ckpt_dir), exit_code=0,
        )


@pytest.fixture()
def parent_checkpoint_dir(tmp_path):
    d = tmp_path / "parent_checkpoint"
    d.mkdir()
    (d / "model.keras").write_bytes(b"fake pretrained weights")
    return str(d)


@pytest.fixture()
def trained_run(tmp_path, parent_checkpoint_dir):
    run_state_dir = tmp_path / "run_state"
    checkpoint_index_path = tmp_path / "checkpoint_index.json"
    run_training_session(
        run_state_dir=run_state_dir, checkpoint_index_path=checkpoint_index_path, epoch_runner=_FakeEpochRunner(),
        train_list_path="train.txt", validation_list_path="val.txt", parent_checkpoint_dir=parent_checkpoint_dir,
        run_id="r1", configuration_hash="h1", random_seed=1, max_wall_clock_seconds=1e9,
        stop_requested=lambda: False, max_epochs_this_call=2,
    )
    return {"run_state_dir": run_state_dir, "checkpoint_index_path": checkpoint_index_path}


def test_reconstructs_a_report_without_rerunning_training(tmp_path, trained_run):
    reports_dir = tmp_path / "reports"
    report = generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert report["run_id"] == "r1"
    assert report["cumulative_epoch"] == 2


def test_writes_both_json_and_markdown(tmp_path, trained_run):
    reports_dir = tmp_path / "reports"
    generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert (reports_dir / "session-report.json").exists()
    assert (reports_dir / "session-report.md").exists()


def test_never_claims_quality_the_markdown_carries_the_disclaimer(tmp_path, trained_run):
    reports_dir = tmp_path / "reports"
    generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    md = (reports_dir / "session-report.md").read_text(encoding="utf-8")
    assert "not a" in md.lower() and "quality" in md.lower()
    assert "reserved test set" in md


def test_latest_and_best_checkpoint_are_included(tmp_path, trained_run):
    reports_dir = tmp_path / "reports"
    report = generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert report["latest_checkpoint"] is not None
    assert report["latest_checkpoint"]["resumable"] is True
    assert report["best_checkpoint"] is not None


def test_gracefully_omits_optional_sections_when_their_files_are_absent(tmp_path, trained_run):
    """Character-compatibility/memory-probe/resume-proof/pilot-split reports are each optional
    inputs -- a report generated before all of them exist must not crash, and must honestly say
    `None` rather than fabricate placeholder content."""
    reports_dir = tmp_path / "reports"
    report = generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert report["pilot_split_summary"] is None
    assert report["character_compatibility"] is None
    assert report["memory_probe"] is None
    assert report["resume_proof"] is None


def test_includes_optional_sections_when_their_files_are_present(tmp_path, trained_run):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "character-compatibility.json").write_text(
        json.dumps({"total_lines_examined": 10999, "compatible": True, "unsupported_character_count": 0}),
        encoding="utf-8",
    )
    (reports_dir / "memory-probe.json").write_text(
        json.dumps({"chosen_batch_size": 16, "chosen_peak_vram_mb": 2422.0, "total_vram_mb": 8192.0, "safety_margin_mb": 5770.0}),
        encoding="utf-8",
    )
    (reports_dir / "resume-proof.json").write_text(
        json.dumps({"proof_passed": True, "epoch_continued_not_restarted": True, "global_step_continued_not_restarted": True}),
        encoding="utf-8",
    )
    report = generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert report["character_compatibility"]["compatible"] is True
    assert report["memory_probe"]["chosen_batch_size"] == 16
    assert report["resume_proof"]["proof_passed"] is True

    md = (reports_dir / "session-report.md").read_text(encoding="utf-8")
    assert "RTX 3070 memory probe" in md
    assert "Full-state resume proof" in md


def test_validation_history_table_reflects_real_recorded_epochs(tmp_path, trained_run):
    reports_dir = tmp_path / "reports"
    report = generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert len(report["validation_history"]) == 2
    assert report["validation_history"][0]["val_cer"] == 0.4


def test_reports_when_no_session_has_ever_run(tmp_path):
    """A report requested before any training has happened must not crash -- honest zeros/`None`s,
    not fabricated progress."""
    reports_dir = tmp_path / "reports"
    report = generate_session_reports(
        run_state_dir=tmp_path / "never_run", checkpoint_index_path=tmp_path / "no_index.json",
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert report["run_id"] is None
    assert report["cumulative_epoch"] == 0
    assert report["latest_checkpoint"] is None
    assert report["last_stop_reason"] is None


def test_reports_the_last_sessions_stop_boundary(tmp_path, trained_run):
    reports_dir = tmp_path / "reports"
    report = generate_session_reports(
        run_state_dir=trained_run["run_state_dir"], checkpoint_index_path=trained_run["checkpoint_index_path"],
        manifests_dir=tmp_path / "manifests_absent", reports_dir=reports_dir, dataset_root="F:/huggingface_dataset",
    )
    assert report["last_stop_reason"] == "target_epochs_reached"
    assert report["last_stopped_mid_epoch"] is False
    assert report["last_stop_boundary"] == "after_validation_at_epoch_boundary"

    md = (reports_dir / "session-report.md").read_text(encoding="utf-8")
    assert "target_epochs_reached" in md
    assert "after_validation_at_epoch_boundary" in md
