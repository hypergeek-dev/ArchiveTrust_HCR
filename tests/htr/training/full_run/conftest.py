"""Shared fixtures for `htr/training/full_run/` tests -- synthetic pilot/full-run directory
builders, never real Docker/GPU."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from archivetrust.htr.training.checkpoint_index import CheckpointEntry, append_checkpoint_entry


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_pilot_fixture(
    pilot_dir: Path,
    *,
    val_cers: list[float],
    train_cers: list[float] | None = None,
    train_losses: list[float | None] | None = None,
    val_losses: list[float | None] | None = None,
    durations: list[float] | None = None,
    batch_size: int = 16,
    train_line_count: int = 9999,
    val_line_count: int = 1000,
    run_id: str = "loghi_training_run_fixture",
) -> Path:
    """Builds a real, synthetic pilot run directory with the exact same file shapes
    `analyze_pilot_run` reads -- `session_state.json`'s `validation_history`, `checkpoint_index.json`,
    `memory-probe.json`, `pilot_split_summary.json`, `training_identity.json`."""
    n = len(val_cers)
    train_cers = train_cers or [v + 0.15 for v in val_cers]
    train_losses = train_losses or [None] * n
    val_losses = val_losses or [None] * n
    durations = durations or [480.0] * n

    run_state_dir = pilot_dir / "run-state"
    run_state_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_state_dir / "training_identity.json",
        {
            "identity": {
                "method_id": "loghi_swedish_finetuned_v1", "parent_method_id": "loghi",
                "parent_checkpoint": "generic-2023-02-15@abc", "training_dataset": "d",
                "training_phase": "pilot_10k", "run_id": run_id, "created_at": "2026-08-01T00:00:00Z",
            },
            "configuration_hash": "h1", "configuration": {},
        },
    )

    validation_history = []
    checkpoint_index_path = run_state_dir / "checkpoint_index.json"
    epoch_output = run_state_dir / "epoch_output"
    best_val_cer = None
    for i in range(n):
        epoch = i + 1
        entry = {
            "epoch": epoch, "train_cer": train_cers[i], "val_cer": val_cers[i],
            "train_wer": train_cers[i] + 0.5, "val_wer": val_cers[i] + 0.5,
            "duration_seconds": durations[i],
        }
        if train_losses[i] is not None:
            entry["train_loss"] = train_losses[i]
        if val_losses[i] is not None:
            entry["val_loss"] = val_losses[i]
        validation_history.append(entry)

        ckpt_dir = epoch_output / f"epoch_{epoch}" / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ckpt_dir / "model.keras", "w") as zf:
            zf.writestr("config.json", "{}" + ("x" * epoch))  # varies real file size per epoch

        append_checkpoint_entry(
            checkpoint_index_path,
            CheckpointEntry(
                checkpoint_id=f"ckpt_latest_{epoch}", run_id=run_id, session_id="s1",
                source_checkpoint="parent", epoch=epoch, global_step=0,
                cumulative_training_seconds=sum(durations[: i + 1]), session_training_seconds=0.0,
                checkpoint_dir=str(ckpt_dir), model_file_hash="h", model_state_present=True,
                optimizer_state_present=True, scheduler_state_present=True, sampler_state_present=False,
                configuration_hash="h1", training_manifest_hash="", validation_manifest_hash="",
                validation_metrics={"val_CER_metric": val_cers[i]}, created_at=f"2026-08-01T00:{epoch:02d}:00Z",
                verification_status="verified", resumable=True, checkpoint_kind="latest",
            ),
        )
        if best_val_cer is None or val_cers[i] < best_val_cer:
            best_val_cer = val_cers[i]
            append_checkpoint_entry(
                checkpoint_index_path,
                CheckpointEntry(
                    checkpoint_id=f"ckpt_best_{epoch}", run_id=run_id, session_id="s1",
                    source_checkpoint="parent", epoch=epoch, global_step=0,
                    cumulative_training_seconds=sum(durations[: i + 1]), session_training_seconds=0.0,
                    checkpoint_dir=str(ckpt_dir), model_file_hash="h", model_state_present=True,
                    optimizer_state_present=True, scheduler_state_present=True, sampler_state_present=False,
                    configuration_hash="h1", training_manifest_hash="", validation_manifest_hash="",
                    validation_metrics={"val_CER_metric": val_cers[i]}, created_at=f"2026-08-01T00:{epoch:02d}:00Z",
                    verification_status="verified", resumable=True, checkpoint_kind="best_val",
                ),
            )

    _write_json(
        run_state_dir / "session_state.json",
        {
            "run_id": run_id, "configuration_hash": "h1", "random_seed": 1, "cumulative_epoch": n,
            "global_step": 0, "best_val_cer": min(val_cers), "best_checkpoint_dir": None,
            "latest_checkpoint_dir": None, "validation_history": validation_history, "session_count": 1,
            "cumulative_training_seconds": sum(durations), "recent_epoch_durations": durations[-5:],
            "epochs_since_improvement": 0,
        },
    )

    _write_json(
        pilot_dir / "reports" / "memory-probe.json",
        {"chosen_batch_size": batch_size, "chosen_peak_vram_mb": 2422.0, "safety_margin_mb": 5770.0, "total_vram_mb": 8192.0},
    )

    _write_json(
        pilot_dir / "manifests" / "pilot_split_summary.json",
        {
            "seed": 1, "target_train": train_line_count, "target_val": val_line_count,
            "target_test_reserved": 100, "actual_train": train_line_count, "actual_val": val_line_count,
            "actual_test_reserved": 100, "max_collection_fraction": 0.25,
            "train_collection_distribution": {}, "val_collection_distribution": {},
            "test_reserved_collection_distribution": {}, "single_file_collections": [],
            "train_manifest_hash": "th", "val_manifest_hash": "vh", "test_reserved_manifest_hash": "tth",
            "shortfall_warnings": [],
        },
    )

    return pilot_dir


@pytest.fixture()
def pilot_fixture_dir(tmp_path):
    return tmp_path / "pilot-fixture"
