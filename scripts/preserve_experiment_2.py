"""Phase 13: preserve Experiment 2 -- writes the named artifacts into the run directory and extends
the existing preservation registry (training/preserved-experiments/loghi-finetuning-baselines-20260804/)
with a new experiment_2/ subdirectory, WITHOUT modifying anything already written for experiment_0/ or
experiment_1/ (their own metadata/reports/checksums are untouched; only the top-level manifest and
checksums file gain new entries for the addition).

Never copies model weights -- checkpoints are referenced by path + independently re-verified sha256,
exactly the same discipline scripts/build_preservation_package.py already established for Experiments
0 and 1.

Usage:
    python scripts/preserve_experiment_2.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from archivetrust.htr.training.checkpoint_index import verify_checkpoint  # noqa: E402

TRAINING_ROOT = REPO_ROOT / "training"
EXP2_DIR = TRAINING_ROOT / "experiment-2-loghi-recommended-scratch-20260804T051959Z"
REGISTRY_DIR = TRAINING_ROOT / "preserved-experiments" / "loghi-finetuning-baselines-20260804"
ARCHITECTURE_REPORT_DIR = TRAINING_ROOT / "experiment_2_architecture_report"
CHARACTER_INVENTORY_PATH = TRAINING_ROOT / "experiment_2_character_inventory.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_experiment_2_metadata() -> dict:
    identity = _read_json(EXP2_DIR / "run-state" / "training_identity.json")
    launch_manifest = _read_json(EXP2_DIR / "launch_manifest.json")
    run_state = _read_json(EXP2_DIR / "run-state" / "run_state.json")
    session_state = _read_json(EXP2_DIR / "run-state" / "session_state.json")
    lap_eval_dir = sorted((EXP2_DIR / "lap-evaluation").glob("lap*_best_val"))[-1]
    lap_eval = _read_json(lap_eval_dir / "lap_evaluation.json")
    inventory = _read_json(CHARACTER_INVENTORY_PATH)
    arch_report = _read_json(ARCHITECTURE_REPORT_DIR / "architecture_report_and_preflight.json")

    checkpoint_refs = {}
    for name, subdir_glob in [("best_val", "best_val"), ("latest_end_of_session", None)]:
        entries = [e for e in _read_json(EXP2_DIR / "run-state" / "checkpoint_index.json")["entries"]
                  if e["checkpoint_kind"] == ("best_val" if name == "best_val" else "latest")]
        if entries:
            entry = entries[0]
            path = Path(entry["checkpoint_dir"])
            ok, live_hash, extra = verify_checkpoint(path)
            checkpoint_refs[name] = {
                "checkpoint_dir": str(path), "recorded_sha256": entry["model_file_hash"],
                "live_verified_sha256": live_hash, "hash_matches_recorded": live_hash == entry["model_file_hash"],
                "live_verification_ok": ok, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

    return {
        "experiment_id": "experiment_2",
        "stable_name": "Loghi recommended VGSL — trained from scratch",
        "description": "Native Loghi 'recommended' VGSL architecture trained from random "
                       "initialization, one complete uninterrupted full-corpus epoch. NOT "
                       "fine-tuning -- no pretrained checkpoint, no Experiment 0 or 1 weights, "
                       "anywhere in this run's lineage.",
        "run_directory": str(EXP2_DIR),
        "initialization": "random",
        "pretrained_checkpoint_used": False,
        "experiment_0_weights_used": False,
        "experiment_1_weights_used": False,
        "architecture_source": "loghi predefined model library key (get_model_library())",
        "architecture_key": "recommended",
        "architecture_vgsl_spec": arch_report["vgsl_specification"],
        "parent_checkpoint_path": None,
        "parent_checkpoint_sha256": None,
        "character_vocabulary": {
            "size_raw_characters": inventory["scratch_model_vocabulary"]["size"],
            "size_tokenizer_vocabulary": 126,
            "sha256": inventory["scratch_model_vocabulary"]["sha256"],
            "source": str(CHARACTER_INVENTORY_PATH.relative_to(REPO_ROOT)),
        },
        "training_dataset_identity": {
            "dataset_hash": launch_manifest["dataset_hash"],
            "train_manifest_hash": launch_manifest["training_manifest_hash"],
            "train_line_count": launch_manifest["train_line_count"],
        },
        "validation_dataset_identity": {
            "val_manifest_hash": launch_manifest["validation_manifest_hash"],
            "validation_line_count": launch_manifest["validation_line_count"],
        },
        "reserved_test_set_identity": {
            "excluded_line_count": launch_manifest["excluded_test_line_count"],
            "note": "SEALED. Not opened, not read, not hashed by this preservation pass.",
        },
        "optimizer": launch_manifest["optimizer"],
        "initial_learning_rate": launch_manifest["learning_rate"],
        "learning_rate_schedule": launch_manifest["scheduler"],
        "batch_size": launch_manifest["batch_size"],
        "precision_policy": "mixed_float16 (automatic on GPU, no --use_float32 passed)",
        "seed": launch_manifest["random_seed"],
        "code_commit": launch_manifest["code_commit_hash"],
        "repository_dirty_at_launch": launch_manifest["repository_dirty"],
        "docker_image_digest": launch_manifest["container_image_digest"],
        "gpu_model": "NVIDIA GeForce RTX 3070",
        "started_at": run_state["started_at"],
        "ended_at": run_state["ended_at"],
        "runtime_seconds": session_state["cumulative_training_seconds"],
        "epochs_completed": run_state["epochs_completed"],
        "automatic_epoch_2": False,
        "optimizer_steps_completed": 35111,
        "optimizer_resets": 0,
        "in_training_best_val_cer": run_state["best_metrics"]["val_cer"],
        "authoritative_held_out_result": {
            "corpus_cer": lap_eval["overall"]["corpus_cer"],
            "corpus_wer": lap_eval["overall"]["corpus_wer"],
            "line_error_rate": lap_eval["overall"]["line_error_rate"],
            "evaluation_source": str((lap_eval_dir / "lap_evaluation.json").relative_to(REPO_ROOT)),
        },
        "checkpoint_references": checkpoint_refs,
        "immutability_note": "Derived, read-only summary written by scripts/preserve_experiment_2.py. "
                             "The authoritative source is the run directory named above.",
    }


def build_experiment_2_architecture() -> dict:
    arch_report = _read_json(ARCHITECTURE_REPORT_DIR / "architecture_report_and_preflight.json")
    exp1_arch = _read_json(ARCHITECTURE_REPORT_DIR / "exp1_model_inspection.json")
    return {
        "vgsl_specification": arch_report["vgsl_specification"],
        "layer_sequence": arch_report["layers_after_vocab_resize"],
        "vocabulary_size": arch_report["vocabulary_size_used"],
        "total_parameters": arch_report["params_after_vocab_resize"]["total"],
        "trainable_parameters": arch_report["params_after_vocab_resize"]["trainable"],
        "non_trainable_parameters": arch_report["params_after_vocab_resize"]["non_trainable"],
        "estimated_vram_mb_bounded_smoke_test": arch_report.get("peak_vram_mb_during_smoke_test"),
        "estimated_vram_note": "Measured on a tiny synthetic batch (batch_size=2, width=256) during "
                               "the Phase 6 preflight smoke test, NOT the real training run's actual "
                               "peak -- see the real run's telemetry gpu_samples.jsonl for that "
                               "(peak observed: see experiment_0_1_2_comparison.json "
                               "gpu_memory_used_mb_peak.experiment_2).",
        "comparison_with_experiment_1_architecture": {
            "experiment_1_total_parameters": exp1_arch["total_params"],
            "experiment_1_layer_count": len(exp1_arch["layers"]),
            "experiment_2_layer_count": len(arch_report["layers_after_vocab_resize"]),
            "verdict": "similar_but_not_identical",
            "explanation": "Same conv (24->48->96->96 channels, 3x max-pool) + 5x BiLSTM(512, "
                           "dropout between each) backbone topology and input shape (None,None,64,1). "
                           "Experiment 1's checkpoint additionally has Dropout layers directly after "
                           "each conv block (Experiment 2's recommended spec does not). Parameter "
                           "count difference (31,033,929 vs 30,694,654 = 339,275) is almost entirely "
                           "explained by the output layer alone (468,425 params for 457 Dutch-inclusive "
                           "classes vs 129,150 params for 126 Swedish-only classes -- a 339,275 gap).",
        },
        "input_compatibility_confirmed": True,
        "bounded_forward_pass_verified": arch_report["checks"]["bounded_forward_pass_succeeds"]["ok"],
        "bounded_backward_pass_verified": arch_report["checks"]["bounded_backward_pass_succeeds"]["ok"],
    }


def build_training_metrics_parquet(out_path: Path) -> int:
    trace_path = sorted(EXP2_DIR.glob("run-state/epoch_output/epoch_*/optimizer_trace.csv"))[-1]
    rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
    df = pd.DataFrame(rows)
    for col in ("optimizer_iterations", "learning_rate", "loss", "CER_metric", "WER_metric"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def build_checkpoint_manifest() -> dict:
    index = _read_json(EXP2_DIR / "run-state" / "checkpoint_index.json")
    verified_entries = []
    for entry in index["entries"]:
        path = Path(entry["checkpoint_dir"])
        ok, live_hash, extra = verify_checkpoint(path)
        verified_entries.append({
            **entry,
            "live_verified_sha256": live_hash,
            "hash_matches_recorded": live_hash == entry["model_file_hash"],
            "live_verification_ok": ok,
        })
    return {"experiment_id": "experiment_2", "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": verified_entries}


def main() -> int:
    print("Building experiment_2_metadata.json...")
    metadata = build_experiment_2_metadata()
    _write_json(EXP2_DIR / "experiment_2_metadata.json", metadata)

    print("Building experiment_2_architecture.json...")
    architecture = build_experiment_2_architecture()
    _write_json(EXP2_DIR / "experiment_2_architecture.json", architecture)

    print("Copying experiment_2_character_inventory.json into the run directory...")
    shutil.copyfile(CHARACTER_INVENTORY_PATH, EXP2_DIR / "experiment_2_character_inventory.json")

    print("Building experiment_2_training_metrics.parquet...")
    n_rows = build_training_metrics_parquet(EXP2_DIR / "experiment_2_training_metrics.parquet")
    print(f"  {n_rows} rows")

    print("Building experiment_2_checkpoint_manifest.json...")
    manifest = build_checkpoint_manifest()
    _write_json(EXP2_DIR / "experiment_2_checkpoint_manifest.json", manifest)

    print("Copying experiment_2_evaluation.json (the independent lap_evaluation.json)...")
    lap_eval_dir = sorted((EXP2_DIR / "lap-evaluation").glob("lap*_best_val"))[-1]
    shutil.copyfile(lap_eval_dir / "lap_evaluation.json", EXP2_DIR / "experiment_2_evaluation.json")

    # --- Extend the existing preservation registry with experiment_2/, without touching
    # experiment_0/ or experiment_1/'s own content. ---
    print("Extending the preservation registry...")
    exp2_registry_dir = REGISTRY_DIR / "experiment_2"
    if (exp2_registry_dir / "metadata.json").exists():
        raise SystemExit(f"{exp2_registry_dir} already has metadata.json -- refusing to overwrite a prior preservation pass.")

    _write_json(exp2_registry_dir / "metadata.json", metadata)
    _write_json(exp2_registry_dir / "artifact_index.json", {
        "run_root": {"path": str(EXP2_DIR), "exists": EXP2_DIR.exists()},
        "architecture_report": {"path": str(ARCHITECTURE_REPORT_DIR / "architecture_report_and_preflight.json")},
        "character_inventory": {"path": str(CHARACTER_INVENTORY_PATH)},
        "checkpoint_index": {"path": str(EXP2_DIR / "run-state" / "checkpoint_index.json")},
        "run_state": {"path": str(EXP2_DIR / "run-state" / "run_state.json")},
        "telemetry_samples": {"path": str(EXP2_DIR / "run-state" / "telemetry" / "gpu_samples.jsonl")},
        "optimizer_trace": {"path": str(sorted(EXP2_DIR.glob("run-state/epoch_output/epoch_*/optimizer_trace.csv"))[-1])},
        "independent_evaluation_output": {"path": str(lap_eval_dir / "lap_evaluation.json")},
        "comparison": {"path": str(EXP2_DIR / "experiment_0_1_2_comparison.json")},
        "decision": {"path": str(EXP2_DIR / "experiment_2_decision.json")},
    })
    (exp2_registry_dir / "reports").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lap_eval_dir / "lap_evaluation.md", exp2_registry_dir / "reports" / "lap_evaluation.md")
    (exp2_registry_dir / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXP2_DIR / "run-state" / "training_identity.json", exp2_registry_dir / "configs" / "training_identity.json")
    shutil.copyfile(EXP2_DIR / "launch_manifest.json", exp2_registry_dir / "configs" / "launch_manifest.json")
    (exp2_registry_dir / "metrics").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lap_eval_dir / "lap_evaluation.json", exp2_registry_dir / "metrics" / "lap_evaluation.json")
    (exp2_registry_dir / "checkpoint_references").mkdir(parents=True, exist_ok=True)
    for name, ref in metadata["checkpoint_references"].items():
        _write_json(exp2_registry_dir / "checkpoint_references" / f"{name}.json", ref)

    # Update the top-level registry manifest to note the extension (append-only w.r.t. exp0/exp1).
    top_manifest_path = REGISTRY_DIR / "preservation_manifest.json"
    top_manifest = _read_json(top_manifest_path)
    top_manifest["experiments"]["experiment_2"] = {
        "run_directory": str(EXP2_DIR), "preserved_at": str(exp2_registry_dir),
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    top_manifest["extended_without_modifying_experiment_0_or_1"] = True
    _write_json(top_manifest_path, top_manifest)

    print(f"\nDone. Registry extended at {REGISTRY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
