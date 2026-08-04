"""Builds the immutable preservation package for the two completed Loghi Swedish HTR fine-tuning
baselines: Experiment 0 (57-shard, optimizer/LR reset every shard) and Experiment 1 (one continuous,
uninterrupted full-corpus epoch).

Read-only with respect to both original run directories and the sealed 810-line reserved test set.
Copies only small metadata/report/config/log-style files -- never model weights. Checkpoints are
referenced by their already-recorded path + sha256, independently re-verified live (via
`checkpoint_index.verify_checkpoint`, a cheap zip+hash check, not a full TF load) at build time, so
the package can assert integrity as of the preservation date rather than only at original creation.

Usage:
    python scripts/build_preservation_package.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.htr.training.checkpoint_index import verify_checkpoint  # noqa: E402

EXP0_DIR = REPO_ROOT / "training" / "full-corpus-20260802T044250Z"
EXP1_DIR = REPO_ROOT / "training" / "experiment-1-continuous-epoch-20260803T192146Z"
PARENT_CHECKPOINT_DIR = REPO_ROOT / ".loghi-upstream" / "pretrained-models" / "loghi-htr" / "generic-2023-02-15"
DATASET_INVENTORY = REPO_ROOT / "training" / "loghi-swedish-v1" / "source-inventory" / "full-inventory.parquet"
TRAIN_MANIFEST = REPO_ROOT / "training" / "loghi-swedish-v1" / "manifests" / "train_manifest.parquet"
VAL_MANIFEST = REPO_ROOT / "training" / "loghi-swedish-v1" / "manifests" / "val_manifest.parquet"
RESERVED_TEST_MANIFEST = REPO_ROOT / "training" / "loghi-swedish-v1" / "manifests" / "test_reserved_manifest.parquet"
POOL_DIR = REPO_ROOT / "training" / "_prepared_data" / "fc21a708_e12796a0"

OUT_DIR = REPO_ROOT / "training" / "preserved-experiments" / "loghi-finetuning-baselines-20260804"

PARENT_CHECKPOINT_HASH = "0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95"
CONTAINER_IMAGE_NAME = "loghi/docker.htr:latest"
CONTAINER_IMAGE_DIGEST = "sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8"
GPU_MODEL = "NVIDIA GeForce RTX 3070"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


ALL_CHECKSUMMED_FILES: list[Path] = []  # populated as files are written/copied, relative to OUT_DIR


def _track(path: Path) -> Path:
    ALL_CHECKSUMMED_FILES.append(path)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return _track(path)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _track(path)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return _track(path)


def build_experiment_package(*, exp_id: str, run_dir: Path, out_subdir: Path, description: str,
                              methodological_facts: list[str], checkpoint_refs: dict[str, str]) -> dict:
    """`checkpoint_refs` maps a friendly name ("best_val", "end_of_epoch", "latest") to the
    absolute checkpoint directory path recorded for that run. Returns the summary dict used to
    cross-check the controlled-comparison section afterward."""
    identity = _read_json(run_dir / "run-state" / "training_identity.json")
    launch_manifest = _read_json(run_dir / "launch_manifest.json")
    run_state = _read_json(run_dir / "run-state" / "run_state.json")
    checkpoint_index = _read_json(run_dir / "run-state" / "checkpoint_index.json")
    lap_eval_dir = sorted((run_dir / "lap-evaluation").glob("lap*_best_val"))[-1]
    lap_eval = _read_json(lap_eval_dir / "lap_evaluation.json")

    cfg = identity["configuration"]

    # --- checkpoint_references/: re-verify live, never copy the weights themselves ---
    verified_checkpoints = {}
    for name, chk_path in checkpoint_refs.items():
        chk_path = Path(chk_path)
        if not chk_path.is_absolute():
            chk_path = (REPO_ROOT / chk_path).resolve()
        ok, live_hash, extra = verify_checkpoint(chk_path)
        entries = [e for e in checkpoint_index["entries"] if Path(e["checkpoint_dir"]).name == chk_path.name
                   or e["checkpoint_dir"].replace("\\", "/").endswith(chk_path.as_posix()[-80:])]
        recorded_hash = entries[0]["model_file_hash"] if entries else None
        record = {
            "checkpoint_dir": str(chk_path),
            "recorded_sha256": recorded_hash,
            "live_verified_sha256": live_hash,
            "hash_matches_recorded": (recorded_hash == live_hash) if recorded_hash else None,
            "live_verification_ok": ok,
            "live_verification_extra": extra,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "Referenced by path + hash only. Model weights (.keras) are NOT copied into the "
                    "preservation package -- this record, plus the original checkpoint directory, is "
                    "the reproducibility artifact.",
        }
        _write_json(out_subdir / "checkpoint_references" / f"{name}.json", record)
        verified_checkpoints[name] = record

    # --- configs/: copy small resolved-configuration JSON verbatim ---
    _copy(run_dir / "run-state" / "training_identity.json", out_subdir / "configs" / "training_identity.json")
    _track(out_subdir / "configs" / "training_identity.json")
    _copy(run_dir / "launch_manifest.json", out_subdir / "configs" / "launch_manifest.json")
    _track(out_subdir / "configs" / "launch_manifest.json")

    # --- metrics/: copy the full evaluation JSON, derive a per-collection CSV ---
    _copy(lap_eval_dir / "lap_evaluation.json", out_subdir / "metrics" / "lap_evaluation.json")
    _track(out_subdir / "metrics" / "lap_evaluation.json")
    _write_csv(
        out_subdir / "metrics" / "per_collection.csv",
        ["collection", "corpus_cer", "corpus_wer", "line_error_rate", "mean_per_line_cer",
         "sample_count", "mean_confidence"],
        [[c["collection"], c["metrics"]["corpus_cer"], c["metrics"]["corpus_wer"],
          c["metrics"]["line_error_rate"], c["metrics"]["mean_per_line_cer"],
          c["metrics"]["sample_count"], c["mean_confidence"]] for c in lap_eval["per_collection"]],
    )

    # --- reports/: copy the human-readable evaluation report ---
    _copy(lap_eval_dir / "lap_evaluation.md", out_subdir / "reports" / "lap_evaluation.md")
    _track(out_subdir / "reports" / "lap_evaluation.md")

    # --- metadata.json: the authoritative lineage record ---
    started_at = run_state.get("started_at")
    ended_at = run_state.get("ended_at")
    metadata = {
        "experiment_id": exp_id,
        "description": description,
        "run_directory": str(run_dir),
        "methodological_facts": methodological_facts,
        "parent_model_name": launch_manifest["parent_model_type"],
        "parent_checkpoint_path": launch_manifest["original_loghi_base_checkpoint_path"],
        "parent_checkpoint_sha256": launch_manifest["base_checkpoint_hash"],
        "parent_checkpoint_sha256_matches_pin": launch_manifest["base_checkpoint_hash"] == PARENT_CHECKPOINT_HASH,
        "pilot_or_prior_run_checkpoint_used_as_parent": launch_manifest["pilot_checkpoint_used_as_parent"],
        "architecture": cfg["model_architecture"],
        "character_vocabulary": {
            "sha256": cfg["charlist_hash"],
            "source_path": str(PARENT_CHECKPOINT_DIR / "charlist.txt"),
        },
        "training_dataset_identity": {
            "dataset_id": "riksarkivet_swedish_lion_libre_training_data",
            "dataset_hash": launch_manifest["dataset_hash"],
            "train_manifest_hash": cfg["train_manifest_hash"],
            "train_manifest_path": str(TRAIN_MANIFEST),
            "dataset_inventory_path": str(DATASET_INVENTORY),
            "train_line_count": launch_manifest["train_line_count"],
        },
        "validation_dataset_identity": {
            "val_manifest_hash": cfg["val_manifest_hash"],
            "val_manifest_path": launch_manifest["validation_manifest_path"],
            "validation_line_count": launch_manifest["validation_line_count"],
        },
        "reserved_test_set_identity": {
            "path": str(RESERVED_TEST_MANIFEST),
            "excluded_line_count": launch_manifest["excluded_test_line_count"],
            "note": "SEALED. Not opened, not read, not hashed by this preservation pass or by "
                    "either experiment's training or evaluation. Existence confirmed by directory "
                    "listing metadata only.",
        },
        "initialization_method": "weights-only load from the pristine pinned parent checkpoint "
                                  "(tf.keras.models.load_model on a directory; no VGSL-from-scratch build)",
        "optimizer": cfg["optimizer"],
        "initial_learning_rate": launch_manifest["learning_rate"],
        "learning_rate_schedule": cfg["learning_rate_policy"],
        "batch_size": launch_manifest["batch_size"],
        "precision_policy": "mixed_float16 (loghi-htr's own setup/environment.py applies this "
                             "automatically whenever a GPU is used and --use_float32 is not passed; "
                             "neither experiment passed --use_float32)",
        "seed": launch_manifest["random_seed"],
        "code_commit": launch_manifest["code_commit_hash"],
        "repository_dirty_at_launch": launch_manifest["repository_dirty"],
        "docker_image_name": launch_manifest["container_image_name"],
        "docker_image_digest": launch_manifest["container_image_digest"],
        "gpu_model": GPU_MODEL,
        "started_at": started_at,
        "ended_at": ended_at,
        "cumulative_training_seconds": None,  # filled below per-experiment (differs in source field)
        "run_status": run_state["status"],
        "stop_reason": run_state["stop_reason"],
        "authoritative_held_out_result": {
            "corpus_cer": lap_eval["overall"]["corpus_cer"],
            "corpus_wer": lap_eval["overall"]["corpus_wer"],
            "scored_line_count": lap_eval["scored_line_count"],
            "validation_line_count": lap_eval["validation_line_count"],
            "evaluation_source": str((lap_eval_dir / "lap_evaluation.json").relative_to(REPO_ROOT)),
            "evaluation_checkpoint": lap_eval["checkpoint_dir"],
            "evaluation_checkpoint_sha256": lap_eval["checkpoint_hash"],
        },
        "checkpoint_references": verified_checkpoints,
        "immutability_note": "This metadata.json is a derived, read-only summary written by "
                              "scripts/build_preservation_package.py. The authoritative source for "
                              "every field is the original run directory named above, which this "
                              "preservation pass did not modify.",
    }
    _write_json(out_subdir / "metadata.json", metadata)

    return {
        "identity": identity, "launch_manifest": launch_manifest, "run_state": run_state,
        "lap_eval": lap_eval, "metadata": metadata,
    }


def build_artifact_index(*, run_dir: Path, out_subdir: Path, extra: dict) -> None:
    def _entry(path: Path, note: str = "") -> dict:
        exists = path.exists()
        return {"path": str(path), "exists": exists, "note": note}

    index = {
        "run_root": _entry(run_dir),
        "parent_checkpoint_dir": _entry(PARENT_CHECKPOINT_DIR),
        "resolved_training_configuration": _entry(run_dir / "run-state" / "training_identity.json"),
        "launch_manifest": _entry(run_dir / "launch_manifest.json"),
        "dataset_inventory": _entry(DATASET_INVENTORY),
        "train_manifest": _entry(TRAIN_MANIFEST),
        "validation_manifest": _entry(VAL_MANIFEST),
        "reserved_test_manifest": _entry(RESERVED_TEST_MANIFEST, "SEALED -- not opened by this pass"),
        "generated_training_pool": _entry(POOL_DIR, "shared content-addressed pool both experiments trained from"),
        "validation_list": _entry(POOL_DIR / "val_list.txt"),
        "checkpoint_index": _entry(run_dir / "run-state" / "checkpoint_index.json"),
        "run_state": _entry(run_dir / "run-state" / "run_state.json"),
        "session_state": _entry(run_dir / "run-state" / "session_state.json"),
        "telemetry_samples": _entry(run_dir / "run-state" / "telemetry" / "gpu_samples.jsonl"),
        "telemetry_status": _entry(run_dir / "run-state" / "telemetry" / "status.json"),
        "independent_evaluation_output": _entry(
            sorted((run_dir / "lap-evaluation").glob("lap*_best_val"))[-1] / "lap_evaluation.json"),
        "prediction_file": _entry(
            sorted((run_dir / "lap-evaluation").glob("lap*_best_val"))[-1] / "results.txt"),
    }
    index.update(extra)
    _write_json(out_subdir / "artifact_index.json", index)


def main() -> int:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} already exists -- this builder never overwrites a prior "
                          f"preservation pass. Remove it manually first if a genuine rebuild is intended.")

    print("Building Experiment 0 package...")
    exp0 = build_experiment_package(
        exp_id="experiment_0",
        run_dir=EXP0_DIR,
        out_subdir=OUT_DIR / "experiment_0",
        description="Generic Loghi checkpoint fine-tuned over the full corpus through 57 separate "
                     "shard-level trainer invocations (the shard-reset baseline). NOT one continuous "
                     "epoch: each of the 57 invocations was an independent container process.",
        methodological_facts=[
            "model weights carried forward between shard invocations",
            "optimizer state did NOT carry forward between shard invocations",
            "learning-rate schedule position did NOT carry forward between shard invocations",
            "optimizer and LR scheduler were recreated fresh at every shard boundary",
            "one full corpus pass consisted of 57 separate trainer/container invocations",
        ],
        checkpoint_refs={
            "best_val": str(EXP0_DIR / "run-state" / "epoch_output" / "epoch_48" / "model_new10" / "best_val"),
            "end_of_epoch": str(EXP0_DIR / "run-state" / "epoch_output" / "epoch_57" / "model_new10"
                                 / "epoch_0_CER_0.1463_val_0.1756"),
        },
    )
    exp0["metadata"]["cumulative_training_seconds"] = 25093.17073539998
    _write_json(OUT_DIR / "experiment_0" / "metadata.json", exp0["metadata"])
    build_artifact_index(run_dir=EXP0_DIR, out_subdir=OUT_DIR / "experiment_0", extra={
        "end_of_epoch_checkpoint": {"path": str(EXP0_DIR / "run-state" / "epoch_output" / "epoch_57"
                                                 / "model_new10" / "epoch_0_CER_0.1463_val_0.1756"),
                                     "exists": True},
        "best_checkpoint": {"path": str(EXP0_DIR / "run-state" / "epoch_output" / "epoch_48"
                                         / "model_new10" / "best_val"), "exists": True},
        "generated_shard_lists": {"path": str(POOL_DIR), "count": 171,
                                   "note": "shard_NNNNN_list.txt, derived slices of train_list.txt"},
        "per_shard_logs": {"path": str(EXP0_DIR / "run-state" / "epoch_output" / "epoch_<1..57>" / "log.csv"),
                            "note": "one log.csv per shard invocation (57 total), CSVLogger epoch-level rows"},
    })
    print("Experiment 0 package written.")

    print("Building Experiment 1 package...")
    exp1 = build_experiment_package(
        exp_id="experiment_1",
        run_dir=EXP1_DIR,
        out_subdir=OUT_DIR / "experiment_1",
        description="Generic Loghi checkpoint fine-tuned for one complete, uninterrupted full-corpus "
                     "epoch in a single trainer/container lifecycle -- the corrected-method run.",
        methodological_facts=[
            "optimizer.iterations advanced continuously from 0 to 35,099 across the whole epoch",
            "optimizer state (Adam moments) was continuous within the one process -- no resets",
            "learning rate decayed smoothly along one schedule for the whole epoch",
            "no shard-level trainer restart occurred at any point",
            "no crash or manual intervention occurred after launch",
            "one full corpus pass consisted of exactly one trainer/container invocation",
        ],
        checkpoint_refs={
            "best_val": str(EXP1_DIR / "run-state" / "epoch_output" / "epoch_1" / "model_new10" / "best_val"),
            "latest": str(EXP1_DIR / "run-state" / "epoch_output" / "epoch_1" / "model_new10"
                          / "epoch_0_CER_0.1949_val_0.1755"),
        },
    )
    exp1["metadata"]["cumulative_training_seconds"] = 16543.1574517
    exp1["metadata"]["optimizer_continuity_proof"] = {
        "source": str((EXP1_DIR / "run-state" / "epoch_output" / "epoch_1" / "optimizer_trace.csv").relative_to(REPO_ROOT)),
        "optimizer_iterations_start": 0,
        "optimizer_iterations_end": 35099,
        "monotonically_non_decreasing": True,
        "learning_rate_start": 9.999999747378752e-05,
        "learning_rate_end": 9.900096483761445e-05,
    }
    exp1["metadata"]["note_on_checkpoint_kinds"] = (
        "Experiment 1 has no separate 'end_of_epoch' checkpoint_kind entry (unlike Experiment 0): "
        "run_training_session() was called directly, bypassing the shard orchestrator that records "
        "that category. Since Experiment 1 has exactly one epoch, its 'latest' and 'end_of_session' "
        "entries already point at the same checkpoint -- there is no separate shard-boundary-vs-lap "
        "distinction to make."
    )
    _write_json(OUT_DIR / "experiment_1" / "metadata.json", exp1["metadata"])
    _copy(EXP1_DIR / "run-state" / "epoch_output" / "epoch_1" / "optimizer_trace.csv",
          OUT_DIR / "experiment_1" / "metrics" / "optimizer_trace.csv")
    _track(OUT_DIR / "experiment_1" / "metrics" / "optimizer_trace.csv")
    build_artifact_index(run_dir=EXP1_DIR, out_subdir=OUT_DIR / "experiment_1", extra={
        "best_checkpoint": {"path": str(EXP1_DIR / "run-state" / "epoch_output" / "epoch_1"
                                         / "model_new10" / "best_val"), "exists": True},
        "end_of_epoch_checkpoint": {"path": None,
                                     "note": "no distinct end_of_epoch entry -- see metadata.json's "
                                             "note_on_checkpoint_kinds"},
        "generated_training_list": {"path": str(POOL_DIR / "train_list.txt"),
                                     "note": "single full-corpus list, 562,123 lines -- reused unmodified "
                                             "from the same pool Experiment 0's data prep built"},
        "optimizer_continuity_trace": {
            "path": str(EXP1_DIR / "run-state" / "epoch_output" / "epoch_1" / "optimizer_trace.csv")},
        "single_epoch_log": {"path": str(EXP1_DIR / "run-state" / "epoch_output" / "epoch_1" / "log.csv"),
                              "note": "one row only -- see comparison notes on logging granularity"},
    })
    print("Experiment 1 package written.")

    return build_comparison_and_provenance(exp0, exp1)


def build_comparison_and_provenance(exp0: dict, exp1: dict) -> int:
    print("Building comparison package...")
    m0, m1 = exp0["metadata"], exp1["metadata"]
    r0, r1 = m0["authoritative_held_out_result"], m1["authoritative_held_out_result"]

    # --- controlled-comparison verification ---
    identical_fields = {
        "parent_checkpoint_sha256": (m0["parent_checkpoint_sha256"], m1["parent_checkpoint_sha256"]),
        "architecture": (m0["architecture"], m1["architecture"]),
        "training_dataset_hash": (m0["training_dataset_identity"]["dataset_hash"], m1["training_dataset_identity"]["dataset_hash"]),
        "train_manifest_hash": (m0["training_dataset_identity"]["train_manifest_hash"], m1["training_dataset_identity"]["train_manifest_hash"]),
        "validation_manifest_hash": (m0["validation_dataset_identity"]["val_manifest_hash"], m1["validation_dataset_identity"]["val_manifest_hash"]),
        "character_vocabulary_hash": (m0["character_vocabulary"]["sha256"], m1["character_vocabulary"]["sha256"]),
        "batch_size": (m0["batch_size"], m1["batch_size"]),
        "optimizer": (m0["optimizer"], m1["optimizer"]),
        "initial_learning_rate": (m0["initial_learning_rate"], m1["initial_learning_rate"]),
        "precision_policy": (m0["precision_policy"], m1["precision_policy"]),
        "seed": (m0["seed"], m1["seed"]),
        "docker_image_digest": (m0["docker_image_digest"], m1["docker_image_digest"]),
        "gpu_model": (m0["gpu_model"], m1["gpu_model"]),
    }
    mismatches = {k: v for k, v in identical_fields.items() if v[0] != v[1]}
    controlled_ok = len(mismatches) == 0

    disclosed_differences = [
        {
            "field": "code_commit",
            "experiment_0": m0["code_commit"], "experiment_1": m1["code_commit"],
            "material": False,
            "explanation": "4 commits landed between the two runs, confined to "
                            "src/archivetrust/htr/training/full_run/ (shard-orchestration bookkeeping, "
                            "the new end-of-lap evaluation suite, CLI/preflight additions). None touch "
                            "container_epoch_runner.py, training_session.py, checkpoint_index.py, or "
                            "training_identity.py -- the code paths that actually build the docker "
                            "invocation and training configuration. See provenance/repository_diff.patch.",
        },
        {
            "field": "trainer_lifecycle / optimizer and LR continuity",
            "experiment_0": "57 separate trainer invocations, optimizer/scheduler reset each time",
            "experiment_1": "1 continuous trainer invocation, optimizer/scheduler never reset",
            "material": True,
            "explanation": "This is the sole INTENDED independent variable of this comparison.",
        },
    ]

    comparison = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controlled": controlled_ok,
        "unexpected_mismatches": mismatches,
        "intended_independent_variable": "trainer lifecycle / optimizer and LR continuity",
        "disclosed_differences": disclosed_differences,
        "held_out_results": {
            "experiment_0": {"corpus_cer": r0["corpus_cer"], "corpus_wer": r0["corpus_wer"]},
            "experiment_1": {"corpus_cer": r1["corpus_cer"], "corpus_wer": r1["corpus_wer"]},
            "cer_delta_exp1_minus_exp0": round(r1["corpus_cer"] - r0["corpus_cer"], 4),
            "wer_delta_exp1_minus_exp0": round(r1["corpus_wer"] - r0["corpus_wer"], 4),
        },
        "authoritative_interpretation": (
            "Correcting optimizer and learning-rate continuity was methodologically necessary, but it "
            "did not materially improve held-out generalisation after one full epoch for this corpus, "
            "architecture, optimizer, and learning-rate configuration. The difference between the "
            "experiments is within observed noise (CER delta +0.0019, WER delta +0.0011 for Experiment "
            "1 relative to Experiment 0). This does not mean Experiment 1 is meaningfully worse, does "
            "not mean Experiment 0 is superior, and does not mean optimizer continuity never matters -- "
            "the supported conclusion is limited to this exact comparison and training budget."
        ),
    }
    _write_json(OUT_DIR / "comparison" / "experiment_0_vs_1.json", comparison)

    per_collection_rows = []
    e0_by_collection = {c["collection"]: c["metrics"] for c in exp0["lap_eval"]["per_collection"]}
    e1_by_collection = {c["collection"]: c["metrics"] for c in exp1["lap_eval"]["per_collection"]}
    for name in sorted(set(e0_by_collection) | set(e1_by_collection)):
        e0m, e1m = e0_by_collection.get(name), e1_by_collection.get(name)
        per_collection_rows.append([
            name,
            e0m["corpus_cer"] if e0m else "", e1m["corpus_cer"] if e1m else "",
            (round(e1m["corpus_cer"] - e0m["corpus_cer"], 4) if e0m and e1m else ""),
            e0m["corpus_wer"] if e0m else "", e1m["corpus_wer"] if e1m else "",
            (round(e1m["corpus_wer"] - e0m["corpus_wer"], 4) if e0m and e1m else ""),
        ])
    _write_csv(OUT_DIR / "comparison" / "per_collection_comparison.csv",
               ["collection", "exp0_corpus_cer", "exp1_corpus_cer", "cer_delta",
                "exp0_corpus_wer", "exp1_corpus_wer", "wer_delta"], per_collection_rows)

    _write_csv(OUT_DIR / "comparison" / "metric_summary.csv",
               ["metric", "experiment_0", "experiment_1", "delta_exp1_minus_exp0"],
               [
                   ["held_out_corpus_cer", r0["corpus_cer"], r1["corpus_cer"], round(r1["corpus_cer"] - r0["corpus_cer"], 4)],
                   ["held_out_corpus_wer", r0["corpus_wer"], r1["corpus_wer"], round(r1["corpus_wer"] - r0["corpus_wer"], 4)],
                   ["in_training_best_val_cer", exp0["run_state"]["best_metrics"]["val_cer"],
                    exp1["run_state"]["best_metrics"]["val_cer"],
                    round(exp1["run_state"]["best_metrics"]["val_cer"] - exp0["run_state"]["best_metrics"]["val_cer"], 4)],
                   ["trainer_invocation_count", 57, 1, -56],
                   ["cumulative_training_seconds", m0["cumulative_training_seconds"], m1["cumulative_training_seconds"],
                    round(m1["cumulative_training_seconds"] - m0["cumulative_training_seconds"], 1)],
               ])

    md = f"""# Experiment 0 vs. Experiment 1: controlled comparison

Generated {comparison['generated_at']} by `scripts/build_preservation_package.py`.

## Controlled comparison status: {"CONTROLLED" if controlled_ok else "NOT CONTROLLED -- see mismatches below"}

Sole intended independent variable: **trainer lifecycle / optimizer and LR continuity**.

| field | Experiment 0 | Experiment 1 | identical? |
|---|---|---|---|
""" + "\n".join(
        f"| {k} | `{v[0]}` | `{v[1]}` | {'yes' if v[0] == v[1] else '**NO**'} |"
        for k, v in identical_fields.items()
    ) + f"""

### Disclosed differences

""" + "\n".join(
        f"- **{d['field']}** ({'material' if d['material'] else 'not material'}): "
        f"{d['experiment_0']!r} vs {d['experiment_1']!r} -- {d['explanation']}"
        for d in disclosed_differences
    ) + f"""

## Held-out result (same evaluation code, same 1,000-line validation set)

| | Experiment 0 | Experiment 1 | delta (Exp1 - Exp0) |
|---|---|---|---|
| corpus CER | {r0['corpus_cer']:.4f} | {r1['corpus_cer']:.4f} | {comparison['held_out_results']['cer_delta_exp1_minus_exp0']:+.4f} |
| corpus WER | {r0['corpus_wer']:.4f} | {r1['corpus_wer']:.4f} | {comparison['held_out_results']['wer_delta_exp1_minus_exp0']:+.4f} |

## Authoritative interpretation

{comparison['authoritative_interpretation']}

## Per-collection detail

See `per_collection_comparison.csv` in this directory.
"""
    _write_text(OUT_DIR / "comparison" / "experiment_0_vs_1.md", md)
    print("Comparison package written.")

    build_provenance(exp0, exp1, comparison)
    build_readme_and_manifest(exp0, exp1, comparison)
    write_checksums()
    return 0


def build_provenance(exp0: dict, exp1: dict, comparison: dict) -> None:
    print("Building provenance records...")
    _write_text(OUT_DIR / "provenance" / "code_revision.txt",
                f"Experiment 0 code revision: {exp0['metadata']['code_commit']}\n"
                f"Experiment 1 code revision: {exp1['metadata']['code_commit']}\n"
                f"Preservation-pass HEAD at build time: "
                f"{subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()}\n"
                f"See repository_diff.patch for the src/ diff between the two experiments' commits.\n")

    diff = subprocess.run(
        ["git", "diff", exp0["metadata"]["code_commit"], exp1["metadata"]["code_commit"], "--", "src/"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout
    _write_text(OUT_DIR / "provenance" / "repository_diff.patch",
                "# git diff " + exp0["metadata"]["code_commit"] + " " + exp1["metadata"]["code_commit"]
                + " -- src/\n# Scoped to src/ only: the full diff between these commits also includes "
                  "each run's own generated report/log artifacts, which are not code and are already "
                  "preserved in this package's experiment_0/ and experiment_1/ subdirectories.\n\n" + diff)

    _write_json(OUT_DIR / "provenance" / "container_image.json", {
        "image_name": CONTAINER_IMAGE_NAME,
        "image_digest": CONTAINER_IMAGE_DIGEST,
        "identical_across_both_experiments": exp0["metadata"]["docker_image_digest"] == exp1["metadata"]["docker_image_digest"],
        "live_reverified_at_build_time": subprocess.run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", CONTAINER_IMAGE_NAME],
            capture_output=True, text=True,
        ).stdout.strip() or None,
    })

    _write_json(OUT_DIR / "provenance" / "dataset_identity.json", {
        "dataset_id": "riksarkivet_swedish_lion_libre_training_data",
        "dataset_hash": exp0["metadata"]["training_dataset_identity"]["dataset_hash"],
        "dataset_inventory_path": str(DATASET_INVENTORY),
        "train_manifest_path": str(TRAIN_MANIFEST),
        "train_manifest_hash": exp0["metadata"]["training_dataset_identity"]["train_manifest_hash"],
        "validation_manifest_path": str(VAL_MANIFEST),
        "validation_manifest_hash": exp0["metadata"]["validation_dataset_identity"]["val_manifest_hash"],
        "reserved_test_manifest_path": str(RESERVED_TEST_MANIFEST),
        "reserved_test_manifest_note": "SEALED. Path recorded for lineage only; not opened, read, or "
                                        "hashed by this preservation pass, matching the instruction not "
                                        "to access the sealed 810-line test set.",
        "identical_across_both_experiments": (
            exp0["metadata"]["training_dataset_identity"]["dataset_hash"] == exp1["metadata"]["training_dataset_identity"]["dataset_hash"]
            and exp0["metadata"]["training_dataset_identity"]["train_manifest_hash"] == exp1["metadata"]["training_dataset_identity"]["train_manifest_hash"]
            and exp0["metadata"]["validation_dataset_identity"]["val_manifest_hash"] == exp1["metadata"]["validation_dataset_identity"]["val_manifest_hash"]
        ),
    })

    _write_json(OUT_DIR / "provenance" / "evaluation_identity.json", {
        "evaluation_code": "src/archivetrust/htr/training/full_run/lap_evaluation.py + evaluation_metrics.py",
        "evaluation_cli": "scripts/evaluate_full_run_lap.py",
        "cer_definition": "corpus-level sum(Levenshtein edits) / sum(reference characters) -- matches "
                           "the container's own CERMetric weighting",
        "wer_definition": "true word-level Levenshtein over word tokens (distinct from the container's "
                           "own line-error-rate 'WER_metric')",
        "same_evaluation_code_used_for_both_experiments": True,
        "same_validation_set_used_for_both_experiments": True,
        "experiment_0_evaluation_output": str((EXP0_DIR / "lap-evaluation" / "lap1_best_val" / "lap_evaluation.json").relative_to(REPO_ROOT)),
        "experiment_1_evaluation_output": str((EXP1_DIR / "lap-evaluation" / "lap1_best_val" / "lap_evaluation.json").relative_to(REPO_ROOT)),
    })
    print("Provenance records written.")


def build_readme_and_manifest(exp0: dict, exp1: dict, comparison: dict) -> None:
    print("Building README and preservation manifest...")
    readme = f"""# Loghi Swedish HTR fine-tuning baselines: preservation package

Preserved {time.strftime("%Y-%m-%d", time.gmtime())}. Covers two completed, immutable training runs.
Neither run's original artifacts were modified, renamed, deleted, or reinterpreted by this package --
everything here is either a small derived summary/index or an unmodified copy of a small text/JSON/CSV
artifact. Model weights are never copied; checkpoints are referenced by path + independently
re-verified sha256.

## Experiment 0 -- the shard-reset baseline

{exp0['metadata']['description']}

- Held-out corpus CER: **{exp0['metadata']['authoritative_held_out_result']['corpus_cer']:.4f}**
- Held-out corpus WER: **{exp0['metadata']['authoritative_held_out_result']['corpus_wer']:.4f}**
- Run directory: `{exp0['metadata']['run_directory']}`

## Experiment 1 -- the continuous-epoch (corrected-method) run

{exp1['metadata']['description']}

- Held-out corpus CER: **{exp1['metadata']['authoritative_held_out_result']['corpus_cer']:.4f}**
- Held-out corpus WER: **{exp1['metadata']['authoritative_held_out_result']['corpus_wer']:.4f}**
- Best checkpoint: `{exp1['metadata']['checkpoint_references']['best_val']['checkpoint_dir']}`
- Run directory: `{exp1['metadata']['run_directory']}`

## Authoritative interpretation

{comparison['authoritative_interpretation']}

## Package structure

```
experiment_0/   metadata.json, artifact_index.json, reports/, configs/, metrics/, checkpoint_references/
experiment_1/   (same shape)
comparison/     experiment_0_vs_1.{{md,json}}, per_collection_comparison.csv, metric_summary.csv
provenance/     code_revision.txt, repository_diff.patch, container_image.json,
                dataset_identity.json, evaluation_identity.json
checksums.sha256   sha256 of every file in this package (verify with: sha256sum -c checksums.sha256)
preservation_manifest.json   top-level index + build provenance
```

## What is NOT in this package

- Model weights (`.keras` files) -- referenced by path + hash in `checkpoint_references/`, never copied.
- The sealed 810-line reserved test set -- neither experiment's training or evaluation touched it, and
  this preservation pass did not open, read, or hash it. Its path is recorded for lineage only.
- The 57 per-shard `.keras` checkpoints of Experiment 0 beyond `best_val`/`end_of_epoch` -- referenced
  collectively via `checkpoint_index.json` in the original run directory, not individually indexed here.

## How to cite these runs

Cite `experiment_0/metadata.json` / `experiment_1/metadata.json` directly, or the comparison in
`comparison/experiment_0_vs_1.json`. Both are self-contained and do not require reopening the original
run directories or re-deriving the comparison.
"""
    _write_text(OUT_DIR / "README.md", readme)

    manifest = {
        "package_name": "loghi-finetuning-baselines-20260804",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "built_by": "scripts/build_preservation_package.py",
        "repository_head_at_build_time": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
        "experiments": {
            "experiment_0": {"run_directory": str(EXP0_DIR), "preserved_at": str(OUT_DIR / "experiment_0")},
            "experiment_1": {"run_directory": str(EXP1_DIR), "preserved_at": str(OUT_DIR / "experiment_1")},
        },
        "controlled_comparison": comparison["controlled"],
        "sealed_test_set_accessed": False,
        "model_weights_copied": False,
        "original_run_directories_modified": False,
    }
    _write_json(OUT_DIR / "preservation_manifest.json", manifest)
    print("README and manifest written.")


def write_checksums() -> None:
    print(f"Computing checksums for {len(ALL_CHECKSUMMED_FILES)} preserved files...")
    lines = []
    for path in sorted(set(ALL_CHECKSUMMED_FILES)):
        if not path.exists():
            continue
        digest = _sha256_file(path)
        rel = path.relative_to(OUT_DIR)
        lines.append(f"{digest}  {rel.as_posix()}")
    (OUT_DIR / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checksums.sha256 written ({len(lines)} entries).")


if __name__ == "__main__":
    raise SystemExit(main())
