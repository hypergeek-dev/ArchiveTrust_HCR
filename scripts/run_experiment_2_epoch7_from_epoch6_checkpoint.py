"""Experiment 2, epoch-7 salvage continuation: a single fresh Keras epoch trained on top of Experiment
2's own verified epoch-6 checkpoint (`training/experiment-2-seven-epochs-20260812T173551Z/run-state/
epoch_output/epoch_1/recommended/best_val`), produced by the epoch-7 run that crashed (`epoch container
exited 1`, exit code 1, no OOM/segfault signature) after ~99% completion -- epochs 1-6 all completed and
checkpointed cleanly in that run before it died very early into the would-be epoch 7.

**This is NOT the same rigor as every other epoch-N script in this lineage.** Every prior
`run_experiment_2_N_epochs.py` insisted on a single, uninterrupted `--epochs N` container invocation so
the optimizer/LR-schedule object never crosses a process boundary -- "true optimizer-continuous
continuation" has been this project's standard since Experiment 0 was rejected for breaking exactly that
property. This script deliberately does NOT meet that standard: it fine-tunes for one epoch from the
epoch-6 checkpoint's *weights only*. There is no `optimizer_state/` checkpoint to restore from (same
limitation flagged at every epoch since epoch 3 -- Experiment 2 has never used the state-saving
mechanism), so the optimizer and LR schedule both restart fresh for this one epoch rather than
continuing where epoch 6 left off.

**Why this tradeoff, here, for the first time.** Every prior epoch used a full from-scratch redo instead
of this weights-only path specifically to avoid this compromise. That calculus changes when a run has
already burned ~13h getting 6 of 7 epochs done and cleanly checkpointed: redoing all 7 from scratch a
second time risks hitting the same unexplained failure again for another ~13-15h, for zero additional
information. Continuing from the salvaged, independently-verified epoch-6 weights costs ~2-3h instead.
The user's explicit call, after the crash: use the finished epoch-6 checkpoint rather than restart blind.

**What this means for comparing this result to prior epochs.** The resulting number is still a genuine
epoch of additional training on top of a real, verified 6-epoch scratch-trained model -- but readers
comparing it to Experiment 1's epoch-over-epoch deltas (all single-continuous-invocation) should treat
this one data point as measuring something adjacent, not identical.

Uses `ContainerEpochRunner` (the same runner Experiment 1 uses to fine-tune from Loghi's pretrained
checkpoint) rather than `ScratchEpochRunner` (which structurally refuses any parent checkpoint) --
mechanically this is a fine-tune, but the parent is Experiment 2's own scratch-trained weights, never
Loghi's pretrained checkpoint or any Experiment 1 lineage. The provenance gate below checks for exactly
that: parent must resolve to the crashed epoch-7 run's own checkpoint, and must NOT resolve to the Loghi
pretrained checkpoint, the pilot, or any Experiment 1 run directory.

Genuinely new run, separate directory: writes to
`training/experiment-2-epoch7-from-epoch6-checkpoint-<timestamp>Z/`, never touches the crashed
`experiment-2-seven-epochs-20260812T173551Z/` run directory (read-only source of the parent checkpoint,
staged to a writable copy before any training happens -- same mechanism `training_session.py` already
uses for Experiment 1's pristine parent checkpoint).

Usage:
    python scripts/run_experiment_2_epoch7_from_epoch6_checkpoint.py --check-only
    python scripts/run_experiment_2_epoch7_from_epoch6_checkpoint.py --dry-run-lines 2000 --confirm
    python scripts/run_experiment_2_epoch7_from_epoch6_checkpoint.py --hours 16 --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.htr.training.checkpoint_index import verify_checkpoint  # noqa: E402
from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner  # noqa: E402
from archivetrust.htr.training.full_run.identity import (  # noqa: E402
    PilotCheckpointRejected,
    RunDirectoryAlreadyExists,
    create_full_run_identity,
)
from archivetrust.htr.training.full_run.launch_manifest import (  # noqa: E402
    build_launch_manifest,
    write_prepared_marker,
)
from archivetrust.htr.training.full_run.orchestrator import _has_nan_or_inf, _oom_signature_in  # noqa: E402
from archivetrust.htr.training.full_run.run_state import (  # noqa: E402
    create_initial_run_state,
    heartbeat,
    mark_completed,
    mark_failed,
    mark_running,
    save_run_state,
)
from archivetrust.htr.training.full_run.shard_training_data import ShardTrainingDataPaths  # noqa: E402
from archivetrust.htr.training.training_identity import TrainingConfiguration  # noqa: E402
from archivetrust.htr.training.training_session import run_training_session  # noqa: E402
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS  # noqa: E402

LOGHI_UPSTREAM = REPO_ROOT / ".loghi-upstream"
LOGHI_PRETRAINED_CHECKPOINT_DIR = LOGHI_UPSTREAM / "pretrained-models" / "loghi-htr" / "generic-2023-02-15"
"""Never used as a parent here -- Experiment 2 has never been fine-tuned from this, and this salvage
continuation doesn't change that. Referenced only so the provenance gate can explicitly reject it."""
TRAINING_ROOT = REPO_ROOT / "training"
DEFAULT_PILOT_RUN_DIR = TRAINING_ROOT / "loghi-swedish-v1"
EXPERIMENT_0_RUN_DIR = TRAINING_ROOT / "full-corpus-20260802T044250Z"
EXPERIMENT_1_EPOCH1_RUN_DIR = TRAINING_ROOT / "experiment-1-continuous-epoch-20260803T192146Z"
CRASHED_EPOCH7_RUN_DIR = TRAINING_ROOT / "experiment-2-seven-epochs-20260812T173551Z"
"""The run this script salvages from. Read-only source of the parent checkpoint -- never written to."""
PARENT_CHECKPOINT_DIR = (
    CRASHED_EPOCH7_RUN_DIR / "run-state" / "epoch_output" / "epoch_1" / "recommended" / "best_val"
)
"""Experiment 2's own verified epoch-6 checkpoint (val_CER 0.1053, independently confirmed via
verify_checkpoint() before this script was written). Weights only -- no optimizer_state/ to restore."""
POOL_DIR = TRAINING_ROOT / "_prepared_data" / "fc21a708_e12796a0"
VAL_MANIFEST_PATH = TRAINING_ROOT / "loghi-swedish-v1" / "manifests" / "val_manifest.parquet"
OVERRIDE_FILE = TRAINING_ROOT / "experiment_1_overrides" / "modes_training_with_optimizer_trace.py"
"""Reused unmodified -- architecture-agnostic trace callback, same as every prior epoch script."""

LEARNING_RATE_POLICY = (
    "epoch7_salvage_single_epoch_fresh_adam_base_0.0001_continuing_experiment2_own_verified_epoch6_"
    "checkpoint_weights_only_no_optimizer_state_restored"
)
BATCH_SIZE_DEFAULT = 16
LEARNING_RATE = 0.0001
OPTIMIZER = "adam"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pool_manifest() -> dict:
    return json.loads((POOL_DIR / "pool_manifest.json").read_text(encoding="utf-8"))


def run_provenance_gate(*, run_dir: Path) -> dict:
    """Inverted from every prior gate in this lineage: the parent checkpoint here is *supposed* to be
    derived from a specific prior Experiment 2 run (the crashed epoch-7 attempt's epoch-6 checkpoint),
    not the pristine Loghi checkpoint. Still checks the negative space just as strictly: must NOT be
    Loghi's pretrained checkpoint, the pilot, Experiment 0, or any Experiment 1 lineage."""
    checks: dict = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ok, digest, extra = verify_checkpoint(PARENT_CHECKPOINT_DIR)
    checks["checkpoint_readable_and_zip_valid"] = ok
    checks["checkpoint_sha256"] = digest
    checks["checkpoint_extra"] = extra

    resolved_parent = PARENT_CHECKPOINT_DIR.resolve()
    checks["parent_checkpoint_path"] = str(resolved_parent)

    resolved_crashed_run = CRASHED_EPOCH7_RUN_DIR.resolve()
    is_from_crashed_run = resolved_parent.is_relative_to(resolved_crashed_run)
    checks["is_experiment_2_epoch6_checkpoint"] = is_from_crashed_run

    rejected_paths = [
        str(p.resolve()) for p in (
            DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR, EXPERIMENT_1_EPOCH1_RUN_DIR,
            LOGHI_PRETRAINED_CHECKPOINT_DIR,
        )
    ]
    not_derived_from_pretrained_or_experiment_1 = all(
        resolved_parent != Path(p) and not resolved_parent.is_relative_to(Path(p)) for p in rejected_paths
    )
    checks["checked_against_rejected_dirs"] = rejected_paths
    checks["not_derived_from_pretrained_or_experiment_1"] = not_derived_from_pretrained_or_experiment_1

    checks["passed"] = bool(ok and is_from_crashed_run and not_derived_from_pretrained_or_experiment_1)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def _build_dry_run_lists(*, n_train: int, n_val: int) -> tuple[Path, Path]:
    real_train = (POOL_DIR / "train_list.txt").read_text(encoding="utf-8").splitlines()
    real_val = (POOL_DIR / "val_list.txt").read_text(encoding="utf-8").splitlines()
    dry_train = POOL_DIR / "experiment_2_epoch7_salvage_dry_run_train_list.txt"
    dry_val = POOL_DIR / "experiment_2_epoch7_salvage_dry_run_val_list.txt"
    dry_train.write_text("\n".join(real_train[:n_train]) + "\n", encoding="utf-8")
    dry_val.write_text("\n".join(real_val[:n_val]) + "\n", encoding="utf-8")
    return dry_train, dry_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--hours", type=float, default=16.0,
                        help="wall-clock ceiling for this single-epoch container invocation (a single "
                             "full-corpus epoch has taken ~2-2.5h in every prior Experiment 2 run; this "
                             "defaults with generous margin). Default: 16.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-lines", type=int, default=None,
                        help="if set, run over only this many train lines (and min(200, n) val lines) "
                             "instead of the full corpus -- for validating the pipeline end-to-end "
                             "before committing real GPU-hours.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        checks = run_provenance_gate(run_dir=REPO_ROOT / "training" / "_experiment_2_epoch7_salvage_check_only_scratch")
        print(json.dumps(checks, indent=2))
        return 0 if checks["passed"] else 1

    if not args.confirm:
        raise SystemExit("Refusing to launch training without --confirm. Use --check-only first.")

    run_name = args.run_name or ("experiment-2-epoch7-from-epoch6-checkpoint-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = TRAINING_ROOT / run_name
    run_state_dir = run_dir / "run-state"

    checks = run_provenance_gate(run_dir=run_dir)
    if not checks["passed"]:
        raise SystemExit(
            f"Provenance gate FAILED -- refusing to launch. See {run_dir / 'provenance_check.json'}\n"
            f"{json.dumps(checks, indent=2)}"
        )
    print(f"Provenance gate passed: parent checkpoint is Experiment 2's own verified epoch-6 checkpoint "
          f"from the crashed epoch-7 run, not Loghi's pretrained checkpoint, the pilot, or any "
          f"Experiment 1 lineage. See {run_dir / 'provenance_check.json'}")

    pool_manifest = _read_pool_manifest()
    val_manifest_hash = _sha256_file(VAL_MANIFEST_PATH)
    charlist_hash = _sha256_file(PARENT_CHECKPOINT_DIR / "tokenizer.json")

    configuration = TrainingConfiguration(
        train_manifest_hash=pool_manifest["training_manifest_hash"],
        val_manifest_hash=val_manifest_hash,
        charlist_hash=charlist_hash,
        preprocessing_version="byte_identical_from_source_parquet",
        model_architecture="recommended",
        parent_checkpoint_hash=checks["checkpoint_sha256"],
        learning_rate_policy=LEARNING_RATE_POLICY,
        optimizer=OPTIMIZER,
        augmentation_policy="none",
    )

    try:
        identity, configuration_hash = create_full_run_identity(
            run_state_dir=run_state_dir,
            parent_checkpoint=f"experiment_2_epoch6_checkpoint@{checks['checkpoint_sha256']}",
            parent_checkpoint_dir=PARENT_CHECKPOINT_DIR,
            configuration=configuration,
            known_pilot_run_dirs=(DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR, EXPERIMENT_1_EPOCH1_RUN_DIR),
        )
    except (PilotCheckpointRejected, RunDirectoryAlreadyExists) as exc:
        raise SystemExit(f"Refusing to launch: {exc}") from exc

    print(f"run_id={identity.run_id} configuration_hash={configuration_hash}")

    if args.dry_run_lines:
        n_val = min(200, args.dry_run_lines)
        train_list_path, val_list_path = _build_dry_run_lists(n_train=args.dry_run_lines, n_val=n_val)
        train_line_count, validation_line_count = args.dry_run_lines, n_val
    else:
        train_list_path = POOL_DIR / "train_list.txt"
        val_list_path = POOL_DIR / "val_list.txt"
        train_line_count = pool_manifest["train_image_count"]
        validation_line_count = pool_manifest["validation_image_count"]

    training_data_paths = ShardTrainingDataPaths(
        pool_dir=str(POOL_DIR),
        validation_list_path=str(val_list_path),
        shard_list_paths=(str(train_list_path),),
        train_image_count=train_line_count,
        validation_image_count=validation_line_count,
        reused_existing_pool=True,
        dataset_hash=pool_manifest["dataset_hash"],
        training_manifest_hash=pool_manifest["training_manifest_hash"],
    )
    (run_dir / "training_data_paths.json").write_text(
        training_data_paths.model_dump_json(indent=2), encoding="utf-8"
    )

    manifest = build_launch_manifest(
        run_id=identity.run_id,
        original_loghi_base_checkpoint_path=PARENT_CHECKPOINT_DIR,
        base_checkpoint_hash=checks["checkpoint_sha256"],
        known_pilot_run_dirs=(DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR, EXPERIMENT_1_EPOCH1_RUN_DIR),
        training_manifest_dir=POOL_DIR,
        training_manifest_hash=pool_manifest["training_manifest_hash"],
        validation_manifest_path=VAL_MANIFEST_PATH,
        validation_manifest_hash=val_manifest_hash,
        dataset_hash=pool_manifest["dataset_hash"],
        train_line_count=train_line_count,
        validation_line_count=validation_line_count,
        excluded_test_line_count=810,
        code_commit_hash=create_initial_run_state(run_id=identity.run_id, configuration_hash=configuration_hash).code_revision,
        container_image_name=CURRENT_PINNED_VERSIONS.docker_image_tag,
        container_image_digest=CURRENT_PINNED_VERSIONS.docker_image_digest,
        batch_size=args.batch_size,
        optimizer=OPTIMIZER,
        scheduler=LEARNING_RATE_POLICY,
        learning_rate=LEARNING_RATE,
        random_seed=args.seed,
        max_epochs=1,
        max_epochs_basis="Salvage continuation: exactly one fresh Keras epoch on top of Experiment 2's "
                          "own verified epoch-6 checkpoint, after the epoch-7 run that would have "
                          "produced this epoch inside a single continuous invocation crashed ~99% "
                          "through (exit code 1, no OOM/segfault signature). NOT single-invocation "
                          "optimizer continuity -- weights-only continuation, optimizer/LR schedule "
                          "restart fresh (see module docstring).",
        early_stopping_patience=0,
        early_stopping_basis="Disabled: a single-epoch run has nothing cross-epoch to early-stop against.",
        wall_clock_policy=f"Single uninterrupted invocation, hard ceiling {args.hours}h.",
        shard_count=1,
        checkpoint_policy="Single epoch: validate once, then checkpoint both 'latest' and 'best_val'.",
        dashboard_config=f"TrainingDashboardViewModel(training_root={TRAINING_ROOT})",
        exact_launch_command=" ".join(sys.argv),
    )
    manifest_dict = manifest.model_dump()
    manifest_dict.update({
        "salvage_continuation": True,
        "salvaged_from_crashed_run": str(CRASHED_EPOCH7_RUN_DIR),
        "crashed_run_failure_detail": "epoch container exited 1 (~99% through epoch 7, 6/7 epochs completed and checkpointed)",
        "optimizer_continuity_mechanism": "weights_only_fresh_optimizer_single_epoch",
        "true_single_invocation_optimizer_continuity": False,
        "cross_process_optimizer_restore_used": False,
        "cross_process_optimizer_restore_mechanism_available_but_unused":
            "training/experiment_epoch2_overrides/main_with_optimizer_restore.py -- not used here either; "
            "no optimizer_state/ checkpoint exists in the crashed run to restore from (same limitation "
            "as every other epoch in this lineage).",
        "parent_is_experiment_2_own_epoch6_checkpoint": True,
        "parent_is_loghi_pretrained_checkpoint": False,
    })
    (run_dir / "launch_manifest.json").write_text(json.dumps(manifest_dict, indent=2, default=str), encoding="utf-8")
    write_prepared_marker(run_dir)

    state = create_initial_run_state(
        run_id=identity.run_id, configuration_hash=configuration_hash, dataset_hash=pool_manifest["dataset_hash"]
    )
    state = state.model_copy(update={"shards_per_epoch": 1})
    state = mark_running(state)
    save_run_state(run_state_dir, state)

    epoch_runner = ContainerEpochRunner(
        batch_size=args.batch_size,
        gradient_accumulation=1,
        precision="mixed_float16",
        max_image_width=65536,
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        beam_width=CURRENT_PINNED_VERSIONS.beam_width,
        timeout_seconds=args.hours * 3600.0,
        run_state_dir=run_state_dir,
        extra_volume_mounts=((str(OVERRIDE_FILE), "/src/loghi-htr/src/modes/training.py"),),
    )

    print(f"Starting the single salvage epoch (ceiling {args.hours}h), fine-tuning from Experiment 2's "
          f"own verified epoch-6 checkpoint. This call does not return until the container exits.")
    started = time.monotonic()
    summary = run_training_session(
        run_state_dir=run_state_dir,
        checkpoint_index_path=run_state_dir / "checkpoint_index.json",
        epoch_runner=epoch_runner,
        train_list_path=str(train_list_path),
        validation_list_path=str(val_list_path),
        parent_checkpoint_dir=str(PARENT_CHECKPOINT_DIR),
        run_id=identity.run_id,
        configuration_hash=configuration_hash,
        random_seed=args.seed,
        max_wall_clock_seconds=args.hours * 3600.0,
        stop_requested=lambda: (run_state_dir / "STOP_REQUESTED").exists(),
        max_epochs_this_call=1,
        early_stopping_patience=None,
    )
    elapsed = time.monotonic() - started
    print(f"Session ended after {elapsed:.0f}s: stop_reason={summary.stop_reason} "
          f"epochs_completed_this_session={summary.epochs_completed_this_session}")

    last_result = summary.epoch_results[-1] if summary.epoch_results else None
    failure_detail = None
    if last_result is None or not last_result.ok:
        failure_detail = (last_result.error_message if last_result else None) or "no epoch result produced"
    elif _has_nan_or_inf(last_result.train_loss, last_result.val_loss, last_result.val_cer):
        failure_detail = "NaN/inf detected in train_loss/val_loss/val_cer"
    else:
        oom = _oom_signature_in(last_result.stderr_tail or "")
        if oom:
            failure_detail = f"likely OOM: stderr contained {oom!r}"

    latest_verified = last_result.checkpoint_dir and verify_checkpoint(last_result.checkpoint_dir)[0]
    best_verified = last_result.best_val_checkpoint_dir and verify_checkpoint(last_result.best_val_checkpoint_dir)[0]

    if failure_detail or not latest_verified:
        state = mark_failed(state, stop_reason=summary.stop_reason, failure_detail=failure_detail or "checkpoint verification failed")
    else:
        state = heartbeat(
            state,
            current_epoch=1,
            global_shards_completed=1,
            epochs_completed=1,
            shards_completed_in_current_epoch=1,
            latest_metrics={k: v for k, v in {
                "train_cer": last_result.train_cer, "val_cer": last_result.val_cer,
                "train_wer": last_result.train_wer, "val_wer": last_result.val_wer,
                "train_loss": last_result.train_loss, "val_loss": last_result.val_loss,
            }.items() if v is not None},
            best_metrics={"val_cer": last_result.val_cer} if last_result.val_cer is not None else {},
            latest_checkpoint=last_result.checkpoint_dir,
            best_checkpoint=last_result.best_val_checkpoint_dir or last_result.checkpoint_dir,
        )
        state = mark_completed(state, stop_reason=summary.stop_reason)
    save_run_state(run_state_dir, state)

    print(f"run_state.json written: status={state.status} epochs_completed={state.epochs_completed}")
    print(f"latest checkpoint (the salvaged epoch 7) verified={latest_verified}: {last_result.checkpoint_dir if last_result else None}")
    print(f"best_val checkpoint verified={best_verified}: {last_result.best_val_checkpoint_dir if last_result else None}")
    epoch_output_dir = run_state_dir / "epoch_output" / "epoch_1"
    print(f"optimizer trace for this single salvage epoch: {epoch_output_dir / 'optimizer_trace.csv'}")
    print(f"run directory: {run_dir}")

    return 0 if not failure_detail and latest_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
