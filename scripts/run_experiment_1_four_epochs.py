"""Experiment 1, epoch-4 continuation: a single continuous `--epochs 4` container invocation, run as
a fresh restart from the same pristine parent checkpoint Experiment 1's original epoch-1 run used
(`training/experiment-1-continuous-epoch-20260803T192146Z/`, preserved untouched) -- the same origin
the epoch-2 run (`training/experiment-1-two-epochs-20260805T181633Z/`) and epoch-3 run
(`training/experiment-1-three-epochs-20260809T020441Z/`, the successful relaunch -- the crashed
2026-08-08 attempt is preserved separately as evidence, never used as an origin) used.

**Why a fresh restart, not a true cross-process resume.** Same rationale as the epoch-2 and epoch-3
drivers (`scripts/run_experiment_1_two_epochs.py`, `scripts/run_experiment_1_three_epochs.py`): true
cross-process optimizer-continuous continuation was built and empirically verified in this project
(`training/experiment_epoch2_overrides/main_with_optimizer_restore.py`), but none of the preserved
prior runs wrote an `optimizer_state/` checkpoint to restore from -- all three predate that mechanism.
Rather than redo already-completed epochs just to produce something to restore from, this follows the
epoch-2/epoch-3 runs' own precedent: one single, uninterrupted container invocation covering all four
epochs from the same original origin, so a fourth epoch is directly comparable to the three-epoch
result at the same level of rigor (`training/epoch_3_continuation_comparison_report.md` names a fourth
epoch as "the natural next check", given Experiment 1's diminishing but still-positive epoch 2->3 gain).

**Why this still satisfies "true optimizer-continuous continuation" in spirit.** A single container
invocation with `--epochs 4` never leaves the Python process across any of the four real Keras epochs
-- `model.fit(epochs=4, ...)` keeps the same optimizer object, the same LR-schedule object, and the same
mixed-precision loss-scale state alive across all three internal epoch boundaries by construction. There
is no cross-process boundary here for state to fail to survive across -- unlike Experiment 0's 57-shard
design (the exact flaw Experiment 1 was built to correct), none of the four epochs ever touch a fresh
optimizer. `ContainerEpochRunner(epochs_per_invocation=4)` is the only mechanism this needs; the
cross-process restore machinery is not invoked.

**Genuinely new run, separate directory.** Writes to `training/experiment-1-four-epochs-<timestamp>Z/`,
never touches `training/experiment-1-continuous-epoch-20260803T192146Z/` (epoch-1 artifacts),
`training/experiment-1-two-epochs-20260805T181633Z/` (epoch-2 artifacts),
`training/experiment-1-three-epochs-20260809T020441Z/` (epoch-3 artifacts), or Experiment 0's directory.

Usage:
    python scripts/run_experiment_1_four_epochs.py --check-only
    python scripts/run_experiment_1_four_epochs.py --dry-run-lines 2000 --confirm
    python scripts/run_experiment_1_four_epochs.py --hours 60 --confirm
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
    write_launch_manifest,
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
PARENT_CHECKPOINT_DIR = LOGHI_UPSTREAM / "pretrained-models" / "loghi-htr" / "generic-2023-02-15"
CHARLIST_PATH = PARENT_CHECKPOINT_DIR / "charlist.txt"
TRAINING_ROOT = REPO_ROOT / "training"
DEFAULT_PILOT_RUN_DIR = TRAINING_ROOT / "loghi-swedish-v1"
EXPERIMENT_0_RUN_DIR = TRAINING_ROOT / "full-corpus-20260802T044250Z"
EXPERIMENT_1_EPOCH1_RUN_DIR = TRAINING_ROOT / "experiment-1-continuous-epoch-20260803T192146Z"
"""Experiment 1's preserved, immutable epoch-1 run. Never used as a parent checkpoint source (this is
a fresh restart from the pristine pinned checkpoint, not a resume of this run), never written to."""
EXPERIMENT_1_EPOCH2_RUN_DIR = TRAINING_ROOT / "experiment-1-two-epochs-20260805T181633Z"
"""Experiment 1's preserved, immutable epoch-2 run. Same treatment as the epoch-1 run dir above: never
a parent checkpoint source, never written to."""
EXPERIMENT_1_EPOCH3_RUN_DIR = TRAINING_ROOT / "experiment-1-three-epochs-20260809T020441Z"
"""Experiment 1's preserved, immutable epoch-3 run (the successful relaunch). Same treatment as the
epoch-1/epoch-2 run dirs above: never a parent checkpoint source, never written to."""
POOL_DIR = TRAINING_ROOT / "_prepared_data" / "fc21a708_e12796a0"
VAL_MANIFEST_PATH = TRAINING_ROOT / "loghi-swedish-v1" / "manifests" / "val_manifest.parquet"
OVERRIDE_FILE = TRAINING_ROOT / "experiment_1_overrides" / "modes_training_with_optimizer_trace.py"
"""Reused unmodified from Experiment 1 epoch 1. `OptimizerTraceCallback` logs every 500 steps for the
whole `model.fit(epochs=4, ...)` call, threshold-based on cumulative `optimizer.iterations` -- it does
not reset at any internal Keras epoch boundary, so the resulting trace is itself the proof that
iterations/LR stayed continuous across all three boundaries within this one container invocation."""

LEARNING_RATE_POLICY = (
    "four_continuous_epochs_single_container_invocation_fresh_adam_base_0.0001_continuous_decay_0.99_"
    "across_full_corpus"
)
BATCH_SIZE_DEFAULT = 16
LEARNING_RATE = 0.0001
OPTIMIZER = "adam"
EPOCHS_PER_INVOCATION = 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pool_manifest() -> dict:
    return json.loads((POOL_DIR / "pool_manifest.json").read_text(encoding="utf-8"))


def run_provenance_gate(*, run_dir: Path) -> dict:
    """Identical discipline to Experiment 1's own gate: the parent checkpoint must be the real,
    pristine pinned checkpoint -- never Experiment 0's fine-tuned output, the pilot's, or Experiment 1
    epoch 1's/epoch 2's/epoch 3's own output (this run does not resume any of those directories, it
    restarts from the same origin all of them used)."""
    checks: dict = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ok, digest, extra = verify_checkpoint(PARENT_CHECKPOINT_DIR)
    checks["checkpoint_readable_and_zip_valid"] = ok
    checks["checkpoint_sha256"] = digest
    checks["checkpoint_extra"] = extra
    checks["expected_sha256"] = CURRENT_PINNED_VERSIONS.model_checkpoint_hash
    checks["hash_matches_pin"] = ok and digest == CURRENT_PINNED_VERSIONS.model_checkpoint_hash

    resolved_parent = PARENT_CHECKPOINT_DIR.resolve()
    rejected_paths = [
        str(p.resolve()) for p in (
            DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR, EXPERIMENT_1_EPOCH1_RUN_DIR, EXPERIMENT_1_EPOCH2_RUN_DIR,
            EXPERIMENT_1_EPOCH3_RUN_DIR,
        )
    ]
    not_derived_from_prior_run = all(
        resolved_parent != Path(p) and not resolved_parent.is_relative_to(Path(p)) for p in rejected_paths
    )
    checks["parent_checkpoint_path"] = str(resolved_parent)
    checks["checked_against_prior_run_dirs"] = rejected_paths
    checks["not_derived_from_prior_run"] = not_derived_from_prior_run

    checks["passed"] = bool(checks["hash_matches_pin"] and not_derived_from_prior_run)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def _build_dry_run_lists(*, n_train: int, n_val: int) -> tuple[Path, Path]:
    real_train = (POOL_DIR / "train_list.txt").read_text(encoding="utf-8").splitlines()
    real_val = (POOL_DIR / "val_list.txt").read_text(encoding="utf-8").splitlines()
    dry_train = POOL_DIR / "experiment_1_four_epochs_dry_run_train_list.txt"
    dry_val = POOL_DIR / "experiment_1_four_epochs_dry_run_val_list.txt"
    dry_train.write_text("\n".join(real_train[:n_train]) + "\n", encoding="utf-8")
    dry_val.write_text("\n".join(real_val[:n_val]) + "\n", encoding="utf-8")
    return dry_train, dry_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--hours", type=float, default=60.0,
                        help="wall-clock ceiling for the single container invocation, covering ALL "
                             "FOUR internal Keras epochs (the three-epoch run used a 45h ceiling for "
                             "three epochs, itself ~1.5x the two-epoch run's 30h ceiling; this defaults "
                             "to the same ~15h/epoch margin scaled to four epochs). Default: 60.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-lines", type=int, default=None,
                        help="if set, run over only this many train lines (and min(200, n) val lines) "
                             "instead of the full corpus -- for validating the --epochs 4 mechanics "
                             "end-to-end before committing real GPU-hours.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        checks = run_provenance_gate(run_dir=REPO_ROOT / "training" / "_experiment_1_four_epochs_check_only_scratch")
        print(json.dumps(checks, indent=2))
        return 0 if checks["passed"] else 1

    if not args.confirm:
        raise SystemExit("Refusing to launch training without --confirm. Use --check-only first.")

    run_name = args.run_name or ("experiment-1-four-epochs-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = TRAINING_ROOT / run_name
    run_state_dir = run_dir / "run-state"

    checks = run_provenance_gate(run_dir=run_dir)
    if not checks["passed"]:
        raise SystemExit(
            f"Provenance gate FAILED -- refusing to launch. See {run_dir / 'provenance_check.json'}\n"
            f"{json.dumps(checks, indent=2)}"
        )
    print(f"Provenance gate passed: parent checkpoint is the pristine pinned {CURRENT_PINNED_VERSIONS.model_checkpoint_id}, "
          f"not derived from Experiment 0, the pilot, or Experiment 1 epoch 1/epoch 2/epoch 3. See {run_dir / 'provenance_check.json'}")

    pool_manifest = _read_pool_manifest()
    val_manifest_hash = _sha256_file(VAL_MANIFEST_PATH)
    charlist_hash = _sha256_file(CHARLIST_PATH)

    configuration = TrainingConfiguration(
        train_manifest_hash=pool_manifest["training_manifest_hash"],
        val_manifest_hash=val_manifest_hash,
        charlist_hash=charlist_hash,
        preprocessing_version="byte_identical_from_source_parquet",
        model_architecture="new10",
        parent_checkpoint_hash=CURRENT_PINNED_VERSIONS.model_checkpoint_hash,
        learning_rate_policy=LEARNING_RATE_POLICY,
        optimizer=OPTIMIZER,
        augmentation_policy="none",
    )

    try:
        identity, configuration_hash = create_full_run_identity(
            run_state_dir=run_state_dir,
            parent_checkpoint=f"{CURRENT_PINNED_VERSIONS.model_checkpoint_id}@{CURRENT_PINNED_VERSIONS.model_checkpoint_hash}",
            parent_checkpoint_dir=PARENT_CHECKPOINT_DIR,
            configuration=configuration,
            known_pilot_run_dirs=(
                DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR, EXPERIMENT_1_EPOCH1_RUN_DIR, EXPERIMENT_1_EPOCH2_RUN_DIR,
                EXPERIMENT_1_EPOCH3_RUN_DIR,
            ),
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
        base_checkpoint_hash=CURRENT_PINNED_VERSIONS.model_checkpoint_hash,
        known_pilot_run_dirs=(
            DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR, EXPERIMENT_1_EPOCH1_RUN_DIR, EXPERIMENT_1_EPOCH2_RUN_DIR,
            EXPERIMENT_1_EPOCH3_RUN_DIR,
        ),
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
        max_epochs=EPOCHS_PER_INVOCATION,
        max_epochs_basis="Experiment 1 epoch-4 continuation: four real Keras epochs inside one single, "
                          "uninterrupted container invocation (--epochs 4), so the optimizer/LR-schedule "
                          "object never crosses a process boundary across any epoch. A fresh restart "
                          "from the pristine parent checkpoint, not a resume of the preserved epoch-1, "
                          "epoch-2, or epoch-3 Experiment 1 runs.",
        early_stopping_patience=0,
        early_stopping_basis="Disabled: fixed at exactly four epochs by design.",
        wall_clock_policy=f"Single uninterrupted invocation covering all four epochs, hard ceiling {args.hours}h.",
        shard_count=1,
        checkpoint_policy="Four internal Keras epochs, each independently validated and checkpointed by "
                          "Loghi's own LoghiCustomCallback (epoch_0_..., epoch_1_..., epoch_2_..., "
                          "epoch_3_...); 'latest' resolves to the higher-numbered (final, epoch_3) "
                          "checkpoint by parsed epoch number, not glob order.",
        dashboard_config=f"TrainingDashboardViewModel(training_root={TRAINING_ROOT})",
        exact_launch_command=" ".join(sys.argv),
    )
    manifest_dict = manifest.model_dump()
    manifest_dict.update({
        "epochs_per_container_invocation": EPOCHS_PER_INVOCATION,
        "container_invocations": 1,
        "optimizer_continuity_mechanism": "in_process_model.fit(epochs=4)_single_invocation",
        "cross_process_optimizer_restore_used": False,
        "cross_process_optimizer_restore_mechanism_available_but_unused":
            "training/experiment_epoch2_overrides/main_with_optimizer_restore.py -- built and "
            "empirically verified (exact optimizer_iterations/learning_rate match across two separate "
            "container invocations), but not needed here: this run never crosses a process boundary "
            "between any of the four epochs, so there is nothing for it to restore.",
        "resumes_experiment_1_epoch_1_run": False,
        "resumes_experiment_1_epoch_2_run": False,
        "resumes_experiment_1_epoch_3_run": False,
        "experiment_1_epoch_1_run_dir_preserved_untouched": str(EXPERIMENT_1_EPOCH1_RUN_DIR),
        "experiment_1_epoch_2_run_dir_preserved_untouched": str(EXPERIMENT_1_EPOCH2_RUN_DIR),
        "experiment_1_epoch_3_run_dir_preserved_untouched": str(EXPERIMENT_1_EPOCH3_RUN_DIR),
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
        epochs_per_invocation=EPOCHS_PER_INVOCATION,
    )

    print(f"Starting the single continuous four-epoch container invocation (ceiling {args.hours}h). "
          f"This call does not return until the container exits -- one process, four real Keras "
          f"epochs, no shard restarts, no cross-process optimizer boundary.")
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
        max_epochs_this_call=1,  # one call to epoch_runner.run_epoch() -- that one call runs 4 real
                                  # Keras epochs internally, via epochs_per_invocation=4 above.
        early_stopping_patience=None,
    )
    elapsed = time.monotonic() - started
    print(f"Session ended after {elapsed:.0f}s: stop_reason={summary.stop_reason} "
          f"container_invocations_this_session={summary.epochs_completed_this_session}")

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
            epochs_completed=EPOCHS_PER_INVOCATION,  # 4 genuine full passes over the corpus, all
                                                      # inside this one container invocation.
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
    print(f"latest checkpoint (final, epoch 4) verified={latest_verified}: {last_result.checkpoint_dir if last_result else None}")
    print(f"best_val checkpoint verified={best_verified}: {last_result.best_val_checkpoint_dir if last_result else None}")
    epoch_output_dir = run_state_dir / "epoch_output" / "epoch_1"
    print(f"optimizer trace across all four internal epochs (if the container honored the mounted override): "
          f"{epoch_output_dir / 'optimizer_trace.csv'}")
    print(f"run directory: {run_dir}")

    return 0 if not failure_detail and latest_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
