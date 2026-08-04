"""Experiment 1: one continuous, uninterrupted full-corpus training epoch, correcting Experiment 0's
per-shard optimizer/LR reset (`training/full-corpus-20260802T044250Z/`, `docs/methods/
loghi-full-corpus-training.md`).

Bypasses `full_run/cli.py`/`orchestrator.py`/`corpus_sharding.py` entirely -- there is no shard plan
here. Calls `training_session.run_training_session()` directly, exactly once, with
`max_epochs_this_call=1` over a `train_list.txt` that already covers the whole ~562k-line corpus
(reused, unmodified, from Experiment 0's own data pool). Everything about how one epoch runs --
staging the pristine parent checkpoint, GPU/system telemetry, checkpoint indexing -- is the existing,
unmodified engine; nothing here reimplements it.

The one thing Loghi does not do natively: prove, empirically, that the optimizer and learning-rate
schedule do not reset mid-epoch. That requires `training/experiment_1_overrides/
modes_training_with_optimizer_trace.py`, bind-mounted over the container's own `modes/training.py`
for this run only via `ContainerEpochRunner`'s `extra_volume_mounts` -- the pinned `.loghi-upstream`
checkout on disk is never touched.

Usage:
    python scripts/run_experiment_1_continuous_epoch.py --check-only
    python scripts/run_experiment_1_continuous_epoch.py --dry-run-lines 2000 --confirm
    python scripts/run_experiment_1_continuous_epoch.py --hours 16 --confirm
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
"""The shard-based run being corrected. Never used as a parent checkpoint source, never written to."""
POOL_DIR = TRAINING_ROOT / "_prepared_data" / "fc21a708_e12796a0"
"""Experiment 0's own full-corpus data pool -- `train_list.txt`/`val_list.txt` here already cover the
whole usable corpus and the same fixed validation set (confirmed via `pool_manifest.json`). Reused
unmodified; no new data extraction for Experiment 1."""
VAL_MANIFEST_PATH = TRAINING_ROOT / "loghi-swedish-v1" / "manifests" / "val_manifest.parquet"
OVERRIDE_FILE = TRAINING_ROOT / "experiment_1_overrides" / "modes_training_with_optimizer_trace.py"

LEARNING_RATE_POLICY = (
    "single_continuous_epoch_fresh_adam_base_0.0001_continuous_decay_0.99_across_full_corpus"
)
BATCH_SIZE_DEFAULT = 16
LEARNING_RATE = 0.0001
OPTIMIZER = "adam"
DECAY_RATE_NOTE = "0.99 (loghi-htr default, unchanged from Experiment 0)"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pool_manifest() -> dict:
    return json.loads((POOL_DIR / "pool_manifest.json").read_text(encoding="utf-8"))


def run_provenance_gate(*, run_dir: Path) -> dict:
    """The one check that must pass before anything else happens: the parent checkpoint is the real,
    pristine pinned checkpoint -- never Experiment 0's fine-tuned output, never the pilot's. Written to
    `provenance_check.json` as an auditable fact, independent of whatever `training_identity.py`'s own
    (separate, later) check also confirms.
    """
    checks: dict = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ok, digest, extra = verify_checkpoint(PARENT_CHECKPOINT_DIR)
    checks["checkpoint_readable_and_zip_valid"] = ok
    checks["checkpoint_sha256"] = digest
    checks["checkpoint_extra"] = extra
    checks["expected_sha256"] = CURRENT_PINNED_VERSIONS.model_checkpoint_hash
    checks["hash_matches_pin"] = ok and digest == CURRENT_PINNED_VERSIONS.model_checkpoint_hash

    resolved_parent = PARENT_CHECKPOINT_DIR.resolve()
    rejected_paths = [str(p.resolve()) for p in (DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR)]
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
    """Small list files sliced from the real pool lists, written as new siblings *inside* the pool
    directory -- the list files' own entries are baked-in `/lists/images/...` container paths, so the
    directory mounted at `/lists` must be the pool directory itself (confirmed by inspecting
    `train_list.txt`'s real contents), not a separate directory copied elsewhere.
    """
    real_train = (POOL_DIR / "train_list.txt").read_text(encoding="utf-8").splitlines()
    real_val = (POOL_DIR / "val_list.txt").read_text(encoding="utf-8").splitlines()
    dry_train = POOL_DIR / "experiment_1_dry_run_train_list.txt"
    dry_val = POOL_DIR / "experiment_1_dry_run_val_list.txt"
    dry_train.write_text("\n".join(real_train[:n_train]) + "\n", encoding="utf-8")
    dry_val.write_text("\n".join(real_val[:n_val]) + "\n", encoding="utf-8")
    return dry_train, dry_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--hours", type=float, default=16.0,
                        help="wall-clock ceiling for the single container invocation and the session "
                             "engine (Experiment 0's own run_profile.py extrapolated ~7.2h for one "
                             "full-corpus pass at batch 16). Default: 16.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-lines", type=int, default=None,
                        help="if set, run over only this many train lines (and min(200, n) val lines) "
                             "instead of the full corpus -- for validating the pipeline end-to-end "
                             "before committing real GPU-hours.")
    parser.add_argument("--check-only", action="store_true",
                        help="run the provenance gate and print the result, then exit without "
                             "creating a run directory or touching Docker.")
    parser.add_argument("--confirm", action="store_true",
                        help="required to actually launch training (not needed with --check-only).")
    args = parser.parse_args()

    if args.check_only:
        checks = run_provenance_gate(run_dir=REPO_ROOT / "training" / "_experiment_1_check_only_scratch")
        print(json.dumps(checks, indent=2))
        return 0 if checks["passed"] else 1

    if not args.confirm:
        raise SystemExit("Refusing to launch training without --confirm. Use --check-only first.")

    run_name = args.run_name or ("experiment-1-continuous-epoch-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = TRAINING_ROOT / run_name
    run_state_dir = run_dir / "run-state"

    # --- Provenance gate: must pass before any identity/state file is created. ---
    checks = run_provenance_gate(run_dir=run_dir)
    if not checks["passed"]:
        raise SystemExit(
            f"Provenance gate FAILED -- refusing to launch. See {run_dir / 'provenance_check.json'}\n"
            f"{json.dumps(checks, indent=2)}"
        )
    print(f"Provenance gate passed: parent checkpoint is the pristine pinned {CURRENT_PINNED_VERSIONS.model_checkpoint_id}, "
          f"not derived from Experiment 0 or the pilot. See {run_dir / 'provenance_check.json'}")

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
            known_pilot_run_dirs=(DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR),
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
        known_pilot_run_dirs=(DEFAULT_PILOT_RUN_DIR, EXPERIMENT_0_RUN_DIR),
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
        max_epochs_basis="Experiment 1: exactly one full-corpus pass in a single uninterrupted "
                          "container invocation, by design. Not a multi-epoch or shard-based plan.",
        early_stopping_patience=0,
        early_stopping_basis="Disabled: a single-epoch run has nothing cross-epoch to early-stop against.",
        wall_clock_policy=f"Single uninterrupted invocation, hard ceiling {args.hours}h "
                          "(container subprocess timeout and session wall-clock budget both set from --hours).",
        shard_count=1,
        checkpoint_policy="Single epoch: validate once, then checkpoint both 'latest' and 'best_val' "
                          "(checkpoint_index.json), atomic promotion via write-temp-then-rename.",
        dashboard_config=f"TrainingDashboardViewModel(training_root={TRAINING_ROOT})",
        exact_launch_command=" ".join(sys.argv),
    )
    write_launch_manifest(manifest, run_dir / "launch_manifest.json")
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

    print(f"Starting the single continuous epoch (ceiling {args.hours}h). This call does not return "
          f"until the container exits -- one process, one epoch, no shard restarts.")
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
    print(f"latest checkpoint verified={latest_verified}: {last_result.checkpoint_dir if last_result else None}")
    print(f"best_val checkpoint verified={best_verified}: {last_result.best_val_checkpoint_dir if last_result else None}")
    epoch_output_dir = run_state_dir / "epoch_output" / "epoch_1"
    print(f"optimizer trace (if the container honored the mounted override): {epoch_output_dir / 'optimizer_trace.csv'}")
    print(f"run directory: {run_dir}")

    return 0 if not failure_detail and latest_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
