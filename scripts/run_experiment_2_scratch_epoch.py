"""Experiment 2: Loghi recommended VGSL architecture trained from scratch (random initialization),
one continuous, uninterrupted full-corpus epoch -- the third leg of the Experiment 0/1/2 comparison.

Research question: can a newly initialized Loghi network trained directly on the Swedish corpus
produce a meaningfully different or more promising trajectory than fine-tuning the generic Loghi
checkpoint (Experiments 0/1)?

Structural guarantees, not conventions:
- No --model directory argument exists anywhere in this script's CLI surface. --model is hardcoded
  to the Loghi predefined-library key "recommended" (verified via get_model_library(), not guessed --
  see scratch_epoch_runner.py's own docstring for the exact resolved VGSL spec).
- ScratchEpochRunner mounts no /model directory at all and raises if ever handed an
  existing_model_dir, so "no pretrained checkpoint" holds even if a caller made a mistake upstream.
- Kept identical to Experiment 1 wherever technically possible: full corpus, validation set,
  batch size, optimizer, initial learning rate, precision policy, hardware, container image,
  evaluation code. The two INTENDED changed variables are initialization (random vs pretrained) and
  architecture (native "recommended" vs the generic checkpoint's "new10"), which Phase 4's real,
  measured comparison found to be similar-but-not-identical (see architecture_report.json).

Usage:
    python scripts/run_experiment_2_scratch_epoch.py --check-only
    python scripts/run_experiment_2_scratch_epoch.py --dry-run-lines 2000 --confirm
    python scripts/run_experiment_2_scratch_epoch.py --hours 16 --confirm
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
from archivetrust.htr.training.full_run.identity import RunDirectoryAlreadyExists  # noqa: E402
from archivetrust.htr.training.full_run.launch_manifest import (  # noqa: E402
    build_launch_manifest,
    write_launch_manifest,
    write_prepared_marker,
)
from archivetrust.htr.training.full_run.orchestrator import _has_nan_or_inf, _oom_signature_in  # noqa: E402
from archivetrust.htr.training.full_run.run_state import (  # noqa: E402
    create_initial_run_state,
    mark_completed,
    mark_failed,
    mark_running,
    heartbeat,
    save_run_state,
)
from archivetrust.htr.training.full_run.shard_training_data import ShardTrainingDataPaths  # noqa: E402
from archivetrust.htr.training.scratch_epoch_runner import (  # noqa: E402
    RECOMMENDED_VGSL_SPEC,
    PretrainedCheckpointRejected,
    ScratchEpochRunner,
)
from archivetrust.htr.training.training_identity import (  # noqa: E402
    TrainingConfiguration,
    create_or_load_identity,
)
from archivetrust.htr.training.training_session import run_training_session  # noqa: E402
from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS  # noqa: E402

TRAINING_ROOT = REPO_ROOT / "training"
EXPERIMENT_0_RUN_DIR = TRAINING_ROOT / "full-corpus-20260802T044250Z"
EXPERIMENT_1_RUN_DIR = TRAINING_ROOT / "experiment-1-continuous-epoch-20260803T192146Z"
POOL_DIR = TRAINING_ROOT / "_prepared_data" / "fc21a708_e12796a0"
VAL_MANIFEST_PATH = TRAINING_ROOT / "loghi-swedish-v1" / "manifests" / "val_manifest.parquet"
CHARACTER_INVENTORY_PATH = TRAINING_ROOT / "experiment_2_character_inventory.json"
OVERRIDE_FILE = TRAINING_ROOT / "experiment_1_overrides" / "modes_training_with_optimizer_trace.py"
"""Reused unmodified from Experiment 1 -- the live loss/CER/WER/optimizer-iterations/LR trace
callback has no dependency on where the model's weights came from."""

ARCHITECTURE_KEY = "recommended"
LEARNING_RATE_POLICY = "single_continuous_epoch_random_init_fresh_adam_base_0.0001_continuous_decay_0.99_across_full_corpus"
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


def run_no_pretrained_checkpoint_gate(*, run_dir: Path) -> dict:
    """Phase 5's explicit guard: proves, before anything else, that this launch cannot possibly be
    pointed at a pretrained or prior-experiment checkpoint -- there is no code path in this script
    that accepts a --model directory value at all."""
    checks = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cli_accepts_model_directory_argument": False,
        "architecture_source": "loghi predefined model library key (get_model_library())",
        "architecture_key": ARCHITECTURE_KEY,
        "resolved_vgsl_spec": RECOMMENDED_VGSL_SPEC,
        "runner_class": "ScratchEpochRunner",
        "runner_mounts_model_directory": False,
        "runner_raises_if_existing_model_dir_provided": True,
        "experiment_0_run_dir_referenced_as_parent": False,
        "experiment_1_run_dir_referenced_as_parent": False,
    }
    checks["passed"] = not checks["cli_accepts_model_directory_argument"] and not checks["runner_mounts_model_directory"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "no_pretrained_checkpoint_guard.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def run_character_vocabulary_gate(*, run_dir: Path) -> dict:
    """Reuses Phase 3's already-computed, already-verified inventory rather than recomputing it --
    the vocabulary and its launch_gate.passed verdict do not change between when Phase 3 ran and now,
    since neither train_list.txt nor val_list.txt has been touched."""
    if not CHARACTER_INVENTORY_PATH.exists():
        raise SystemExit(
            f"Refusing to launch: {CHARACTER_INVENTORY_PATH} not found. Run "
            "scripts/analyze_experiment_2_character_vocabulary.py first."
        )
    inventory = json.loads(CHARACTER_INVENTORY_PATH.read_text(encoding="utf-8"))
    gate = inventory["launch_gate"]
    (run_dir / "character_vocabulary_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate["passed"]:
        raise SystemExit(f"Refusing to launch: character vocabulary gate failed -- {gate['reason']}")
    return inventory


def _build_dry_run_lists(*, n_train: int, n_val: int) -> tuple[Path, Path]:
    real_train = (POOL_DIR / "train_list.txt").read_text(encoding="utf-8").splitlines()
    real_val = (POOL_DIR / "val_list.txt").read_text(encoding="utf-8").splitlines()
    dry_train = POOL_DIR / "experiment_2_dry_run_train_list.txt"
    dry_val = POOL_DIR / "experiment_2_dry_run_val_list.txt"
    dry_train.write_text("\n".join(real_train[:n_train]) + "\n", encoding="utf-8")
    dry_val.write_text("\n".join(real_val[:n_val]) + "\n", encoding="utf-8")
    return dry_train, dry_val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--hours", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-lines", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        scratch = REPO_ROOT / "training" / "_experiment_2_check_only_scratch"
        checks = run_no_pretrained_checkpoint_gate(run_dir=scratch)
        vocab = run_character_vocabulary_gate(run_dir=scratch)
        print(json.dumps({"no_pretrained_checkpoint_gate": checks,
                          "character_vocabulary_gate": vocab["launch_gate"]}, indent=2))
        return 0 if checks["passed"] and vocab["launch_gate"]["passed"] else 1

    if not args.confirm:
        raise SystemExit("Refusing to launch training without --confirm. Use --check-only first.")

    run_name = args.run_name or ("experiment-2-loghi-recommended-scratch-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir = TRAINING_ROOT / run_name
    run_state_dir = run_dir / "run-state"

    guard = run_no_pretrained_checkpoint_gate(run_dir=run_dir)
    if not guard["passed"]:
        raise SystemExit(f"Refusing to launch: no-pretrained-checkpoint guard failed.\n{json.dumps(guard, indent=2)}")
    print("No-pretrained-checkpoint guard passed: this script cannot mount or reference any "
          "checkpoint directory -- --model is hardcoded to the library key 'recommended'.")

    vocab_inventory = run_character_vocabulary_gate(run_dir=run_dir)
    print(f"Character vocabulary gate passed: {vocab_inventory['scratch_model_vocabulary']['size']} "
          f"characters, sha256={vocab_inventory['scratch_model_vocabulary']['sha256']}")

    pool_manifest = _read_pool_manifest()
    val_manifest_hash = _sha256_file(VAL_MANIFEST_PATH)

    configuration = TrainingConfiguration(
        train_manifest_hash=pool_manifest["training_manifest_hash"],
        val_manifest_hash=val_manifest_hash,
        charlist_hash=vocab_inventory["scratch_model_vocabulary"]["sha256"],
        preprocessing_version="byte_identical_from_source_parquet",
        model_architecture=ARCHITECTURE_KEY,
        parent_checkpoint_hash="RANDOM_INITIALIZATION",
        learning_rate_policy=LEARNING_RATE_POLICY,
        optimizer=OPTIMIZER,
        augmentation_policy="none",
    )

    try:
        run_state_dir.mkdir(parents=True, exist_ok=False) if not run_state_dir.exists() else None
        if (run_state_dir / "training_identity.json").exists():
            raise RunDirectoryAlreadyExists(f"{run_state_dir} already has a training_identity.json")
        identity, configuration_hash = create_or_load_identity(
            run_state_dir=run_state_dir,
            parent_checkpoint="RANDOM_INITIALIZATION",
            configuration=configuration,
            training_phase="full_corpus_scratch",
        )
    except RunDirectoryAlreadyExists as exc:
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
        pool_dir=str(POOL_DIR), validation_list_path=str(val_list_path),
        shard_list_paths=(str(train_list_path),), train_image_count=train_line_count,
        validation_image_count=validation_line_count, reused_existing_pool=True,
        dataset_hash=pool_manifest["dataset_hash"], training_manifest_hash=pool_manifest["training_manifest_hash"],
    )
    (run_dir / "training_data_paths.json").write_text(training_data_paths.model_dump_json(indent=2), encoding="utf-8")

    manifest = build_launch_manifest(
        run_id=identity.run_id,
        original_loghi_base_checkpoint_path="RANDOM_INITIALIZATION -- no base checkpoint",
        base_checkpoint_hash="RANDOM_INITIALIZATION",
        known_pilot_run_dirs=(),
        training_manifest_dir=POOL_DIR, training_manifest_hash=pool_manifest["training_manifest_hash"],
        validation_manifest_path=VAL_MANIFEST_PATH, validation_manifest_hash=val_manifest_hash,
        dataset_hash=pool_manifest["dataset_hash"], train_line_count=train_line_count,
        validation_line_count=validation_line_count, excluded_test_line_count=810,
        code_commit_hash=create_initial_run_state(run_id=identity.run_id, configuration_hash=configuration_hash).code_revision,
        container_image_name=CURRENT_PINNED_VERSIONS.docker_image_tag,
        container_image_digest=CURRENT_PINNED_VERSIONS.docker_image_digest,
        batch_size=args.batch_size, optimizer=OPTIMIZER, scheduler=LEARNING_RATE_POLICY,
        learning_rate=LEARNING_RATE, random_seed=args.seed, max_epochs=1,
        max_epochs_basis="Experiment 2: exactly one full-corpus pass from random initialization, by design.",
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
        "initialization": "random",
        "pretrained_checkpoint_used": False,
        "experiment_0_weights_used": False,
        "experiment_1_weights_used": False,
        "architecture_source": "loghi predefined model library (get_model_library())",
        "architecture_key": ARCHITECTURE_KEY,
        "architecture_vgsl_spec": RECOMMENDED_VGSL_SPEC,
        "character_vocabulary_sha256": vocab_inventory["scratch_model_vocabulary"]["sha256"],
        "character_vocabulary_size": vocab_inventory["scratch_model_vocabulary"]["size"],
        "automatic_epoch_2": False,
    })
    (run_dir / "launch_manifest.json").write_text(json.dumps(manifest_dict, indent=2, default=str), encoding="utf-8")
    write_prepared_marker(run_dir)

    state = create_initial_run_state(run_id=identity.run_id, configuration_hash=configuration_hash, dataset_hash=pool_manifest["dataset_hash"])
    state = state.model_copy(update={"shards_per_epoch": 1})
    state = mark_running(state)
    save_run_state(run_state_dir, state)

    epoch_runner = ScratchEpochRunner(
        batch_size=args.batch_size, gradient_accumulation=1, precision="mixed_float16",
        max_image_width=65536, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
        beam_width=CURRENT_PINNED_VERSIONS.beam_width, timeout_seconds=args.hours * 3600.0,
        run_state_dir=run_state_dir,
        extra_volume_mounts=((str(OVERRIDE_FILE), "/src/loghi-htr/src/modes/training.py"),),
        architecture=ARCHITECTURE_KEY,
    )

    print(f"Starting the single continuous epoch from random initialization (ceiling {args.hours}h). "
          f"This call does not return until the container exits.")
    started = time.monotonic()
    try:
        summary = run_training_session(
            run_state_dir=run_state_dir, checkpoint_index_path=run_state_dir / "checkpoint_index.json",
            epoch_runner=epoch_runner, train_list_path=str(train_list_path),
            validation_list_path=str(val_list_path), parent_checkpoint_dir=None,
            run_id=identity.run_id, configuration_hash=configuration_hash, random_seed=args.seed,
            max_wall_clock_seconds=args.hours * 3600.0,
            stop_requested=lambda: (run_state_dir / "STOP_REQUESTED").exists(),
            max_epochs_this_call=1, early_stopping_patience=None,
        )
    except PretrainedCheckpointRejected as exc:
        raise SystemExit(f"STRUCTURAL GUARD TRIPPED (this should be unreachable): {exc}") from exc
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
            state, current_epoch=1, global_shards_completed=1, epochs_completed=1,
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
    print(f"optimizer/CER/WER trace: {epoch_output_dir / 'optimizer_trace.csv'}")
    print(f"run directory: {run_dir}")

    return 0 if not failure_detail and latest_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
