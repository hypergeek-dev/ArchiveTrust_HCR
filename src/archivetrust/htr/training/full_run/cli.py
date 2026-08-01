"""`python -m archivetrust.htr.training.full_run <subcommand>` -- the operator-controlled
full-corpus training workflow's command surface: `analyze-pilot | preflight | prepare | start |
status | stop | resume | gui`. `start` never runs automatically from `prepare` -- always a separate,
explicit invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = Path(r"F:\huggingface_dataset")
LOGHI_UPSTREAM = REPO_ROOT / ".loghi-upstream"
PARENT_CHECKPOINT_DIR = LOGHI_UPSTREAM / "pretrained-models" / "loghi-htr" / "generic-2023-02-15"
CHARLIST_PATH = PARENT_CHECKPOINT_DIR / "charlist.txt"
TRAINING_ROOT = REPO_ROOT / "training"
DEFAULT_PILOT_RUN_DIR = TRAINING_ROOT / "loghi-swedish-v1"
INVENTORY_PATH = DEFAULT_PILOT_RUN_DIR / "source-inventory" / "full-inventory.parquet"
CONFIG_DIR = TRAINING_ROOT / "config"
"""`training/config/` -- exactly the location the brief names for `pilot_derived_monitoring.json`."""

STOP_SENTINEL_NAME = "STOP_REQUESTED"


def _pilot_analysis_path() -> Path:
    return CONFIG_DIR / "pilot_analysis.json"


def _monitoring_config_path() -> Path:
    return CONFIG_DIR / "pilot_derived_monitoring.json"


def _dataset_hash_record_path() -> Path:
    return CONFIG_DIR / "dataset_hash_record.json"


def _read_recorded_dataset_hash() -> str | None:
    path = _dataset_hash_record_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("dataset_hash")


def _write_dataset_hash_record(dataset_hash: str | None) -> None:
    import time

    path = _dataset_hash_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"dataset_hash": dataset_hash, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2),
        encoding="utf-8",
    )


def cmd_analyze_pilot(args: argparse.Namespace) -> int:
    from archivetrust.htr.training.full_run.monitoring_config import (
        derive_monitoring_config,
        write_monitoring_config,
    )
    from archivetrust.htr.training.full_run.pilot_analysis import analyze_pilot_run, write_pilot_analysis

    pilot_run_dir = Path(args.pilot_run)
    analysis = analyze_pilot_run(pilot_run_dir)
    write_pilot_analysis(analysis, _pilot_analysis_path())

    config = derive_monitoring_config(
        analysis,
        shard_line_count=args.shard_line_count,
        total_valid_corpus_line_count=args.corpus_line_count,
        patience_override=args.patience,
        max_full_run_epochs_override=args.max_epochs,
    )
    write_monitoring_config(config, _monitoring_config_path())

    # Records the dataset hash as of analysis time -- preflight and the launch guard both compare
    # against this to detect a dataset change between analysis and (attempted) launch.
    _write_dataset_hash_record(_dataset_hash())

    print(f"Pilot analysis written to {_pilot_analysis_path()}")
    print(f"Derived monitoring config written to {_monitoring_config_path()}")
    print(f"Dataset hash recorded to {_dataset_hash_record_path()}")
    print()
    print(f"pilot_run_id: {analysis.pilot_run_id}")
    print(f"base_checkpoint_identity: {analysis.base_checkpoint_identity}")
    print(f"pilot_epoch_count: {analysis.pilot_epoch_count}")
    print(f"best_pilot_epoch: {analysis.best_pilot_epoch} (val_CER={analysis.best_pilot_val_cer})")
    print(f"estimated_plateau_epoch: {analysis.estimated_plateau_epoch} "
          f"(extrapolated={analysis.plateau_is_extrapolated})")
    print(f"genuine_plateau_occurred: {analysis.genuine_plateau_occurred}")
    print(f"early_stopping_triggered: {analysis.early_stopping_triggered} ({analysis.early_stopping_note})")
    print(f"stop_reason: {analysis.stop_reason}")
    print(f"measured_vs_extrapolated_summary: {analysis.measured_vs_extrapolated_summary}")
    print(f"shard_line_count: {config.shard_line_count}")
    print(f"min_exposure_steps: {config.min_exposure_steps} ({config.min_exposure_basis})")
    print(f"recommended_patience: {config.recommended_patience} ({config.patience_basis})")
    print(f"max_full_run_epochs: {config.max_full_run_epochs}")
    return 0


def _known_pilot_run_dirs() -> tuple[Path, ...]:
    return tuple(p for p in TRAINING_ROOT.iterdir() if p.is_dir() and (p / "manifests" / "pilot_split_summary.json").exists()) \
        if TRAINING_ROOT.exists() else ()


def _dataset_hash() -> str | None:
    if not INVENTORY_PATH.exists():
        return None
    digest = hashlib.sha256()
    with INVENTORY_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_preflight(args: argparse.Namespace) -> int:
    from archivetrust.htr.training.full_run.monitoring_config import load_monitoring_config
    from archivetrust.htr.training.full_run.preflight import run_preflight
    from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

    config = load_monitoring_config(args.config)
    run_dir = Path(args.run) if args.run else TRAINING_ROOT / "preflight-check"

    val_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "val_manifest.parquet"
    test_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "test_reserved_manifest.parquet"
    resume_proof_path = DEFAULT_PILOT_RUN_DIR / "reports" / "resume-proof.json"

    smoke_test_runner = None
    probe_train_list = probe_val_list = None
    if not args.no_smoke_test:
        from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner

        smoke_test_runner = ContainerEpochRunner(
            batch_size=args.batch_size, gradient_accumulation=1, precision="mixed_float16",
            max_image_width=65536, optimizer="adam", learning_rate=0.0001, timeout_seconds=900,
        )
        probe_train_list = str(DEFAULT_PILOT_RUN_DIR / "prepared-data" / "probe_train_list.txt")
        probe_val_list = str(DEFAULT_PILOT_RUN_DIR / "prepared-data" / "probe_val_list.txt")

    report = run_preflight(
        base_model_dir=PARENT_CHECKPOINT_DIR,
        base_checkpoint_pinned_hash=CURRENT_PINNED_VERSIONS.model_checkpoint_hash,
        parent_checkpoint_dir=PARENT_CHECKPOINT_DIR,
        known_pilot_run_dirs=_known_pilot_run_dirs(),
        pilot_val_manifest_path=val_manifest_path,
        pilot_test_manifest_path=test_manifest_path,
        output_dir=run_dir,
        training_root=TRAINING_ROOT,
        monitoring_config=config,
        dataset_hash=_dataset_hash(),
        recorded_dataset_hash=_read_recorded_dataset_hash(),
        random_seed=args.seed,
        pilot_resume_proof_path=resume_proof_path,
        container_image_tag=CURRENT_PINNED_VERSIONS.docker_image_tag,
        container_image_digest=CURRENT_PINNED_VERSIONS.docker_image_digest,
        require_gpu=not args.no_gpu,
        min_free_disk_gb=args.min_free_disk_gb,
        run_smoke_test=not args.no_smoke_test,
        check_docker_daemon=not args.no_docker_check,
        check_container_image=not args.no_docker_check,
        smoke_test_runner=smoke_test_runner,
        probe_train_list=probe_train_list,
        probe_val_list=probe_val_list,
    )

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.message}")
    print()
    print(f"Preflight: {report.summary} passed")
    print(f"All critical checks passed: {report.all_critical_passed}")
    _write_preflight_status(passed=report.all_critical_passed, summary=report.summary)
    return 0 if report.all_critical_passed else 1


def _preflight_status_path() -> Path:
    return CONFIG_DIR / "preflight_status.json"


def _write_preflight_status(*, passed: bool, summary: str) -> None:
    import time

    path = _preflight_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"all_critical_passed": passed, "summary": summary, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_preflight_status() -> tuple[bool, float | None]:
    """`(all_critical_passed, age_seconds)` -- `(False, None)` if no preflight has ever been recorded."""
    import calendar
    import time

    path = _preflight_status_path()
    if not path.exists():
        return False, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        checked_at = calendar.timegm(time.strptime(payload["checked_at"], "%Y-%m-%dT%H:%M:%SZ"))
    except (KeyError, ValueError):
        return bool(payload.get("all_critical_passed")), None
    return bool(payload.get("all_critical_passed")), max(0.0, time.time() - checked_at)


def cmd_prepare(args: argparse.Namespace) -> int:
    from archivetrust.htr.training.full_run.corpus_sharding import build_full_corpus_shards
    from archivetrust.htr.training.full_run.identity import create_full_run_identity, generate_run_name
    from archivetrust.htr.training.full_run.launch_manifest import (
        build_launch_manifest,
        write_launch_manifest,
        write_prepared_marker,
    )
    from archivetrust.htr.training.full_run.launch_preview import build_launch_preview, write_launch_preview_file
    from archivetrust.htr.training.full_run.monitoring_config import load_monitoring_config, write_monitoring_config
    from archivetrust.htr.training.full_run.run_state import create_initial_run_state, display_status, save_run_state
    from archivetrust.htr.training.training_identity import TrainingConfiguration
    from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

    config = load_monitoring_config(args.config)
    run_name = generate_run_name(base_name=args.run_name)
    run_dir = TRAINING_ROOT / run_name
    run_state_dir = run_dir / "run-state"
    known_pilot_run_dirs = _known_pilot_run_dirs()

    val_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "val_manifest.parquet"
    test_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "test_reserved_manifest.parquet"
    val_manifest_hash = hashlib.sha256(val_manifest_path.read_bytes()).hexdigest() if val_manifest_path.exists() else "unknown"
    val_line_count = pq.read_table(val_manifest_path, columns=["line_id"]).num_rows if val_manifest_path.exists() else 0
    test_line_count = pq.read_table(test_manifest_path, columns=["line_id"]).num_rows if test_manifest_path.exists() else 0

    # Shards are built *before* identity/configuration-hash creation so the real
    # `sharding_summary.line_id_set_hash` -- not a placeholder string -- is what goes into the
    # configuration hash and the launch manifest.
    shards_dir = run_dir / "shards"
    sharding_summary = build_full_corpus_shards(
        inventory_path=INVENTORY_PATH, output_dir=shards_dir,
        exclude_manifest_paths=(val_manifest_path, test_manifest_path),
        shard_line_count=config.shard_line_count, seed=args.seed, target_shard_count=config.max_full_run_epochs,
    )

    charlist_hash = hashlib.sha256(CHARLIST_PATH.read_bytes()).hexdigest() if CHARLIST_PATH.exists() else "unknown"
    training_configuration = TrainingConfiguration(
        train_manifest_hash=sharding_summary.line_id_set_hash, val_manifest_hash=val_manifest_hash,
        charlist_hash=charlist_hash, preprocessing_version="byte_identical_from_source_parquet",
        model_architecture="new10", parent_checkpoint_hash=CURRENT_PINNED_VERSIONS.model_checkpoint_hash,
        # Accurate as of 2026-08-01, after empirical verification. The previous label
        # ("constant_0.0001_decay_0.99") implied one continuous decaying schedule across the whole
        # run; that is false. The pinned container saves no optimizer state (clone_model) and
        # rebuilds a fresh Adam + schedule on every invocation, so decay only ever acts *within* one
        # shard's ~625 steps and then resets to the base rate. See training_session.py's module
        # docstring for the measured evidence.
        learning_rate_policy="per_shard_fresh_adam_base_0.0001_intra_shard_decay_0.99_no_cross_shard_continuity",
        optimizer="adam", augmentation_policy="none",
    )

    identity, configuration_hash = create_full_run_identity(
        run_state_dir=run_state_dir,
        parent_checkpoint=f"{CURRENT_PINNED_VERSIONS.model_checkpoint_id}@{CURRENT_PINNED_VERSIONS.model_checkpoint_hash}",
        parent_checkpoint_dir=PARENT_CHECKPOINT_DIR,
        configuration=training_configuration,
        known_pilot_run_dirs=known_pilot_run_dirs,
    )

    dataset_hash = _dataset_hash()
    run_state = create_initial_run_state(run_id=identity.run_id, configuration_hash=configuration_hash, dataset_hash=dataset_hash)
    save_run_state(run_state_dir, run_state)

    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    write_monitoring_config(config, run_dir / "config" / "full_run_monitoring.json")

    preview = build_launch_preview(run_dir=run_dir, hours=args.hours, batch_size=args.batch_size)
    write_launch_preview_file(preview, run_dir / "LAUNCH_COMMANDS.txt")

    manifest = build_launch_manifest(
        run_id=identity.run_id,
        original_loghi_base_checkpoint_path=PARENT_CHECKPOINT_DIR,
        base_checkpoint_hash=CURRENT_PINNED_VERSIONS.model_checkpoint_hash,
        known_pilot_run_dirs=known_pilot_run_dirs,
        training_manifest_dir=shards_dir,
        training_manifest_hash=sharding_summary.line_id_set_hash,
        validation_manifest_path=val_manifest_path,
        validation_manifest_hash=val_manifest_hash,
        dataset_hash=dataset_hash,
        train_line_count=sharding_summary.usable_line_count,
        validation_line_count=val_line_count,
        excluded_test_line_count=test_line_count,
        code_commit_hash=run_state.code_revision,
        container_image_name=CURRENT_PINNED_VERSIONS.docker_image_tag,
        container_image_digest=CURRENT_PINNED_VERSIONS.docker_image_digest,
        batch_size=args.batch_size,
        optimizer="adam",
        scheduler=training_configuration.learning_rate_policy,
        learning_rate=0.0001,
        random_seed=args.seed,
        max_epochs=config.max_full_run_epochs,
        max_epochs_basis=config.assumptions_and_method,
        early_stopping_patience=config.recommended_patience,
        early_stopping_basis=config.patience_basis,
        wall_clock_policy=(
            "Operator-specified per invocation via --hours (not fixed at prepare time); a run "
            "resumes across many bounded invocations via `resume`, never one unbounded process."
        ),
        shard_count=len(sharding_summary.shards),
        checkpoint_policy="Every shard: validate, then checkpoint both 'latest' and 'best_val' (checkpoint_index.json), atomic promotion via write-temp-then-rename.",
        dashboard_config=f"TrainingDashboardViewModel(training_root={TRAINING_ROOT})",
        exact_launch_command=preview.launch_command,
    )
    write_launch_manifest(manifest, run_dir / "launch_manifest.json")
    write_prepared_marker(run_dir)

    print("== Preparation report ==")
    print(f"Run name: {run_name}")
    print(f"Run ID: {identity.run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Status: {display_status(run_state)}")
    print(f"Selected base model: {PARENT_CHECKPOINT_DIR}")
    print(f"Base checkpoint hash: {CURRENT_PINNED_VERSIONS.model_checkpoint_hash}")
    print("Checkpoint will be loaded: yes (pristine pinned base checkpoint, staged fresh on first epoch)")
    print("Pilot checkpoint used as parent: no")
    print(f"Full dataset size (usable, excluding pilot val/test): {sharding_summary.usable_line_count}")
    print(f"Training shard size: {config.shard_line_count} lines/shard, {len(sharding_summary.shards)} shard(s) prepared")
    print(f"Training manifest hash: {sharding_summary.line_id_set_hash}")
    print(f"Validation split: {val_manifest_path} (reused from the pilot, never used for training), {val_line_count} lines")
    print(f"Validation manifest hash: {val_manifest_hash}")
    print(f"Excluded reserved test lines: {test_line_count}")
    print(f"Dataset hash: {dataset_hash}")
    print(f"Estimated steps per shard: {config.steps_per_shard}")
    print(f"Configured maximum epochs (shards): {config.max_full_run_epochs}")
    print(f"Configured minimum training exposure: {config.min_exposure_steps} steps ({config.min_exposure_basis})")
    print(f"Early-stopping patience: {config.recommended_patience}")
    print("Validation interval: every shard")
    print("Checkpoint interval: every shard")
    print(f"Code commit: {run_state.code_revision} (repository dirty: {manifest.repository_dirty})")
    print(f"Container image: {manifest.container_image_name}")
    print(f"Output directory: {run_dir}")
    print(f"Launch manifest: {run_dir / 'launch_manifest.json'}")
    print(f"Launch commands file (DO NOT EXECUTE AUTOMATICALLY): {run_dir / 'LAUNCH_COMMANDS.txt'}")
    print()
    print("Pilot weights will not be used. Pilot metrics are used only for monitoring configuration.")
    print()
    print(f"FULL-CORPUS STATUS: {display_status(run_state)}")
    print()
    print("Exact manual launch command (NOT executed):")
    print(f"  {preview.launch_command}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    return _run_session(args, resume=False)


def cmd_resume(args: argparse.Namespace) -> int:
    return _run_session(args, resume=True)


def _run_session(args: argparse.Namespace, *, resume: bool) -> int:
    from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner
    from archivetrust.htr.training.full_run.corpus_sharding import load_sharding_summary
    from archivetrust.htr.training.full_run.launch_guard import LaunchGuardRejected, enforce_launch_guard
    from archivetrust.htr.training.full_run.launch_manifest import load_launch_manifest
    from archivetrust.htr.training.full_run.monitoring_config import load_monitoring_config
    from archivetrust.htr.training.full_run.orchestrator import run_full_corpus_session
    from archivetrust.htr.training.full_run.run_state import (
        STATUS_PREPARED,
        get_code_revision,
        load_run_state,
        mark_resumed,
        save_run_state,
    )
    from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

    run_dir = Path(args.run)
    run_state_dir = run_dir / "run-state"
    verb = "resume" if resume else "start"

    state = load_run_state(run_state_dir)
    if state is None:
        print(f"No run_state.json under {run_state_dir} -- run `prepare` first.", file=sys.stderr)
        return 1
    if not resume and state.status != STATUS_PREPARED:
        print(f"Run status is {state.status!r}, not 'prepared' -- use `resume` instead of `start`.", file=sys.stderr)
        return 1

    launch_manifest = load_launch_manifest(run_dir / "launch_manifest.json") if (run_dir / "launch_manifest.json").exists() else None
    preflight_passed, preflight_age_seconds = _read_preflight_status()
    sharding_summary = load_sharding_summary(run_dir / "shards")

    from archivetrust.htr.training.full_run.launch_guard import recompute_real_shard_hash

    guard_kwargs = dict(
        confirmed=args.confirm_full_corpus_run,
        run_dir=run_dir,
        run_state=state,
        launch_manifest=launch_manifest,
        current_dataset_hash=_dataset_hash(),
        # A real rehash of every real lap-0 shard file's actual line_id column -- not just a second
        # read of sharding_summary.json's own recorded hash field (which would never catch a shard
        # file replaced or corrupted after `prepare` without the summary itself being touched).
        current_training_manifest_hash=recompute_real_shard_hash(run_dir / "shards"),
        test_manifest_path=DEFAULT_PILOT_RUN_DIR / "manifests" / "test_reserved_manifest.parquet",
        preflight_passed=preflight_passed,
        preflight_age_seconds=preflight_age_seconds,
        max_preflight_age_seconds=args.max_preflight_age_seconds,
        min_free_disk_gb=20.0,
        container_image_tag=CURRENT_PINNED_VERSIONS.docker_image_tag,
        container_image_digest=CURRENT_PINNED_VERSIONS.docker_image_digest,
        allow_dirty_repository=args.allow_dirty_repository,
        check_docker=not args.no_docker_check,
        current_code_revision=get_code_revision(),
        allow_code_revision_drift=args.allow_code_revision_drift,
        is_resume=resume,
    )

    if args.dry_run:
        from archivetrust.htr.training.full_run.launch_guard import evaluate_launch_guard

        problems = evaluate_launch_guard(**guard_kwargs)
        print(f"DRY RUN ({verb}): launch-guard evaluation for {run_dir}:")
        if problems:
            for p in problems:
                print(f"  [WOULD REJECT] {p}")
        else:
            print("  All launch-guard conditions currently pass (confirmation flag as supplied).")
        print("DRY RUN: configuration parsed, manifests loaded, shard set discovered "
              f"({len(sharding_summary.shards)} shard(s)), container command would use image "
              f"{CURRENT_PINNED_VERSIONS.docker_image_tag}, volume/env construction not yet exercised "
              "beyond this point.")
        print(f"DRY RUN: STOPPING HERE -- before {'mark_resumed()' if resume else 'mark_running()'}, "
              "before invoking run_full_corpus_session(), before any container process is created, "
              "before any optimizer step. No model weights were touched.")
        return 0

    try:
        enforce_launch_guard(**guard_kwargs)
    except LaunchGuardRejected as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if resume:
        save_run_state(run_state_dir, mark_resumed(state))

    monitoring_config = load_monitoring_config(run_dir / "config" / "full_run_monitoring.json")

    stop_sentinel = run_state_dir / STOP_SENTINEL_NAME
    stop_sentinel.unlink(missing_ok=True)

    runner = ContainerEpochRunner(
        batch_size=args.batch_size, gradient_accumulation=1, precision="mixed_float16",
        max_image_width=65536, optimizer="adam", learning_rate=0.0001, timeout_seconds=7200,
        run_state_dir=run_state_dir,
    )

    summary = run_full_corpus_session(
        run_state_dir=run_state_dir,
        checkpoint_index_path=run_state_dir / "checkpoint_index.json",
        epoch_runner=runner,
        shards=sharding_summary.shards,
        validation_list_path=str(DEFAULT_PILOT_RUN_DIR / "manifests" / "val_manifest.parquet"),
        parent_checkpoint_dir=str(PARENT_CHECKPOINT_DIR),
        run_id=state.run_id,
        configuration_hash=state.configuration_hash,
        random_seed=args.seed,
        monitoring_config=monitoring_config,
        max_wall_clock_seconds=args.hours * 3600.0,
        stop_requested=lambda: stop_sentinel.exists(),
    )

    print(f"stop_reason: {summary.stop_reason}")
    print(f"shards_completed_this_call: {summary.shards_completed_this_call}")
    print(f"cumulative_shards_completed: {summary.cumulative_shards_completed}")
    print(f"completed: {summary.completed}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from archivetrust.htr.training.full_run.run_state import display_status, load_run_state

    run_state_dir = Path(args.run) / "run-state"
    state = load_run_state(run_state_dir)
    if state is None:
        print(f"No run_state.json under {run_state_dir}.")
        return 1
    print(f"display_status: {display_status(state)}")
    print(json.dumps(state.model_dump(), indent=2, ensure_ascii=False))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    run_state_dir = Path(args.run) / "run-state"
    stop_sentinel = run_state_dir / STOP_SENTINEL_NAME
    run_state_dir.mkdir(parents=True, exist_ok=True)
    stop_sentinel.write_text("", encoding="utf-8")
    print(f"Graceful stop requested: {stop_sentinel}")
    print("The run will stop after the shard currently in progress finishes -- never mid-epoch.")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from archivetrust.htr.training.full_run.gui.app import main as gui_main

    return gui_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="archivetrust.htr.training.full_run")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze-pilot", help="analyze a completed pilot run's real behavior")
    analyze.add_argument("--pilot-run", default=str(DEFAULT_PILOT_RUN_DIR))
    analyze.add_argument("--shard-line-count", type=int, default=None)
    analyze.add_argument("--corpus-line-count", type=int, default=None)
    analyze.add_argument("--patience", type=int, default=None)
    analyze.add_argument("--max-epochs", type=int, default=None)
    analyze.set_defaults(func=cmd_analyze_pilot)

    preflight = subparsers.add_parser("preflight", help="validate the environment before preparing a run")
    preflight.add_argument("--config", default=str(_monitoring_config_path()))
    preflight.add_argument("--run", default=None)
    preflight.add_argument("--batch-size", type=int, default=16)
    preflight.add_argument("--seed", type=int, default=42)
    preflight.add_argument("--min-free-disk-gb", type=float, default=20.0)
    preflight.add_argument("--no-gpu", action="store_true")
    preflight.add_argument("--no-smoke-test", action="store_true")
    preflight.add_argument("--no-docker-check", action="store_true", help="skip docker daemon/image checks (for environments without Docker)")
    preflight.set_defaults(func=cmd_preflight)

    prepare = subparsers.add_parser("prepare", help="create a fresh full-corpus run directory")
    prepare.add_argument("--config", default=str(_monitoring_config_path()))
    prepare.add_argument("--run-name", default=None)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--hours", type=float, default=5.0, help="default --hours recorded in the generated launch preview commands")
    prepare.add_argument("--batch-size", type=int, default=16)
    prepare.set_defaults(func=cmd_prepare)

    start = subparsers.add_parser("start", help="start a prepared full-corpus run")
    start.add_argument("--run", required=True)
    start.add_argument("--hours", type=float, default=5.0)
    start.add_argument("--batch-size", type=int, default=16)
    start.add_argument("--seed", type=int, default=42)
    start.add_argument(
        "--confirm-full-corpus-run", action="store_true",
        help="required: the one explicit flag that allows real full-corpus training to start",
    )
    start.add_argument("--dry-run", action="store_true", help="validate everything short of invoking the trainer; never starts real training")
    start.add_argument("--max-preflight-age-seconds", type=float, default=6 * 3600.0)
    start.add_argument("--allow-dirty-repository", action="store_true")
    start.add_argument("--allow-code-revision-drift", action="store_true", help="allow the current git commit to differ from the one recorded at `prepare` time")
    start.add_argument("--no-docker-check", action="store_true", help="skip the launch guard's docker daemon/image checks")
    start.set_defaults(func=cmd_start)

    status = subparsers.add_parser("status", help="show a run's current state")
    status.add_argument("--run", required=True)
    status.set_defaults(func=cmd_status)

    stop = subparsers.add_parser("stop", help="request a graceful stop")
    stop.add_argument("--run", required=True)
    stop.set_defaults(func=cmd_stop)

    resume = subparsers.add_parser("resume", help="resume an interrupted full-corpus run")
    resume.add_argument("--run", required=True)
    resume.add_argument("--hours", type=float, default=5.0)
    resume.add_argument("--batch-size", type=int, default=16)
    resume.add_argument("--seed", type=int, default=42)
    resume.add_argument(
        "--confirm-full-corpus-run", action="store_true",
        help="required: the one explicit flag that allows real full-corpus training to resume",
    )
    resume.add_argument("--dry-run", action="store_true", help="validate everything short of invoking the trainer; never resumes real training")
    resume.add_argument("--max-preflight-age-seconds", type=float, default=6 * 3600.0)
    resume.add_argument("--allow-dirty-repository", action="store_true")
    resume.add_argument("--allow-code-revision-drift", action="store_true", help="allow the current git commit to differ from the one recorded at `prepare` time")
    resume.add_argument("--no-docker-check", action="store_true", help="skip the launch guard's docker daemon/image checks")
    resume.set_defaults(func=cmd_resume)

    gui = subparsers.add_parser("gui", help="open the PySide6 GUI")
    gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
