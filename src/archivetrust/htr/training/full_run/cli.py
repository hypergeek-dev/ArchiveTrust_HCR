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

    print(f"Pilot analysis written to {_pilot_analysis_path()}")
    print(f"Derived monitoring config written to {_monitoring_config_path()}")
    print()
    print(f"pilot_epoch_count: {analysis.pilot_epoch_count}")
    print(f"best_pilot_epoch: {analysis.best_pilot_epoch} (val_CER={analysis.best_pilot_val_cer})")
    print(f"estimated_plateau_epoch: {analysis.estimated_plateau_epoch} "
          f"(extrapolated={analysis.plateau_is_extrapolated})")
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

    config = load_monitoring_config(args.config)
    run_dir = Path(args.run) if args.run else TRAINING_ROOT / "preflight-check"

    train_manifest_paths: tuple[Path, ...] = ()
    val_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "val_manifest.parquet"
    sharding_dir = run_dir / "shards"
    if (sharding_dir / "sharding_summary.json").exists():
        from archivetrust.htr.training.full_run.corpus_sharding import load_sharding_summary

        summary = load_sharding_summary(sharding_dir)
        train_manifest_paths = tuple(Path(s.manifest_path) for s in summary.shards[:1])

    smoke_test_runner = None
    probe_train_list = probe_val_list = None
    if not args.no_smoke_test:
        from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner
        from archivetrust.providers.loghi.environment import probe_loghi_environment

        if probe_loghi_environment().docker_cli_present:
            smoke_test_runner = ContainerEpochRunner(
                batch_size=args.batch_size, gradient_accumulation=1, precision="mixed_float16",
                max_image_width=65536, optimizer="adam", learning_rate=0.0001, timeout_seconds=900,
            )
            probe_train_list = str(DEFAULT_PILOT_RUN_DIR / "prepared-data" / "probe_train_list.txt")
            probe_val_list = str(DEFAULT_PILOT_RUN_DIR / "prepared-data" / "probe_val_list.txt")

    report = run_preflight(
        base_model_dir=PARENT_CHECKPOINT_DIR,
        parent_checkpoint_dir=PARENT_CHECKPOINT_DIR,
        known_pilot_run_dirs=_known_pilot_run_dirs(),
        train_manifest_paths=train_manifest_paths,
        val_manifest_path=val_manifest_path,
        output_dir=run_dir,
        monitoring_config=config,
        dataset_hash=_dataset_hash(),
        require_gpu=not args.no_gpu,
        run_smoke_test=not args.no_smoke_test,
        smoke_test_runner=smoke_test_runner,
        probe_train_list=probe_train_list,
        probe_val_list=probe_val_list,
    )

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.message}")
    print()
    print(f"All critical checks passed: {report.all_critical_passed}")
    return 0 if report.all_critical_passed else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    from archivetrust.htr.training.full_run.corpus_sharding import build_full_corpus_shards
    from archivetrust.htr.training.full_run.identity import create_full_run_identity, generate_run_name
    from archivetrust.htr.training.full_run.monitoring_config import load_monitoring_config
    from archivetrust.htr.training.full_run.run_state import create_initial_run_state, save_run_state
    from archivetrust.htr.training.training_identity import TrainingConfiguration
    from archivetrust.providers.loghi.pinned_versions import CURRENT_PINNED_VERSIONS

    config = load_monitoring_config(args.config)
    run_name = generate_run_name(base_name=args.run_name)
    run_dir = TRAINING_ROOT / run_name
    run_state_dir = run_dir / "run-state"

    charlist_hash = hashlib.sha256(CHARLIST_PATH.read_bytes()).hexdigest() if CHARLIST_PATH.exists() else "unknown"
    training_configuration = TrainingConfiguration(
        train_manifest_hash="full_corpus_shards", val_manifest_hash="pilot_val_manifest_reused",
        charlist_hash=charlist_hash, preprocessing_version="byte_identical_from_source_parquet",
        model_architecture="new10", parent_checkpoint_hash=CURRENT_PINNED_VERSIONS.model_checkpoint_hash,
        learning_rate_policy="constant_0.0001_decay_0.99", optimizer="adam", augmentation_policy="none",
    )

    identity, configuration_hash = create_full_run_identity(
        run_state_dir=run_state_dir,
        parent_checkpoint=f"{CURRENT_PINNED_VERSIONS.model_checkpoint_id}@{CURRENT_PINNED_VERSIONS.model_checkpoint_hash}",
        parent_checkpoint_dir=PARENT_CHECKPOINT_DIR,
        configuration=training_configuration,
        known_pilot_run_dirs=_known_pilot_run_dirs(),
    )

    val_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "val_manifest.parquet"
    test_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "test_reserved_manifest.parquet"
    shards_dir = run_dir / "shards"
    sharding_summary = build_full_corpus_shards(
        inventory_path=INVENTORY_PATH, output_dir=shards_dir,
        exclude_manifest_paths=(val_manifest_path, test_manifest_path),
        shard_line_count=config.shard_line_count, seed=42, target_shard_count=config.max_full_run_epochs,
    )

    dataset_hash = _dataset_hash()
    run_state = create_initial_run_state(run_id=identity.run_id, configuration_hash=configuration_hash, dataset_hash=dataset_hash)
    save_run_state(run_state_dir, run_state)

    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    from archivetrust.htr.training.full_run.monitoring_config import write_monitoring_config

    write_monitoring_config(config, run_dir / "config" / "full_run_monitoring.json")

    print("== Preparation report ==")
    print(f"Run name: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"Selected base model: {PARENT_CHECKPOINT_DIR}")
    print("Checkpoint will be loaded: yes (pristine pinned base checkpoint, staged fresh on first epoch)")
    print(f"Full dataset size (usable, excluding pilot val/test): {sharding_summary.usable_line_count}")
    print(f"Training shard size: {config.shard_line_count} lines/shard, {len(sharding_summary.shards)} shard(s) prepared")
    print(f"Validation split: {val_manifest_path} (reused from the pilot, never used for training)")
    print(f"Estimated steps per shard: {config.steps_per_shard}")
    print(f"Configured maximum epochs (shards): {config.max_full_run_epochs}")
    print(f"Configured minimum training exposure: {config.min_exposure_steps} steps ({config.min_exposure_basis})")
    print(f"Early-stopping patience: {config.recommended_patience}")
    print("Validation interval: every shard")
    print("Checkpoint interval: every shard")
    print(f"Output directory: {run_dir}")
    print()
    print("Pilot weights will not be used. Pilot metrics are used only for monitoring configuration.")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    return _run_session(args, resume=False)


def cmd_resume(args: argparse.Namespace) -> int:
    return _run_session(args, resume=True)


def _run_session(args: argparse.Namespace, *, resume: bool) -> int:
    from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner
    from archivetrust.htr.training.full_run.corpus_sharding import load_sharding_summary
    from archivetrust.htr.training.full_run.monitoring_config import load_monitoring_config
    from archivetrust.htr.training.full_run.orchestrator import run_full_corpus_session
    from archivetrust.htr.training.full_run.run_state import (
        STATUS_PREPARED,
        load_run_state,
        mark_resumed,
        save_run_state,
    )

    run_dir = Path(args.run)
    run_state_dir = run_dir / "run-state"
    state = load_run_state(run_state_dir)
    if state is None:
        print(f"No run_state.json under {run_state_dir} -- run `prepare` first.", file=sys.stderr)
        return 1
    if not resume and state.status != STATUS_PREPARED:
        print(f"Run status is {state.status!r}, not 'prepared' -- use `resume` instead of `start`.", file=sys.stderr)
        return 1
    if resume:
        save_run_state(run_state_dir, mark_resumed(state))

    monitoring_config = load_monitoring_config(run_dir / "config" / "full_run_monitoring.json")
    sharding_summary = load_sharding_summary(run_dir / "shards")

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
        random_seed=42,
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
    from archivetrust.htr.training.full_run.run_state import load_run_state

    run_state_dir = Path(args.run) / "run-state"
    state = load_run_state(run_state_dir)
    if state is None:
        print(f"No run_state.json under {run_state_dir}.")
        return 1
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
    preflight.add_argument("--no-gpu", action="store_true")
    preflight.add_argument("--no-smoke-test", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    prepare = subparsers.add_parser("prepare", help="create a fresh full-corpus run directory")
    prepare.add_argument("--config", default=str(_monitoring_config_path()))
    prepare.add_argument("--run-name", default=None)
    prepare.set_defaults(func=cmd_prepare)

    start = subparsers.add_parser("start", help="start a prepared full-corpus run")
    start.add_argument("--run", required=True)
    start.add_argument("--hours", type=float, default=5.0)
    start.add_argument("--batch-size", type=int, default=16)
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
