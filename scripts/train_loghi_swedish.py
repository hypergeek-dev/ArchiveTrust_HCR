#!/usr/bin/env python
"""Swedish Loghi fine-tuning launcher -- resumable pilot training for `loghi_swedish_finetuned_v1`.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/train_loghi_swedish.py

Separate from `scripts/run_lion_loghi_comparison.py` on purpose: that launcher creates method-
comparison experiments over already-trained methods; this one creates a *new* candidate checkpoint
through real, resumable, multi-hour training sessions. Building `loghi_swedish_finetuned_v1` does not
touch `htr/research_status.py` -- it is a future candidate model version under the existing `loghi`
pipeline family, not a new active method (docs/methods/loghi-swedish-finetuning.md).

Reuses this repository's existing patterns throughout: `DurableHtrResearchStore` + `FileTelemetrySink`
for durable state, `_confirm()`-before-acting from `scripts/reliability_test.py`, real per-option
checks that report honestly rather than assume readiness.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from archivetrust.htr.persistence import DurableHtrResearchStore  # noqa: E402
from archivetrust.htr.training.character_inventory import (  # noqa: E402
    build_character_compatibility_report,
    render_human_readable,
    write_character_compatibility_report,
)
from archivetrust.htr.training.checkpoint_index import (  # noqa: E402
    best_validation_checkpoint,
    latest_resumable_checkpoint,
    load_index,
    verify_checkpoint,
)
from archivetrust.htr.training.container_epoch_runner import ContainerEpochRunner  # noqa: E402
from archivetrust.htr.training.loghi_training_data import prepare_loghi_training_data  # noqa: E402
from archivetrust.htr.training.memory_probe import run_memory_probe  # noqa: E402
from archivetrust.htr.training.pilot_split import build_pilot_split, load_pilot_split_summary  # noqa: E402
from archivetrust.htr.training.session_report import (  # noqa: E402
    generate_session_reports,
)
from archivetrust.htr.training.swedish_dataset_inventory import (  # noqa: E402
    build_source_inventory,
)
from archivetrust.htr.training.training_identity import (  # noqa: E402
    TrainingConfiguration,
    create_or_load_identity,
)
from archivetrust.htr.training.training_session import (  # noqa: E402
    load_session_state,
    run_training_session,
)
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402
from archivetrust.providers.loghi.adapter import LoghiAdapter  # noqa: E402
from archivetrust.providers.swedish_lion.adapter import SwedishLionAdapter  # noqa: E402

DATASET_ROOT = Path(r"F:\huggingface_dataset")
LOGHI_UPSTREAM = REPO_ROOT / ".loghi-upstream"
PARENT_CHECKPOINT_DIR = LOGHI_UPSTREAM / "pretrained-models" / "loghi-htr" / "generic-2023-02-15"
CHARLIST_PATH = PARENT_CHECKPOINT_DIR / "charlist.txt"

TRAINING_ROOT = REPO_ROOT / "training" / "loghi-swedish-v1"
INVENTORY_PATH = TRAINING_ROOT / "source-inventory" / "full-inventory.parquet"
MANIFESTS_DIR = TRAINING_ROOT / "manifests"
PREPARED_DATA_DIR = TRAINING_ROOT / "prepared-data"
REPORTS_DIR = TRAINING_ROOT / "reports"
RUN_STATE_DIR = TRAINING_ROOT / "run-state"
CHECKPOINT_INDEX_PATH = RUN_STATE_DIR / "checkpoint_index.json"
STOP_SENTINEL_PATH = RUN_STATE_DIR / "STOP_REQUESTED"

SESSION_TARGET_SECONDS = 5 * 3600.0


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def do_dataset_discovery() -> None:
    print("\n-- Dataset discovery --")
    if not DATASET_ROOT.exists():
        print(f"MISSING: {DATASET_ROOT}")
        return
    collections = [p for p in DATASET_ROOT.iterdir() if p.is_dir()]
    total_files = sum(1 for c in collections for _ in c.glob("*.parquet"))
    print(f"{DATASET_ROOT}: {len(collections)} collections, {total_files} parquet files")
    print(f"Inventory exists: {INVENTORY_PATH.exists()}")
    print(f"Parent checkpoint present: {PARENT_CHECKPOINT_DIR.exists()}")
    loghi_ready = LoghiAdapter().validate_environment()
    print(f"Loghi environment ready: {loghi_ready.valid}")


def do_validate_source_inventory() -> None:
    print("\n-- Validate source inventory --")
    if INVENTORY_PATH.exists():
        print(f"Inventory already exists at {INVENTORY_PATH} -- not regenerating.")
        return
    if not _confirm(f"Build the full source inventory from {DATASET_ROOT}? (~5 minutes)"):
        return
    summary = build_source_inventory(
        dataset_root=DATASET_ROOT, output_path=INVENTORY_PATH, charlist_path=CHARLIST_PATH,
        progress_every=100_000,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:2000])


def do_prepare_pilot_manifests() -> None:
    print("\n-- Prepare or verify pilot manifests --")
    summary_path = MANIFESTS_DIR / "pilot_split_summary.json"
    if summary_path.exists():
        summary = load_pilot_split_summary(MANIFESTS_DIR)
        print(f"Manifests already exist: train={summary.actual_train} val={summary.actual_val} "
              f"test_reserved={summary.actual_test_reserved}")
        return
    if not INVENTORY_PATH.exists():
        print("Run option 2 first (no inventory).")
        return
    if not _confirm("Build the deterministic pilot split?"):
        return
    summary = build_pilot_split(inventory_path=INVENTORY_PATH, output_dir=MANIFESTS_DIR)
    print(f"train={summary.actual_train} val={summary.actual_val} test_reserved={summary.actual_test_reserved}")
    print(f"shortfalls: {summary.shortfall_warnings}")

    char_report_path = REPORTS_DIR / "character-compatibility.json"
    if not char_report_path.exists():
        report = build_character_compatibility_report(
            inventory_path=INVENTORY_PATH,
            manifest_paths=[MANIFESTS_DIR / "train_manifest.parquet", MANIFESTS_DIR / "val_manifest.parquet"],
            charlist_path=CHARLIST_PATH,
        )
        write_character_compatibility_report(report, char_report_path)
        (REPORTS_DIR / "character-compatibility.txt").write_text(
            render_human_readable(report), encoding="utf-8"
        )
        print(f"character compatibility: {report.compatible} ({report.unsupported_character_count} unsupported)")
        if not report.compatible:
            print("WARNING: vocabulary incompatible -- do not proceed to training until resolved.")

    prep_report = prepare_loghi_training_data(
        dataset_root=DATASET_ROOT,
        manifest_paths={"train": MANIFESTS_DIR / "train_manifest.parquet", "val": MANIFESTS_DIR / "val_manifest.parquet"},
        output_dir=PREPARED_DATA_DIR,
    )
    for split in prep_report.splits:
        print(f"{split.split_name}: {split.sample_count} samples prepared")


def do_validate_environment() -> None:
    print("\n-- Validate Loghi training environment --")
    validation = LoghiAdapter().validate_environment()
    print(f"valid={validation.valid}")
    for m in validation.messages:
        print(f"  - {m}")


def do_smoke_test() -> None:
    print("\n-- Run training smoke test --")
    probe_train = PREPARED_DATA_DIR / "probe_train_list.txt"
    probe_val = PREPARED_DATA_DIR / "probe_val_list.txt"
    if not probe_train.exists():
        train_list = PREPARED_DATA_DIR / "train_list.txt"
        val_list = PREPARED_DATA_DIR / "val_list.txt"
        if not train_list.exists():
            print("Run option 3 first (no prepared training data).")
            return
        probe_train.write_text("\n".join(train_list.read_text(encoding="utf-8").splitlines()[:20]), encoding="utf-8")
        probe_val.write_text("\n".join(val_list.read_text(encoding="utf-8").splitlines()[:5]), encoding="utf-8")

    if not _confirm("Run a real, single-epoch smoke test against the pinned container?"):
        return
    runner = ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16",
        max_image_width=65536, optimizer="adam", learning_rate=0.0001, timeout_seconds=900,
    )
    result = runner.run_epoch(
        existing_model_dir=str(PARENT_CHECKPOINT_DIR),
        output_dir=str(RUN_STATE_DIR / "smoke-test-output"),
        train_list_path=str(probe_train),
        validation_list_path=str(probe_val),
        epoch_seed=1,
    )
    print(f"ok={result.ok} val_cer={result.val_cer} checkpoint_dir={result.checkpoint_dir}")
    if not result.ok:
        print(f"error: {result.error_message}")
        print(result.stderr_tail[-2000:])


def _resolve_training_configuration() -> TrainingConfiguration:
    import hashlib

    summary = load_pilot_split_summary(MANIFESTS_DIR)
    charlist_hash = hashlib.sha256(CHARLIST_PATH.read_bytes()).hexdigest()
    return TrainingConfiguration(
        train_manifest_hash=summary.train_manifest_hash,
        val_manifest_hash=summary.val_manifest_hash,
        charlist_hash=charlist_hash,
        preprocessing_version="byte_identical_from_source_parquet",
        model_architecture="new10",
        parent_checkpoint_hash="0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95",
        learning_rate_policy="constant_0.0001_decay_0.99",
        optimizer="adam",
        augmentation_policy="none",
    )


def do_start_or_resume_session(store: DurableHtrResearchStore) -> None:
    print("\n-- Start or resume a training session --")
    if not (MANIFESTS_DIR / "pilot_split_summary.json").exists() or not (PREPARED_DATA_DIR / "train_list.txt").exists():
        print("Run options 2-3 first.")
        return

    configuration = _resolve_training_configuration()
    identity, configuration_hash = create_or_load_identity(
        run_state_dir=RUN_STATE_DIR, parent_checkpoint="generic-2023-02-15@" + configuration.parent_checkpoint_hash,
        configuration=configuration,
    )
    state = load_session_state(RUN_STATE_DIR)
    session_number = (state.session_count + 1) if state else 1
    restored_epoch = state.cumulative_epoch if state else 0
    restored_step = state.global_step if state else 0
    best_checkpoint = state.best_checkpoint_dir if state else None

    print(f"run_id: {identity.run_id}")
    print(f"session_number: {session_number}")
    print(f"source_checkpoint: {identity.parent_checkpoint}")
    print(f"restored_epoch: {restored_epoch}")
    print(f"restored_global_step: {restored_step}")
    print(f"configuration_hash: {configuration_hash}")
    print(f"current_best_checkpoint: {best_checkpoint}")

    hours_raw = input("How many hours do you want to run for? (blank = ~5 hours): ").strip()
    session_target_seconds = float(hours_raw) * 3600.0 if hours_raw else SESSION_TARGET_SECONDS

    epoch_cap_raw = input("Also cap at N epochs regardless of time (blank = no epoch cap): ").strip()
    max_epochs_this_call = int(epoch_cap_raw) if epoch_cap_raw else None

    patience_raw = input(
        "Stop early after N consecutive epochs with no val_CER improvement (blank = no early stopping): "
    ).strip()
    early_stopping_patience = int(patience_raw) if patience_raw else None

    budget_desc = f"up to {session_target_seconds / 3600.0:.2f} hour(s)"
    if max_epochs_this_call:
        budget_desc += f", capped at {max_epochs_this_call} epoch(s)"
    if early_stopping_patience:
        budget_desc += f", early-stopping patience {early_stopping_patience}"
    if not _confirm(f"Start session {session_number} targeting {budget_desc}?"):
        return

    STOP_SENTINEL_PATH.unlink(missing_ok=True)
    print("A sentinel-file stop is supported: create the file below to request a safe stop after "
          "the current epoch finishes.")
    print(f"  {STOP_SENTINEL_PATH}")

    runner = ContainerEpochRunner(
        batch_size=16, gradient_accumulation=1, precision="mixed_float16",
        max_image_width=65536, optimizer="adam", learning_rate=0.0001, timeout_seconds=7200,
        run_state_dir=RUN_STATE_DIR,
    )

    store.record_training_session_started(
        run_id=identity.run_id, session_id="pending", configuration_hash=configuration_hash,
        initial_epoch=restored_epoch, initial_global_step=restored_step,
        source_checkpoint=identity.parent_checkpoint,
    )

    summary = run_training_session(
        run_state_dir=RUN_STATE_DIR,
        checkpoint_index_path=CHECKPOINT_INDEX_PATH,
        epoch_runner=runner,
        train_list_path=str(PREPARED_DATA_DIR / "train_list.txt"),
        validation_list_path=str(PREPARED_DATA_DIR / "val_list.txt"),
        parent_checkpoint_dir=str(PARENT_CHECKPOINT_DIR),
        run_id=identity.run_id,
        configuration_hash=configuration_hash,
        random_seed=42,
        max_wall_clock_seconds=session_target_seconds,
        stop_requested=lambda: STOP_SENTINEL_PATH.exists(),
        max_epochs_this_call=max_epochs_this_call,
        early_stopping_patience=early_stopping_patience,
    )

    for result in summary.epoch_results:
        store.record_training_session_checkpointed(
            run_id=identity.run_id, session_id=summary.session_id,
            epoch=summary.final_epoch, global_step=summary.final_global_step,
            checkpoint_dir=result.checkpoint_dir or "", checkpoint_kind="latest",
            train_cer=result.train_cer, val_cer=result.val_cer,
            train_wer=result.train_wer, val_wer=result.val_wer,
            duration_seconds=result.duration_seconds,
        )
    store.record_training_session_completed(
        run_id=identity.run_id, session_id=summary.session_id, stop_reason=summary.stop_reason,
        final_epoch=summary.final_epoch, final_global_step=summary.final_global_step,
        session_training_seconds=summary.session_training_seconds,
        cumulative_training_seconds=summary.cumulative_training_seconds,
        latest_checkpoint_dir=summary.latest_checkpoint_dir, best_checkpoint_dir=summary.best_checkpoint_dir,
    )

    print(f"stop_reason: {summary.stop_reason}")
    print(f"epochs_completed_this_session: {summary.epochs_completed_this_session}")
    print(f"final_epoch: {summary.final_epoch}")
    print(f"latest_checkpoint_dir: {summary.latest_checkpoint_dir}")
    print(f"best_checkpoint_dir: {summary.best_checkpoint_dir}")


def do_show_status() -> None:
    print("\n-- Training status --")
    state = load_session_state(RUN_STATE_DIR)
    if state is None:
        print("No session has run yet.")
        return
    print(f"run_id: {state.run_id}")
    print(f"cumulative_epoch: {state.cumulative_epoch}")
    print(f"global_step: {state.global_step}")
    print(f"session_count: {state.session_count}")
    print(f"cumulative_training_seconds: {state.cumulative_training_seconds:.1f}")
    print(f"best_val_cer: {state.best_val_cer}")
    print(f"latest_checkpoint_dir: {state.latest_checkpoint_dir}")
    print(f"best_checkpoint_dir: {state.best_checkpoint_dir}")


def do_validate_latest_checkpoint() -> None:
    print("\n-- Validate latest checkpoint --")
    entry = latest_resumable_checkpoint(CHECKPOINT_INDEX_PATH)
    if entry is None:
        print("No resumable checkpoint recorded yet.")
        return
    ok, model_hash, extra = verify_checkpoint(entry.checkpoint_dir)
    print(f"checkpoint_id: {entry.checkpoint_id}")
    print(f"epoch: {entry.epoch}")
    print(f"reopened_and_readable: {ok}")
    print(f"model_file_hash: {model_hash}")
    print(f"extra: {extra}")


def do_generate_report() -> None:
    print("\n-- Generate progress report --")
    report = generate_session_reports(
        run_state_dir=RUN_STATE_DIR,
        checkpoint_index_path=CHECKPOINT_INDEX_PATH,
        manifests_dir=MANIFESTS_DIR,
        reports_dir=REPORTS_DIR,
        dataset_root=DATASET_ROOT,
    )
    print(f"Wrote {REPORTS_DIR / 'session-report.json'} and {REPORTS_DIR / 'session-report.md'}")


MENU = """
Swedish Loghi fine-tuning -- loghi_swedish_finetuned_v1
1. Show dataset discovery status
2. Validate source inventory
3. Prepare or verify pilot manifests
4. Validate Loghi training environment
5. Run training smoke test
6. Start or resume a training session (choose duration)
7. Show training status
8. Validate latest checkpoint
9. Generate progress report
0. Exit
"""


def main() -> int:
    RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sink = FileTelemetrySink(RUN_STATE_DIR / "events.jsonl")
    store = DurableHtrResearchStore(sink)

    while True:
        print(MENU)
        choice = input("> ").strip()
        if choice == "1":
            do_dataset_discovery()
        elif choice == "2":
            do_validate_source_inventory()
        elif choice == "3":
            do_prepare_pilot_manifests()
        elif choice == "4":
            do_validate_environment()
        elif choice == "5":
            do_smoke_test()
        elif choice == "6":
            do_start_or_resume_session(store)
        elif choice == "7":
            do_show_status()
        elif choice == "8":
            do_validate_latest_checkpoint()
        elif choice == "9":
            do_generate_report()
        elif choice == "0":
            print("Exiting.")
            return 0
        else:
            print("Unrecognized choice.")


if __name__ == "__main__":
    raise SystemExit(main())
