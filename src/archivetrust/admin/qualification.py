"""Crash-resilient qualification runner for staged ArchiveTrust evidence collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from archivetrust.acquisition.manual_import import ManualImportSource
from archivetrust.admin.telemetry_freeze import load_active_telemetry_epoch
from archivetrust.application.progress import FileProcessingProgressSink, ProcessingProgressKind
from archivetrust.composition import AppContext
from archivetrust.infrastructure.storage.integrity import (
    default_hash_chain_sidecar_path,
    verify_hash_chain_sidecar,
)
from archivetrust.replay.distribution import ReplayArchive, ReplayArchiveError
from archivetrust.worker.client import LocalWorkerClient
from archivetrust.worker.models import WorkerCommandSource
from archivetrust.workspace.models import Workspace
from archivetrust.workspace.store import WorkspaceStore

CONFIG_PATH = Path("qualification_run_config.json")
TELEMETRY_PROFILE_ID = "archivetrust.telemetry.standard.v1"
NOTE = "qualification evidence, not production approval"
EVIDENCE_CONTRACT_SCHEMA_ID = "archivetrust.qualification_evidence_contract.v1"


class EvidenceState(StrEnum):
    IMPLEMENTED = "implemented"
    UNIT_TESTED = "unit_tested"
    INTEGRATION_TESTED = "integration_tested"
    SIMULATED = "simulated"
    EXERCISED_ON_COPIED_REAL_DATA = "exercised_on_copied_real_data"
    EXERCISED_ON_REAL_QUALIFICATION_DATA = "exercised_on_real_qualification_data"
    INDEPENDENTLY_VALIDATED = "independently_validated"
    PRODUCTION_QUALIFIED = "production_qualified"
    FAILED = "failed"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"
    PARTIAL = "partial"
    NOT_EXECUTED = "not_executed"
    NOT_AVAILABLE = "not_available"


class InvalidationReason(StrEnum):
    REPOSITORY_CODE_CHANGED = "repository_code_changed"
    DIRTY_WORKING_TREE_CHANGED = "dirty_working_tree_changed"
    QUALIFICATION_RUNNER_CHANGED = "qualification_runner_changed"
    TELEMETRY_SCHEMA_CHANGED = "telemetry_schema_changed"
    CONFIGURATION_CHANGED = "configuration_changed"
    PROVIDER_VERSION_CHANGED = "provider_version_changed"
    MODEL_VERSION_CHANGED = "model_version_changed"
    DEPENDENCY_CHANGED = "dependency_changed"
    CORPUS_CONTENT_CHANGED = "corpus_content_changed"
    CORPUS_SELECTION_CHANGED = "corpus_selection_changed"
    LEGAL_OR_SAMPLING_BASIS_CHANGED = "legal_or_sampling_basis_changed"
    DEPLOYMENT_PROFILE_CHANGED = "deployment_profile_changed"
    WORKER_SUPERVISION_MODEL_CHANGED = "worker_supervision_model_changed"
    THRESHOLD_CHANGED = "threshold_changed"
    EVALUATION_PROTOCOL_CHANGED = "evaluation_protocol_changed"
    TELEMETRY_CONTAMINATION = "telemetry_contamination"
    STALE_WORKSPACE_STATE = "stale_workspace_state"
    CONFIGURATION_DRIFT = "configuration_drift"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CORPUS_HASH_MISMATCH = "corpus_hash_mismatch"
    LOST_CHECKPOINT_CONTINUITY = "lost_checkpoint_continuity"
    REDUCED_SCOPE_AFTER_FAILURE = "reduced_scope_after_failure"


class EvidenceAvailability(BaseModel):
    model_config = ConfigDict(frozen=True)

    field_name: str
    state: EvidenceState
    reason: str | None = None
    required_for: tuple[str, ...] = ()
    optional_for: tuple[str, ...] = ()


class InvalidationFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: InvalidationReason
    detail: str
    affects_execution: bool = False
    affects_interpretation: bool = False
    affects_comparability: bool = False
    affects_integrity: bool = False
    affects_approval_scope: bool = False

    @property
    def invalidates_campaign(self) -> bool:
        return any(
            (
                self.affects_execution,
                self.affects_interpretation,
                self.affects_comparability,
                self.affects_integrity,
                self.affects_approval_scope,
            )
        )


class QualificationEvidenceContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = EVIDENCE_CONTRACT_SCHEMA_ID
    evidence_states: tuple[str, ...]
    invalidation_reasons: tuple[str, ...]
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    continuation_rule: str
    failed_campaign_rule: str
    compatibility_note: str


class FailureInjectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    worker_kill_after: int | None = None
    cancel_after: int | None = None
    provider_timeout: int | None = None


class QualificationRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace: str
    corpus_manifest: str
    run_profile: str
    output_root: str = "qualification_runs"
    deployment_root: str = "archivetrust_data"
    target_count: int = 500
    stages: tuple[int, ...] = (10, 50, 100, 250, 500)
    providers: str | tuple[str, ...] = "all_available"
    required_providers: tuple[str, ...] = ()
    campaign_id: str | None = None
    calibration_eligible: bool = True
    dataset_role: str = "qualification"
    failure_injection: FailureInjectionConfig = Field(default_factory=FailureInjectionConfig)


class CorpusDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    path: str
    source_hash: str


class ProviderSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    available: bool


class QualificationRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.qualification_run.v2"
    run_id: str
    campaign_id: str = "legacy-campaign-not-recorded"
    continuation_of_run_id: str | None = None
    continuation_id: str | None = None
    created_at: str
    git_commit: str | None
    dirty_working_tree: bool
    archivetrust_version: str
    telemetry_profile_id: str
    run_profile_id: str
    providers: tuple[ProviderSummary, ...]
    corpus_manifest_digest: str
    target_document_count: int
    stages: tuple[int, ...]
    workspace_path: str
    workspace_id: str
    deployment_root: str
    output_path: str
    calibration_eligible: bool
    dataset_role: str
    evidence_contract_schema_id: str = EVIDENCE_CONTRACT_SCHEMA_ID
    note: str
    config_snapshot: dict[str, Any]


class QualificationCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    campaign_id: str = "legacy-campaign-not-recorded"
    continuation_id: str | None = None
    resume_sequence: int = 0
    status: str = "initialized"
    current_stage: int | None = None
    pending_document_ids: tuple[str, ...] = ()
    running_document_ids: tuple[str, ...] = ()
    completed_document_ids: tuple[str, ...] = ()
    failed_document_ids: tuple[str, ...] = ()
    skipped_document_ids: tuple[str, ...] = ()
    interrupted_stale_document_ids: tuple[str, ...] = ()
    imported_source_hashes: tuple[str, ...] = ()
    last_checkpoint_time: str | None = None
    failure_injection_performed: bool = False
    worker_restarts: int = 0
    last_worker_command_id: str | None = None
    invalidation_findings: tuple[InvalidationFinding, ...] = ()
    storage_baseline_bytes: dict[str, int] = Field(default_factory=dict)


class PreparedRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: QualificationRunConfig
    workspace: Workspace
    workspace_path: str
    deployment_root: str
    corpus: tuple[CorpusDocument, ...]
    run_profile: dict[str, Any]
    run_profile_id: str
    providers: tuple[ProviderSummary, ...]
    old_telemetry_safe: bool
    old_telemetry_reason: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="archivetrust-admin qualification")
    parser.add_argument("command", choices=("qualification", "qualification-run", "qualification-status"))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    runner = QualificationRunner(Path(args.config))
    if args.command == "qualification":
        return run_menu(runner)
    if args.command == "qualification-status":
        return runner.print_status()
    if args.resume:
        return runner.resume_latest(confirm=True)
    prepared = runner.prepare()
    runner.print_plan(prepared)
    return runner.start(prepared)


def run_menu(runner: "QualificationRunner") -> int:
    while True:
        print()
        print("ArchiveTrust Qualification Runner")
        print()
        print("1. Start new qualification run")
        print("2. Resume interrupted run")
        print("3. Exit")
        choice = input("> ").strip()
        if choice == "1":
            prepared = runner.prepare()
            runner.print_plan(prepared)
            if input("Start this qualification run? [y/N] ").strip().lower() != "y":
                print("Qualification run not started.")
                return 0
            return runner.start(prepared)
        if choice == "2":
            return runner.resume_latest(confirm=False)
        if choice == "3":
            return 0
        print("Choose 1, 2, or 3.")


class QualificationRunner:
    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path

    def load_config(self) -> QualificationRunConfig:
        if not self.config_path.exists():
            raise SystemExit(f"qualification config not found: {self.config_path}")
        return QualificationRunConfig.model_validate_json(
            self.config_path.read_text(encoding="utf-8-sig")
        )

    def prepare(self) -> PreparedRun:
        config = self.load_config()
        deployment_root, workspace = _resolve_workspace(config)
        store = WorkspaceStore(deployment_root / "workspaces")
        layout = store.layout_for(workspace.id)
        workspace_path = str(layout.root)
        corpus = _load_corpus_manifest(Path(config.corpus_manifest), config.target_count)
        run_profile = _load_json(Path(config.run_profile), "run profile")
        run_profile_id = str(run_profile.get("run_profile_id") or run_profile.get("id") or Path(config.run_profile).stem)
        providers = _resolve_providers(deployment_root, workspace.id, config.providers, config.required_providers)
        _validate_required_providers(config.required_providers, providers)
        old_telemetry_safe, old_telemetry_reason = _old_telemetry_safe(layout, config.calibration_eligible)
        if config.calibration_eligible and not old_telemetry_safe:
            raise SystemExit(old_telemetry_reason)
        if not providers:
            raise SystemExit("no providers resolved from the current provider registry")
        return PreparedRun(
            config=config,
            workspace=workspace,
            workspace_path=workspace_path,
            deployment_root=str(deployment_root),
            corpus=corpus,
            run_profile=run_profile,
            run_profile_id=run_profile_id,
            providers=providers,
            old_telemetry_safe=old_telemetry_safe,
            old_telemetry_reason=old_telemetry_reason,
        )

    def print_plan(self, prepared: PreparedRun) -> None:
        config = prepared.config
        print()
        print("Qualification run plan")
        print()
        print("Workspace:")
        print(f"  {prepared.workspace_path}")
        print()
        print("Corpus manifest:")
        print(f"  {config.corpus_manifest}")
        print()
        print("Run profile:")
        print(f"  {config.run_profile}")
        print()
        print("Stages:")
        print(f"  {' -> '.join(str(stage) for stage in config.stages)}")
        print()
        print("Providers:")
        if config.providers == "all_available":
            print("  all available providers from registry")
        else:
            print(f"  {', '.join(provider.provider_id for provider in prepared.providers)}")
        print()
        print("Output root:")
        print(f"  {config.output_root}")
        print()
        print("Calibration eligible:")
        print(f"  {'yes' if config.calibration_eligible else 'no'}")
        print()
        print("Failure injection:")
        print(f"  {_failure_injection_config_summary(config.failure_injection)}")
        print()

    def start(self, prepared: PreparedRun) -> int:
        run_dir = _new_run_dir(Path(prepared.config.output_root))
        run_dir.mkdir(parents=True, exist_ok=False)
        record = _run_record(prepared, run_dir)
        _atomic_write_json(run_dir / "qualification_run.json", record.model_dump(mode="json"))
        _atomic_write_json(run_dir / "evidence_contract.json", _evidence_contract().model_dump(mode="json"))
        _append_event(run_dir, "qualification_run_created", run_id=record.run_id)
        checkpoint = QualificationCheckpoint(
            run_id=record.run_id,
            campaign_id=record.campaign_id,
            status="running",
            pending_document_ids=tuple(doc.document_id for doc in prepared.corpus),
            last_checkpoint_time=_now(),
        )
        _write_checkpoint(run_dir, checkpoint)
        return self._execute(record, checkpoint, prepared.corpus, resumed=False)

    def resume_latest(self, *, confirm: bool) -> int:
        config = self.load_config()
        incomplete = _incomplete_runs(Path(config.output_root))
        if not incomplete:
            print("No incomplete qualification runs found.")
            return 0
        run_dir = incomplete[0]
        if len(incomplete) > 1 and not confirm:
            print("Incomplete qualification runs:")
            for index, candidate in enumerate(incomplete, start=1):
                checkpoint = _load_checkpoint(candidate)
                print(
                    f"{index}. {candidate.name} stage={checkpoint.current_stage} "
                    f"completed={len(checkpoint.completed_document_ids)} failed={len(checkpoint.failed_document_ids)}"
                )
            choice = input("Resume which run? ").strip()
            try:
                run_dir = incomplete[int(choice) - 1]
            except (ValueError, IndexError):
                print("No run resumed.")
                return 1
        record = QualificationRunRecord.model_validate_json(
            (run_dir / "qualification_run.json").read_text(encoding="utf-8")
        )
        checkpoint = _load_checkpoint(run_dir)
        _print_resume_summary(run_dir, checkpoint)
        if not confirm and input("Resume this run? [y/N] ").strip().lower() != "y":
            print("Qualification run not resumed.")
            return 0
        corpus = _load_corpus_manifest(Path(record.config_snapshot["corpus_manifest"]), record.target_document_count)
        if checkpoint.running_document_ids:
            resume_sequence = checkpoint.resume_sequence + 1
            continuation_id = f"{record.run_id}-resume-{resume_sequence:03d}"
            stale = tuple(
                sorted(set(checkpoint.interrupted_stale_document_ids) | set(checkpoint.running_document_ids))
            )
            checkpoint = checkpoint.model_copy(
                update={
                    "campaign_id": record.campaign_id,
                    "continuation_id": continuation_id,
                    "resume_sequence": resume_sequence,
                    "running_document_ids": (),
                    "interrupted_stale_document_ids": stale,
                    "worker_restarts": checkpoint.worker_restarts + 1,
                    "last_checkpoint_time": _now(),
                }
            )
            _write_checkpoint(run_dir, checkpoint)
            _append_event(
                run_dir,
                "stale_running_documents_detected",
                continuation_id=continuation_id,
                document_ids=stale,
            )
            _write_resume_report(run_dir, checkpoint)
        return self._execute(record, checkpoint, corpus, resumed=True)

    def print_status(self) -> int:
        config = self.load_config()
        run_dir = _latest_run(Path(config.output_root))
        if run_dir is None:
            print("No qualification runs found.")
            return 0
        missing = [
            name
            for name in ("qualification_run.json", "checkpoint.json", "qualification_events.jsonl")
            if not (run_dir / name).exists()
        ]
        if missing:
            print(f"Latest qualification run is incomplete on disk: {run_dir}")
            print(f"Missing: {', '.join(missing)}")
            return 1
        record = QualificationRunRecord.model_validate_json(
            (run_dir / "qualification_run.json").read_text(encoding="utf-8-sig")
        )
        checkpoint = _load_checkpoint(run_dir)
        print(_status_text(run_dir, record, checkpoint))
        return 0

    def _execute(
        self,
        record: QualificationRunRecord,
        checkpoint: QualificationCheckpoint,
        corpus: tuple[CorpusDocument, ...],
        *,
        resumed: bool,
    ) -> int:
        run_dir = Path(record.output_path)
        deployment_root = Path(record.deployment_root)
        store = WorkspaceStore(deployment_root / "workspaces")
        layout = store.layout_for(record.workspace_id)
        context = AppContext(
            deployment_root=deployment_root,
            auto_create_default_workspace=False,
            use_process_worker=False,
        )
        context.open_workspace(record.workspace_id)
        _ensure_vision_providers_bound(context, self.load_config().required_providers)
        manual_source = ManualImportSource(source_id="qualification-runner")
        context.acquisition_manager.add_source(manual_source)
        if not checkpoint.storage_baseline_bytes:
            checkpoint = checkpoint.model_copy(
                update={
                    "storage_baseline_bytes": _storage_snapshot(layout, run_dir),
                    "last_checkpoint_time": _now(),
                }
            )
            _write_checkpoint(run_dir, checkpoint)
        previous_target = 0
        for stage in record.stages:
            if stage <= len(checkpoint.completed_document_ids):
                previous_target = stage
                continue
            stage_docs = corpus[previous_target : min(stage, len(corpus))]
            if not stage_docs:
                previous_target = stage
                continue
            checkpoint = checkpoint.model_copy(
                update={"current_stage": stage, "status": "running", "last_checkpoint_time": _now()}
            )
            _write_checkpoint(run_dir, checkpoint)
            stage_dir = run_dir / f"stage_{stage:03d}"
            stage_dir.mkdir(exist_ok=True)
            _append_event(run_dir, "stage_started", stage=stage, document_count=len(stage_docs))
            imported_hashes = set(checkpoint.imported_source_hashes)
            to_import = [
                Path(doc.path)
                for doc in stage_docs
                if doc.source_hash not in imported_hashes
            ]
            if to_import:
                imported = context.acquisition_manager.import_files(manual_source, tuple(to_import))
                imported_hashes.update(doc.content_hash for doc in imported)
                checkpoint = checkpoint.model_copy(
                    update={
                        "imported_source_hashes": tuple(sorted(imported_hashes)),
                        "pending_document_ids": tuple(
                            doc.document_id
                            for doc in corpus
                            if doc.source_hash not in imported_hashes
                        ),
                        "last_checkpoint_time": _now(),
                    }
                )
                _write_checkpoint(run_dir, checkpoint)
            try:
                checkpoint, terminal_ok = _run_worker_for_stage(
                    run_dir=run_dir,
                    deployment_root=deployment_root,
                    workspace_id=record.workspace_id,
                    layout=layout,
                    checkpoint=checkpoint,
                    target_document_count=record.target_document_count,
                    failure_injection=FailureInjectionConfig.model_validate(
                        record.config_snapshot.get("failure_injection", {})
                    ),
                )
            except KeyboardInterrupt:
                checkpoint = _load_checkpoint(run_dir).model_copy(
                    update={"status": "interrupted", "last_checkpoint_time": _now()}
                )
                _write_checkpoint(run_dir, checkpoint)
                _generate_reports(run_dir, record, checkpoint, status="interrupted")
                print()
                print("Qualification run interrupted safely.")
                print("Resume with: archivetrust-admin qualification")
                return 130
            if not terminal_ok:
                checkpoint = checkpoint.model_copy(update={"status": "interrupted", "last_checkpoint_time": _now()})
                _write_checkpoint(run_dir, checkpoint)
                _generate_reports(run_dir, record, checkpoint, status="interrupted")
                print(f"Qualification run interrupted. Resume with: archivetrust-admin qualification")
                return 130
            if checkpoint.failed_document_ids:
                checkpoint = checkpoint.model_copy(update={"status": "failed", "last_checkpoint_time": _now()})
                _write_checkpoint(run_dir, checkpoint)
                _append_event(run_dir, "stage_failed", stage=stage)
                _generate_reports(run_dir, record, checkpoint, status="failed")
                return 1
            _append_event(run_dir, "stage_completed", stage=stage)
            previous_target = stage
        checkpoint = checkpoint.model_copy(update={"status": "completed", "last_checkpoint_time": _now()})
        _write_checkpoint(run_dir, checkpoint)
        _append_event(run_dir, "qualification_run_completed")
        _generate_reports(run_dir, record, checkpoint, status="completed")
        print(f"Qualification evidence collected in {run_dir}")
        return 0


def _resolve_workspace(config: QualificationRunConfig) -> tuple[Path, Workspace]:
    candidate = Path(config.workspace)
    if candidate.exists() and (candidate / "workspace.json").exists():
        workspace = Workspace.model_validate_json((candidate / "workspace.json").read_text(encoding="utf-8"))
        deployment_root = candidate.parent.parent if candidate.parent.name == "workspaces" else Path(config.deployment_root)
        return deployment_root, workspace
    deployment_root = Path(config.deployment_root)
    store = WorkspaceStore(deployment_root / "workspaces")
    workspace = next((item for item in store.list() if item.id == config.workspace or item.name == config.workspace), None)
    if workspace is None:
        raise SystemExit(f"workspace not found: {config.workspace}")
    return deployment_root, workspace


def _load_corpus_manifest(path: Path, target_count: int) -> tuple[CorpusDocument, ...]:
    if not path.exists():
        raise SystemExit(f"corpus manifest not found: {path}")
    payload = _load_json(path, "corpus manifest")
    items = payload.get("documents") or payload.get("items") or payload.get("files") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise SystemExit("corpus manifest must be a JSON list or contain documents/items/files")
    base = path.parent
    docs: list[CorpusDocument] = []
    for index, item in enumerate(items[:target_count], start=1):
        if isinstance(item, str):
            source = item
            document_id = f"document_{index:04d}"
            expected_hash = None
        elif isinstance(item, dict):
            source = item.get("path") or item.get("source") or item.get("file")
            document_id = str(item.get("document_id") or item.get("id") or f"document_{index:04d}")
            expected_hash = item.get("sha256") or item.get("source_hash")
        else:
            raise SystemExit(f"unsupported corpus manifest item at index {index}")
        if not source:
            raise SystemExit(f"corpus manifest item {index} has no path/source/file")
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = base / source_path
        if not source_path.exists():
            raise SystemExit(f"corpus document not found: {source_path}")
        digest = _sha256_file(source_path)
        if expected_hash and str(expected_hash).lower() != digest:
            raise SystemExit(f"corpus document hash mismatch: {source_path}")
        docs.append(CorpusDocument(document_id=document_id, path=str(source_path), source_hash=digest))
    if len(docs) < target_count:
        raise SystemExit(f"corpus manifest has {len(docs)} documents; target_count is {target_count}")
    return tuple(docs)


def _load_json(path: Path, label: str) -> Any:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{label} is not valid JSON: {error}") from error


def _ensure_vision_providers_bound(context: AppContext, provider_ids: tuple[str, ...]) -> None:
    """Binds whichever vision-capable providers in `provider_ids` aren't registered yet (P4-P10
    Prequalification, 2026-07-20). A `required_providers` list naming two vision-capable providers
    (e.g. `paddleocr-vl` and `surya`) cannot be satisfied by the desktop's single "Primary Vision
    Provider" wizard binding alone -- the first missing one found here is bound as Primary if no
    Primary binding exists yet, and any other missing ones are bound via
    `bind_additional_vision_provider` (Secondary). Never touches an already-registered provider or
    an existing operator-configured binding.
    """
    from archivetrust.presentation.first_launch_viewmodel import (
        PRIMARY_VISION_PROVIDER,
        default_model_id_for_provider,
    )

    registered_ids = {adapter.provider_id for adapter in context.provider_registry.all()}
    vision_ids_needed = [
        provider_id
        for provider_id in provider_ids
        if provider_id not in registered_ids and default_model_id_for_provider(provider_id) is not None
    ]
    if not vision_ids_needed:
        return
    has_primary_binding = any(
        binding.logical_name == PRIMARY_VISION_PROVIDER for binding in context.model_registry.bindings()
    )
    for provider_id in vision_ids_needed:
        if not has_primary_binding:
            context.bind_additional_vision_provider(provider_id, logical_name=PRIMARY_VISION_PROVIDER)
            has_primary_binding = True
        else:
            context.bind_additional_vision_provider(provider_id)


def _resolve_providers(
    deployment_root: Path,
    workspace_id: str,
    requested: str | tuple[str, ...],
    required: tuple[str, ...] = (),
) -> tuple[ProviderSummary, ...]:
    context = AppContext(
        deployment_root=deployment_root,
        auto_create_default_workspace=False,
        use_process_worker=False,
    )
    context.open_workspace(workspace_id)
    ids_to_bind = set(required) | (set(requested) if requested != "all_available" else set())
    if ids_to_bind:
        _ensure_vision_providers_bound(context, tuple(sorted(ids_to_bind)))
    requested_ids = None if requested == "all_available" else set(requested)
    rows = []
    for adapter in context.provider_registry.all():
        if requested_ids is not None and adapter.provider_id not in requested_ids:
            continue
        available = context.provider_availability.get(adapter.provider_id, False)
        if requested == "all_available" and not available:
            continue
        rows.append(
            ProviderSummary(
                provider_id=adapter.provider_id,
                provider_version=str(getattr(adapter, "_provider_version", "not_available")),
                available=available,
            )
        )
    return tuple(rows)


def _validate_required_providers(
    required_provider_ids: tuple[str, ...], providers: tuple[ProviderSummary, ...]
) -> None:
    if not required_provider_ids:
        return
    by_id = {provider.provider_id: provider for provider in providers}
    missing = [provider_id for provider_id in required_provider_ids if provider_id not in by_id]
    unavailable = [
        provider_id
        for provider_id in required_provider_ids
        if provider_id in by_id and not by_id[provider_id].available
    ]
    if missing or unavailable:
        parts = []
        if missing:
            parts.append(f"missing/unregistered required providers: {', '.join(missing)}")
        if unavailable:
            parts.append(f"unavailable required providers: {', '.join(unavailable)}")
        raise SystemExit(
            "qualification run refused: " + "; ".join(parts)
            + ". Start/configure the required provider runtime before running full qualification."
        )


def _evidence_contract() -> QualificationEvidenceContract:
    return QualificationEvidenceContract(
        evidence_states=tuple(state.value for state in EvidenceState),
        invalidation_reasons=tuple(reason.value for reason in InvalidationReason),
        required_fields=(
            "run_id",
            "campaign_id",
            "continuation_id",
            "status",
            "evidence_state",
            "invalidation_findings",
            "repository_commit",
            "dirty_working_tree",
            "configuration",
            "providers",
            "corpus_manifest_digest",
            "document_results",
            "evidence_availability",
        ),
        optional_fields=(
            "resource_samples",
            "replay_summary",
            "export_summary",
            "failure_injection_log",
            "artifact_digest_manifest",
            "human_approval_record",
            "cryptographic_signature",
        ),
        continuation_rule=(
            "A resumed campaign records a continuation_id and resume_sequence. The original "
            "run remains interrupted, failed, partial, or blocked evidence as applicable."
        ),
        failed_campaign_rule=(
            "A failed or reduced-scope campaign must not be reclassified as success. Remediation "
            "starts a new campaign or an explicit continuation with preserved lineage."
        ),
        compatibility_note=(
            "Legacy qualification_run.json and checkpoint.json files without campaign or "
            "continuation fields load with legacy defaults and remain historical evidence."
        ),
    )


def _campaign_id(
    *,
    workspace_id: str,
    run_profile_id: str,
    corpus_manifest_digest: str,
    target_count: int,
    stages: tuple[int, ...],
    required_providers: tuple[str, ...],
) -> str:
    payload = {
        "workspace_id": workspace_id,
        "run_profile_id": run_profile_id,
        "corpus_manifest_digest": corpus_manifest_digest,
        "target_count": target_count,
        "stages": stages,
        "required_providers": required_providers,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"campaign_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _run_evidence_state(status: str) -> EvidenceState:
    mapping = {
        "completed": EvidenceState.EXERCISED_ON_REAL_QUALIFICATION_DATA,
        "failed": EvidenceState.FAILED,
        "interrupted": EvidenceState.INTERRUPTED,
        "blocked": EvidenceState.BLOCKED,
    }
    return mapping.get(status, EvidenceState.PARTIAL)


def _evidence_availability(
    record: QualificationRunRecord | None,
    checkpoint: QualificationCheckpoint | None,
    *,
    status: str,
    artifact_states: dict[str, tuple[EvidenceState, str | None]] | None = None,
) -> tuple[EvidenceAvailability, ...]:
    state = _run_evidence_state(status)
    artifact_states = artifact_states or {}

    def artifact_state(field_name: str, default_reason: str) -> tuple[EvidenceState, str | None]:
        return artifact_states.get(field_name, (EvidenceState.NOT_AVAILABLE, default_reason))

    resource_state, resource_reason = artifact_state(
        "resource_samples",
        "Resource and storage observation was not produced for this run.",
    )
    replay_state, replay_reason = artifact_state(
        "replay_summary",
        "Replay verification was not produced for this run.",
    )
    export_state, export_reason = artifact_state(
        "export_summary",
        "Export verification was not produced for this run.",
    )
    bundle_state, bundle_reason = artifact_state(
        "artifact_digest_manifest",
        "Bundle integrity manifest was not produced for this run.",
    )
    rows = [
        EvidenceAvailability(
            field_name="qualification_run_manifest",
            state=EvidenceState.IMPLEMENTED if record is not None else EvidenceState.NOT_AVAILABLE,
            required_for=("P4", "P5", "P10"),
        ),
        EvidenceAvailability(
            field_name="checkpoint",
            state=EvidenceState.IMPLEMENTED if checkpoint is not None else EvidenceState.NOT_AVAILABLE,
            required_for=("P4", "P5", "P10"),
        ),
        EvidenceAvailability(
            field_name="document_results",
            state=state,
            required_for=("P4", "P5", "P10"),
        ),
        EvidenceAvailability(
            field_name="resource_samples",
            state=resource_state,
            reason=resource_reason,
            required_for=("P5", "P10"),
        ),
        EvidenceAvailability(
            field_name="replay_summary",
            state=replay_state,
            reason=replay_reason,
            required_for=("P5", "P7", "P10"),
        ),
        EvidenceAvailability(
            field_name="export_summary",
            state=export_state,
            reason=export_reason,
            required_for=("P5", "P7", "P10"),
        ),
        EvidenceAvailability(
            field_name="failure_injection",
            state=(
                EvidenceState.IMPLEMENTED
                if checkpoint is not None and checkpoint.failure_injection_performed
                else EvidenceState.NOT_EXECUTED
            ),
            reason=(
                None
                if checkpoint is not None and checkpoint.failure_injection_performed
                else "Configured trigger was not reached or injection was not requested."
            ),
            required_for=("P4", "P10"),
        ),
        EvidenceAvailability(
            field_name="artifact_digest_manifest",
            state=bundle_state,
            reason=bundle_reason,
            required_for=("P4", "P5", "P10"),
        ),
        EvidenceAvailability(
            field_name="human_approval_record",
            state=EvidenceState.NOT_EXECUTED,
            reason="Human approval is collected after campaign review, not by the runner.",
            required_for=("P4", "P5", "P10"),
        ),
    ]
    return tuple(rows)


def _old_telemetry_safe(layout, calibration_eligible: bool) -> tuple[bool, str]:
    if not calibration_eligible:
        return True, "calibration eligibility disabled"
    epoch = load_active_telemetry_epoch(layout)
    if epoch is not None:
        return True, f"active telemetry epoch present: {epoch.epoch_id}"
    telemetry_files = (layout.telemetry_dir / "events.jsonl", layout.telemetry_dir / "processing.jsonl")
    has_history = any(path.exists() and path.stat().st_size > 0 for path in telemetry_files)
    if has_history:
        return (
            False,
            "calibration-eligible qualification run refused: workspace has existing telemetry but "
            "no active telemetry epoch freeze. Run freeze-telemetry-history first.",
        )
    return True, "workspace telemetry is empty"


def _run_record(prepared: PreparedRun, run_dir: Path) -> QualificationRunRecord:
    corpus_manifest_digest = _sha256_file(Path(prepared.config.corpus_manifest))
    campaign_id = prepared.config.campaign_id or _campaign_id(
        workspace_id=prepared.workspace.id,
        run_profile_id=prepared.run_profile_id,
        corpus_manifest_digest=corpus_manifest_digest,
        target_count=prepared.config.target_count,
        stages=prepared.config.stages,
        required_providers=prepared.config.required_providers,
    )
    return QualificationRunRecord(
        run_id=run_dir.name.replace("p4_p10_qualification_", ""),
        campaign_id=campaign_id,
        created_at=_now(),
        git_commit=_git(["rev-parse", "HEAD"]),
        dirty_working_tree=bool(_git(["status", "--porcelain"])),
        archivetrust_version=_package_version(),
        telemetry_profile_id=TELEMETRY_PROFILE_ID,
        run_profile_id=prepared.run_profile_id,
        providers=prepared.providers,
        corpus_manifest_digest=corpus_manifest_digest,
        target_document_count=prepared.config.target_count,
        stages=prepared.config.stages,
        workspace_path=prepared.workspace_path,
        workspace_id=prepared.workspace.id,
        deployment_root=prepared.deployment_root,
        output_path=str(run_dir),
        calibration_eligible=prepared.config.calibration_eligible,
        dataset_role=prepared.config.dataset_role,
        note=NOTE,
        config_snapshot=prepared.config.model_dump(mode="json"),
    )


def _run_worker_for_stage(
    *,
    run_dir: Path,
    deployment_root: Path,
    workspace_id: str,
    layout,
    checkpoint: QualificationCheckpoint,
    target_document_count: int,
    failure_injection: FailureInjectionConfig,
) -> tuple[QualificationCheckpoint, bool]:
    progress_path = layout.telemetry_dir / "processing.jsonl"
    start_count = len(FileProcessingProgressSink(progress_path).events())
    handle = LocalWorkerClient(deployment_root=deployment_root, workspace_id=workspace_id).start_process_queue(
        source=WorkerCommandSource.QUALIFICATION_RUNNER,
        campaign_id=checkpoint.campaign_id,
        continuation_id=checkpoint.continuation_id,
    )
    checkpoint = checkpoint.model_copy(update={"last_worker_command_id": handle.command.command_id})
    _write_checkpoint(run_dir, checkpoint)
    _append_resource_sample(run_dir, layout, checkpoint, "worker_started")
    killed = False
    last_printed: tuple[int, int, int, int] | None = None
    try:
        while True:
            progress = FileProcessingProgressSink(progress_path).events()[start_count:]
            checkpoint = _checkpoint_from_progress(checkpoint, progress)
            _write_checkpoint(run_dir, checkpoint)
            _append_resource_sample(run_dir, layout, checkpoint, "worker_poll")
            counts = (
                len(checkpoint.completed_document_ids),
                len(checkpoint.failed_document_ids),
                len(checkpoint.skipped_document_ids),
                len(checkpoint.running_document_ids),
            )
            if counts != last_printed:
                processed = counts[0] + counts[1] + counts[2]
                print(
                    f"Progress: stage {checkpoint.current_stage} | "
                    f"processed {processed}/{target_document_count} | "
                    f"completed={counts[0]} failed={counts[1]} "
                    f"skipped={counts[2]} running={counts[3]}",
                    flush=True,
                )
                last_printed = counts
            completed_total = len(checkpoint.completed_document_ids) + len(checkpoint.failed_document_ids)
            if (
                failure_injection.enabled
                and failure_injection.worker_kill_after is not None
                and not checkpoint.failure_injection_performed
                and completed_total >= failure_injection.worker_kill_after
            ):
                handle.process.kill()
                killed = True
                checkpoint = checkpoint.model_copy(
                    update={"failure_injection_performed": True, "last_checkpoint_time": _now()}
                )
                _write_checkpoint(run_dir, checkpoint)
                _append_event(
                    run_dir,
                    "failure_injection_worker_killed",
                    command_id=handle.command.command_id,
                    after_documents=completed_total,
                )
                _append_resource_sample(run_dir, layout, checkpoint, "worker_killed_by_injection")
                break
            if (
                failure_injection.enabled
                and failure_injection.cancel_after is not None
                and not checkpoint.failure_injection_performed
                and completed_total >= failure_injection.cancel_after
            ):
                handle.cancel()
                checkpoint = checkpoint.model_copy(
                    update={"failure_injection_performed": True, "last_checkpoint_time": _now()}
                )
                _write_checkpoint(run_dir, checkpoint)
                _append_event(
                    run_dir,
                    "failure_injection_worker_cancelled",
                    command_id=handle.command.command_id,
                    after_documents=completed_total,
                )
                _append_resource_sample(run_dir, layout, checkpoint, "worker_cancelled_by_injection")
                break
            result = handle.result()
            if result is not None and result.status.value not in {"queued", "running"}:
                progress = FileProcessingProgressSink(progress_path).events()[start_count:]
                checkpoint = _checkpoint_from_progress(checkpoint, progress)
                _write_checkpoint(run_dir, checkpoint)
                _append_resource_sample(run_dir, layout, checkpoint, f"worker_{result.status.value}")
                return checkpoint, result.status.value == "completed"
            if handle.process.poll() is not None and result is None:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        _append_event(run_dir, "qualification_runner_interrupted", command_id=handle.command.command_id)
        handle.cancel()
        raise
    running = checkpoint.running_document_ids
    stale = tuple(sorted(set(checkpoint.interrupted_stale_document_ids) | set(running)))
    checkpoint = checkpoint.model_copy(
        update={
            "running_document_ids": (),
            "interrupted_stale_document_ids": stale,
            "last_checkpoint_time": _now(),
        }
    )
    _write_checkpoint(run_dir, checkpoint)
    _append_event(
        run_dir,
        "worker_process_ended_without_terminal_result",
        command_id=handle.command.command_id,
        killed_by_injection=killed,
        stale_document_ids=stale,
    )
    _append_resource_sample(run_dir, layout, checkpoint, "worker_ended_without_terminal_result")
    return checkpoint, False


def _checkpoint_from_progress(
    checkpoint: QualificationCheckpoint, progress_events: tuple[Any, ...]
) -> QualificationCheckpoint:
    running = set(checkpoint.running_document_ids)
    completed = set(checkpoint.completed_document_ids)
    failed = set(checkpoint.failed_document_ids)
    skipped = set(checkpoint.skipped_document_ids)
    for event in progress_events:
        doc_id = event.archive_object_id
        if not doc_id:
            continue
        if event.kind is ProcessingProgressKind.DOCUMENT_STARTED:
            if doc_id not in completed and doc_id not in failed:
                running.add(doc_id)
        elif event.kind is ProcessingProgressKind.DOCUMENT_COMPLETED:
            running.discard(doc_id)
            if event.had_failure:
                failed.add(doc_id)
            else:
                completed.add(doc_id)
        elif event.kind in {
            ProcessingProgressKind.DOCUMENT_CANCELLED,
            ProcessingProgressKind.DOCUMENT_INTERRUPTED,
        }:
            running.discard(doc_id)
            skipped.add(doc_id)
    return checkpoint.model_copy(
        update={
            "running_document_ids": tuple(sorted(running)),
            "completed_document_ids": tuple(sorted(completed)),
            "failed_document_ids": tuple(sorted(failed)),
            "skipped_document_ids": tuple(sorted(skipped)),
            "last_checkpoint_time": _now(),
        }
    )


def _append_resource_sample(run_dir: Path, layout, checkpoint: QualificationCheckpoint, phase: str) -> None:
    sample = {
        "recorded_at": _now(),
        "phase": phase,
        "current_stage": checkpoint.current_stage,
        "completed_documents": len(checkpoint.completed_document_ids),
        "failed_documents": len(checkpoint.failed_document_ids),
        "skipped_documents": len(checkpoint.skipped_document_ids),
        "running_documents": len(checkpoint.running_document_ids),
        "process_memory_rss_bytes": _process_memory_rss_bytes(),
        "process_cpu_time_seconds": round(time.process_time(), 6),
        "cpu_logical_count": os.cpu_count(),
        "disk_free_bytes": _disk_free_bytes(run_dir),
        "storage_bytes": _storage_snapshot(layout, run_dir),
        "gpu": {
            "state": EvidenceState.NOT_AVAILABLE.value,
            "reason": "No dependency-free GPU/VRAM probe is available in the qualification runner.",
        },
    }
    path = run_dir / "resource_samples.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, sort_keys=True))
        handle.write("\n")


def _resource_summary(run_dir: Path, layout, checkpoint: QualificationCheckpoint) -> dict[str, Any]:
    samples = _load_jsonl(run_dir / "resource_samples.jsonl")
    current = _storage_snapshot(layout, run_dir)
    baseline = checkpoint.storage_baseline_bytes or current
    growth = {
        key: current.get(key, 0) - baseline.get(key, 0)
        for key in sorted(set(current) | set(baseline))
    }
    rss_values = [
        sample.get("process_memory_rss_bytes")
        for sample in samples
        if isinstance(sample.get("process_memory_rss_bytes"), int)
    ]
    summary = {
        "schema_id": "archivetrust.qualification_resource_summary.v1",
        "sample_path": str(run_dir / "resource_samples.jsonl"),
        "sample_count": len(samples),
        "state": EvidenceState.IMPLEMENTED.value if samples else EvidenceState.NOT_EXECUTED.value,
        "storage_baseline_bytes": baseline,
        "storage_current_bytes": current,
        "storage_growth_bytes": growth,
        "peak_process_memory_rss_bytes": max(rss_values) if rss_values else "not_available",
        "cpu_logical_count": os.cpu_count(),
        "disk_free_bytes": _disk_free_bytes(run_dir),
        "gpu_vram": {
            "state": EvidenceState.NOT_AVAILABLE.value,
            "reason": "GPU and VRAM metrics require a host-specific collector outside the core runner.",
        },
    }
    _atomic_write_json(run_dir / "resource_summary.json", summary)
    return summary


def _replay_verification(layout) -> dict[str, Any]:
    try:
        replay = ReplayArchive.open(layout.root)
        summary = replay.summary()
    except ReplayArchiveError as error:
        return {
            "schema_id": "archivetrust.qualification_replay_summary.v1",
            "state": EvidenceState.NOT_EXECUTED.value,
            "status": "not_executed",
            "reason": str(error),
        }
    except Exception as error:  # pragma: no cover - defensive evidence capture
        return {
            "schema_id": "archivetrust.qualification_replay_summary.v1",
            "state": EvidenceState.FAILED.value,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
        }
    payload = summary.model_dump(mode="json")
    payload.update(
        {
            "schema_id": "archivetrust.qualification_replay_summary.v1",
            "state": EvidenceState.IMPLEMENTED.value,
            "status": "passed" if summary.integrity.ok is not False else "failed",
        }
    )
    return payload


def _export_verification(run_dir: Path, record: QualificationRunRecord, layout) -> dict[str, Any]:
    export_dir = run_dir / "verification_exports"
    export_dir.mkdir(exist_ok=True)
    output_path = export_dir / "workspace_export.json"
    try:
        replay = ReplayArchive.open(layout.root)
        replay.write_export_json(
            output_path,
            workspace_id=record.workspace_id,
            workspace_name=record.config_snapshot.get("workspace", record.workspace_id),
            integrity_sidecar=True,
        )
        sidecar_path = default_hash_chain_sidecar_path(output_path)
        sidecar_result = (
            verify_hash_chain_sidecar(output_path, sidecar_path)
            if sidecar_path.exists()
            else None
        )
    except ReplayArchiveError as error:
        return {
            "schema_id": "archivetrust.qualification_export_summary.v1",
            "state": EvidenceState.NOT_EXECUTED.value,
            "status": "not_executed",
            "reason": str(error),
        }
    except Exception as error:  # pragma: no cover - defensive evidence capture
        return {
            "schema_id": "archivetrust.qualification_export_summary.v1",
            "state": EvidenceState.FAILED.value,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
        }
    return {
        "schema_id": "archivetrust.qualification_export_summary.v1",
        "state": EvidenceState.IMPLEMENTED.value,
        "status": "passed" if sidecar_result is None or sidecar_result.ok else "failed",
        "export_path": str(output_path),
        "export_size_bytes": output_path.stat().st_size,
        "integrity_sidecar_path": str(sidecar_path) if sidecar_path.exists() else None,
        "integrity_ok": None if sidecar_result is None else sidecar_result.ok,
        "integrity_reason": None if sidecar_result is None else sidecar_result.reason,
    }


def _failure_injection_summary(
    run_dir: Path, record: QualificationRunRecord, checkpoint: QualificationCheckpoint
) -> dict[str, Any]:
    config = FailureInjectionConfig.model_validate(record.config_snapshot.get("failure_injection", {}))
    events = [
        event for event in _load_jsonl(run_dir / "qualification_events.jsonl")
        if str(event.get("event_type", "")).startswith("failure_injection_")
        or event.get("event_type") == "worker_process_ended_without_terminal_result"
    ]
    event_types = {event.get("event_type") for event in events}
    summary = {
        "schema_id": "archivetrust.qualification_failure_injection_summary.v1",
        "enabled": config.enabled,
        "events": events,
        "worker_termination": (
            EvidenceState.SIMULATED.value
            if "failure_injection_worker_killed" in event_types
            else EvidenceState.NOT_EXECUTED.value
        ),
        "application_cancel": (
            EvidenceState.SIMULATED.value
            if "failure_injection_worker_cancelled" in event_types
            else EvidenceState.NOT_EXECUTED.value
        ),
        "provider_timeout": (
            EvidenceState.NOT_EXECUTED.value
            if config.provider_timeout is None
            else EvidenceState.BLOCKED.value
        ),
        "provider_timeout_reason": (
            "No provider_timeout trigger configured."
            if config.provider_timeout is None
            else "Provider timeout injection needs provider/runtime-specific hooks before it can be exercised."
        ),
    }
    _atomic_write_json(run_dir / "failure_injection_summary.json", summary)
    return summary


def _write_human_approval_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "human_approval_record.md"
    if not path.exists():
        path.write_text(
            """# Human Approval Record

Status: not_approved

This file is a placeholder for independent campaign review. The qualification runner does not
approve production readiness automatically.

- Reviewer:
- Reviewed at:
- Decision:
- Scope:
- Notes:
""",
            encoding="utf-8",
        )
    return {
        "path": str(path),
        "state": EvidenceState.NOT_EXECUTED.value,
        "reason": "Awaiting independent human review and signature.",
    }


def _artifact_state_from_status(payload: dict[str, Any]) -> tuple[EvidenceState, str | None]:
    state = EvidenceState(str(payload.get("state", EvidenceState.NOT_AVAILABLE.value)))
    status = payload.get("status")
    if status in {None, "passed"} and state is not EvidenceState.NOT_EXECUTED:
        return state, None
    return state, str(payload.get("reason") or payload.get("status") or "Evidence not executed.")


def _storage_snapshot(layout, run_dir: Path) -> dict[str, int]:
    return {
        "workspace_root": _directory_size(layout.root),
        "telemetry": _directory_size(layout.telemetry_dir),
        "telemetry_blobs": _directory_size(layout.telemetry_dir / "blobs"),
        "archive": _directory_size(layout.archive_dir),
        "derived": _directory_size(layout.derived_dir),
        "run_output": _directory_size(run_dir),
    }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _disk_free_bytes(path: Path) -> int | str:
    try:
        return shutil.disk_usage(path if path.exists() else path.parent).free
    except OSError:
        return "not_available"


def _process_memory_rss_bytes() -> int | str:
    try:
        import resource  # type: ignore

        usage = resource.getrusage(resource.RUSAGE_SELF)
        value = int(usage.ru_maxrss)
        return value * 1024 if value and os.name == "posix" else value
    except Exception:
        pass
    if os.name == "nt":
        return _windows_process_memory_rss_bytes()
    return "not_available"


def _windows_process_memory_rss_bytes() -> int | str:
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else "not_available"
    except Exception:
        return "not_available"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _generate_reports(
    run_dir: Path, record: QualificationRunRecord, checkpoint: QualificationCheckpoint, *, status: str
) -> None:
    docs = _document_results(record, checkpoint)
    total_completed = len(checkpoint.completed_document_ids)
    total_failed = len(checkpoint.failed_document_ids)
    durations = sorted(d["duration_seconds"] for d in docs if isinstance(d["duration_seconds"], (int, float)))
    layout = WorkspaceStore(Path(record.deployment_root) / "workspaces").layout_for(record.workspace_id)
    resource_summary = _resource_summary(run_dir, layout, checkpoint)
    replay_summary = _replay_verification(layout)
    export_summary = _export_verification(run_dir, record, layout)
    failure_summary = _failure_injection_summary(run_dir, record, checkpoint)
    human_approval = _write_human_approval_record(run_dir)
    _atomic_write_json(run_dir / "replay_summary.json", replay_summary)
    _atomic_write_json(run_dir / "export_summary.json", export_summary)
    artifact_states = {
        "resource_samples": (
            EvidenceState.IMPLEMENTED if resource_summary["sample_count"] else EvidenceState.NOT_EXECUTED,
            None if resource_summary["sample_count"] else "No resource samples were captured in this run.",
        ),
        "replay_summary": _artifact_state_from_status(replay_summary),
        "export_summary": _artifact_state_from_status(export_summary),
        "artifact_digest_manifest": (
            EvidenceState.IMPLEMENTED,
            None,
        ),
    }
    availability = _evidence_availability(
        record,
        checkpoint,
        status=status,
        artifact_states=artifact_states,
    )
    summary = {
        "schema_id": "archivetrust.qualification_summary.v1",
        "run_id": record.run_id,
        "campaign_id": record.campaign_id,
        "continuation_id": checkpoint.continuation_id,
        "resume_sequence": checkpoint.resume_sequence,
        "evidence_contract_schema_id": record.evidence_contract_schema_id,
        "status": status,
        "evidence_state": _run_evidence_state(status).value,
        "invalidation_findings": [
            finding.model_dump(mode="json") | {"invalidates_campaign": finding.invalidates_campaign}
            for finding in checkpoint.invalidation_findings
        ],
        "evidence_availability": [item.model_dump(mode="json") for item in availability],
        "note": NOTE,
        "total_selected": record.target_document_count,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "total_skipped": len(checkpoint.skipped_document_ids),
        "total_interrupted_or_stale": len(checkpoint.interrupted_stale_document_ids),
        "throughput_documents_per_hour": "not_available" if not durations else round(total_completed / (sum(durations) / 3600), 3) if sum(durations) else "not_available",
        "average_document_duration_seconds": "not_available" if not durations else round(sum(durations) / len(durations), 3),
        "p50_duration_seconds": _percentile(durations, 50),
        "p95_duration_seconds": _percentile(durations, 95),
        "p99_duration_seconds": _percentile(durations, 99),
        "telemetry_size_growth": resource_summary["storage_growth_bytes"]["telemetry"],
        "blob_store_growth": resource_summary["storage_growth_bytes"]["telemetry_blobs"],
        "peak_memory": resource_summary["peak_process_memory_rss_bytes"],
        "resource_samples": {
            "path": resource_summary["sample_path"],
            "sample_count": resource_summary["sample_count"],
            "summary_path": str(run_dir / "resource_summary.json"),
        },
        "worker_restarts": checkpoint.worker_restarts,
        "failure_injection_evidence": "implemented" if checkpoint.failure_injection_performed else "not_executed: trigger not reached",
        "failure_injection_summary": failure_summary,
        "replay_pass_fail_summary": replay_summary,
        "export_pass_fail_summary": export_summary,
        "human_approval_record": human_approval,
    }
    _atomic_write_json(run_dir / "qualification_summary.json", summary)
    _atomic_write_json(run_dir / "run_manifest.json", record.model_dump(mode="json"))
    with (run_dir / "document_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(docs[0].keys()) if docs else ["document_id"])
        writer.writeheader()
        writer.writerows(docs)
    with (run_dir / "failed_documents.jsonl").open("w", encoding="utf-8") as handle:
        for row in docs:
            if row["status"] == "failed":
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
    (run_dir / "evidence_gaps.md").write_text(_evidence_gaps_markdown(availability), encoding="utf-8")
    (run_dir / "qualification_summary.md").write_text(
        _summary_markdown(record, checkpoint, summary),
        encoding="utf-8",
    )
    bundle_manifest = _write_artifact_digest_manifest(run_dir)
    _atomic_write_json(run_dir / "bundle_verification.json", _verify_artifact_digest_manifest(run_dir, bundle_manifest))


def _write_artifact_digest_manifest(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_digest_manifest.json", "bundle_verification.json"}
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema_id": "archivetrust.qualification_artifact_digest_manifest.v1",
        "created_at": _now(),
        "root": str(run_dir),
        "artifact_count": len(files),
        "hash_algorithm": "sha256",
        "artifacts": files,
        "state": EvidenceState.IMPLEMENTED.value,
        "note": "Manifest excludes artifact_digest_manifest.json and bundle_verification.json to avoid self-reference.",
    }
    _atomic_write_json(run_dir / "artifact_digest_manifest.json", manifest)
    return manifest


def _verify_artifact_digest_manifest(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    missing = []
    for item in manifest.get("artifacts", []):
        rel_path = item.get("path")
        expected = item.get("sha256")
        if not rel_path or not expected:
            continue
        path = run_dir / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        actual = _sha256_file(path)
        if actual != expected:
            mismatches.append({"path": rel_path, "expected": expected, "actual": actual})
    ok = not missing and not mismatches
    return {
        "schema_id": "archivetrust.qualification_bundle_verification.v1",
        "verified_at": _now(),
        "manifest_path": str(run_dir / "artifact_digest_manifest.json"),
        "artifact_count": len(manifest.get("artifacts", [])),
        "ok": ok,
        "missing": missing,
        "mismatches": mismatches,
        "state": EvidenceState.IMPLEMENTED.value if ok else EvidenceState.FAILED.value,
    }


def _document_results(record: QualificationRunRecord, checkpoint: QualificationCheckpoint) -> list[dict[str, Any]]:
    layout = WorkspaceStore(Path(record.deployment_root) / "workspaces").layout_for(record.workspace_id)
    progress = FileProcessingProgressSink(layout.telemetry_dir / "processing.jsonl").events()
    domain_counts = _domain_counts(layout.telemetry_dir / "events.jsonl")
    source_hashes = _source_hashes(layout.telemetry_dir / "acquisition.jsonl")
    rows = []
    for doc_id in sorted(
        set(checkpoint.completed_document_ids)
        | set(checkpoint.failed_document_ids)
        | set(checkpoint.interrupted_stale_document_ids)
        | set(checkpoint.skipped_document_ids)
    ):
        events = [event for event in progress if event.archive_object_id == doc_id]
        started = next((event for event in events if event.kind is ProcessingProgressKind.DOCUMENT_STARTED), None)
        completed = next((event for event in reversed(events) if event.kind is ProcessingProgressKind.DOCUMENT_COMPLETED), None)
        status = (
            "completed"
            if doc_id in checkpoint.completed_document_ids
            else "failed"
            if doc_id in checkpoint.failed_document_ids
            else "interrupted_stale"
            if doc_id in checkpoint.interrupted_stale_document_ids
            else "skipped"
        )
        counts = domain_counts.get(doc_id, {})
        rows.append(
            {
                "document_id": doc_id,
                "source_hash": source_hashes.get(doc_id, "not_available"),
                "status": status,
                "start_time": started.recorded_at if started else "not_available",
                "end_time": completed.recorded_at if completed else "not_available",
                "duration_seconds": completed.duration_seconds if completed and completed.duration_seconds is not None else "not_available",
                "provider_attempts": counts.get("ProviderObservationAttempted", 0),
                "provider_success_count": counts.get("provider_success_count", "not_available"),
                "provider_failure_count": counts.get("provider_failure_count", "not_available"),
                "provider_no_observation_count": counts.get("provider_no_observation_count", "not_available"),
                "evidence_count": counts.get("EvidenceCreated", 0),
                "observation_count": counts.get("ObservationCreated", 0),
                "alignment_count": counts.get("AlignmentAttempted", 0),
                "comparison_count": counts.get("ObservationCompared", 0),
                "canonical_decision_count": counts.get("CanonicalDecisionCreated", 0),
                "canonical_document_created": counts.get("CanonicalDocumentCreated", 0) > 0,
                "replay_status": "not_available",
                "integrity_status": _integrity_status(layout),
                "error_classification": completed.failure_disposition.value if completed and completed.failure_disposition else "not_available",
            }
        )
    return rows


def _source_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not path.exists():
        return hashes
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if payload.get("kind") != "ArchiveObjectRegistered":
                continue
            archive_object = payload.get("archive_object")
            if isinstance(archive_object, dict):
                object_id = archive_object.get("id")
                content_hash = archive_object.get("content_hash")
                if object_id and content_hash:
                    hashes[str(object_id)] = str(content_hash)
    return hashes


def _domain_counts(path: Path) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            doc = payload.get("document_ref")
            kind = payload.get("kind")
            if not doc or not kind:
                continue
            row = counts.setdefault(doc, {})
            row[kind] = row.get(kind, 0) + 1
            if kind == "ProviderObservationAttempted":
                outcome = payload.get("outcome")
                if outcome == "produced_observations":
                    row["provider_success_count"] = row.get("provider_success_count", 0) + 1
                elif outcome == "failed":
                    row["provider_failure_count"] = row.get("provider_failure_count", 0) + 1
                elif outcome == "no_observations":
                    row["provider_no_observation_count"] = row.get("provider_no_observation_count", 0) + 1
    return counts


def _integrity_status(layout) -> str:
    event_path = layout.telemetry_dir / "events.jsonl"
    sidecar = default_hash_chain_sidecar_path(event_path)
    if not sidecar.exists():
        return "not_available"
    result = verify_hash_chain_sidecar(event_path, sidecar)
    return "ok" if result.ok else f"failed: {result.reason}"


def _summary_markdown(
    record: QualificationRunRecord, checkpoint: QualificationCheckpoint, summary: dict[str, Any]
) -> str:
    return f"""# ArchiveTrust Qualification Summary

Status: {summary["status"]}

This report records qualification evidence only. It does not declare ArchiveTrust production-ready.

## Implemented

- Staged runner: {' -> '.join(str(stage) for stage in record.stages)}
- Durable run directory, checkpoint, event log, manifest, summary, document CSV, failed JSONL
- Existing worker process orchestration
- Stale running document classification on resume

## Tested

- Completed documents recorded: {summary["total_completed"]}
- Failed documents recorded: {summary["total_failed"]}
- Interrupted/stale documents recorded: {summary["total_interrupted_or_stale"]}

## Simulated

- None automatically claimed.

## Failed

- Run status: {summary["status"]}
- Evidence state: {summary["evidence_state"]}
- Campaign id: {summary["campaign_id"]}
- Continuation id: {summary["continuation_id"] or "not_applicable"}
- Resume sequence: {summary["resume_sequence"]}

## Not Tested

- Independent UX/accessibility qualification
- Clean-machine disaster recovery inside this runner

## Not Available

- Peak memory: {summary["peak_memory"]}
- Resource samples: {summary["resource_samples"]}
- Replay summary: {summary["replay_pass_fail_summary"]}
- Export summary: {summary["export_pass_fail_summary"]}

## Checkpoint

- Current stage: {checkpoint.current_stage}
- Last checkpoint: {checkpoint.last_checkpoint_time}
- Worker restarts: {checkpoint.worker_restarts}
"""


def _evidence_gaps_markdown(availability: tuple[EvidenceAvailability, ...] | None = None) -> str:
    availability = availability or _evidence_availability(None, None, status="not_executed")
    rows = "\n".join(
        f"- `{item.field_name}`: {item.state.value}"
        + (f" - {item.reason}" if item.reason else "")
        for item in availability
    )
    return f"""# Evidence Gaps

{rows}

`not_available` means current tooling cannot produce the evidence yet. `not_executed` means the
exercise or observation was planned or required but not run in this campaign.

- The runner collects evidence and never upgrades the production-readiness verdict automatically.
"""


def _write_resume_report(run_dir: Path, checkpoint: QualificationCheckpoint) -> None:
    (run_dir / "resume_report.md").write_text(
        f"""# Resume Report

Run id: {checkpoint.run_id}
Campaign id: {checkpoint.campaign_id}
Continuation id: {checkpoint.continuation_id or "not_available"}
Resume sequence: {checkpoint.resume_sequence}
Resumed at: {_now()}
Interrupted/stale documents: {len(checkpoint.interrupted_stale_document_ids)}
Last checkpoint before resume: {checkpoint.last_checkpoint_time}
""",
        encoding="utf-8",
    )


def _print_resume_summary(run_dir: Path, checkpoint: QualificationCheckpoint) -> None:
    print()


def _status_text(
    run_dir: Path, record: QualificationRunRecord, checkpoint: QualificationCheckpoint
) -> str:
    completed = len(checkpoint.completed_document_ids)
    failed = len(checkpoint.failed_document_ids)
    skipped = len(checkpoint.skipped_document_ids)
    stale = len(checkpoint.interrupted_stale_document_ids)
    running = len(checkpoint.running_document_ids)
    done = completed + failed + skipped
    total = record.target_document_count
    percent = (done / total * 100) if total else 0
    worker_status = "not_available"
    if checkpoint.last_worker_command_id:
        result_path = (
            Path(record.deployment_root)
            / "worker"
            / "results"
            / f"{checkpoint.last_worker_command_id}.json"
        )
        if result_path.exists():
            try:
                worker_status = json.loads(result_path.read_text(encoding="utf-8-sig")).get(
                    "status", "not_available"
                )
            except ValueError:
                worker_status = "unreadable"
    stage = checkpoint.current_stage if checkpoint.current_stage is not None else "not_started"
    return "\n".join(
        (
            "Qualification run status",
            "",
            f"Run id: {record.run_id}",
            f"Campaign id: {record.campaign_id}",
            f"Continuation id: {checkpoint.continuation_id or 'not_applicable'}",
            f"Status: {checkpoint.status}",
            f"Stage: {stage}",
            f"Progress: {done}/{total} ({percent:.1f}%)",
            f"Completed: {completed}",
            f"Failed: {failed}",
            f"Skipped/cancelled: {skipped}",
            f"Interrupted/stale: {stale}",
            f"Running: {running}",
            f"Worker status: {worker_status}",
            f"Last checkpoint: {checkpoint.last_checkpoint_time}",
            f"Output: {run_dir}",
        )
    )
    print("Resume candidate")
    print(f"  run id: {checkpoint.run_id}")
    print(f"  path: {run_dir}")
    print(f"  stage: {checkpoint.current_stage}")
    print(f"  completed: {len(checkpoint.completed_document_ids)}")
    print(f"  failed: {len(checkpoint.failed_document_ids)}")
    print(f"  pending: {len(checkpoint.pending_document_ids)}")
    print(f"  interrupted/stale: {len(checkpoint.interrupted_stale_document_ids) + len(checkpoint.running_document_ids)}")
    print(f"  last checkpoint: {checkpoint.last_checkpoint_time}")
    print()


def _new_run_dir(output_root: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"p4_p10_qualification_{run_id}"


def _latest_run(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    candidates = sorted(
        output_root.glob("p4_p10_qualification_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _incomplete_runs(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    candidates = []
    for path in sorted(output_root.glob("p4_p10_qualification_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not all((path / name).exists() for name in ("qualification_run.json", "checkpoint.json", "qualification_events.jsonl")):
            continue
        checkpoint = _load_checkpoint(path)
        if checkpoint.status not in {"completed", "failed"}:
            candidates.append(path)
    return candidates


def _load_checkpoint(run_dir: Path) -> QualificationCheckpoint:
    return QualificationCheckpoint.model_validate_json((run_dir / "checkpoint.json").read_text(encoding="utf-8-sig"))


def _write_checkpoint(run_dir: Path, checkpoint: QualificationCheckpoint) -> None:
    checkpoint = checkpoint.model_copy(update={"last_checkpoint_time": _now()})
    _atomic_write_json(run_dir / "checkpoint.json", checkpoint.model_dump(mode="json"))


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, target: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(12):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _append_event(run_dir: Path, event_type: str, **fields: Any) -> None:
    path = run_dir / "qualification_events.jsonl"
    payload = {"event_type": event_type, "recorded_at": _now(), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("archivetrust")
    except Exception:
        return "not_available"


def _failure_injection_config_summary(config: FailureInjectionConfig) -> str:
    if not config.enabled:
        return "disabled"
    if config.worker_kill_after is not None:
        return f"worker kill after {config.worker_kill_after} documents"
    if config.cancel_after is not None:
        return f"cancel after {config.cancel_after} documents"
    if config.provider_timeout is not None:
        return "provider timeout injection not_available"
    return "enabled, no configured trigger"


def _percentile(values: list[float], percentile: int) -> float | str:
    if not values:
        return "not_available"
    index = min(len(values) - 1, round((percentile / 100) * (len(values) - 1)))
    return round(values[index], 3)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
