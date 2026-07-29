from __future__ import annotations

import json
from pathlib import Path

from archivetrust.admin import qualification
from archivetrust.admin.cli import main as admin_main
from archivetrust.admin.qualification import (
    EvidenceState,
    InvalidationFinding,
    InvalidationReason,
    ProviderSummary,
    QualificationCheckpoint,
    QualificationRunRecord,
    QualificationRunner,
    _atomic_write_json,
    _evidence_contract,
    _failure_injection_summary,
    _generate_reports,
)
from archivetrust.workspace.store import WorkspaceStore


def test_qualification_menu_has_only_three_options(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "3")

    assert admin_main(["qualification"]) == 0

    output = capsys.readouterr().out
    assert "ArchiveTrust Qualification Runner" in output
    assert "1. Start new qualification run" in output
    assert "2. Resume interrupted run" in output
    assert "3. Exit" in output
    assert "4." not in output


def test_start_new_run_creates_durable_run_files_before_execution(tmp_path, monkeypatch) -> None:
    deployment = tmp_path / "deployment"
    store = WorkspaceStore(deployment / "workspaces")
    workspace = store.create("Qualification")
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    first = corpus_dir / "one.txt"
    second = corpus_dir / "two.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"documents": [{"id": "one", "path": str(first)}, {"id": "two", "path": str(second)}]}),
        encoding="utf-8",
    )
    profile = tmp_path / "profile.json"
    profile.write_text('{"id":"test-profile"}', encoding="utf-8")
    config = tmp_path / "qualification_run_config.json"
    output_root = tmp_path / "runs"
    config.write_text(
        json.dumps(
            {
                "workspace": workspace.id,
                "deployment_root": str(deployment),
                "corpus_manifest": str(manifest),
                "run_profile": str(profile),
                "output_root": str(output_root),
                "target_count": 2,
                "stages": [1, 2],
                "providers": "all_available",
                "calibration_eligible": True,
                "dataset_role": "qualification",
                "failure_injection": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification,
        "_resolve_providers",
        lambda *_args, **_kwargs: (
            ProviderSummary(provider_id="docling", provider_version="2.x", available=True),
        ),
    )

    def fake_execute(self, record, checkpoint, corpus, *, resumed):
        run_dir = Path(record.output_path)
        assert (run_dir / "qualification_run.json").exists()
        assert (run_dir / "qualification_events.jsonl").exists()
        assert (run_dir / "checkpoint.json").exists()
        assert checkpoint.pending_document_ids == ("one", "two")
        assert tuple(doc.document_id for doc in corpus) == ("one", "two")
        assert (run_dir / "evidence_contract.json").exists()
        contract = json.loads((run_dir / "evidence_contract.json").read_text(encoding="utf-8"))
        assert contract["schema_id"] == "archivetrust.qualification_evidence_contract.v1"
        assert resumed is False
        return 0

    monkeypatch.setattr(QualificationRunner, "_execute", fake_execute)

    runner = QualificationRunner(config)
    prepared = runner.prepare()
    assert runner.start(prepared) == 0


def test_prepare_refuses_missing_required_provider(tmp_path, monkeypatch) -> None:
    deployment = tmp_path / "deployment"
    store = WorkspaceStore(deployment / "workspaces")
    workspace = store.create("Qualification")
    doc = tmp_path / "one.txt"
    doc.write_text("one", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [{"id": "one", "path": str(doc)}]}), encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text('{"id":"test-profile"}', encoding="utf-8")
    config = tmp_path / "qualification_run_config.json"
    config.write_text(
        json.dumps(
            {
                "workspace": workspace.id,
                "deployment_root": str(deployment),
                "corpus_manifest": str(manifest),
                "run_profile": str(profile),
                "output_root": str(tmp_path / "runs"),
                "target_count": 1,
                "stages": [1],
                "providers": "all_available",
                "required_providers": ["docling", "paddleocr-vl", "surya"],
                "calibration_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        qualification,
        "_resolve_providers",
        lambda *_args, **_kwargs: (
            ProviderSummary(provider_id="docling", provider_version="2.x", available=True),
        ),
    )

    try:
        QualificationRunner(config).prepare()
    except SystemExit as exc:
        assert "paddleocr-vl" in str(exc)
        assert "surya" in str(exc)
    else:
        raise AssertionError("required provider guard did not refuse missing provider")


def test_resume_classifies_stale_running_documents_before_execution(tmp_path, monkeypatch) -> None:
    config = tmp_path / "qualification_run_config.json"
    output_root = tmp_path / "runs"
    run_dir = output_root / "p4_p10_qualification_20260717T000000Z"
    run_dir.mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    doc = tmp_path / "doc.txt"
    doc.write_text("doc", encoding="utf-8")
    manifest.write_text(json.dumps({"documents": [{"id": "doc", "path": str(doc)}]}), encoding="utf-8")
    config.write_text(
        json.dumps(
            {
                "workspace": "unused",
                "corpus_manifest": str(manifest),
                "run_profile": str(tmp_path / "profile.json"),
                "output_root": str(output_root),
                "target_count": 1,
            }
        ),
        encoding="utf-8",
    )
    record = QualificationRunRecord(
        run_id="20260717T000000Z",
        created_at="2026-07-17T00:00:00+00:00",
        git_commit=None,
        dirty_working_tree=False,
        archivetrust_version="0.1.0",
        telemetry_profile_id="archivetrust.telemetry.standard.v1",
        run_profile_id="profile",
        providers=(ProviderSummary(provider_id="docling", provider_version="2.x", available=True),),
        corpus_manifest_digest="digest",
        target_document_count=1,
        stages=(1,),
        workspace_path="workspace",
        workspace_id="workspace-id",
        deployment_root=str(tmp_path / "deployment"),
        output_path=str(run_dir),
        calibration_eligible=True,
        dataset_role="qualification",
        note="qualification evidence, not production approval",
        config_snapshot={"corpus_manifest": str(manifest), "target_count": 1},
    )
    (run_dir / "qualification_run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "qualification_events.jsonl").write_text("", encoding="utf-8")
    checkpoint = QualificationCheckpoint(
        run_id=record.run_id,
        status="running",
        current_stage=1,
        running_document_ids=("archive-object-1",),
        last_checkpoint_time="2026-07-17T00:01:00+00:00",
    )
    (run_dir / "checkpoint.json").write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")

    def fake_execute(self, _record, resumed_checkpoint, corpus, *, resumed):
        assert resumed is True
        assert resumed_checkpoint.campaign_id == "legacy-campaign-not-recorded"
        assert resumed_checkpoint.continuation_id == "20260717T000000Z-resume-001"
        assert resumed_checkpoint.resume_sequence == 1
        assert resumed_checkpoint.running_document_ids == ()
        assert resumed_checkpoint.interrupted_stale_document_ids == ("archive-object-1",)
        resume_report = (run_dir / "resume_report.md").read_text(encoding="utf-8")
        assert "Continuation id: 20260717T000000Z-resume-001" in resume_report
        return 0

    monkeypatch.setattr(QualificationRunner, "_execute", fake_execute)

    assert QualificationRunner(config).resume_latest(confirm=True) == 0


def test_atomic_write_retries_transient_windows_permission_error(tmp_path, monkeypatch) -> None:
    calls = {"count": 0}
    real_replace = qualification.os.replace

    def flaky_replace(source, target):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("transient lock")
        return real_replace(source, target)

    monkeypatch.setattr(qualification.os, "replace", flaky_replace)
    target = tmp_path / "checkpoint.json"

    _atomic_write_json(target, {"ok": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert calls["count"] == 2


def test_qualification_status_prints_progress_summary(tmp_path, capsys) -> None:
    output_root = tmp_path / "runs"
    run_dir = output_root / "p4_p10_qualification_20260717T010000Z"
    run_dir.mkdir(parents=True)
    config = tmp_path / "qualification_run_config.json"
    config.write_text(
        json.dumps(
            {
                "workspace": "unused",
                "corpus_manifest": "unused",
                "run_profile": "unused",
                "output_root": str(output_root),
            }
        ),
        encoding="utf-8",
    )
    record = QualificationRunRecord(
        run_id="20260717T010000Z",
        created_at="2026-07-17T01:00:00+00:00",
        git_commit=None,
        dirty_working_tree=False,
        archivetrust_version="0.1.0",
        telemetry_profile_id="archivetrust.telemetry.standard.v1",
        run_profile_id="profile",
        providers=(ProviderSummary(provider_id="docling", provider_version="2.x", available=True),),
        corpus_manifest_digest="digest",
        target_document_count=500,
        stages=(10, 50, 100, 250, 500),
        workspace_path="workspace",
        workspace_id="workspace-id",
        deployment_root=str(tmp_path / "deployment"),
        output_path=str(run_dir),
        calibration_eligible=True,
        dataset_role="qualification",
        note="qualification evidence, not production approval",
        config_snapshot={"corpus_manifest": "unused", "target_count": 500},
    )
    (run_dir / "qualification_run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "qualification_events.jsonl").write_text("", encoding="utf-8")
    checkpoint = QualificationCheckpoint(
        run_id=record.run_id,
        status="running",
        current_stage=50,
        completed_document_ids=("a", "b"),
        running_document_ids=("c",),
        last_checkpoint_time="2026-07-17T01:05:00+00:00",
    )
    (run_dir / "checkpoint.json").write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")

    assert QualificationRunner(config).print_status() == 0

    output = capsys.readouterr().out
    assert "Qualification run status" in output
    assert "Campaign id: legacy-campaign-not-recorded" in output
    assert "Progress: 2/500 (0.4%)" in output
    assert "Running: 1" in output


def test_evidence_contract_defines_qec1_vocabulary_and_rules() -> None:
    contract = _evidence_contract()

    assert contract.schema_id == "archivetrust.qualification_evidence_contract.v1"
    assert EvidenceState.FAILED.value in contract.evidence_states
    assert EvidenceState.NOT_AVAILABLE.value in contract.evidence_states
    assert InvalidationReason.REDUCED_SCOPE_AFTER_FAILURE.value in contract.invalidation_reasons
    assert "campaign_id" in contract.required_fields
    assert "continuation_id" in contract.required_fields
    assert "must not be reclassified as success" in contract.failed_campaign_rule
    assert "Legacy qualification_run.json" in contract.compatibility_note


def test_legacy_qualification_artifacts_load_with_contract_defaults() -> None:
    record = QualificationRunRecord.model_validate(
        {
            "run_id": "legacy",
            "created_at": "2026-07-17T00:00:00+00:00",
            "git_commit": None,
            "dirty_working_tree": False,
            "archivetrust_version": "0.1.0",
            "telemetry_profile_id": "archivetrust.telemetry.standard.v1",
            "run_profile_id": "profile",
            "providers": [{"provider_id": "docling", "provider_version": "2.x", "available": True}],
            "corpus_manifest_digest": "digest",
            "target_document_count": 1,
            "stages": [1],
            "workspace_path": "workspace",
            "workspace_id": "workspace-id",
            "deployment_root": "deployment",
            "output_path": "output",
            "calibration_eligible": True,
            "dataset_role": "qualification",
            "note": "qualification evidence, not production approval",
            "config_snapshot": {},
        }
    )
    checkpoint = QualificationCheckpoint.model_validate({"run_id": "legacy"})

    assert record.schema_id == "archivetrust.qualification_run.v2"
    assert record.campaign_id == "legacy-campaign-not-recorded"
    assert record.continuation_id is None
    assert checkpoint.campaign_id == "legacy-campaign-not-recorded"
    assert checkpoint.resume_sequence == 0
    assert checkpoint.invalidation_findings == ()


def test_generated_reports_record_evidence_contract_states_and_invalidation(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = QualificationRunRecord(
        run_id="run-1",
        campaign_id="campaign-1",
        created_at="2026-07-17T00:00:00+00:00",
        git_commit=None,
        dirty_working_tree=False,
        archivetrust_version="0.1.0",
        telemetry_profile_id="archivetrust.telemetry.standard.v1",
        run_profile_id="profile",
        providers=(ProviderSummary(provider_id="docling", provider_version="2.x", available=True),),
        corpus_manifest_digest="digest",
        target_document_count=1,
        stages=(1,),
        workspace_path="workspace",
        workspace_id="workspace-id",
        deployment_root=str(tmp_path / "deployment"),
        output_path=str(run_dir),
        calibration_eligible=True,
        dataset_role="qualification",
        note="qualification evidence, not production approval",
        config_snapshot={},
    )
    checkpoint = QualificationCheckpoint(
        run_id="run-1",
        campaign_id="campaign-1",
        continuation_id="run-1-resume-001",
        resume_sequence=1,
        status="interrupted",
        interrupted_stale_document_ids=("doc-1",),
        invalidation_findings=(
            InvalidationFinding(
                reason=InvalidationReason.TELEMETRY_CONTAMINATION,
                detail="test contamination",
                affects_integrity=True,
            ),
        ),
    )
    monkeypatch.setattr(qualification, "_document_results", lambda _record, _checkpoint: [])

    _generate_reports(run_dir, record, checkpoint, status="interrupted")

    summary = json.loads((run_dir / "qualification_summary.json").read_text(encoding="utf-8"))
    assert summary["campaign_id"] == "campaign-1"
    assert summary["continuation_id"] == "run-1-resume-001"
    assert summary["evidence_state"] == EvidenceState.INTERRUPTED.value
    assert summary["invalidation_findings"][0]["invalidates_campaign"] is True
    availability = {row["field_name"]: row for row in summary["evidence_availability"]}
    assert availability["resource_samples"]["state"] == EvidenceState.NOT_EXECUTED.value
    assert availability["replay_summary"]["state"] == EvidenceState.NOT_EXECUTED.value
    assert availability["export_summary"]["state"] == EvidenceState.NOT_EXECUTED.value
    assert availability["artifact_digest_manifest"]["state"] == EvidenceState.IMPLEMENTED.value
    assert availability["human_approval_record"]["state"] == EvidenceState.NOT_EXECUTED.value
    assert (run_dir / "resource_summary.json").exists()
    assert (run_dir / "replay_summary.json").exists()
    assert (run_dir / "export_summary.json").exists()
    assert (run_dir / "failure_injection_summary.json").exists()
    assert (run_dir / "artifact_digest_manifest.json").exists()
    assert (run_dir / "bundle_verification.json").exists()
    assert (run_dir / "human_approval_record.md").exists()
    bundle = json.loads((run_dir / "bundle_verification.json").read_text(encoding="utf-8"))
    assert bundle["ok"] is True
    gaps = (run_dir / "evidence_gaps.md").read_text(encoding="utf-8")
    assert "`resource_samples`: not_executed" in gaps
    assert "`human_approval_record`: not_executed" in gaps


def test_failure_injection_summary_uses_event_types_and_blocks_provider_timeout(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "qualification_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "failure_injection_worker_killed"}),
                json.dumps({"event_type": "failure_injection_worker_cancelled"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    record = QualificationRunRecord(
        run_id="run-1",
        campaign_id="campaign-1",
        created_at="2026-07-17T00:00:00+00:00",
        git_commit=None,
        dirty_working_tree=False,
        archivetrust_version="0.1.0",
        telemetry_profile_id="archivetrust.telemetry.standard.v1",
        run_profile_id="profile",
        providers=(ProviderSummary(provider_id="docling", provider_version="2.x", available=True),),
        corpus_manifest_digest="digest",
        target_document_count=1,
        stages=(1,),
        workspace_path="workspace",
        workspace_id="workspace-id",
        deployment_root=str(tmp_path / "deployment"),
        output_path=str(run_dir),
        calibration_eligible=True,
        dataset_role="qualification",
        note="qualification evidence, not production approval",
        config_snapshot={
            "failure_injection": {
                "enabled": True,
                "worker_kill_after": 1,
                "cancel_after": 1,
                "provider_timeout": 1,
            }
        },
    )
    checkpoint = QualificationCheckpoint(run_id="run-1", campaign_id="campaign-1")

    summary = _failure_injection_summary(run_dir, record, checkpoint)

    assert summary["worker_termination"] == EvidenceState.SIMULATED.value
    assert summary["application_cancel"] == EvidenceState.SIMULATED.value
    assert summary["provider_timeout"] == EvidenceState.BLOCKED.value
