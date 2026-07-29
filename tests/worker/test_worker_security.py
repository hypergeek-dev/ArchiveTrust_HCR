from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from archivetrust.admin.identity import AuthenticatedSession, IdentityRole, LocalAccount, LocalAccountStore
from archivetrust.admin.identity import PermissionDenied
from archivetrust.worker.client import LocalWorkerClient
from archivetrust.worker.models import (
    WorkerCommand,
    WorkerCommandResult,
    WorkerCommandSource,
    WorkerCommandStatus,
    WorkerControlAction,
    WorkerRejectionReason,
)
from archivetrust.worker.security import (
    CommandFreshnessStore,
    DeploymentIdentityProvider,
    FilesystemSecurityInspector,
    WorkerCommandAuthorizer,
    WorkerControlVerifier,
    WorkerResultVerifier,
    WorkerSecurityError,
    command_digest,
    write_json_atomic,
)
from archivetrust.worker.service import execute_command
from archivetrust.workspace.store import WorkspaceStore


def _deployment(tmp_path):
    deployment = tmp_path / "deployment"
    workspace = WorkspaceStore(deployment / "workspaces").create("Worker security")
    DeploymentIdentityProvider(deployment).ensure()
    return deployment, workspace


def _account(deployment, *, roles=(IdentityRole.OPERATOR,)):
    store = LocalAccountStore(deployment / "identity" / "accounts.json")
    account = LocalAccount.with_password(
        password="Municipal-Archive-2026",
        display_name="Olivia Operator",
        reviewer_ref="reviewer.olivia",
        roles=roles,
    )
    store.add(account)
    return store, account, AuthenticatedSession.start(account)


def _write_command(deployment, command: WorkerCommand) -> None:
    write_json_atomic(
        deployment / "worker" / "commands" / f"{command.command_id}.json",
        command,
    )


def test_valid_authorized_command_executes_and_records_verification_context(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    _store, _account_obj, session = _account(deployment)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.DESKTOP,
        session=session,
    )
    _write_command(deployment, command)

    result = execute_command(deployment_root=deployment, command_id=command.command_id)

    assert result.status is WorkerCommandStatus.COMPLETED
    assert result.command_digest == command_digest(command)
    assert result.submitted_by_account_id == session.account.account_id
    assert result.authorization_mechanism == "hmac_sha256_deployment_key"
    state = CommandFreshnessStore.for_deployment(deployment).load()
    assert command.command_id in state.completed_command_ids


def test_modified_command_payload_is_rejected(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.TEST,
    )
    tampered = command.model_copy(update={"workspace_id": "other-workspace"})
    _write_command(deployment, tampered)

    result = execute_command(deployment_root=deployment, command_id=command.command_id)

    assert result.status is WorkerCommandStatus.REJECTED
    assert result.rejection_reason is WorkerRejectionReason.MISSING_OR_INVALID_AUTHENTICATION


def test_command_copied_to_another_deployment_is_rejected(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path / "one")
    other, _other_workspace = _deployment(tmp_path / "two")
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.TEST,
    )
    _write_command(other, command)

    result = execute_command(deployment_root=other, command_id=command.command_id)

    assert result.status is WorkerCommandStatus.REJECTED
    assert result.rejection_reason is WorkerRejectionReason.WRONG_DEPLOYMENT


def test_expired_command_is_rejected(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.TEST,
        ttl=timedelta(seconds=-1),
    )
    _write_command(deployment, command)

    result = execute_command(deployment_root=deployment, command_id=command.command_id)

    assert result.status is WorkerCommandStatus.REJECTED
    assert result.rejection_reason is WorkerRejectionReason.EXPIRED_COMMAND


def test_expired_session_cannot_issue_worker_command(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    _store, account, session = _account(deployment)
    expired = session.model_copy(
        update={"expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    )

    with pytest.raises(PermissionDenied):
        WorkerCommandAuthorizer(deployment).create_command(
            workspace_id=workspace.id,
            source=WorkerCommandSource.DESKTOP,
            session=expired,
        )


def test_completed_command_replay_is_rejected(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.TEST,
    )
    _write_command(deployment, command)
    assert execute_command(deployment_root=deployment, command_id=command.command_id).status is WorkerCommandStatus.COMPLETED

    replay = execute_command(deployment_root=deployment, command_id=command.command_id)

    assert replay.status is WorkerCommandStatus.REJECTED
    assert replay.rejection_reason is WorkerRejectionReason.REPLAYED_COMMAND


def test_duplicate_active_command_is_rejected(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.TEST,
    )
    CommandFreshnessStore.for_deployment(deployment).reserve_command(command)
    _write_command(deployment, command)

    result = execute_command(deployment_root=deployment, command_id=command.command_id)

    assert result.status is WorkerCommandStatus.REJECTED
    assert result.rejection_reason is WorkerRejectionReason.DUPLICATE_ACTIVE_COMMAND


def test_disabled_account_and_insufficient_permission_are_rejected(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    store, account, session = _account(deployment)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.DESKTOP,
        session=session,
    )
    store.replace(account.model_copy(update={"active": False}))
    _write_command(deployment, command)

    disabled = execute_command(deployment_root=deployment, command_id=command.command_id)
    assert disabled.rejection_reason is WorkerRejectionReason.DISABLED_OR_UNKNOWN_ACCOUNT

    deployment2, workspace2 = _deployment(tmp_path / "second")
    store2, account2, session2 = _account(deployment2)
    command2 = WorkerCommandAuthorizer(deployment2).create_command(
        workspace_id=workspace2.id,
        source=WorkerCommandSource.DESKTOP,
        session=session2,
    )
    store2.replace(account2.model_copy(update={"roles": (IdentityRole.REVIEWER,)}))
    _write_command(deployment2, command2)

    denied = execute_command(deployment_root=deployment2, command_id=command2.command_id)
    assert denied.rejection_reason is WorkerRejectionReason.INSUFFICIENT_PERMISSION


def test_configuration_drift_rejects_command(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.TEST,
    )
    (deployment / "workspaces" / workspace.id / "config" / "changed.json").write_text("{}", encoding="utf-8")
    _write_command(deployment, command)

    result = execute_command(deployment_root=deployment, command_id=command.command_id)

    assert result.rejection_reason is WorkerRejectionReason.CONFIGURATION_DRIFT


def test_valid_invalid_and_reused_control_requests(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    authorizer = WorkerCommandAuthorizer(deployment)
    command = authorizer.create_command(workspace_id=workspace.id, source=WorkerCommandSource.TEST)
    verifier = WorkerControlVerifier(deployment)
    request = authorizer.create_control(command_id=command.command_id, action=WorkerControlAction.CANCEL)

    assert verifier.verify(request, command_id=command.command_id).request.action is WorkerControlAction.CANCEL
    with pytest.raises(WorkerSecurityError) as reused:
        verifier.verify(request, command_id=command.command_id)
    assert reused.value.reason is WorkerRejectionReason.INVALID_CONTROL_REQUEST

    other = authorizer.create_control(command_id=command.command_id, action=WorkerControlAction.PAUSE)
    with pytest.raises(WorkerSecurityError) as wrong:
        verifier.verify(other, command_id="another-command")
    assert wrong.value.reason is WorkerRejectionReason.INVALID_CONTROL_REQUEST


def test_forged_success_result_is_not_trusted(tmp_path) -> None:
    deployment, workspace = _deployment(tmp_path)
    command = WorkerCommandAuthorizer(deployment).create_command(
        workspace_id=workspace.id,
        source=WorkerCommandSource.TEST,
    )
    result = {
        "command_id": command.command_id,
        "workspace_id": workspace.id,
        "status": "completed",
        "recorded_at": "2026-07-19T00:00:00+00:00",
    }

    with pytest.raises(WorkerSecurityError) as error:
        WorkerResultVerifier().verify(WorkerCommandResult.model_validate(result), command=command)
    assert error.value.reason is WorkerRejectionReason.RESULT_INTEGRITY_FAILURE


def test_filesystem_security_and_diagnostics_redact_secret_material(tmp_path) -> None:
    from archivetrust.admin.health import create_diagnostic_bundle, deployment_health
    import zipfile

    deployment, _workspace = _deployment(tmp_path)
    health = deployment_health(deployment)
    assert any(check.name == "worker_command_security_mode" for check in health.checks)
    assert FilesystemSecurityInspector().inspect(deployment).schema_id == "archivetrust.worker_filesystem_security.v1"

    bundle = tmp_path / "diagnostics.zip"
    create_diagnostic_bundle(deployment, bundle)
    with zipfile.ZipFile(bundle) as archive:
        assert "worker-security.json" in archive.namelist()
        combined = b"".join(archive.read(name) for name in archive.namelist())
        identity = json.loads((deployment / "identity" / "deployment_identity.json").read_text(encoding="utf-8"))
        assert identity["hmac_key_hex"].encode("utf-8") not in combined


def test_qualification_source_command_uses_same_worker_verification_path(tmp_path, monkeypatch) -> None:
    deployment, workspace = _deployment(tmp_path)
    launched = {}

    class _Process:
        pass

    def fake_popen(arguments, **kwargs):
        launched["arguments"] = arguments
        return _Process()

    monkeypatch.setattr("archivetrust.worker.client.subprocess.Popen", fake_popen)
    handle = LocalWorkerClient(deployment_root=deployment, workspace_id=workspace.id).start_process_queue(
        source=WorkerCommandSource.QUALIFICATION_RUNNER,
        campaign_id="campaign-1",
        continuation_id="continuation-1",
    )

    assert handle.command.command_source is WorkerCommandSource.QUALIFICATION_RUNNER
    assert handle.command.qualification_campaign_id == "campaign-1"
    result = execute_command(deployment_root=deployment, command_id=handle.command.command_id)
    assert result.status is WorkerCommandStatus.COMPLETED
