from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from archivetrust.admin.configuration import ConfigurationDriftError, verify_configuration_lock, write_configuration_lock
from archivetrust.admin.identity import IdentityRole, LocalAccountStore, Permission, PermissionDenied, require_permission
from archivetrust.domain.shared.ids import new_id
from archivetrust.worker.models import (
    WorkerCommand,
    WorkerCommandResult,
    WorkerCommandSource,
    WorkerCommandStatus,
    WorkerControlAction,
    WorkerControlRequest,
    WorkerRejectionReason,
    VerifiedWorkerCommand,
    VerifiedWorkerControl,
    canonical_json,
    utc_now,
)


COMMAND_SCHEMA_ID = "archivetrust.worker_command.v2"
CONTROL_SCHEMA_ID = "archivetrust.worker_control_request.v1"
RESULT_SCHEMA_ID = "archivetrust.worker_command_result.v2"
AUTH_MECHANISM = "hmac_sha256_deployment_key"
AUTH_VERSION = "v1"
DEFAULT_COMMAND_TTL = timedelta(minutes=15)
DEFAULT_CONTROL_TTL = timedelta(minutes=5)


class WorkerSecurityError(PermissionError):
    def __init__(self, reason: WorkerRejectionReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class DeploymentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.deployment_identity.v1"
    deployment_id: str = Field(default_factory=lambda: new_id("deployment"))
    hmac_key_hex: str = Field(default_factory=lambda: secrets.token_hex(32), repr=False)
    created_at: str = Field(default_factory=utc_now)


class FreshnessState(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.worker_freshness_state.v1"
    used_command_nonces: tuple[str, ...] = ()
    used_control_nonces: tuple[str, ...] = ()
    active_command_ids: tuple[str, ...] = ()
    completed_command_ids: tuple[str, ...] = ()
    rejected_command_ids: tuple[str, ...] = ()
    last_successful_verification: dict[str, Any] | None = None
    replay_attempts: int = 0
    rejected_commands: int = 0


class FilesystemSecurityFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    status: str
    detail: str


class FilesystemSecurityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.worker_filesystem_security.v1"
    checked_at: str
    platform: str
    mode: str
    processing_blocked: bool
    findings: tuple[FilesystemSecurityFinding, ...]


class DeploymentIdentityProvider:
    def __init__(self, deployment_root: Path) -> None:
        self.deployment_root = deployment_root
        self.path = deployment_root / "identity" / "deployment_identity.json"

    def ensure(self) -> DeploymentIdentity:
        if self.path.exists():
            return self.load()
        identity = DeploymentIdentity()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, identity.model_dump_json(indent=2))
        return identity

    def load(self) -> DeploymentIdentity:
        if not self.path.exists():
            raise WorkerSecurityError(
                WorkerRejectionReason.WRONG_DEPLOYMENT,
                "deployment identity is missing",
            )
        return DeploymentIdentity.model_validate_json(self.path.read_text(encoding="utf-8"))


class CommandFreshnessStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_deployment(cls, deployment_root: Path) -> "CommandFreshnessStore":
        return cls(deployment_root / "worker" / "freshness_state.json")

    def load(self) -> FreshnessState:
        if not self.path.exists():
            return FreshnessState()
        return FreshnessState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def reserve_command(self, command: WorkerCommand) -> FreshnessState:
        with self._lock:
            state = self.load()
            active = set(state.active_command_ids)
            completed = set(state.completed_command_ids)
            used = set(state.used_command_nonces)
            rejected = set(state.rejected_command_ids)
            replay_attempts = state.replay_attempts
            if command.command_id in active:
                raise WorkerSecurityError(WorkerRejectionReason.DUPLICATE_ACTIVE_COMMAND, "command is already active")
            if command.command_id in completed or command.command_id in rejected or command.nonce in used:
                replay_attempts += 1
                self._write(state.model_copy(update={"replay_attempts": replay_attempts}))
                raise WorkerSecurityError(WorkerRejectionReason.REPLAYED_COMMAND, "command was already used")
            active.add(command.command_id)
            if command.nonce:
                used.add(command.nonce)
            updated = state.model_copy(
                update={
                    "active_command_ids": tuple(sorted(active)),
                    "used_command_nonces": tuple(sorted(used)),
                    "replay_attempts": replay_attempts,
                }
            )
            self._write(updated)
            return updated

    def complete_command(self, command_id: str, *, verification: dict[str, Any] | None = None) -> FreshnessState:
        with self._lock:
            state = self.load()
            active = set(state.active_command_ids)
            completed = set(state.completed_command_ids)
            active.discard(command_id)
            completed.add(command_id)
            updated = state.model_copy(
                update={
                    "active_command_ids": tuple(sorted(active)),
                    "completed_command_ids": tuple(sorted(completed)),
                    "last_successful_verification": verification or state.last_successful_verification,
                }
            )
            self._write(updated)
            return updated

    def reject_command(self, command_id: str | None) -> FreshnessState:
        with self._lock:
            state = self.load()
            rejected = set(state.rejected_command_ids)
            if command_id:
                rejected.add(command_id)
            updated = state.model_copy(
                update={
                    "rejected_command_ids": tuple(sorted(rejected)),
                    "rejected_commands": state.rejected_commands + 1,
                }
            )
            self._write(updated)
            return updated

    def reserve_control(self, request: WorkerControlRequest) -> FreshnessState:
        with self._lock:
            state = self.load()
            used = set(state.used_control_nonces)
            if request.nonce in used:
                raise WorkerSecurityError(WorkerRejectionReason.INVALID_CONTROL_REQUEST, "control nonce was already used")
            used.add(request.nonce)
            updated = state.model_copy(update={"used_control_nonces": tuple(sorted(used))})
            self._write(updated)
            return updated

    def _write(self, state: FreshnessState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, state.model_dump_json(indent=2))


class FilesystemSecurityInspector:
    """Delegates to `archivetrust.worker.acl_policy.verify()` on Windows (real SID-based ACL
    verification, added 2026-07-20 to close the "Authenticated Users: Modify" boundary gap found
    by `artifacts/qec_boundary_validation_20260720/`). A path with no ACL policy yet applied is a
    `warning`, not `critical` -- it does not block processing on its own, matching current/legacy
    deployments that have not yet run `archivetrust-admin acl-apply`. Once a policy has been
    applied, an unauthorized-write finding on a critical-class path is `critical` and blocks
    processing (`enforce_filesystem_security` in `WorkerCommandVerifier`)."""

    def inspect(self, deployment_root: Path) -> FilesystemSecurityReport:
        paths = (
            deployment_root,
            deployment_root / "worker" / "commands",
            deployment_root / "worker" / "control",
            deployment_root / "worker" / "results",
            deployment_root / "worker" / "freshness_state.json",
            deployment_root / "identity",
        )
        if os.name == "nt":
            return self._inspect_windows(deployment_root, paths)
        findings = tuple(self._finding_posix(path) for path in paths)
        blocked = any(item.status == "critical" for item in findings)
        return FilesystemSecurityReport(
            checked_at=utc_now(),
            platform=os.name,
            mode="posix_advisory_only",
            processing_blocked=blocked,
            findings=findings,
        )

    def _inspect_windows(self, deployment_root: Path, paths: tuple[Path, ...]) -> FilesystemSecurityReport:
        from archivetrust.worker.acl_policy import verify as acl_verify

        result = acl_verify(deployment_root)
        if not result.platform_supported:
            findings = tuple(
                FilesystemSecurityFinding(
                    path=str(path),
                    status="warning",
                    detail="Windows ACL inspection unavailable (pywin32 not importable)",
                )
                for path in paths
            )
            return FilesystemSecurityReport(
                checked_at=utc_now(), platform=os.name, mode="windows_acl_required", processing_blocked=False, findings=findings
            )
        findings_by_path: dict[str, list[str]] = {}
        for finding in result.findings:
            severity = "critical" if finding.critical else "warning"
            findings_by_path.setdefault(finding.path, []).append(f"{severity}:{finding.code}:{finding.detail}")
        report_findings = []
        for path in paths:
            path_str = str(path)
            if not path.exists():
                report_findings.append(FilesystemSecurityFinding(path=path_str, status="warning", detail="path does not exist yet"))
                continue
            entries = findings_by_path.get(path_str)
            if not entries:
                report_findings.append(FilesystemSecurityFinding(path=path_str, status="ok", detail="ACL policy verified, no drift"))
                continue
            status = "critical" if any(entry.startswith("critical:") for entry in entries) else "warning"
            report_findings.append(FilesystemSecurityFinding(path=path_str, status=status, detail="; ".join(entries)))
        return FilesystemSecurityReport(
            checked_at=utc_now(),
            platform=os.name,
            mode="windows_acl_required",
            processing_blocked=result.processing_blocked,
            findings=tuple(report_findings),
        )

    def _finding_posix(self, path: Path) -> FilesystemSecurityFinding:
        if not path.exists():
            return FilesystemSecurityFinding(path=str(path), status="warning", detail="path does not exist yet")
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError as error:
            return FilesystemSecurityFinding(path=str(path), status="critical", detail=f"cannot inspect: {error}")
        if mode & 0o002:
            return FilesystemSecurityFinding(path=str(path), status="critical", detail="world-writable path")
        return FilesystemSecurityFinding(path=str(path), status="ok", detail=f"posix mode {mode:o}")


class WorkerCommandAuthorizer:
    def __init__(self, deployment_root: Path) -> None:
        self.deployment_root = deployment_root
        self.identity = DeploymentIdentityProvider(deployment_root).ensure()

    def create_command(
        self,
        *,
        workspace_id: str,
        source: WorkerCommandSource,
        session=None,
        campaign_id: str | None = None,
        continuation_id: str | None = None,
        run_profile_digest: str | None = None,
        ttl: timedelta = DEFAULT_COMMAND_TTL,
    ) -> WorkerCommand:
        issued = datetime.now(timezone.utc)
        account = session.require_active() if session is not None else None
        if account is not None:
            require_permission(account, Permission.PROCESSING_CONTROL)
        workspace_root = self.deployment_root / "workspaces" / workspace_id
        config_lock = write_configuration_lock(workspace_root, workspace_id=workspace_id)
        command = WorkerCommand(
            command_id=new_id("worker_command"),
            deployment_id=self.identity.deployment_id,
            workspace_id=workspace_id,
            submitted_at=issued.isoformat(),
            submitted_by_account_id=None if account is None else account.account_id,
            submitted_by_display_name=None if account is None else account.display_name,
            submitted_by_reviewer_ref=None if account is None else account.reviewer_ref,
            submitted_roles=tuple(role.value for role in account.roles) if account is not None else (IdentityRole.ADMINISTRATOR.value,),
            permission=Permission.PROCESSING_CONTROL.value,
            session_id=None if session is None else session.session_id,
            issued_at=issued.isoformat(),
            expires_at=(issued + ttl).isoformat(),
            nonce=secrets.token_hex(16),
            configuration_lock_digest=config_lock.configuration_hash,
            run_profile_digest=run_profile_digest,
            qualification_campaign_id=campaign_id,
            qualification_continuation_id=continuation_id,
            authorization_mechanism=AUTH_MECHANISM,
            authorization_version=AUTH_VERSION,
            command_source=source,
        )
        payload_digest = _payload_digest(command, exclude={"auth_tag", "payload_digest"})
        command = command.model_copy(update={"payload_digest": payload_digest})
        return command.model_copy(update={"auth_tag": _tag(command, self.identity)})

    def create_control(
        self,
        *,
        command_id: str,
        action: WorkerControlAction,
        session=None,
        ttl: timedelta = DEFAULT_CONTROL_TTL,
    ) -> WorkerControlRequest:
        issued = datetime.now(timezone.utc)
        account = session.require_active() if session is not None else None
        if account is not None:
            require_permission(account, Permission.PROCESSING_CONTROL)
        request = WorkerControlRequest(
            control_id=new_id("worker_control"),
            command_id=command_id,
            deployment_id=self.identity.deployment_id,
            action=action,
            submitted_by_account_id=None if account is None else account.account_id,
            submitted_by_display_name=None if account is None else account.display_name,
            submitted_by_reviewer_ref=None if account is None else account.reviewer_ref,
            submitted_roles=tuple(role.value for role in account.roles) if account is not None else (IdentityRole.ADMINISTRATOR.value,),
            permission=Permission.PROCESSING_CONTROL.value,
            session_id=None if session is None else session.session_id,
            issued_at=issued.isoformat(),
            expires_at=(issued + ttl).isoformat(),
            nonce=secrets.token_hex(16),
            authorization_mechanism=AUTH_MECHANISM,
            authorization_version=AUTH_VERSION,
            auth_tag="",
        )
        return request.model_copy(update={"auth_tag": _tag(request, self.identity)})


class WorkerCommandVerifier:
    def __init__(
        self,
        deployment_root: Path,
        *,
        freshness: CommandFreshnessStore | None = None,
        filesystem_inspector: FilesystemSecurityInspector | None = None,
        enforce_filesystem_security: bool = True,
    ) -> None:
        self.deployment_root = deployment_root
        self.identity = DeploymentIdentityProvider(deployment_root).load()
        self.freshness = freshness or CommandFreshnessStore.for_deployment(deployment_root)
        self.filesystem_inspector = filesystem_inspector or FilesystemSecurityInspector()
        self.enforce_filesystem_security = enforce_filesystem_security

    def verify(self, command: WorkerCommand) -> VerifiedWorkerCommand:
        if command.schema_id != COMMAND_SCHEMA_ID:
            raise WorkerSecurityError(WorkerRejectionReason.UNSUPPORTED_SCHEMA, "unsupported command schema")
        _require(command.deployment_id == self.identity.deployment_id, WorkerRejectionReason.WRONG_DEPLOYMENT, "wrong deployment")
        _require(command.authorization_mechanism == AUTH_MECHANISM, WorkerRejectionReason.MISSING_OR_INVALID_AUTHENTICATION, "unsupported authorization mechanism")
        _require(command.authorization_version == AUTH_VERSION, WorkerRejectionReason.MISSING_OR_INVALID_AUTHENTICATION, "unsupported authorization version")
        _require(command.auth_tag and hmac.compare_digest(command.auth_tag, _tag(command, self.identity)), WorkerRejectionReason.MISSING_OR_INVALID_AUTHENTICATION, "invalid command authentication")
        _require(command.payload_digest == _payload_digest(command, exclude={"auth_tag", "payload_digest"}), WorkerRejectionReason.MISSING_OR_INVALID_AUTHENTICATION, "payload digest mismatch")
        _require(command.expires_at is not None and datetime.fromisoformat(command.expires_at) > datetime.now(timezone.utc), WorkerRejectionReason.EXPIRED_COMMAND, "command expired")
        workspace_root = self.deployment_root / "workspaces" / command.workspace_id
        _require(workspace_root.exists(), WorkerRejectionReason.UNKNOWN_WORKSPACE, "workspace does not exist")
        if command.submitted_by_account_id:
            self._verify_account(command)
        try:
            actual_lock = verify_configuration_lock(workspace_root)
        except ConfigurationDriftError as error:
            raise WorkerSecurityError(WorkerRejectionReason.CONFIGURATION_DRIFT, "configuration lock drift") from error
        _require(actual_lock.configuration_hash == command.configuration_lock_digest, WorkerRejectionReason.CONFIGURATION_DRIFT, "configuration lock changed after command issue")
        report = self.filesystem_inspector.inspect(self.deployment_root)
        if self.enforce_filesystem_security and report.processing_blocked:
            raise WorkerSecurityError(WorkerRejectionReason.INSECURE_FILESYSTEM_BOUNDARY, "filesystem boundary failed inspection")
        self.freshness.reserve_command(command)
        return VerifiedWorkerCommand(
            command=command,
            command_digest=command_digest(command),
            verified_at=utc_now(),
            configuration_lock_digest=actual_lock.configuration_hash,
        )

    def _verify_account(self, command: WorkerCommand) -> None:
        store_path = self.deployment_root / "identity" / "accounts.json"
        store = LocalAccountStore(store_path)
        try:
            account = store.get(command.submitted_by_account_id or "")
        except KeyError as error:
            raise WorkerSecurityError(WorkerRejectionReason.DISABLED_OR_UNKNOWN_ACCOUNT, "unknown account") from error
        if not account.active:
            raise WorkerSecurityError(WorkerRejectionReason.DISABLED_OR_UNKNOWN_ACCOUNT, "account disabled")
        try:
            require_permission(account, Permission.PROCESSING_CONTROL)
        except PermissionDenied as error:
            raise WorkerSecurityError(WorkerRejectionReason.INSUFFICIENT_PERMISSION, "insufficient permission") from error
        if command.expires_at is None or datetime.fromisoformat(command.expires_at) <= datetime.now(timezone.utc):
            raise WorkerSecurityError(WorkerRejectionReason.EXPIRED_OR_INVALID_SESSION, "command authorization expired")


class WorkerControlVerifier:
    def __init__(self, deployment_root: Path, *, freshness: CommandFreshnessStore | None = None) -> None:
        self.deployment_root = deployment_root
        self.identity = DeploymentIdentityProvider(deployment_root).load()
        self.freshness = freshness or CommandFreshnessStore.for_deployment(deployment_root)

    def verify(self, request: WorkerControlRequest, *, command_id: str) -> VerifiedWorkerControl:
        _require(request.schema_id == CONTROL_SCHEMA_ID, WorkerRejectionReason.INVALID_CONTROL_REQUEST, "unsupported control schema")
        _require(request.command_id == command_id, WorkerRejectionReason.INVALID_CONTROL_REQUEST, "control is for another command")
        _require(request.deployment_id == self.identity.deployment_id, WorkerRejectionReason.WRONG_DEPLOYMENT, "control is for another deployment")
        _require(request.authorization_mechanism == AUTH_MECHANISM and request.authorization_version == AUTH_VERSION, WorkerRejectionReason.INVALID_CONTROL_REQUEST, "unsupported control authorization")
        _require(hmac.compare_digest(request.auth_tag, _tag(request, self.identity)), WorkerRejectionReason.INVALID_CONTROL_REQUEST, "invalid control authentication")
        _require(datetime.fromisoformat(request.expires_at) > datetime.now(timezone.utc), WorkerRejectionReason.INVALID_CONTROL_REQUEST, "control expired")
        self.freshness.reserve_control(request)
        return VerifiedWorkerControl(request=request, verified_at=utc_now())


class WorkerResultVerifier:
    def verify(self, result: WorkerCommandResult, *, command: WorkerCommand | None = None) -> WorkerCommandResult:
        if result.schema_id != RESULT_SCHEMA_ID:
            raise WorkerSecurityError(WorkerRejectionReason.RESULT_INTEGRITY_FAILURE, "unsupported result schema")
        if result.status is WorkerCommandStatus.COMPLETED and not result.command_digest:
            raise WorkerSecurityError(WorkerRejectionReason.RESULT_INTEGRITY_FAILURE, "successful result lacks command digest")
        if command is not None and result.command_digest != command_digest(command):
            raise WorkerSecurityError(WorkerRejectionReason.RESULT_INTEGRITY_FAILURE, "result command digest mismatch")
        return result


def command_digest(command: WorkerCommand) -> str:
    return hashlib.sha256(canonical_json(command.model_dump(mode="json"))).hexdigest()


def result_digest(result: WorkerCommandResult) -> str:
    payload = result.model_dump(mode="json", exclude={"result_digest", "result_auth_tag"})
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def result_with_digest(result: WorkerCommandResult, identity: DeploymentIdentity | None = None) -> WorkerCommandResult:
    digest = result_digest(result)
    update: dict[str, Any] = {"result_digest": digest}
    if identity is not None:
        update["result_auth_tag"] = _hmac_hex(bytes.fromhex(identity.hmac_key_hex), digest.encode("utf-8"))
    return result.model_copy(update=update)


def read_control_request(path: Path) -> WorkerControlRequest:
    return WorkerControlRequest.model_validate_json(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, payload.model_dump_json(indent=2))


def _tag(model: BaseModel, identity: DeploymentIdentity) -> str:
    payload = model.model_dump(mode="json", exclude={"auth_tag"})
    return _hmac_hex(bytes.fromhex(identity.hmac_key_hex), canonical_json(payload))


def _payload_digest(command: WorkerCommand, *, exclude: set[str]) -> str:
    return hashlib.sha256(canonical_json(command.model_dump(mode="json", exclude=exclude))).hexdigest()


def _hmac_hex(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _require(condition: bool, reason: WorkerRejectionReason, detail: str) -> None:
    if not condition:
        raise WorkerSecurityError(reason, detail)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
