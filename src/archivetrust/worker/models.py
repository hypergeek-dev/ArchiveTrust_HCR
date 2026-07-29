from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkerCommandStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class WorkerCommandSource(str, Enum):
    DESKTOP = "desktop"
    ADMIN = "admin"
    QUALIFICATION_RUNNER = "qualification_runner"
    TEST = "test"


class WorkerControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class WorkerRejectionReason(str, Enum):
    MALFORMED_COMMAND = "malformed_command"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    MISSING_OR_INVALID_AUTHENTICATION = "missing_or_invalid_authentication"
    EXPIRED_COMMAND = "expired_command"
    REPLAYED_COMMAND = "replayed_command"
    WRONG_DEPLOYMENT = "wrong_deployment"
    UNKNOWN_WORKSPACE = "unknown_workspace"
    CONFIGURATION_DRIFT = "configuration_drift"
    DISABLED_OR_UNKNOWN_ACCOUNT = "disabled_or_unknown_account"
    EXPIRED_OR_INVALID_SESSION = "expired_or_invalid_session"
    INSUFFICIENT_PERMISSION = "insufficient_permission"
    DUPLICATE_ACTIVE_COMMAND = "duplicate_active_command"
    INSECURE_FILESYSTEM_BOUNDARY = "insecure_filesystem_boundary"
    INVALID_CONTROL_REQUEST = "invalid_control_request"
    RESULT_INTEGRITY_FAILURE = "result_integrity_failure"


class WorkerCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.worker_command.v2"
    command_id: str
    command_kind: str = "process_queue"
    deployment_id: str | None = None
    workspace_id: str
    action: str = "process_queue"
    submitted_at: str
    submitted_by_account_id: str | None = None
    submitted_by_display_name: str | None = None
    submitted_by_reviewer_ref: str | None = None
    submitted_roles: tuple[str, ...] = ()
    permission: str | None = None
    session_id: str | None = None
    issued_at: str | None = None
    expires_at: str | None = None
    nonce: str | None = None
    configuration_lock_digest: str | None = None
    run_profile_digest: str | None = None
    qualification_campaign_id: str | None = None
    qualification_continuation_id: str | None = None
    payload_digest: str | None = None
    authorization_mechanism: str | None = None
    authorization_version: str | None = None
    command_source: WorkerCommandSource = WorkerCommandSource.DESKTOP
    auth_tag: str | None = Field(default=None, repr=False)


class WorkerCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.worker_command_result.v2"
    command_id: str
    workspace_id: str
    status: WorkerCommandStatus
    recorded_at: str
    run_id: str | None = None
    documents_done: int = 0
    documents_total: int = 0
    error: str | None = None
    rejection_reason: WorkerRejectionReason | None = None
    command_digest: str | None = None
    command_schema_id: str | None = None
    authorization_mechanism: str | None = None
    authorization_version: str | None = None
    verification_status: str | None = None
    verified_at: str | None = None
    verified_configuration_lock: str | None = None
    submitted_by_account_id: str | None = None
    submitted_by_display_name: str | None = None
    command_source: WorkerCommandSource | None = None
    result_digest: str | None = None
    result_auth_tag: str | None = Field(default=None, repr=False)


class WorkerControlRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.worker_control_request.v1"
    control_id: str
    command_id: str
    deployment_id: str
    action: WorkerControlAction
    submitted_by_account_id: str | None = None
    submitted_by_display_name: str | None = None
    submitted_by_reviewer_ref: str | None = None
    submitted_roles: tuple[str, ...] = ()
    permission: str | None = None
    session_id: str | None = None
    issued_at: str
    expires_at: str
    nonce: str
    authorization_mechanism: str
    authorization_version: str
    auth_tag: str = Field(repr=False)


class VerifiedWorkerCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: WorkerCommand
    command_digest: str
    verified_at: str
    configuration_lock_digest: str


class VerifiedWorkerControl(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: WorkerControlRequest
    verified_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def canonical_json(payload: dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
