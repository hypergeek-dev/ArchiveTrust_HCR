"""Durable, versioned installation-state file for the Windows bootstrap
(`scripts/setup-archivetrust.ps1`).

This is a development bootstrap / installation-candidate concern, not a production installer.
Nothing here implements Docker, provider, or deployment-root logic itself -- it only tracks
whether each phase of that eventual work has run, so `-Mode Resume` can pick up where a prior
run left off instead of restarting from scratch. See `docs/WINDOWS_BOOTSTRAP.md` for the phase
list and what each one is expected to mean once implemented.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

STATE_SCHEMA_ID = "archivetrust.bootstrap_install_state.v1"
STATE_SCHEMA_VERSION = 1


class InstallStateError(RuntimeError):
    pass


class InstallStateCorruptionError(InstallStateError):
    """Raised when the state file exists but cannot be trusted (invalid JSON, unsupported
    schema version, or a phase record that fails validation). The caller must surface this as a
    `BLOCKED`/manual-action condition -- silently discarding and recreating the file would hide
    real drift from the operator."""


class PhaseStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    BLOCKED = "blocked"
    REBOOT_REQUIRED = "reboot_required"
    MANUAL_ACTION_REQUIRED = "manual_action_required"


class Phase(str, Enum):
    """Every phase Part 4/5/6/8/9-14 of the bootstrap task eventually needs. Only `PREFLIGHT`
    (and the state machinery itself) has real logic behind it right now -- see `implemented` on
    `PhaseRecord`. The rest exist so the schema does not need a breaking change when their real
    implementation lands, and so `-Mode Verify`/`-Mode Status` can honestly report
    "not implemented yet" instead of a misleading `not_started`."""

    PREFLIGHT = "preflight_complete"
    PREREQUISITES = "prerequisites_complete"
    PYTHON_ENVIRONMENT = "python_environment_complete"
    ARCHIVETRUST_INSTALLED = "archivetrust_installed"
    DOCKER_DESKTOP_DETECTED = "docker_desktop_detected"
    DOCKER_RUNTIME_AVAILABLE = "docker_runtime_available"
    DOCKER_COMPOSE_AVAILABLE = "docker_compose_available"
    PROVIDER_IMAGES_ACQUIRED = "provider_images_acquired"
    PROVIDER_CONTAINERS_CONFIGURED = "provider_containers_configured"
    PROVIDER_HEALTH_VERIFIED = "provider_health_verified"
    DEPLOYMENT_ROOT_INITIALIZED = "deployment_root_initialized"
    ADMINISTRATOR_BOOTSTRAPPED = "administrator_bootstrapped"
    ACL_POLICY_APPLIED = "acl_policy_applied"
    ACL_POLICY_VERIFIED = "acl_policy_verified"
    CONFIGURATION_VALIDATED = "configuration_validated"
    CONFIGURATION_LOCKED = "configuration_locked"
    HEALTH_VERIFIED = "health_verified"
    SMOKE_TEST_READY = "smoke_test_ready"
    SMOKE_TEST_EXECUTED = "smoke_test_executed"


IMPLEMENTED_PHASES = frozenset({Phase.PREFLIGHT})
"""Phases with real, tested logic behind them as of this bootstrap skeleton (Parts 3-5 of the
Windows bootstrap task). Every other phase is a placeholder -- `InstallState.record()` still
accepts updates for them (so later work does not need a schema migration), but
`PhaseRecord.implemented` stays `False` until real logic lands, and the PowerShell/CLI surfaces
must say so rather than claiming completion."""


class PhaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: Phase
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    implemented: bool = False
    attempts: int = 0
    started_at: str | None = None
    updated_at: str | None = None
    last_error_classification: str | None = None
    detail: str | None = None
    """Short, non-secret, non-document-content human-readable status. Callers must not put
    credentials, tokens, or document paths/content here -- see the module-level "no secrets"
    tests in `tests/bootstrap/test_install_state.py`."""


class InstallState(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = STATE_SCHEMA_ID
    schema_version: int = STATE_SCHEMA_VERSION
    deployment_root: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    phases: dict[str, PhaseRecord] = Field(default_factory=dict)

    def phase_record(self, phase: Phase) -> PhaseRecord:
        return self.phases.get(phase.value, PhaseRecord(phase=phase))

    def with_phase_result(
        self,
        phase: Phase,
        status: PhaseStatus,
        *,
        detail: str | None = None,
        error_classification: str | None = None,
        implemented: bool | None = None,
    ) -> "InstallState":
        existing = self.phase_record(phase)
        now = datetime.now(timezone.utc).isoformat()
        updated = existing.model_copy(
            update={
                "status": status,
                "implemented": existing.implemented if implemented is None else implemented,
                "attempts": existing.attempts + 1,
                "started_at": existing.started_at or now,
                "updated_at": now,
                "detail": detail,
                "last_error_classification": error_classification if status is PhaseStatus.FAILED else existing.last_error_classification,
            }
        )
        phases = dict(self.phases)
        phases[phase.value] = updated
        return self.model_copy(update={"phases": phases, "updated_at": now})

    def is_complete(self, phase: Phase) -> bool:
        return self.phase_record(phase).status is PhaseStatus.COMPLETED

    def overall_blocked(self) -> bool:
        return any(record.status in (PhaseStatus.BLOCKED, PhaseStatus.FAILED) for record in self.phases.values())

    def reboot_required(self) -> bool:
        return any(record.status is PhaseStatus.REBOOT_REQUIRED for record in self.phases.values())


class InstallStateStore:
    """Atomic read/write for one deployment root's install-state file. Concurrent-safe within a
    process via a lock; cross-process safety comes from the atomic rename, matching the pattern
    already used by `worker/security.py`'s durable command/result files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    @classmethod
    def for_deployment(cls, deployment_root: Path) -> "InstallStateStore":
        return cls(deployment_root / "bootstrap" / "install_state.json")

    def exists(self) -> bool:
        return self.path.exists()

    def load(self, *, deployment_root: str | None = None) -> InstallState:
        if not self.path.exists():
            if deployment_root is None:
                raise InstallStateError(f"no install state at {self.path} and no deployment_root given to create one")
            return InstallState(deployment_root=deployment_root)
        text = self.path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except ValueError as error:
            raise InstallStateCorruptionError(f"install state at {self.path} is not valid JSON: {error}") from error
        version = payload.get("schema_version")
        if version != STATE_SCHEMA_VERSION:
            raise InstallStateCorruptionError(
                f"install state at {self.path} has unsupported schema_version {version!r}; expected {STATE_SCHEMA_VERSION}"
            )
        try:
            return InstallState.model_validate(payload)
        except Exception as error:  # noqa: BLE001 - corruption boundary, not a crash
            raise InstallStateCorruptionError(f"install state at {self.path} failed validation: {error}") from error

    def save(self, state: InstallState) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(dir=self.path.parent)
            temp = Path(temp_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(state.model_dump_json(indent=2))
                temp.replace(self.path)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
