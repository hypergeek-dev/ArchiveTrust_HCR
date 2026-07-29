"""A5 local identity, roles, and operator audit log.

This is deliberately local-first and application-facing. It does not change Trust Engine telemetry
or pretend to be enterprise IAM; it gives the desktop/admin layer named actors and auditable
operator actions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class IdentityRole(str, Enum):
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    ADMINISTRATOR = "administrator"
    AUDITOR = "auditor"


class AdminAction(str, Enum):
    WORKSPACE_CREATED = "workspace_created"
    WORKSPACE_RENAMED = "workspace_renamed"
    WORKSPACE_RETIRED = "workspace_retired"
    WORKSPACE_REACTIVATED = "workspace_reactivated"
    WORKSPACE_DELETED = "workspace_deleted"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_CANCELLED = "processing_cancelled"
    PROCESSING_RETRIED = "processing_retried"
    REVIEW_CLAIMED = "review_claimed"
    REVIEW_DECIDED = "review_decided"
    EVALUATION_ANNOTATED = "evaluation_annotated"
    EVALUATION_ADJUDICATED = "evaluation_adjudicated"
    EXPORT_CREATED = "export_created"
    CONFIGURATION_CHANGED = "configuration_changed"
    INTEGRITY_VERIFIED = "integrity_verified"
    ACCOUNT_CHANGED = "account_changed"
    POLICY_CHANGED = "policy_changed"


class Permission(str, Enum):
    WORKSPACE_CREATE = "workspace.create"
    WORKSPACE_RETIRE = "workspace.retire"
    WORKSPACE_DELETE = "workspace.delete"
    PROVIDER_CONFIGURE = "provider.configure"
    PROCESSING_CONTROL = "processing.control"
    REVIEW = "review"
    EVALUATION_ANNOTATE = "evaluation.annotate"
    EVALUATION_ADJUDICATE = "evaluation.adjudicate"
    EXPORT = "export"
    INTEGRITY_VERIFY = "integrity.verify"
    ACCOUNT_ADMINISTER = "account.administer"
    POLICY_CHANGE = "policy.change"
    AUDIT_READ = "audit.read"


class PermissionDenied(PermissionError):
    """Raised when a local account lacks the role required for an operator action."""


class LocalAccount(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    display_name: str
    reviewer_ref: str
    roles: tuple[IdentityRole, ...]
    active: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    password_salt: str | None = Field(default=None, repr=False)
    password_digest: str | None = Field(default=None, repr=False)
    password_iterations: int = Field(default=600_000, ge=300_000, repr=False)

    def has_role(self, role: IdentityRole) -> bool:
        return role in self.roles

    @classmethod
    def with_password(cls, *, password: str, **account_fields) -> "LocalAccount":
        _validate_password(password)
        salt = secrets.token_bytes(32)
        iterations = 600_000
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return cls(
            **account_fields,
            password_salt=salt.hex(),
            password_digest=digest.hex(),
            password_iterations=iterations,
        )

    def verifies_password(self, password: str) -> bool:
        if not self.active or self.password_salt is None or self.password_digest is None:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(self.password_salt),
            self.password_iterations,
        )
        return hmac.compare_digest(candidate, bytes.fromhex(self.password_digest))


class AuthenticatedSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    account: LocalAccount
    authenticated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str

    @classmethod
    def start(cls, account: LocalAccount, *, lifetime: timedelta = timedelta(hours=8)) -> "AuthenticatedSession":
        if not account.active:
            raise PermissionDenied("account is disabled")
        return cls(account=account, expires_at=(datetime.now(timezone.utc) + lifetime).isoformat())

    def require_active(self) -> LocalAccount:
        if not self.account.active or datetime.fromisoformat(self.expires_at) <= datetime.now(timezone.utc):
            raise PermissionDenied("authenticated session has expired or the account is disabled")
        return self.account


class AdminAuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    actor_account_id: str
    actor_display_name: str
    action: AdminAction
    target_ref: str
    target_label: str | None = None
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, str] = Field(default_factory=dict)
    actor_roles: tuple[IdentityRole, ...] = ()
    prior_state_ref: str | None = None
    resulting_state_ref: str | None = None
    success: bool = True
    reason: str | None = None
    session_id: str | None = None
    previous_hash: str | None = None
    event_hash: str | None = None


_PERMISSION_ROLES: dict[Permission, frozenset[IdentityRole]] = {
    Permission.WORKSPACE_CREATE: frozenset({IdentityRole.OPERATOR, IdentityRole.ADMINISTRATOR}),
    Permission.WORKSPACE_RETIRE: frozenset({IdentityRole.OPERATOR, IdentityRole.ADMINISTRATOR}),
    Permission.WORKSPACE_DELETE: frozenset({IdentityRole.ADMINISTRATOR}),
    Permission.PROVIDER_CONFIGURE: frozenset({IdentityRole.ADMINISTRATOR}),
    Permission.PROCESSING_CONTROL: frozenset({IdentityRole.OPERATOR, IdentityRole.ADMINISTRATOR}),
    Permission.REVIEW: frozenset({IdentityRole.REVIEWER, IdentityRole.ADMINISTRATOR}),
    Permission.EVALUATION_ANNOTATE: frozenset({IdentityRole.REVIEWER, IdentityRole.ADMINISTRATOR}),
    Permission.EVALUATION_ADJUDICATE: frozenset({IdentityRole.ADMINISTRATOR}),
    Permission.EXPORT: frozenset({IdentityRole.OPERATOR, IdentityRole.ADMINISTRATOR}),
    Permission.INTEGRITY_VERIFY: frozenset({IdentityRole.AUDITOR, IdentityRole.ADMINISTRATOR}),
    Permission.ACCOUNT_ADMINISTER: frozenset({IdentityRole.ADMINISTRATOR}),
    Permission.POLICY_CHANGE: frozenset({IdentityRole.ADMINISTRATOR}),
    Permission.AUDIT_READ: frozenset({IdentityRole.AUDITOR, IdentityRole.ADMINISTRATOR}),
}


def _validate_password(password: str) -> None:
    if len(password) < 14 or password.lower() == password or password.upper() == password or not any(c.isdigit() for c in password):
        raise ValueError("password must be at least 14 characters and include upper/lower case and a digit")


def require_role(account: LocalAccount, role: IdentityRole) -> None:
    if not account.active or role not in account.roles:
        raise PermissionDenied(f"{account.display_name!r} requires role {role.value!r}")


def require_permission(account: LocalAccount, permission: Permission) -> None:
    allowed = _PERMISSION_ROLES[permission]
    if not account.active or not allowed.intersection(account.roles):
        raise PermissionDenied(f"{account.display_name!r} lacks permission {permission.value!r}")


def _audit_hash(event: AdminAuditEvent) -> str:
    payload = event.model_dump(mode="json", exclude={"event_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AdminAuditLog:
    """Append-only JSONL audit log for operator actions."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._events: list[AdminAuditEvent] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not self._path.exists():
                self._path.touch()
            self._events = [
                AdminAuditEvent.model_validate(json.loads(line))
                for line in self._path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def append(self, event: AdminAuditEvent) -> AdminAuditEvent:
        with self._lock:
            previous_hash = self._events[-1].event_hash if self._events else None
            chained = event.model_copy(update={"previous_hash": previous_hash, "event_hash": None})
            chained = chained.model_copy(update={"event_hash": _audit_hash(chained)})
            self._events.append(chained)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(chained.model_dump_json())
                    handle.write("\n")
            return chained

    def record(
        self,
        *,
        actor: LocalAccount,
        action: AdminAction,
        target_ref: str,
        target_label: str | None = None,
        metadata: dict[str, str] | None = None,
        prior_state_ref: str | None = None,
        resulting_state_ref: str | None = None,
        success: bool = True,
        reason: str | None = None,
        session_id: str | None = None,
    ) -> AdminAuditEvent:
        event = AdminAuditEvent(
            actor_account_id=actor.account_id,
            actor_display_name=actor.display_name,
            action=action,
            target_ref=target_ref,
            target_label=target_label,
            metadata=metadata or {},
            actor_roles=actor.roles,
            prior_state_ref=prior_state_ref,
            resulting_state_ref=resulting_state_ref,
            success=success,
            reason=reason,
            session_id=session_id,
        )
        return self.append(event)

    def all_events(self) -> tuple[AdminAuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def verify_integrity(self) -> tuple[bool, str]:
        with self._lock:
            previous_hash: str | None = None
            for index, event in enumerate(self._events):
                if event.previous_hash != previous_hash:
                    return False, f"audit event {index} has an invalid previous hash"
                if event.event_hash != _audit_hash(event):
                    return False, f"audit event {index} hash mismatch"
                previous_hash = event.event_hash
        return True, "audit hash chain verified"


class LocalAccountStore:
    """Small JSON-backed account store for local-first deployments."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    def list(self) -> tuple[LocalAccount, ...]:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return tuple(LocalAccount.model_validate(row) for row in payload)

    def add(self, account: LocalAccount) -> LocalAccount:
        accounts = list(self.list())
        if any(existing.reviewer_ref == account.reviewer_ref for existing in accounts):
            raise ValueError(f"duplicate reviewer_ref {account.reviewer_ref!r}")
        accounts.append(account)
        self._write(tuple(accounts))
        return account

    def get(self, account_id: str) -> LocalAccount:
        for account in self.list():
            if account.account_id == account_id:
                return account
        raise KeyError(account_id)

    def authenticate(self, reviewer_ref: str, password: str, *, lifetime: timedelta = timedelta(hours=8)) -> AuthenticatedSession:
        for account in self.list():
            if account.reviewer_ref == reviewer_ref and account.verifies_password(password):
                return AuthenticatedSession.start(account, lifetime=lifetime)
        raise PermissionDenied("invalid account name or password")

    def replace(self, account: LocalAccount) -> LocalAccount:
        accounts = list(self.list())
        for index, existing in enumerate(accounts):
            if existing.account_id == account.account_id:
                accounts[index] = account
                self._write(tuple(accounts))
                return account
        raise KeyError(account.account_id)

    def reset_password(self, account_id: str, password: str) -> LocalAccount:
        existing = self.get(account_id)
        replacement = LocalAccount.with_password(
            password=password,
            account_id=existing.account_id,
            display_name=existing.display_name,
            reviewer_ref=existing.reviewer_ref,
            roles=existing.roles,
            active=existing.active,
            created_at=existing.created_at,
        )
        return self.replace(replacement)

    def _write(self, accounts: tuple[LocalAccount, ...]) -> None:
        self._path.write_text(
            json.dumps([account.model_dump(mode="json") for account in accounts], indent=2),
            encoding="utf-8",
        )
