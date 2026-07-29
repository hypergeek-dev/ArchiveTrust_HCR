"""Windows least-privilege ACL policy for the accepted `SPAWNED_WORKER` deployment profile
(D-01, `docs/PRODUCTION_WORKER_SUPERVISION_DECISION.md`).

Closes the 2026-07-20 validation finding (`artifacts/qec_boundary_validation_20260720/`) that
`worker/commands`, `worker/control`, `worker/results`, `identity`, and `workspaces` inherited
`NT AUTHORITY\\Authenticated Users:(M)` -- Modify for every authenticated local account, not only
the operator.

One versioned, machine-readable policy (`AclPolicy`), one path classifier
(`classify_deployment_paths`), and one platform adapter (`WindowsAclEngine`) implementing plan,
apply, and verify. Non-Windows platforms and environments without `pywin32` report
`unsupported_platform` rather than silently no-op'ing or raising.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

POLICY_SCHEMA_ID = "archivetrust.acl_policy.v1"
POLICY_VERSION = "v1"

# Well-known SIDs, independent of locale/display-name (Windows-native semantics section).
SID_ADMINISTRATORS = "S-1-5-32-544"
SID_SYSTEM = "S-1-5-18"
SID_AUTHENTICATED_USERS = "S-1-5-11"
SID_EVERYONE = "S-1-1-0"
SID_BUILTIN_USERS = "S-1-5-32-545"

# Principals that are never acceptable as an explicit or inherited write-capable ACE on a
# protected path once this policy has been applied -- these are exactly the broad groups the
# 2026-07-20 validation found holding Modify.
DISALLOWED_WRITE_PRINCIPAL_SIDS = {SID_AUTHENTICATED_USERS, SID_EVERYONE, SID_BUILTIN_USERS}


class AclPrincipalRole(str, Enum):
    ADMINISTRATORS = "administrators"
    SYSTEM = "system"
    OPERATOR = "operator"


class AclAccessLevel(str, Enum):
    FULL_CONTROL = "full_control"
    MODIFY = "modify"
    READ_EXECUTE = "read_execute"


class AclPathClass(str, Enum):
    ADMINISTRATION = "administration"
    """Installer/policy/model-manifest paths: writable only by administrators/SYSTEM."""
    OPERATOR_RUNTIME = "operator_runtime"
    """Worker command/control/result/freshness paths: operator + admin write, no one else."""
    AUTHORITATIVE_EVIDENCE = "authoritative_evidence"
    """Archive, telemetry, blobs, identity, config-lock: operator + admin write, no one else."""
    REBUILDABLE = "rebuildable"
    """Caches, logs, derived projections, reports: same principals, but drift is non-critical."""
    QUALIFICATION_EVIDENCE = "qualification_evidence"
    """Qualification run output roots: operator + admin write, no one else."""


CRITICAL_PATH_CLASSES = {
    AclPathClass.ADMINISTRATION,
    AclPathClass.OPERATOR_RUNTIME,
    AclPathClass.AUTHORITATIVE_EVIDENCE,
    AclPathClass.QUALIFICATION_EVIDENCE,
}
"""`REBUILDABLE` is deliberately excluded: a stale/broad ACE on a cache or log directory cannot
by itself compromise authoritative evidence or the worker command boundary (Path Classifications,
'Rebuildable/output paths' -- do not give broad permissions to authoritative paths just because
some projections are rebuildable; the inverse also holds -- do not block production processing
over a rebuildable path)."""

CLASS_ACCESS: dict[AclPathClass, dict[AclPrincipalRole, AclAccessLevel]] = {
    AclPathClass.ADMINISTRATION: {
        AclPrincipalRole.ADMINISTRATORS: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.SYSTEM: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.OPERATOR: AclAccessLevel.READ_EXECUTE,
    },
    AclPathClass.OPERATOR_RUNTIME: {
        AclPrincipalRole.ADMINISTRATORS: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.SYSTEM: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.OPERATOR: AclAccessLevel.MODIFY,
    },
    AclPathClass.AUTHORITATIVE_EVIDENCE: {
        AclPrincipalRole.ADMINISTRATORS: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.SYSTEM: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.OPERATOR: AclAccessLevel.MODIFY,
    },
    AclPathClass.REBUILDABLE: {
        AclPrincipalRole.ADMINISTRATORS: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.SYSTEM: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.OPERATOR: AclAccessLevel.MODIFY,
    },
    AclPathClass.QUALIFICATION_EVIDENCE: {
        AclPrincipalRole.ADMINISTRATORS: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.SYSTEM: AclAccessLevel.FULL_CONTROL,
        AclPrincipalRole.OPERATOR: AclAccessLevel.MODIFY,
    },
}


class AclPolicy(BaseModel):
    """Versioned, deterministic, idempotent description of the expected ACL state. Does not
    itself know how to apply/verify on any platform -- that is `WindowsAclEngine`'s job."""

    model_config = ConfigDict(frozen=True)

    schema_id: str = POLICY_SCHEMA_ID
    version: str = POLICY_VERSION
    class_access: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            cls.value: {role.value: level.value for role, level in access.items()}
            for cls, access in CLASS_ACCESS.items()
        }
    )
    critical_classes: tuple[str, ...] = tuple(sorted(cls.value for cls in CRITICAL_PATH_CLASSES))
    disallowed_write_principal_sids: tuple[str, ...] = tuple(sorted(DISALLOWED_WRITE_PRINCIPAL_SIDS))


class ClassifiedPath(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    path_class: AclPathClass
    exists: bool


class AceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    sid: str
    account: str | None = None
    access_mask: int
    ace_type: str
    inherited: bool


class AclDriftFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    path_class: AclPathClass
    critical: bool
    code: str
    detail: str


class AclVerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.acl_verification_result.v1"
    checked_at: str
    policy_version: str
    platform_supported: bool
    operator_sid: str | None
    paths_checked: int
    findings: tuple[AclDriftFinding, ...]
    processing_blocked: bool
    """True only when a *critical* finding exists. `REBUILDABLE`-class findings never set this."""

    @property
    def status(self) -> str:
        if not self.platform_supported:
            return "unsupported_platform"
        if self.processing_blocked:
            return "fail"
        if self.findings:
            return "warning"
        return "pass"


class AclApplicationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.acl_application_record.v1"
    applied_at: str
    actor: str
    deployment_id: str | None
    policy_version: str
    operator_sid: str | None
    paths_planned: tuple[str, ...]
    paths_applied: tuple[str, ...]
    paths_failed: tuple[str, ...]
    result: str  # "success" | "partial_failure"


# -- Path classification -----------------------------------------------------------------------

_DEPLOYMENT_ADMINISTRATION_SUBDIRS = ("config", "models", "plugins")
_DEPLOYMENT_REBUILDABLE_SUBDIRS = ("cache", "logs")
_DEPLOYMENT_EVIDENCE_SUBDIRS = ("identity", "telemetry")
_DEPLOYMENT_OPERATOR_RUNTIME_SUBDIRS = ("worker",)

_WORKSPACE_EVIDENCE_SUBDIRS = ("archive", "telemetry", "config", "input")
_WORKSPACE_REBUILDABLE_SUBDIRS = ("derived", "reports", "cache", "logs", "evaluation")
_WORKSPACE_ADMINISTRATION_SUBDIRS = ("models", "plugins")


def classify_deployment_paths(
    deployment_root: Path, *, qualification_roots: tuple[Path, ...] = ()
) -> tuple[ClassifiedPath, ...]:
    """Enumerates every path this policy governs and its class. Does not require the paths to
    exist yet (see 'New-path behavior') -- `exists` records whether `apply()` can act on it now
    versus only once it is created by normal operation."""

    root = deployment_root
    # The root container itself must stay OPERATOR_RUNTIME (operator = Modify), not
    # ADMINISTRATION (operator = read-execute only): a 2026-07-20 smoke test of this exact
    # policy proved that classifying the root as admin-only locks a non-elevated operator out
    # of creating *any* new child under it -- new workspaces (`WorkspaceStore.create()`), the
    # identity file (`DeploymentIdentityProvider.ensure()`), and `worker/` itself are all
    # created directly under root by ordinary operator-privilege code, not an installer. Only
    # the specific administration subdirectories below (config/models/plugins, populated by
    # install/update, not by runtime operator actions) are admin-only.
    entries: list[tuple[Path, AclPathClass]] = [(root, AclPathClass.OPERATOR_RUNTIME)]
    for name in _DEPLOYMENT_ADMINISTRATION_SUBDIRS:
        entries.append((root / name, AclPathClass.ADMINISTRATION))
    for name in _DEPLOYMENT_REBUILDABLE_SUBDIRS:
        entries.append((root / name, AclPathClass.REBUILDABLE))
    for name in _DEPLOYMENT_EVIDENCE_SUBDIRS:
        entries.append((root / name, AclPathClass.AUTHORITATIVE_EVIDENCE))
    for name in _DEPLOYMENT_OPERATOR_RUNTIME_SUBDIRS:
        worker_root = root / name
        entries.append((worker_root, AclPathClass.OPERATOR_RUNTIME))
        for sub in ("commands", "control", "results"):
            entries.append((worker_root / sub, AclPathClass.OPERATOR_RUNTIME))
        entries.append((worker_root / "freshness_state.json", AclPathClass.OPERATOR_RUNTIME))

    workspaces_root = root / "workspaces"
    entries.append((workspaces_root, AclPathClass.AUTHORITATIVE_EVIDENCE))
    if workspaces_root.exists():
        for workspace_dir in sorted(p for p in workspaces_root.iterdir() if p.is_dir()):
            for name in _WORKSPACE_EVIDENCE_SUBDIRS:
                entries.append((workspace_dir / name, AclPathClass.AUTHORITATIVE_EVIDENCE))
            for name in _WORKSPACE_REBUILDABLE_SUBDIRS:
                entries.append((workspace_dir / name, AclPathClass.REBUILDABLE))
            for name in _WORKSPACE_ADMINISTRATION_SUBDIRS:
                entries.append((workspace_dir / name, AclPathClass.ADMINISTRATION))

    for extra_root in qualification_roots:
        entries.append((extra_root, AclPathClass.QUALIFICATION_EVIDENCE))

    return tuple(
        ClassifiedPath(path=str(path), path_class=path_class, exists=path.exists())
        for path, path_class in entries
    )


# -- Windows engine -----------------------------------------------------------------------------

DANGEROUS_ROOTS = {"c:\\", "c:\\windows", "c:\\users", "d:\\", "\\"}
"""Reject applying this policy to a drive root, the Windows directory, or a bare user-profile
root -- these are never a valid `deployment_root` and applying protected/non-inherited ACLs there
would be catastrophic (Safety and lockout prevention)."""


class AclPolicyError(RuntimeError):
    pass


def _normalize(path: Path) -> str:
    return str(Path(path).resolve()).rstrip("\\").lower() + "\\"


def is_dangerous_root(path: Path) -> bool:
    normalized = _normalize(path)
    for dangerous in DANGEROUS_ROOTS:
        candidate = dangerous if dangerous.endswith("\\") else dangerous + "\\"
        if normalized == candidate.lower():
            return True
    try:
        home = _normalize(Path.home())
    except RuntimeError:
        home = None
    if home is not None and normalized == home:
        return True
    return False


class WindowsAclEngine:
    """The one platform adapter. Everything above this class is pure policy/data and works
    without `pywin32` or Windows so it can be unit-tested cross-platform; only this class touches
    the Windows security API (or `icacls` as a documented fallback -- not used here because
    `pywin32`'s `win32security` module is already a project dependency and gives SID-based,
    locale-independent access, not English-string parsing)."""

    def __init__(self) -> None:
        self._win32security = None
        self._win32api = None
        self._ntsecuritycon = None
        self._import_error: str | None = None
        if os.name == "nt":
            try:
                import ntsecuritycon
                import win32api
                import win32security

                self._win32security = win32security
                self._win32api = win32api
                self._ntsecuritycon = ntsecuritycon
            except ImportError as error:  # pragma: no cover - depends on environment
                self._import_error = str(error)

    @property
    def available(self) -> bool:
        return self._win32security is not None

    @property
    def unavailable_reason(self) -> str | None:
        if os.name != "nt":
            return "unsupported_platform: not Windows"
        if not self.available:
            return f"unsupported_platform: pywin32 not importable ({self._import_error})"
        return None

    def operator_sid(self) -> str:
        if not self.available:
            raise AclPolicyError(self.unavailable_reason or "windows ACL engine unavailable")
        win32security = self._win32security
        win32api = self._win32api
        sid, _domain, _type = win32security.LookupAccountName(None, win32api.GetUserName())
        return win32security.ConvertSidToStringSid(sid)

    def _access_mask(self, level: AclAccessLevel) -> int:
        ntsecuritycon = self._ntsecuritycon
        if level is AclAccessLevel.FULL_CONTROL:
            return ntsecuritycon.FILE_ALL_ACCESS
        if level is AclAccessLevel.MODIFY:
            return (
                ntsecuritycon.FILE_GENERIC_READ
                | ntsecuritycon.FILE_GENERIC_WRITE
                | ntsecuritycon.FILE_GENERIC_EXECUTE
                | ntsecuritycon.DELETE
            )
        return ntsecuritycon.FILE_GENERIC_READ | ntsecuritycon.FILE_GENERIC_EXECUTE

    def _build_dacl(self, path_class: AclPathClass, operator_sid_string: str):
        win32security = self._win32security
        dacl = win32security.ACL()
        inherit_flags = (
            self._ntsecuritycon.CONTAINER_INHERIT_ACE | self._ntsecuritycon.OBJECT_INHERIT_ACE
        )
        access = CLASS_ACCESS[path_class]
        principal_sids = {
            AclPrincipalRole.ADMINISTRATORS: win32security.ConvertStringSidToSid(SID_ADMINISTRATORS),
            AclPrincipalRole.SYSTEM: win32security.ConvertStringSidToSid(SID_SYSTEM),
            AclPrincipalRole.OPERATOR: win32security.ConvertStringSidToSid(operator_sid_string),
        }
        for role in (AclPrincipalRole.ADMINISTRATORS, AclPrincipalRole.SYSTEM, AclPrincipalRole.OPERATOR):
            level = access[role]
            dacl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION_DS,
                inherit_flags,
                self._access_mask(level),
                principal_sids[role],
            )
        return dacl

    def apply_path(self, path: Path, path_class: AclPathClass, operator_sid_string: str) -> None:
        if not self.available:
            raise AclPolicyError(self.unavailable_reason or "windows ACL engine unavailable")
        if not path.exists():
            return
        win32security = self._win32security
        dacl = self._build_dacl(path_class, operator_sid_string)
        security_descriptor = win32security.GetFileSecurity(
            str(path), win32security.DACL_SECURITY_INFORMATION
        )
        security_descriptor.SetSecurityDescriptorDacl(1, dacl, 0)
        win32security.SetFileSecurity(
            str(path),
            win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            security_descriptor,
        )

    def read_aces(self, path: Path) -> tuple[AceRecord, ...]:
        if not self.available:
            raise AclPolicyError(self.unavailable_reason or "windows ACL engine unavailable")
        win32security = self._win32security
        security_descriptor = win32security.GetFileSecurity(
            str(path), win32security.DACL_SECURITY_INFORMATION
        )
        dacl = security_descriptor.GetSecurityDescriptorDacl()
        records: list[AceRecord] = []
        if dacl is None:
            return tuple(records)
        for index in range(dacl.GetAceCount()):
            ace = dacl.GetAce(index)
            (ace_type_flags, ace_flags), mask, sid = ace
            sid_string = win32security.ConvertSidToStringSid(sid)
            try:
                name, domain, _type = win32security.LookupAccountSid(None, sid)
                account = f"{domain}\\{name}"
            except Exception:  # pragma: no cover - unresolvable SID (deleted account)
                account = None
            ace_type = "allow" if ace_type_flags == self._ntsecuritycon.ACCESS_ALLOWED_ACE_TYPE else "deny"
            inherited = bool(ace_flags & win32security.INHERITED_ACE)
            records.append(
                AceRecord(sid=sid_string, account=account, access_mask=mask, ace_type=ace_type, inherited=inherited)
            )
        return tuple(records)

    def has_write_bits(self, access_mask: int) -> bool:
        ntsecuritycon = self._ntsecuritycon
        write_mask = ntsecuritycon.FILE_GENERIC_WRITE | ntsecuritycon.FILE_ALL_ACCESS | ntsecuritycon.DELETE
        return bool(access_mask & write_mask)


# -- Plan / apply / verify orchestration --------------------------------------------------------


def plan(deployment_root: Path, *, qualification_roots: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read-only: what would change. Never mutates the filesystem."""
    engine = WindowsAclEngine()
    classified = classify_deployment_paths(deployment_root, qualification_roots=qualification_roots)
    if not engine.available:
        return {
            "policy_version": POLICY_VERSION,
            "platform_supported": False,
            "reason": engine.unavailable_reason,
            "paths": [item.model_dump(mode="json") for item in classified],
        }
    operator_sid = engine.operator_sid()
    changes = []
    for item in classified:
        path = Path(item.path)
        if not path.exists():
            changes.append({"path": item.path, "path_class": item.path_class.value, "action": "skip_missing"})
            continue
        try:
            current = engine.read_aces(path)
        except AclPolicyError as error:
            changes.append({"path": item.path, "path_class": item.path_class.value, "action": "error", "detail": str(error)})
            continue
        disallowed = [
            record.model_dump(mode="json")
            for record in current
            if record.sid in DISALLOWED_WRITE_PRINCIPAL_SIDS and engine.has_write_bits(record.access_mask)
        ]
        changes.append(
            {
                "path": item.path,
                "path_class": item.path_class.value,
                "action": "protect_and_reset_dacl",
                "current_disallowed_write_principals": disallowed,
                "expected_principals": {
                    "administrators": SID_ADMINISTRATORS,
                    "system": SID_SYSTEM,
                    "operator": operator_sid,
                },
            }
        )
    return {
        "policy_version": POLICY_VERSION,
        "platform_supported": True,
        "operator_sid": operator_sid,
        "paths": changes,
    }


def apply(
    deployment_root: Path,
    *,
    actor: str,
    deployment_id: str | None = None,
    qualification_roots: tuple[Path, ...] = (),
) -> AclApplicationRecord:
    """Applies the policy. Fails closed: if any *critical*-class path fails to apply, the record
    is `partial_failure` and the caller must not present it as success (Safety and lockout
    prevention; Apply requirements)."""
    if is_dangerous_root(deployment_root):
        raise AclPolicyError(f"refusing to apply ACL policy to a dangerous root: {deployment_root}")
    engine = WindowsAclEngine()
    if not engine.available:
        raise AclPolicyError(engine.unavailable_reason or "windows ACL engine unavailable")
    operator_sid = engine.operator_sid()
    classified = classify_deployment_paths(deployment_root, qualification_roots=qualification_roots)
    planned = tuple(item.path for item in classified)
    applied: list[str] = []
    failed: list[str] = []
    # Safe ordering: apply non-critical/rebuildable paths first, then evidence/runtime, then the
    # top-level administration root last, so a failure never leaves the operator locked out of a
    # path they still needed mid-run to diagnose the failure.
    ordering = {
        AclPathClass.REBUILDABLE: 0,
        AclPathClass.QUALIFICATION_EVIDENCE: 1,
        AclPathClass.AUTHORITATIVE_EVIDENCE: 2,
        AclPathClass.OPERATOR_RUNTIME: 3,
        AclPathClass.ADMINISTRATION: 4,
    }
    ordered = sorted(classified, key=lambda item: ordering[item.path_class])
    for item in ordered:
        path = Path(item.path)
        if not path.exists():
            continue
        try:
            engine.apply_path(path, item.path_class, operator_sid)
            applied.append(item.path)
        except Exception as error:  # noqa: BLE001 - durable failure evidence, not a crash
            failed.append(f"{item.path}: {type(error).__name__}: {error}")
    result = "success" if not failed else "partial_failure"
    record = AclApplicationRecord(
        applied_at=datetime.now(timezone.utc).isoformat(),
        actor=actor,
        deployment_id=deployment_id,
        policy_version=POLICY_VERSION,
        operator_sid=operator_sid,
        paths_planned=planned,
        paths_applied=tuple(applied),
        paths_failed=tuple(failed),
        result=result,
    )
    _append_operation_log(deployment_root, record)
    _write_policy_state(deployment_root, record)
    return record


def verify(deployment_root: Path, *, qualification_roots: tuple[Path, ...] = ()) -> AclVerificationResult:
    engine = WindowsAclEngine()
    classified = classify_deployment_paths(deployment_root, qualification_roots=qualification_roots)
    checked_at = datetime.now(timezone.utc).isoformat()
    if not engine.available:
        return AclVerificationResult(
            checked_at=checked_at,
            policy_version=POLICY_VERSION,
            platform_supported=False,
            operator_sid=None,
            paths_checked=0,
            findings=(),
            processing_blocked=False,
        )
    policy_state = _read_policy_state(deployment_root)
    operator_sid = policy_state.get("operator_sid") if policy_state else None
    findings: list[AclDriftFinding] = []
    checked = 0
    for item in classified:
        path = Path(item.path)
        if not path.exists():
            continue
        checked += 1
        critical = item.path_class in CRITICAL_PATH_CLASSES
        if policy_state is None:
            findings.append(
                AclDriftFinding(
                    path=item.path,
                    path_class=item.path_class,
                    critical=False,
                    code="policy_not_applied",
                    detail="no ACL policy application record found; run `archivetrust-admin acl-apply` for this deployment root",
                )
            )
            continue
        try:
            aces = engine.read_aces(path)
        except AclPolicyError as error:
            findings.append(
                AclDriftFinding(path=item.path, path_class=item.path_class, critical=critical, code="verification_unavailable", detail=str(error))
            )
            continue
        for record in aces:
            if record.sid in DISALLOWED_WRITE_PRINCIPAL_SIDS and engine.has_write_bits(record.access_mask):
                findings.append(
                    AclDriftFinding(
                        path=item.path,
                        path_class=item.path_class,
                        critical=critical,
                        code="unauthorized_principal_write",
                        detail=f"{record.account or record.sid} has write access via {'inherited' if record.inherited else 'explicit'} ACE",
                    )
                )
        allowed_sids = {SID_ADMINISTRATORS, SID_SYSTEM, operator_sid} if operator_sid else {SID_ADMINISTRATORS, SID_SYSTEM}
        if operator_sid and not any(record.sid == operator_sid for record in aces):
            findings.append(
                AclDriftFinding(
                    path=item.path,
                    path_class=item.path_class,
                    critical=critical,
                    code="operator_access_missing",
                    detail="operator SID has no ACE on this protected path",
                )
            )
        if not any(record.sid == SID_ADMINISTRATORS for record in aces):
            findings.append(
                AclDriftFinding(
                    path=item.path,
                    path_class=item.path_class,
                    critical=critical,
                    code="administrator_recovery_access_missing",
                    detail="BUILTIN\\Administrators has no ACE on this protected path",
                )
            )
    processing_blocked = any(finding.critical for finding in findings)
    return AclVerificationResult(
        checked_at=checked_at,
        policy_version=POLICY_VERSION,
        platform_supported=True,
        operator_sid=operator_sid,
        paths_checked=checked,
        findings=tuple(findings),
        processing_blocked=processing_blocked,
    )


# -- Durable evidence ----------------------------------------------------------------------------

_LOCK = threading.Lock()


def _policy_state_path(deployment_root: Path) -> Path:
    return deployment_root / "worker" / "acl" / "policy_state.json"


def _operation_log_path(deployment_root: Path) -> Path:
    return deployment_root / "worker" / "acl" / "acl_operations.jsonl"


def _write_policy_state(deployment_root: Path, record: AclApplicationRecord) -> None:
    path = _policy_state_path(deployment_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        descriptor, temp_name = tempfile.mkstemp(dir=path.parent)
        temp = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(record.model_dump_json(indent=2))
            temp.replace(path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise


def _read_policy_state(deployment_root: Path) -> dict[str, Any] | None:
    path = _policy_state_path(deployment_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _append_operation_log(deployment_root: Path, record: AclApplicationRecord) -> None:
    path = _operation_log_path(deployment_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")
