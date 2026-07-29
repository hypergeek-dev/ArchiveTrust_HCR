from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.admin.configuration import environment_reproducibility_report
from archivetrust.worker.security import (
    AUTH_MECHANISM,
    AUTH_VERSION,
    CommandFreshnessStore,
    DeploymentIdentityProvider,
    FilesystemSecurityInspector,
)
from archivetrust.infrastructure.storage.integrity import (
    default_hash_chain_sidecar_path,
    verify_hash_chain_sidecar,
)


class HealthCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: str
    detail: str


class DeploymentHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = "archivetrust.deployment_health.v1"
    checked_at: str
    overall: str
    checks: tuple[HealthCheck, ...]


def _writable_check(name: str, directory: Path) -> HealthCheck:
    try:
        if not directory.is_dir():
            return HealthCheck(name=name, status="critical", detail=f"directory is missing: {directory}")
        with tempfile.NamedTemporaryFile(prefix=".health-", dir=directory, delete=True):
            pass
        return HealthCheck(name=name, status="ok", detail=f"writable: {directory}")
    except OSError as exc:
        return HealthCheck(name=name, status="critical", detail=f"not writable: {exc}")


def _acl_policy_check(root: Path) -> HealthCheck:
    """Surfaces the same `acl_policy.verify()` result `FilesystemSecurityInspector` already
    consumes -- no duplicate ACL logic, just more of the fields callers need (policy version,
    critical/noncritical split, recommended action) than the terse `worker_filesystem_boundary`
    string affords."""
    from archivetrust.worker.acl_policy import verify as acl_verify

    if os.name != "nt":
        return HealthCheck(name="acl_policy", status="ok", detail="unsupported_platform: not Windows; ACL policy is Windows-only, no action needed")
    result = acl_verify(root)
    if not result.platform_supported:
        return HealthCheck(name="acl_policy", status="warning", detail="unsupported_platform: pywin32 not importable")
    critical = [f for f in result.findings if f.critical]
    noncritical = [f for f in result.findings if not f.critical]
    if result.processing_blocked:
        action = "run `archivetrust-admin acl-verify` for details, then `archivetrust-admin acl-apply --actor <you>` to re-secure the drifted paths"
        return HealthCheck(
            name="acl_policy",
            status="critical",
            detail=f"policy={result.policy_version}; {len(critical)} critical finding(s), {len(noncritical)} noncritical; action: {action}",
        )
    if not result.findings:
        return HealthCheck(name="acl_policy", status="ok", detail=f"policy={result.policy_version}; {result.paths_checked} path(s) verified, no drift")
    return HealthCheck(
        name="acl_policy",
        status="warning",
        detail=f"policy={result.policy_version}; {len(noncritical)} noncritical finding(s) (e.g. policy not yet applied); run `archivetrust-admin acl-apply` to secure this deployment",
    )


def deployment_health(deployment_root: Path) -> DeploymentHealth:
    root = deployment_root.resolve()
    checks: list[HealthCheck] = [_writable_check("deployment_store", root)]
    DeploymentIdentityProvider(root).ensure()
    security = FilesystemSecurityInspector().inspect(root)
    checks.append(
        HealthCheck(
            name="worker_command_security_mode",
            status="ok",
            detail=f"{AUTH_MECHANISM}:{AUTH_VERSION}",
        )
    )
    checks.append(
        HealthCheck(
            name="worker_filesystem_boundary",
            status="critical" if security.processing_blocked else "warning" if any(item.status == "warning" for item in security.findings) else "ok",
            detail=f"{security.mode}; {len([item for item in security.findings if item.status != 'ok'])} finding(s)",
        )
    )
    checks.append(_acl_policy_check(root))
    freshness = CommandFreshnessStore.for_deployment(root).load()
    checks.append(
        HealthCheck(
            name="worker_replay_state",
            status="ok",
            detail=(
                f"active={len(freshness.active_command_ids)} completed={len(freshness.completed_command_ids)} "
                f"rejected={freshness.rejected_commands} replay_attempts={freshness.replay_attempts}"
            ),
        )
    )
    workspace_roots = tuple((root / "workspaces").glob("*")) if (root / "workspaces").exists() else ()
    for workspace in workspace_roots:
        if not workspace.is_dir():
            continue
        checks.extend(
            (
                _writable_check(f"archive_store:{workspace.name}", workspace / "archive"),
                _writable_check(f"event_store:{workspace.name}", workspace / "telemetry"),
                _writable_check(f"blob_store:{workspace.name}", workspace / "telemetry" / "blobs"),
            )
        )
        event_path = workspace / "telemetry" / "events.jsonl"
        if event_path.exists():
            integrity = verify_hash_chain_sidecar(event_path, default_hash_chain_sidecar_path(event_path))
            checks.append(
                HealthCheck(
                    name=f"integrity:{workspace.name}",
                    status="ok" if integrity.ok else "critical",
                    detail=integrity.reason or "hash chain verified",
                )
            )
        lock = workspace / "config" / "configuration.lock.json"
        checks.append(
            HealthCheck(
                name=f"configuration_lock:{workspace.name}",
                status="ok" if lock.exists() else "warning",
                detail="present" if lock.exists() else "not yet created",
            )
        )
    usage = shutil.disk_usage(root)
    free_ratio = usage.free / usage.total if usage.total else 0
    checks.append(
        HealthCheck(
            name="storage_capacity",
            status="critical" if free_ratio < 0.05 else "warning" if free_ratio < 0.15 else "ok",
            detail=f"{usage.free} bytes free of {usage.total}",
        )
    )
    commands = {path.stem for path in (root / "worker" / "commands").glob("*.json")} if (root / "worker" / "commands").exists() else set()
    results = {path.stem for path in (root / "worker" / "results").glob("*.json")} if (root / "worker" / "results").exists() else set()
    checks.append(
        HealthCheck(
            name="queue_health",
            status="warning" if commands - results else "ok",
            detail=f"{len(commands - results)} command(s) without a result",
        )
    )
    rejected_results = []
    for path in sorted((root / "worker" / "results").glob("*.json")) if (root / "worker" / "results").exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if payload.get("status") == "rejected":
            rejected_results.append(payload.get("rejection_reason") or payload.get("error") or "unknown")
    checks.append(
        HealthCheck(
            name="worker_rejected_commands",
            status="warning" if rejected_results else "ok",
            detail=f"{len(rejected_results)} rejected command(s)",
        )
    )
    backups = sorted((root / "backups").glob("*.atbackup"), key=lambda path: path.stat().st_mtime) if (root / "backups").exists() else []
    checks.append(
        HealthCheck(
            name="backup_freshness",
            status="warning" if not backups else "ok",
            detail="no deployment backup found" if not backups else datetime.fromtimestamp(backups[-1].stat().st_mtime, timezone.utc).isoformat(),
        )
    )
    rank = {"ok": 0, "unknown": 1, "warning": 2, "critical": 3}
    overall = max((item.status for item in checks), key=rank.get, default="unknown")
    return DeploymentHealth(
        checked_at=datetime.now(timezone.utc).isoformat(), overall=overall, checks=tuple(checks)
    )


def create_diagnostic_bundle(deployment_root: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    health = deployment_health(deployment_root)
    security = FilesystemSecurityInspector().inspect(deployment_root)
    freshness = CommandFreshnessStore.for_deployment(deployment_root).load()
    locks = []
    for path in sorted(deployment_root.glob("workspaces/*/config/configuration.lock.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        locks.append(
            {
                "workspace_id": payload.get("workspace_id"),
                "configuration_hash": payload.get("configuration_hash"),
                "files": payload.get("files", []),
            }
        )
    failures = []
    for path in sorted(deployment_root.glob("worker/results/*.json"), reverse=True)[:20]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "failed":
            failures.append(payload)
    from archivetrust.worker.acl_policy import verify as acl_verify

    acl_verification = acl_verify(deployment_root) if os.name == "nt" else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("health.json", health.model_dump_json(indent=2))
        archive.writestr("environment.json", json.dumps(environment_reproducibility_report(), indent=2))
        archive.writestr("worker-security.json", json.dumps({
            "authorization_mechanism": AUTH_MECHANISM,
            "authorization_version": AUTH_VERSION,
            "filesystem": security.model_dump(mode="json"),
            "freshness": freshness.model_dump(
                mode="json",
                exclude={"used_command_nonces", "used_control_nonces"},
            ),
            "secrets_included": False,
        }, indent=2))
        if acl_verification is not None:
            archive.writestr(
                "acl-policy.json",
                json.dumps(
                    {
                        **acl_verification.model_dump(mode="json"),
                        "note": "operator_sid and account SIDs identify a local Windows account, not a document-content secret; drop this file before sharing outside the deployment's own administrators if that identification is undesired.",
                    },
                    indent=2,
                ),
            )
        archive.writestr("configuration-manifests.json", json.dumps(locks, indent=2))
        archive.writestr("recent-worker-failures.json", json.dumps(failures, indent=2))
        archive.writestr(
            "CONTENT.txt",
            "Sanitized ArchiveTrust diagnostic bundle. No archive objects, raw telemetry, annotations, secrets, or document text are included. "
            "acl-policy.json (Windows only) contains local account SIDs for ACL diagnosis, not secrets.\n",
        )
    return destination
