"""Supported ArchiveTrust administration command surface."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from archivetrust.admin.backup import create_backup, restore_backup, verify_backup
from archivetrust.admin.configuration import (
    environment_reproducibility_report,
    verify_configuration_lock,
    write_configuration_lock,
)
from archivetrust.admin.identity import IdentityRole, LocalAccount, LocalAccountStore
from archivetrust.admin.health import create_diagnostic_bundle, deployment_health
from archivetrust.admin.telemetry_freeze import freeze_workspace_telemetry
from archivetrust.operations.monitoring import AlertStore, record_health_alerts
from archivetrust.workspace.store import WorkspaceStore
from archivetrust.worker import acl_policy


def _workspace_root(deployment_root: Path, identity: str) -> tuple[str, Path]:
    store = WorkspaceStore(deployment_root / "workspaces")
    workspace = next(
        (item for item in store.list() if item.id == identity or item.name == identity), None
    )
    if workspace is None:
        raise SystemExit(f"workspace {identity!r} not found")
    return workspace.id, store.layout_for(workspace.id).root


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"regenerate-current-state", "reconcile-runs"}:
        from archivetrust.admin.regenerate import main as regenerate_main

        return regenerate_main(argv)
    if argv and argv[0] in {"qualification", "qualification-run", "qualification-status"}:
        from archivetrust.admin.qualification import main as qualification_main

        return qualification_main(argv)

    parser = argparse.ArgumentParser(prog="archivetrust-admin")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("regenerate-current-state", help="supersede stale canonical state")
    subparsers.add_parser("reconcile-runs", help="close historical incomplete processing runs")
    subparsers.add_parser("qualification", help="open qualification runner menu")
    subparsers.add_parser("qualification-run", help="run or resume qualification from config")
    subparsers.add_parser("qualification-status", help="show latest qualification run progress")
    backup = subparsers.add_parser("backup")
    backup.add_argument("--deployment-root", default="archivetrust_data")
    backup.add_argument("--output", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("backup")
    restore.add_argument("--target", required=True)
    verify = subparsers.add_parser("verify-backup")
    verify.add_argument("backup")
    lock = subparsers.add_parser("lock-config")
    lock.add_argument("workspace")
    lock.add_argument("--deployment-root", default="archivetrust_data")
    lock.add_argument("--replace", action="store_true")
    verify_lock = subparsers.add_parser("verify-config")
    verify_lock.add_argument("workspace")
    verify_lock.add_argument("--deployment-root", default="archivetrust_data")
    environment = subparsers.add_parser("environment-report")
    environment.add_argument("--output", required=True)
    health = subparsers.add_parser("health")
    health.add_argument("--deployment-root", default="archivetrust_data")
    diagnostics = subparsers.add_parser("diagnostic-bundle")
    diagnostics.add_argument("--deployment-root", default="archivetrust_data")
    diagnostics.add_argument("--output", required=True)
    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--deployment-root", default="archivetrust_data")
    freeze = subparsers.add_parser("freeze-telemetry-history")
    freeze.add_argument("workspace")
    freeze.add_argument("--deployment-root", default="archivetrust_data")
    freeze.add_argument("--label", default="recent-stable")
    freeze.add_argument("--stable-since", default="2026-07-16T00:00:00+00:00")
    freeze.add_argument(
        "--basis",
        default="Release baseline 2026-07-16: stable runnable profile documented as Docling + Tesseract.",
    )
    freeze.add_argument("--report", default=None)
    freeze.add_argument("--no-activate", action="store_true")
    acknowledge = subparsers.add_parser("acknowledge-alert")
    acknowledge.add_argument("alert_id")
    acknowledge.add_argument("--actor", required=True)
    acknowledge.add_argument("--deployment-root", default="archivetrust_data")
    bootstrap = subparsers.add_parser("bootstrap-admin")
    bootstrap.add_argument("account_name")
    bootstrap.add_argument("display_name")
    bootstrap.add_argument("--deployment-root", default="archivetrust_data")
    acl_plan = subparsers.add_parser("acl-plan", help="show ACL changes without applying them")
    acl_plan.add_argument("--deployment-root", default="archivetrust_data")
    acl_apply = subparsers.add_parser("acl-apply", help="apply the least-privilege Windows ACL policy")
    acl_apply.add_argument("--deployment-root", default="archivetrust_data")
    acl_apply.add_argument("--actor", required=True)
    acl_apply.add_argument("--yes", action="store_true", help="apply without an interactive confirmation prompt")
    acl_verify = subparsers.add_parser("acl-verify", help="compare actual ACLs against policy")
    acl_verify.add_argument("--deployment-root", default="archivetrust_data")
    args = parser.parse_args(argv)

    if args.command == "backup":
        result = create_backup(Path(args.deployment_root), Path(args.output))
    elif args.command == "restore":
        result = restore_backup(Path(args.backup), Path(args.target))
    elif args.command == "verify-backup":
        result = verify_backup(Path(args.backup))
    elif args.command in {"lock-config", "verify-config"}:
        workspace_id, root = _workspace_root(Path(args.deployment_root), args.workspace)
        result = (
            write_configuration_lock(root, workspace_id=workspace_id, replace=args.replace)
            if args.command == "lock-config"
            else verify_configuration_lock(root)
        )
    elif args.command == "environment-report":
        result = environment_reproducibility_report()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "health":
        result = deployment_health(Path(args.deployment_root))
    elif args.command == "diagnostic-bundle":
        result = {"path": str(create_diagnostic_bundle(Path(args.deployment_root), Path(args.output)))}
    elif args.command == "monitor":
        root = Path(args.deployment_root)
        store = AlertStore(root / "operations" / "alerts.jsonl")
        created = record_health_alerts(deployment_health(root), store)
        result = {"created": [item.model_dump(mode="json") for item in created], "active": [item.model_dump(mode="json") for item in store.active()]}
    elif args.command == "freeze-telemetry-history":
        workspace_id, _ = _workspace_root(Path(args.deployment_root), args.workspace)
        result = freeze_workspace_telemetry(
            WorkspaceStore(Path(args.deployment_root) / "workspaces").layout_for(workspace_id),
            workspace_id=workspace_id,
            label=args.label,
            stable_since=args.stable_since,
            stable_basis=args.basis,
            activate=not args.no_activate,
        )
        if args.report:
            output = Path(args.report)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    elif args.command == "acknowledge-alert":
        store = AlertStore(Path(args.deployment_root) / "operations" / "alerts.jsonl")
        result = store.acknowledge(args.alert_id, actor=args.actor)
    elif args.command == "acl-plan":
        result = acl_policy.plan(Path(args.deployment_root))
    elif args.command == "acl-apply":
        root = Path(args.deployment_root)
        if acl_policy.is_dangerous_root(root):
            raise SystemExit(f"refusing to apply ACL policy to a dangerous root: {root}")
        if not args.yes:
            confirmation = input(
                f"Apply least-privilege ACL policy {acl_policy.POLICY_VERSION} to {root}? [y/N] "
            )
            if confirmation.strip().lower() not in {"y", "yes"}:
                raise SystemExit("acl-apply cancelled: confirmation not given (use --yes for non-interactive)")
        from archivetrust.worker.security import DeploymentIdentityProvider

        deployment_id = DeploymentIdentityProvider(root).ensure().deployment_id
        record = acl_policy.apply(root, actor=args.actor, deployment_id=deployment_id)
        result = record
        if record.result != "success":
            print(record.model_dump_json(indent=2))
            raise SystemExit(f"acl-apply completed with partial_failure: {len(record.paths_failed)} path(s) failed")
    elif args.command == "acl-verify":
        verification = acl_policy.verify(Path(args.deployment_root))
        result = verification
        print(result.model_dump_json(indent=2) if hasattr(result, "model_dump_json") else json.dumps(result, indent=2))
        return 0 if verification.status in {"pass", "unsupported_platform"} else 1
    else:
        store = LocalAccountStore(Path(args.deployment_root) / "identity" / "accounts.json")
        if store.list():
            raise SystemExit("bootstrap-admin is allowed only when no accounts exist")
        password = getpass.getpass("New administrator password: ")
        confirmation = getpass.getpass("Repeat password: ")
        if password != confirmation:
            raise SystemExit("passwords did not match")
        result = store.add(
            LocalAccount.with_password(
                password=password,
                display_name=args.display_name,
                reviewer_ref=args.account_name,
                roles=tuple(IdentityRole),
            )
        )
    print(result.model_dump_json(indent=2) if hasattr(result, "model_dump_json") else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
