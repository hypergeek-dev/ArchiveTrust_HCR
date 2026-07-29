"""`archivetrust-bootstrap` -- the Python logic behind `scripts/setup-archivetrust.ps1`.

PowerShell stays the operator-facing entry point and does the OS-level orchestration (elevation
re-launch, Docker Desktop install triggers, reboot handling); this CLI owns the durable
install-state file and the preflight checks, both of which are easier to test and reason about in
Python than in PowerShell. See `docs/WINDOWS_BOOTSTRAP.md`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from archivetrust.bootstrap.install_state import (
    IMPLEMENTED_PHASES,
    InstallState,
    InstallStateCorruptionError,
    InstallStateStore,
    Phase,
    PhaseStatus,
)
from archivetrust.bootstrap.preflight import PreflightStatus, real_command_runner, run_all


def _default_deployment_root() -> Path:
    return Path.cwd() / "archivetrust_data"


def cmd_preflight(args: argparse.Namespace) -> int:
    deployment_root = Path(args.deployment_root)
    checks = run_all(
        installation_path=Path(args.installation_path or deployment_root),
        deployment_root=deployment_root,
        env_path=Path(args.env_path) if args.env_path else deployment_root / ".env",
        ports=tuple(args.port or ()),
        runner=real_command_runner,
    )
    store = InstallStateStore.for_deployment(deployment_root)
    try:
        state = store.load(deployment_root=str(deployment_root))
    except InstallStateCorruptionError as error:
        print(json.dumps({"error": "install_state_corrupted", "detail": str(error)}, indent=2))
        return 2

    blocking = [check for check in checks if check.status not in (PreflightStatus.OK,)]
    status = PhaseStatus.FAILED if any(check.status in (PreflightStatus.MISSING_PREREQUISITE, PreflightStatus.INCOMPATIBLE_VERSION, PreflightStatus.CONFIGURATION_CONFLICT, PreflightStatus.PORT_CONFLICT) for check in checks) else (
        PhaseStatus.COMPLETED if not blocking else PhaseStatus.BLOCKED
    )
    detail = "; ".join(f"{c.name}={c.status.value}" for c in blocking) or "all checks ok"
    state = state.with_phase_result(Phase.PREFLIGHT, status, detail=detail[:2000], implemented=True)
    store.save(state)

    payload = {
        "checks": [dataclasses.asdict(check) | {"status": check.status.value} for check in checks],
        "overall_status": status.value,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if status is PhaseStatus.COMPLETED else 1


def cmd_status(args: argparse.Namespace) -> int:
    deployment_root = Path(args.deployment_root)
    store = InstallStateStore.for_deployment(deployment_root)
    try:
        state = store.load(deployment_root=str(deployment_root))
    except InstallStateCorruptionError as error:
        print(json.dumps({"error": "install_state_corrupted", "detail": str(error)}, indent=2))
        return 2
    report = {
        "schema_version": state.schema_version,
        "deployment_root": state.deployment_root,
        "updated_at": state.updated_at,
        "phases": {
            phase.value: _phase_summary(state, phase) for phase in Phase
        },
        "overall_blocked": state.overall_blocked(),
        "reboot_required": state.reboot_required(),
    }
    print(json.dumps(report, indent=2))
    return 0


def _phase_summary(state: InstallState, phase: Phase) -> dict:
    record = state.phase_record(phase)
    return {
        "status": record.status.value,
        "implemented": phase in IMPLEMENTED_PHASES,
        "attempts": record.attempts,
        "updated_at": record.updated_at,
        "detail": record.detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="archivetrust-bootstrap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="run preflight checks and record the result")
    preflight.add_argument("--deployment-root", default=str(_default_deployment_root()))
    preflight.add_argument("--installation-path", default=None)
    preflight.add_argument("--env-path", default=None)
    preflight.add_argument("--port", action="append", type=int, default=[])
    preflight.set_defaults(func=cmd_preflight)

    status = subparsers.add_parser("status", help="print the current install-state phases")
    status.add_argument("--deployment-root", default=str(_default_deployment_root()))
    status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
