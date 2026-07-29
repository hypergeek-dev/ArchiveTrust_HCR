from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from archivetrust.acquisition.queue_runner import run_queue
from archivetrust.composition import AppContext
from archivetrust.worker.models import (
    WorkerCommand,
    WorkerCommandResult,
    WorkerCommandStatus,
    WorkerRejectionReason,
    utc_now,
)
from archivetrust.worker.security import (
    CommandFreshnessStore,
    DeploymentIdentityProvider,
    WorkerCommandVerifier,
    WorkerControlVerifier,
    WorkerSecurityError,
    read_control_request,
    result_with_digest,
)


def execute_command(*, deployment_root: Path, command_id: str) -> WorkerCommandResult:
    worker_root = deployment_root / "worker"
    command_path = worker_root / "commands" / f"{command_id}.json"
    results = worker_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    result_path = results / f"{command_id}.json"
    identity = DeploymentIdentityProvider(deployment_root).load()
    freshness = CommandFreshnessStore.for_deployment(deployment_root)

    try:
        command = WorkerCommand.model_validate_json(command_path.read_text(encoding="utf-8"))
        verified = WorkerCommandVerifier(deployment_root, freshness=freshness).verify(command)
    except WorkerSecurityError as error:
        freshness.reject_command(command_id)
        result = WorkerCommandResult(
            command_id=command_id,
            workspace_id="unknown",
            status=WorkerCommandStatus.REJECTED,
            recorded_at=utc_now(),
            rejection_reason=error.reason,
            error=error.reason.value,
            verification_status="rejected",
        )
        _write_result(result_path, result_with_digest(result, identity))
        return result
    except Exception:
        freshness.reject_command(command_id)
        result = WorkerCommandResult(
            command_id=command_id,
            workspace_id="unknown",
            status=WorkerCommandStatus.REJECTED,
            recorded_at=utc_now(),
            rejection_reason=WorkerRejectionReason.MALFORMED_COMMAND,
            error="malformed_command",
            verification_status="rejected",
        )
        _write_result(result_path, result_with_digest(result, identity))
        return result

    command = verified.command
    control_verifier = WorkerControlVerifier(deployment_root, freshness=freshness)

    def write(status: WorkerCommandStatus, **fields) -> WorkerCommandResult:
        result = WorkerCommandResult(
            command_id=command.command_id,
            workspace_id=command.workspace_id,
            status=status,
            recorded_at=utc_now(),
            command_digest=verified.command_digest,
            command_schema_id=command.schema_id,
            authorization_mechanism=command.authorization_mechanism,
            authorization_version=command.authorization_version,
            verification_status="accepted",
            verified_at=verified.verified_at,
            verified_configuration_lock=verified.configuration_lock_digest,
            submitted_by_account_id=command.submitted_by_account_id,
            submitted_by_display_name=command.submitted_by_display_name,
            command_source=command.command_source,
            **fields,
        )
        secured = result_with_digest(result, identity)
        _write_result(result_path, secured)
        return secured

    write(WorkerCommandStatus.RUNNING)
    cancelled = threading.Event()
    paused = threading.Event()
    paused.set()
    stopped = threading.Event()

    def monitor() -> None:
        control = worker_root / "control"
        while not stopped.wait(0.2):
            for action in ("cancel", "pause", "resume"):
                request_path = control / f"{command_id}.{action}.json"
                if not request_path.exists():
                    continue
                try:
                    verified_control = control_verifier.verify(
                        read_control_request(request_path),
                        command_id=command_id,
                    )
                except WorkerSecurityError:
                    _record_rejected_control(worker_root, request_path)
                    continue
                if verified_control.request.action.value == "cancel":
                    cancelled.set()
                    paused.set()
                elif verified_control.request.action.value == "pause":
                    paused.clear()
                elif verified_control.request.action.value == "resume":
                    paused.set()

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        context = AppContext(
            deployment_root=deployment_root,
            auto_create_default_workspace=False,
            use_process_worker=False,
        )
        context.open_workspace(command.workspace_id)
        outcome = run_queue(context, paused=paused, cancelled=cancelled)
        result = write(
            WorkerCommandStatus.COMPLETED,
            run_id=outcome.run_id,
            documents_done=outcome.documents_done,
            documents_total=outcome.documents_total,
        )
        _complete_command(freshness, command.command_id, verified)
        return result
    except Exception as error:  # noqa: BLE001 - durable failed command boundary
        result = write(WorkerCommandStatus.FAILED, error=f"{type(error).__name__}: {error}")
        _complete_command(freshness, command.command_id, verified)
        return result
    finally:
        stopped.set()
        monitor_thread.join(timeout=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="archivetrust-worker")
    parser.add_argument("--deployment-root", required=True)
    parser.add_argument("--command-id", required=True)
    args = parser.parse_args(argv)
    result = execute_command(
        deployment_root=Path(args.deployment_root), command_id=args.command_id
    )
    return 0 if result.status is WorkerCommandStatus.COMPLETED else 1


def _write_result(path: Path, result: WorkerCommandResult) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _complete_command(freshness: CommandFreshnessStore, command_id: str, verified) -> None:
    freshness.complete_command(
        command_id,
        verification={
            "command_digest": verified.command_digest,
            "verified_at": verified.verified_at,
            "configuration_lock_digest": verified.configuration_lock_digest,
        },
    )


def _record_rejected_control(worker_root: Path, request_path: Path) -> None:
    log_path = worker_root / "rejected_controls.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": utc_now(),
        "request_path": str(request_path),
        "reason": "invalid_control_request",
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
