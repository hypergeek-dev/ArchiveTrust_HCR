from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from archivetrust.worker.models import (
    WorkerCommand,
    WorkerCommandResult,
    WorkerCommandSource,
    WorkerControlAction,
)
from archivetrust.worker.security import (
    WorkerCommandAuthorizer,
    WorkerResultVerifier,
    write_json_atomic,
)
from archivetrust.admin.configuration import write_configuration_lock


class LocalWorkerHandle:
    def __init__(self, *, root: Path, command: WorkerCommand, process, authorizer: WorkerCommandAuthorizer | None = None, session=None) -> None:
        self.root = root
        self.command = command
        self.process = process
        self._authorizer = authorizer
        self._session = session

    @property
    def result_path(self) -> Path:
        return self.root / "results" / f"{self.command.command_id}.json"

    def result(self) -> WorkerCommandResult | None:
        if not self.result_path.exists():
            return None
        result = WorkerCommandResult.model_validate_json(
            self.result_path.read_text(encoding="utf-8")
        )
        return WorkerResultVerifier().verify(result, command=self.command)

    def is_running(self) -> bool:
        result = self.result()
        return result is None or result.status.value in ("queued", "running")

    def pause(self) -> None:
        self._control("pause")

    def resume(self) -> None:
        self._control("resume")

    def cancel(self) -> None:
        self._control("cancel")

    def _control(self, action: str) -> None:
        if self._authorizer is None:
            raise RuntimeError("worker controls require an authenticated command authorizer")
        control = self.root / "control"
        control.mkdir(parents=True, exist_ok=True)
        request = self._authorizer.create_control(
            command_id=self.command.command_id,
            action=WorkerControlAction(action),
            session=self._session,
        )
        write_json_atomic(control / f"{self.command.command_id}.{action}.json", request)


class LocalWorkerClient:
    def __init__(self, *, deployment_root: Path, workspace_id: str, session=None) -> None:
        self.deployment_root = deployment_root
        self.workspace_id = workspace_id
        self.root = deployment_root / "worker"
        self.session = session

    def start_process_queue(
        self,
        *,
        source: WorkerCommandSource = WorkerCommandSource.DESKTOP,
        session=None,
        campaign_id: str | None = None,
        continuation_id: str | None = None,
        run_profile_digest: str | None = None,
    ) -> LocalWorkerHandle:
        workspace_root = self.deployment_root / "workspaces" / self.workspace_id
        write_configuration_lock(workspace_root, workspace_id=self.workspace_id)
        authorizer = WorkerCommandAuthorizer(self.deployment_root)
        command = authorizer.create_command(
            workspace_id=self.workspace_id,
            source=source,
            session=session or self.session,
            campaign_id=campaign_id,
            continuation_id=continuation_id,
            run_profile_digest=run_profile_digest,
        )
        commands = self.root / "commands"
        commands.mkdir(parents=True, exist_ok=True)
        command_path = commands / f"{command.command_id}.json"
        write_json_atomic(command_path, command)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "archivetrust.worker",
                "--deployment-root",
                str(self.deployment_root),
                "--command-id",
                command.command_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return LocalWorkerHandle(root=self.root, command=command, process=process, authorizer=authorizer, session=session or self.session)

    def latest_results(self) -> tuple[WorkerCommandResult, ...]:
        results = self.root / "results"
        if not results.exists():
            return ()
        return tuple(
            WorkerCommandResult.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(results.glob("*.json"))
        )
