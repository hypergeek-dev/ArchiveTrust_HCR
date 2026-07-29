from __future__ import annotations

from archivetrust.worker.client import LocalWorkerClient
from archivetrust.worker.models import WorkerCommandStatus
from archivetrust.worker.service import execute_command
from archivetrust.workspace.store import WorkspaceStore


def test_worker_command_is_durable_and_executes_outside_desktop_context(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment"
    workspace = WorkspaceStore(deployment / "workspaces").create("Worker test")
    launched = {}

    class _Process:
        pass

    def fake_popen(arguments, **kwargs):
        launched["arguments"] = arguments
        launched["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr("archivetrust.worker.client.subprocess.Popen", fake_popen)
    handle = LocalWorkerClient(
        deployment_root=deployment, workspace_id=workspace.id
    ).start_process_queue()

    assert (deployment / "worker" / "commands" / f"{handle.command.command_id}.json").exists()
    assert "archivetrust.worker" in launched["arguments"]
    assert handle.result() is None

    result = execute_command(
        deployment_root=deployment, command_id=handle.command.command_id
    )
    assert result.status is WorkerCommandStatus.COMPLETED
    assert result.documents_total == 0
    assert handle.result().status is WorkerCommandStatus.COMPLETED


def test_worker_controls_are_durable_files(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment"
    workspace = WorkspaceStore(deployment / "workspaces").create("Worker test")
    monkeypatch.setattr(
        "archivetrust.worker.client.subprocess.Popen", lambda *args, **kwargs: object()
    )
    handle = LocalWorkerClient(
        deployment_root=deployment, workspace_id=workspace.id
    ).start_process_queue()

    handle.pause()
    handle.resume()
    handle.cancel()

    control = deployment / "worker" / "control"
    assert (control / f"{handle.command.command_id}.pause.json").exists()
    assert (control / f"{handle.command.command_id}.resume.json").exists()
    assert (control / f"{handle.command.command_id}.cancel.json").exists()
