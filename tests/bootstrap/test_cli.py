from __future__ import annotations

import json

from archivetrust.bootstrap.cli import main
from archivetrust.bootstrap.install_state import InstallStateStore, Phase, PhaseStatus


def test_preflight_command_records_state_and_prints_json(tmp_path, capsys) -> None:
    root = tmp_path / "deployment"
    exit_code = main(["preflight", "--deployment-root", str(root), "--port", "58301"])
    output = json.loads(capsys.readouterr().out)
    assert "checks" in output
    assert "overall_status" in output
    store = InstallStateStore.for_deployment(root)
    state = store.load()
    record = state.phase_record(Phase.PREFLIGHT)
    assert record.attempts == 1
    assert record.implemented is True
    assert exit_code in (0, 1)


def test_preflight_command_is_idempotent_in_attempts(tmp_path, capsys) -> None:
    root = tmp_path / "deployment"
    main(["preflight", "--deployment-root", str(root)])
    capsys.readouterr()
    main(["preflight", "--deployment-root", str(root)])
    capsys.readouterr()
    store = InstallStateStore.for_deployment(root)
    state = store.load()
    assert state.phase_record(Phase.PREFLIGHT).attempts == 2


def test_status_command_reports_every_phase(tmp_path, capsys) -> None:
    root = tmp_path / "deployment"
    main(["preflight", "--deployment-root", str(root)])
    capsys.readouterr()
    exit_code = main(["status", "--deployment-root", str(root)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert set(output["phases"].keys()) == {phase.value for phase in Phase}
    assert output["phases"][Phase.PREFLIGHT.value]["implemented"] is True
    assert output["phases"][Phase.DOCKER_DESKTOP_DETECTED.value]["implemented"] is False


def test_status_command_reports_corruption_without_crashing(tmp_path, capsys) -> None:
    root = tmp_path / "deployment"
    store = InstallStateStore.for_deployment(root)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("not json", encoding="utf-8")
    exit_code = main(["status", "--deployment-root", str(root)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["error"] == "install_state_corrupted"


def test_status_on_fresh_root_without_preflight_reports_not_started(tmp_path, capsys) -> None:
    root = tmp_path / "deployment"
    exit_code = main(["status", "--deployment-root", str(root)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["phases"][Phase.PREFLIGHT.value]["status"] == PhaseStatus.NOT_STARTED.value
