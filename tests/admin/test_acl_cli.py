from __future__ import annotations

import json
import os

import pytest

from archivetrust.admin.cli import main


windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows ACL engine is Windows-only")


def test_acl_plan_help(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["acl-plan", "--help"])


def test_acl_verify_reports_unsupported_platform_on_non_windows(tmp_path, capsys) -> None:
    if os.name == "nt":
        pytest.skip("this exercises the non-Windows unsupported_platform path")
    exit_code = main(["acl-verify", "--deployment-root", str(tmp_path / "deployment")])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["platform_supported"] is False


@windows_only
def test_acl_apply_and_verify_cli_round_trip(tmp_path, capsys) -> None:
    root = tmp_path / "deployment"
    (root / "worker" / "commands").mkdir(parents=True)
    (root / "worker" / "control").mkdir(parents=True)
    (root / "worker" / "results").mkdir(parents=True)

    exit_code = main(["acl-apply", "--deployment-root", str(root), "--actor", "pytest", "--yes"])
    assert exit_code == 0
    apply_output = json.loads(capsys.readouterr().out)
    assert apply_output["result"] == "success"

    exit_code = main(["acl-verify", "--deployment-root", str(root)])
    assert exit_code == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["processing_blocked"] is False
    assert verify_output["findings"] == []


@windows_only
def test_acl_apply_refuses_dangerous_root(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["acl-apply", "--deployment-root", "C:\\", "--actor", "pytest", "--yes"])


@windows_only
def test_acl_apply_without_yes_requires_confirmation(tmp_path, monkeypatch) -> None:
    root = tmp_path / "deployment"
    (root / "worker").mkdir(parents=True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with pytest.raises(SystemExit):
        main(["acl-apply", "--deployment-root", str(root), "--actor", "pytest"])
