"""Tests `load_local_env` (`clients/desktop/main.py`) in isolation -- never touches the real repo
`.env` or prints any secret value; uses a throwaway variable in a temp directory instead.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("dotenv")

from archivetrust.clients.desktop.main import load_local_env  # noqa: E402


def test_loads_a_dot_env_file_from_the_working_directory(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("ARCHIVETRUST_TEST_PROBE=loaded\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARCHIVETRUST_TEST_PROBE", raising=False)

    found = load_local_env()

    assert found is True
    assert os.environ["ARCHIVETRUST_TEST_PROBE"] == "loaded"


def test_never_overrides_an_already_set_environment_variable(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("ARCHIVETRUST_TEST_PROBE=from_dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARCHIVETRUST_TEST_PROBE", "from_real_environment")

    load_local_env()

    assert os.environ["ARCHIVETRUST_TEST_PROBE"] == "from_real_environment"


def test_is_a_silent_no_op_when_no_dot_env_file_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    found = load_local_env()  # must not raise

    assert found is False
