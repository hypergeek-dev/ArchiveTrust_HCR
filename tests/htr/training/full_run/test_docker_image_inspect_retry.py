from __future__ import annotations

import subprocess

import pytest

import archivetrust.htr.training.full_run.preflight as pf


class _Result:
    def __init__(self, returncode=0, stdout="[{}]", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(pf.subprocess, "run", lambda *a, **k: _Result())
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)


def _run_with(monkeypatch, side_effects):
    """`side_effects` is a list of _Result or exception instances, consumed one per attempt."""
    calls = {"n": 0}

    def _fake(*a, **k):
        i = calls["n"]; calls["n"] += 1
        item = side_effects[min(i, len(side_effects) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(pf.subprocess, "run", _fake)
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    ok, msg = pf._check_container_image_available("img:tag", None)
    return ok, msg, calls["n"]


def test_succeeds_first_attempt_without_noise(monkeypatch):
    ok, msg, n = _run_with(monkeypatch, [_Result()])
    assert ok is True and n == 1
    assert "succeeded on attempt" not in msg


def test_timeout_then_success_reports_the_transient_rather_than_hiding_it(monkeypatch):
    """Six real occurrences of this timeout all succeeded on retry. Retrying is right; silently
    swallowing the warning is not -- a degrading daemon must stay visible."""
    ok, msg, n = _run_with(monkeypatch, [subprocess.TimeoutExpired("docker", 30.0), _Result()])
    assert ok is True
    assert n == 2
    assert "succeeded on attempt 2" in msg
    assert "timeout" in msg


def test_persistent_timeout_fails_after_the_retry_policy_is_exhausted(monkeypatch):
    ok, msg, n = _run_with(monkeypatch, [subprocess.TimeoutExpired("docker", 30.0)])
    assert ok is False
    assert n == pf.IMAGE_INSPECT_MAX_ATTEMPTS
    assert f"after {pf.IMAGE_INSPECT_MAX_ATTEMPTS} attempts" in msg


def test_missing_image_fails_immediately_without_pointless_retries(monkeypatch):
    """A genuinely absent image will not appear on retry -- retrying only delays the real answer."""
    ok, msg, n = _run_with(monkeypatch, [_Result(returncode=1, stderr="Error: No such image: img:tag")])
    assert ok is False
    assert n == 1
    assert "image_absent" in msg
    assert "docker pull" in msg


def test_daemon_unreachable_is_distinguished_from_a_missing_image(monkeypatch):
    ok, msg, n = _run_with(
        monkeypatch,
        [_Result(returncode=1, stderr="error during connect: cannot connect to the Docker daemon")])
    assert ok is False
    assert "daemon_unreachable" in msg
    assert "image_absent" not in msg


def test_malformed_response_is_distinguished_from_success(monkeypatch):
    """Exit 0 with unparseable output must not be read as 'image present'."""
    ok, msg, n = _run_with(monkeypatch, [_Result(returncode=0, stdout="not json at all")])
    assert ok is False
    assert "malformed_response" in msg


def test_malformed_then_valid_response_recovers(monkeypatch):
    ok, msg, n = _run_with(monkeypatch, [_Result(returncode=0, stdout="garbage"), _Result()])
    assert ok is True
    assert n == 2


def test_docker_cli_missing_fails_without_retrying(monkeypatch):
    ok, msg, n = _run_with(monkeypatch, [OSError("docker not found")])
    assert ok is False
    assert n == 1
    assert "could not be executed" in msg


def test_launch_guard_uses_the_same_retrying_check_not_a_private_copy(monkeypatch):
    """A real false negative: the guard kept its own 15s-timeout copy that mapped *every* failure --
    including a timeout on a contended daemon -- to "image not available", and refused a legitimate
    resume while the pinned image was present locally with the exact expected digest. Both call sites
    must now share one classified implementation."""
    import archivetrust.htr.training.full_run.launch_guard as lg

    attempts = {"n": 0}

    def _fake(*a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise subprocess.TimeoutExpired(cmd="docker", timeout=30.0)
        return _Result()

    monkeypatch.setattr(pf.subprocess, "run", _fake)
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)

    ok, msg = lg._container_image_available("img:tag", None)
    assert ok is True, "a transient timeout must not be reported as a missing image"
    assert attempts["n"] == 2, "the guard must retry, not fail on the first timeout"
    assert "timeout" in msg, "the transient failure must still be surfaced, not silently swallowed"


def test_launch_guard_reports_a_genuinely_absent_image_distinctly(monkeypatch):
    import archivetrust.htr.training.full_run.launch_guard as lg

    monkeypatch.setattr(
        pf.subprocess, "run",
        lambda *a, **k: _Result(returncode=1, stderr="Error: No such image: img:tag"),
    )
    ok, msg = lg._container_image_available("img:tag", None)
    assert ok is False
    assert "image_absent" in msg
    assert "docker pull" in msg, "an absent image needs an actionable fix, unlike a timeout"
