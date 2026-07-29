from __future__ import annotations

from archivetrust.acquisition.provider_health_guard import (
    ProviderHealthGuard,
    ProviderHealthGuardPolicy,
    RecoveryAttempt,
)


def test_all_required_providers_healthy_returns_healthy_with_no_recovery_attempted():
    guard = ProviderHealthGuard(
        required_provider_ids=frozenset({"docling", "tesseract_layoutparser"}),
        probes={"docling": lambda: True, "tesseract_layoutparser": lambda: True},
    )
    result = guard.check()
    assert result.healthy is True
    assert result.recovery_attempts == ()
    assert result.reason == ""
    assert all(c.healthy for c in result.checks)


def test_unhealthy_provider_with_no_recovery_callable_stays_unhealthy():
    guard = ProviderHealthGuard(
        required_provider_ids=frozenset({"docling", "paddleocr-vl"}),
        probes={"docling": lambda: True, "paddleocr-vl": lambda: False},
    )
    result = guard.check()
    assert result.healthy is False
    assert result.recovery_attempts == ()
    assert "paddleocr-vl" in result.reason


def test_recovery_succeeds_on_bounded_retry_makes_guard_healthy():
    state = {"healthy": False, "recover_calls": 0}

    def probe() -> bool:
        return state["healthy"]

    def recover() -> RecoveryAttempt:
        state["recover_calls"] += 1
        state["healthy"] = True
        return RecoveryAttempt(provider_id="paddleocr-vl", attempted=True, succeeded=True, detail="recovered")

    guard = ProviderHealthGuard(
        required_provider_ids=frozenset({"paddleocr-vl"}),
        probes={"paddleocr-vl": probe},
        recovery={"paddleocr-vl": recover},
        policy=ProviderHealthGuardPolicy(recovery_wait_seconds=0.0),
    )
    result = guard.check()
    assert result.healthy is True
    assert len(result.recovery_attempts) == 1
    assert result.recovery_attempts[0].attempted is True
    assert result.recovery_attempts[0].succeeded is True
    assert state["recover_calls"] == 1


def test_recovery_attempted_but_still_unhealthy_is_bounded_not_infinite():
    calls = {"count": 0}

    def recover() -> RecoveryAttempt:
        calls["count"] += 1
        return RecoveryAttempt(provider_id="paddleocr-vl", attempted=True, succeeded=False, detail="still down")

    guard = ProviderHealthGuard(
        required_provider_ids=frozenset({"paddleocr-vl"}),
        probes={"paddleocr-vl": lambda: False},
        recovery={"paddleocr-vl": recover},
        policy=ProviderHealthGuardPolicy(max_recovery_attempts=1, recovery_wait_seconds=0.0),
    )
    result = guard.check()
    assert result.healthy is False
    assert calls["count"] == 1
    assert len(result.recovery_attempts) == 1
    assert result.recovery_attempts[0].succeeded is False


def test_custom_policy_bounds_recovery_attempts_to_exactly_configured_count():
    calls = {"count": 0}

    def recover() -> RecoveryAttempt:
        calls["count"] += 1
        return RecoveryAttempt(provider_id="paddleocr-vl", attempted=True, succeeded=False, detail="still down")

    guard = ProviderHealthGuard(
        required_provider_ids=frozenset({"paddleocr-vl"}),
        probes={"paddleocr-vl": lambda: False},
        recovery={"paddleocr-vl": recover},
        policy=ProviderHealthGuardPolicy(max_recovery_attempts=2, recovery_wait_seconds=0.0),
    )
    guard.check()
    assert calls["count"] == 2


def test_a_provider_shrinking_the_required_set_is_detected_via_its_own_probe_going_unhealthy():
    """The real "3 providers -> 2 providers" scenario: `required_provider_ids` is a fixed
    snapshot taken once at run start, so a provider that becomes unreachable mid-run is still
    checked against, even though a fresh `enabled_provider_ids()` call would no longer include it."""
    guard = ProviderHealthGuard(
        required_provider_ids=frozenset({"docling", "tesseract_layoutparser", "paddleocr-vl"}),
        probes={
            "docling": lambda: True,
            "tesseract_layoutparser": lambda: True,
            "paddleocr-vl": lambda: False,  # became unreachable mid-run
        },
    )
    result = guard.check()
    assert result.healthy is False
    unhealthy_ids = {c.provider_id for c in result.checks if not c.healthy}
    assert unhealthy_ids == {"paddleocr-vl"}


def test_a_provider_with_no_registered_probe_is_reported_unhealthy_not_assumed_available():
    guard = ProviderHealthGuard(
        required_provider_ids=frozenset({"docling"}),
        probes={},
    )
    result = guard.check()
    assert result.healthy is False
    assert result.checks[0].healthy is False
    assert "no health probe registered" in result.checks[0].detail


def test_a_probe_that_raises_is_treated_as_unhealthy_not_a_crash():
    def bad_probe() -> bool:
        raise RuntimeError("boom")

    guard = ProviderHealthGuard(required_provider_ids=frozenset({"docling"}), probes={"docling": bad_probe})
    result = guard.check()
    assert result.healthy is False
    assert "boom" in result.checks[0].detail
