from __future__ import annotations

from archivetrust.presentation.observable import Command, Observable


def test_subscriber_notified_on_change() -> None:
    obs: Observable[int] = Observable(0)
    seen: list[int] = []
    obs.subscribe(seen.append)
    obs.value = 1
    obs.value = 2
    assert seen == [1, 2]


def test_no_notification_on_equal_write() -> None:
    obs: Observable[str] = Observable("a")
    seen: list[str] = []
    obs.subscribe(seen.append)
    obs.value = "a"  # equal — must not churn the View
    assert seen == []


def test_immediate_subscription_fires_now() -> None:
    obs: Observable[int] = Observable(7)
    seen: list[int] = []
    obs.subscribe(seen.append, immediate=True)
    assert seen == [7]


def test_unsubscribe_stops_notifications() -> None:
    obs: Observable[int] = Observable(0)
    seen: list[int] = []
    unsubscribe = obs.subscribe(seen.append)
    obs.value = 1
    unsubscribe()
    obs.value = 2
    assert seen == [1]


def test_command_respects_can_execute() -> None:
    calls: list[tuple] = []
    enabled = {"ok": False}
    command = Command(lambda *a: calls.append(a), can_execute=lambda: enabled["ok"])
    command.execute("x")
    assert calls == []  # disabled -> no-op
    enabled["ok"] = True
    command.execute("x")
    assert calls == [("x",)]
