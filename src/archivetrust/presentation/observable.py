"""Minimal, framework-independent binding primitives for ViewModels.

`Observable` is the one mechanism a View uses to learn that ViewModel state changed. It is
deliberately tiny — a value plus a set of subscribers notified on change — so ViewModels can be
tested headless (no Qt event loop) and so the Qt layer stays a thin adapter that forwards these
notifications into Qt signals rather than owning any logic.

This is not a reactive framework; it is the smallest thing that lets `tests/presentation/` observe
every ViewModel state transition without a GUI, which is what keeps HR-6 (reproducible, testable
behavior) true of the presentation layer as well as the services beneath it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Observable(Generic[T]):
    """A single observable value. Subscribers are called with the new value on every change.

    Change is detected by identity/equality (`!=`): setting a value equal to the current one does
    not notify, so a View is not churned by no-op writes. Subscription returns an unsubscribe
    callable; subscribers are held in insertion order and notified in that order for determinism.
    """

    def __init__(self, initial: T) -> None:
        self._value: T = initial
        self._subscribers: list[Callable[[T], None]] = []

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        if new_value != self._value:
            self._value = new_value
            for subscriber in tuple(self._subscribers):
                subscriber(new_value)

    def subscribe(self, callback: Callable[[T], None], *, immediate: bool = False) -> Callable[[], None]:
        """Register a callback. With `immediate=True`, it is called once now with the current
        value (the common "bind and render initial state" case). Returns an unsubscribe callable.
        """
        self._subscribers.append(callback)
        if immediate:
            callback(self._value)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe


class Command:
    """A bindable action with an optional enabled-predicate — the uniform surface a View binds a
    button or menu item to, so the View never contains the decision of *whether* an action is
    available (that stays in the ViewModel, HR-12/§14: the View only orchestrates).
    """

    def __init__(
        self,
        execute: Callable[..., None],
        *,
        can_execute: Callable[[], bool] | None = None,
    ) -> None:
        self._execute = execute
        self._can_execute = can_execute

    def can_execute(self) -> bool:
        return self._can_execute() if self._can_execute is not None else True

    def execute(self, *args: object, **kwargs: object) -> None:
        if not self.can_execute():
            return
        self._execute(*args, **kwargs)
