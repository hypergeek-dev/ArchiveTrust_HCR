"""Provider-neutral checks for VLM responses that try to leave their output contract."""

from __future__ import annotations

import json


class PromptInjectionSuspected(ValueError):
    """A model response attempted tool use or instruction control."""


_CONTROL_KEYS = frozenset(
    {"tool_calls", "tool_call", "function_call", "commands", "shell", "execute", "instructions"}
)


def reject_control_output(raw_text: str) -> None:
    """Reject valid JSON control structures; ordinary schema errors remain importer rejections."""

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    forbidden = _CONTROL_KEYS.intersection(key.lower() for key in payload)
    if forbidden:
        raise PromptInjectionSuspected(
            f"model response attempted a forbidden control field: {sorted(forbidden)[0]}"
        )
