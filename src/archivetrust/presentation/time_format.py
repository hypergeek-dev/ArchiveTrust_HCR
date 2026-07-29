"""Small date/time formatting helpers for operator-facing UI."""

from __future__ import annotations

from datetime import datetime, timezone


def format_timestamp(value: datetime | str | None, *, now: datetime | None = None) -> str:
    if value is None:
        return "never"
    parsed = _parse_datetime(value)
    if parsed is None:
        return str(value)

    reference = now or datetime.now(parsed.tzinfo or timezone.utc)
    if parsed.date() == reference.date():
        return parsed.strftime("%H:%M")
    if parsed.year == reference.year:
        return parsed.strftime("%d %b, %H:%M")
    return parsed.strftime("%d %b %Y, %H:%M")


def _parse_datetime(value: datetime | str) -> datetime | None:
    if isinstance(value, datetime):
        return value
    raw = value.strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
