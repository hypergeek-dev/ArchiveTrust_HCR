from __future__ import annotations

from datetime import datetime, timezone

from archivetrust.presentation.time_format import format_timestamp


def test_format_timestamp_uses_short_operator_friendly_dates() -> None:
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

    assert format_timestamp("2026-07-16T10:30:00+00:00", now=now) == "10:30"
    assert format_timestamp("2026-07-15T10:30:00+00:00", now=now) == "15 Jul, 10:30"
    assert format_timestamp("2025-07-15T10:30:00+00:00", now=now) == "15 Jul 2025, 10:30"
    assert format_timestamp(None, now=now) == "never"


def test_format_timestamp_preserves_unparseable_values() -> None:
    assert format_timestamp("scan pending") == "scan pending"
