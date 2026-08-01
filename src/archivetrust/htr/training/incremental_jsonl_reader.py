"""`read_new_lines`: reads only the bytes appended to a JSONL file since a previous call, so the
dashboard never re-parses a whole growing `events.jsonl`/`gpu_samples.jsonl` on every 5-15s refresh
(brief §"Performance requirements"). A malformed individual line is skipped and counted, never
allowed to make the whole read fail -- the file keeps growing for the lifetime of a multi-day
training run, and one corrupted line (e.g. a write torn by a concurrent process crash) must not take
down every future read of it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class JsonlReadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: tuple[dict, ...]
    new_offset: int
    malformed_count: int


def read_new_lines(path: str | Path, *, since_offset: int = 0) -> JsonlReadResult:
    """Reads whole lines starting at byte `since_offset`. If the file is shorter than
    `since_offset` (truncated/replaced since the last read -- e.g. a new run reusing a path a
    caller's cursor was still pointed at), starts over from the beginning rather than raising, since
    a stale offset pointing past the real end of a smaller file is not a malformed-data problem."""
    path = Path(path)
    if not path.exists():
        return JsonlReadResult(rows=(), new_offset=0, malformed_count=0)

    size = path.stat().st_size
    start = since_offset if since_offset <= size else 0

    rows: list[dict] = []
    malformed_count = 0
    with path.open("rb") as f:
        f.seek(start)
        raw = f.read()

    # Only fully-terminated lines are consumed -- a line still being written (no trailing "\n" yet)
    # is left for the next call, so a reader never sees a torn, partially-written JSON object.
    last_newline = raw.rfind(b"\n")
    if last_newline == -1:
        return JsonlReadResult(rows=(), new_offset=start, malformed_count=0)

    complete_bytes = raw[: last_newline + 1]
    new_offset = start + len(complete_bytes)

    for line in complete_bytes.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed_count += 1

    return JsonlReadResult(rows=tuple(rows), new_offset=new_offset, malformed_count=malformed_count)
