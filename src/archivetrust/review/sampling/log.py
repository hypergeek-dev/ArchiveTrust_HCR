"""The sampling decision log: records *why* a slot was selected, before review happens.

Kept as its own append-only stream (mirrors `infrastructure/storage/telemetry_sink.py`'s
JSON-Lines convention) rather than a new field on `HumanCorrectionSubmitted`, so the frozen
Milestone-6 correction contract (`domain/feedback/*`, `domain/telemetry/events.py`) never has to
change for this milestone. Calibration code correlates a later `HumanCorrectionSubmitted.
correction_id` back to the `SamplingDecision` that caused the review to happen, by matching
`semantic_slot_id` (and, once populated, the resulting `correction_id`) -- see
`review/sampling/coordinator.py`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from archivetrust.review.sampling.intent import ReviewIntent


class SamplingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    semantic_slot_id: str
    document_ref: str
    intent: ReviewIntent
    strategy_name: str
    pattern: str
    explanation: str
    sampled_at: str
    """UTC ISO-8601, matching `HumanCorrectionSubmitted.submitted_at`'s convention."""
    correction_id: str | None = None
    """Filled in once the resulting review is actually submitted -- `None` until then."""


@runtime_checkable
class SamplingLogSink(Protocol):
    """Append-only, like `domain.telemetry.sink.TelemetrySink` -- a sampling decision, once
    logged, is never rewritten except to attach the `correction_id` it eventually produced."""

    def append(self, decision: SamplingDecision) -> None: ...

    def record_correction(self, semantic_slot_id: str, correction_id: str) -> None: ...

    def all_decisions(self) -> Iterable[SamplingDecision]: ...


class InMemorySamplingLogSink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._decisions: list[SamplingDecision] = []

    def append(self, decision: SamplingDecision) -> None:
        with self._lock:
            self._decisions.append(decision)

    def record_correction(self, semantic_slot_id: str, correction_id: str) -> None:
        """Attach a submitted correction's id to the most recent undecided `SamplingDecision` for
        this slot, so calibration code can later filter `HumanCorrectionSubmitted` events down to
        the ones a statistically-justified sample actually produced."""
        with self._lock:
            for decision in reversed(self._decisions):
                if decision.semantic_slot_id == semantic_slot_id and decision.correction_id is None:
                    idx = self._decisions.index(decision)
                    self._decisions[idx] = decision.model_copy(update={"correction_id": correction_id})
                    return

    def all_decisions(self) -> Iterable[SamplingDecision]:
        with self._lock:
            return tuple(self._decisions)


class FileSamplingLogSink(InMemorySamplingLogSink):
    """Durable, append-only JSON-Lines file, same convention as `FileTelemetrySink`."""

    def __init__(self, path: Path | str) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        self._decisions = self._load()

    def _load(self) -> list[SamplingDecision]:
        text = self._path.read_text(encoding="utf-8")
        return [
            SamplingDecision.model_validate(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]

    def _rewrite(self) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            for decision in self._decisions:
                handle.write(decision.model_dump_json())
                handle.write("\n")

    def append(self, decision: SamplingDecision) -> None:
        super().append(decision)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(decision.model_dump_json())
            handle.write("\n")

    def record_correction(self, semantic_slot_id: str, correction_id: str) -> None:
        super().record_correction(semantic_slot_id, correction_id)
        self._rewrite()
