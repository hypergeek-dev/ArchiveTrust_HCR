"""The passive review-telemetry sink (ROADMAP_V2.md §9.4, LP-3; HUMAN_REVIEW_SPECIFICATION.md §11).

This is the Learning Platform's *own* telemetry stream — deliberately separate from the Trust
Engine's frozen, timeless knowledge-evolution events (`archivetrust.domain.telemetry`). Review
interactions are temporal (durations, timestamps) and must never be merged into the Trust Engine's
event set (LP-3); this sink is where they live instead.

Append-only, exactly like the Trust Engine's `TelemetrySink` and for the same reason: observed
behavior is a record that accumulates, never one that is rewritten. The interface defines only the
Protocol (Dependency Inversion); the in-memory implementation is the concrete store used by tests
and by any client that has not yet chosen a durable backend.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from archivetrust.learning.review.interaction import ReviewInteraction


@runtime_checkable
class ReviewInteractionSink(Protocol):
    """Append-only store for passive review interactions. Mirrors
    `archivetrust.domain.telemetry.sink.TelemetrySink`, but for the Learning Platform's own
    temporal stream — it has no `update`/`delete`, by design (HR-4: every review action is
    replayable, which append-only is what guarantees).
    """

    def append(self, interaction: ReviewInteraction) -> None:
        """Persist one interaction, preserving insertion order within a `review_id`."""
        ...

    def interactions_for_review(self, review_id: str) -> Iterator[ReviewInteraction]:
        """Every interaction for one review, in append order — the input to `ReviewSession`."""
        ...

    def all_interactions(self) -> Iterable[ReviewInteraction]:
        """Every interaction this sink holds, across all reviews, in append order."""
        ...


class InMemoryReviewInteractionSink:
    """Process-local, non-durable passive-telemetry store. Thread-safe append/read, matching the
    Trust Engine's `InMemoryTelemetrySink` so both streams behave identically where it matters.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_review: dict[str, list[ReviewInteraction]] = defaultdict(list)
        self._all: list[ReviewInteraction] = []

    def append(self, interaction: ReviewInteraction) -> None:
        with self._lock:
            self._by_review[interaction.review_id].append(interaction)
            self._all.append(interaction)

    def interactions_for_review(self, review_id: str) -> Iterator[ReviewInteraction]:
        with self._lock:
            snapshot = tuple(self._by_review.get(review_id, ()))
        return iter(snapshot)

    def all_interactions(self) -> Iterable[ReviewInteraction]:
        with self._lock:
            return tuple(self._all)


class FileReviewInteractionSink:
    """Durable JSONL interaction stream scoped to one Workspace."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def append(self, interaction: ReviewInteraction) -> None:
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(interaction.model_dump_json())
            handle.write("\n")

    def interactions_for_review(self, review_id: str) -> Iterator[ReviewInteraction]:
        return iter(
            tuple(
                interaction
                for interaction in self.all_interactions()
                if interaction.review_id == review_id
            )
        )

    def all_interactions(self) -> Iterable[ReviewInteraction]:
        with self._lock, self._path.open("r", encoding="utf-8") as handle:
            return tuple(
                ReviewInteraction.model_validate_json(line)
                for line in handle
                if line.strip()
            )
