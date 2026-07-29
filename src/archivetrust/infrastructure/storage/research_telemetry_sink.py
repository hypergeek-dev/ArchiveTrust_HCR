"""Concrete `ResearchTelemetrySink` implementations (Constitution Article 34), mirroring
`infrastructure.storage.telemetry_sink`'s in-memory + file-backed split exactly, as two
implementations of a deliberately separate Protocol from `TelemetrySink`.

`FileResearchTelemetrySink` is the persistence mechanism Research Telemetry uses in production:
always a repository-local, version-controlled path (e.g. `benchmarks/research_telemetry.jsonl`) --
**never** a real deployment's `archivetrust_data/workspaces/*/telemetry` directory, which may hold
live state this standard has no authority to mutate (Article 34).
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from archivetrust.domain.research.events import ResearchEvent, parse_research_event


class InMemoryResearchTelemetrySink:
    """Process-local, non-durable. Useful for tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events_by_corpus: dict[str, list[ResearchEvent]] = defaultdict(list)
        self._all_events: list[ResearchEvent] = []

    def append(self, event: ResearchEvent) -> None:
        with self._lock:
            self._events_by_corpus[event.corpus_ref].append(event)
            self._all_events.append(event)

    def events_for_corpus(self, corpus_ref: str) -> Iterator[ResearchEvent]:
        with self._lock:
            snapshot = tuple(self._events_by_corpus.get(corpus_ref, ()))
        return iter(snapshot)

    def all_events(self) -> Iterable[ResearchEvent]:
        with self._lock:
            snapshot = tuple(self._all_events)
        return snapshot


class FileResearchTelemetrySink:
    """Durable, append-only JSON-Lines file -- one event per line, exactly mirroring
    `FileTelemetrySink`'s own format and read-through-cache discipline, for a deliberately separate
    vocabulary and file.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        self._events: list[ResearchEvent] = self._load()

    def _load(self) -> list[ResearchEvent]:
        text = self._path.read_text(encoding="utf-8")
        events = []
        for line in text.splitlines():
            if not line.strip():
                continue
            events.append(parse_research_event(json.loads(line)))
        return events

    def append(self, event: ResearchEvent) -> None:
        line = event.model_dump_json()
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
            self._events.append(event)

    def events_for_corpus(self, corpus_ref: str) -> Iterator[ResearchEvent]:
        with self._lock:
            snapshot = tuple(e for e in self._events if e.corpus_ref == corpus_ref)
        return iter(snapshot)

    def all_events(self) -> Iterable[ResearchEvent]:
        with self._lock:
            return tuple(self._events)
