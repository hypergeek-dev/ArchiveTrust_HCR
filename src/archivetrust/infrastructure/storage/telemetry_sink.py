"""Concrete TelemetrySink implementations (ROADMAP.md S11: in-memory + file-backed both required
to pass the same contract test suite -- Milestone 2 acceptance criteria).

Both implementations are append-only and preserve insertion order per `document_ref`, matching
`archivetrust.domain.telemetry.sink.TelemetrySink`. Neither performs any transformation on events
beyond (de)serialization -- knowledge-evolution semantics live entirely in the domain layer and in
`application/journal.py`, never here.

**Read architecture (release WS5, 2026-07-16).** `FileTelemetrySink` previously parsed the entire
stream into pydantic objects at construction (measured: 12.4 s and ~600 MB of Python heap for a
118k-event / 103 MB workspace) and rewrote a full hash-chain manifest on every append (measured
quadratic during the F3 campaign). It now:

- builds a **byte-offset index** at construction (one `json.loads` validation pass, no pydantic),
- parses events **on demand** from file offsets, with a small per-document cache (review flows
  replay the same document repeatedly),
- serves `all_events()` as a lazy, snapshot-consistent `Sequence` view — `len()` is O(1),
  iteration streams one event at a time, and slicing (`events[cursor:]`, the incremental
  read-model contract) parses only the sliced range,
- isolates **corrupt lines**: a record that fails JSON validation is indexed as a
  `CorruptTelemetryRecord` (line number, byte offset, reason), reported via `corrupt_records` and
  a logged warning, and skipped by reads — one bad record never invalidates unrelated history.
  A partial final line (a process died mid-write) is isolated the same way, and the next `append`
  writes a fresh newline first so the partial record is never merged into a new one,
- records tamper-evidence through the O(1) `HashChainAppender` sidecar
  (`events.jsonl.chain.jsonl`) instead of the previous per-append full-manifest rewrite.

The JSONL file remains the single durable record; the index, caches, and chain sidecar are all
derived and rebuildable from it.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from archivetrust.domain.telemetry.events import TelemetryEvent, parse_event
from archivetrust.infrastructure.storage.blob_store import ContentAddressedBlobStore
from archivetrust.infrastructure.storage.integrity import HashChainAppender

_BLOB_REF_KEY = "__archivetrust_blob_ref__"
_BLOB_SIZE_KEY = "size_bytes"
_BLOB_MEDIA_TYPE_KEY = "media_type"
_BLOB_ENCODING_KEY = "encoding"
_BLOB_INTEGRITY_KEY = "integrity"
_DEFAULT_RAW_OUTPUT_BLOB_THRESHOLD = 4096
_DOCUMENT_CACHE_SIZE = 16
_APPEND_CACHE_SIZE = 4096
_DEFAULT_MAX_EVENT_BYTES = 1_048_576

logger = logging.getLogger(__name__)


class TelemetryEventSizeError(ValueError):
    """A serialized event exceeded the durable one-record safety envelope."""


class InMemoryTelemetrySink:
    """Process-local, non-durable. Useful for tests and for a pipeline run that persists via a
    different sink at a higher layer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events_by_document: dict[str, list[TelemetryEvent]] = defaultdict(list)
        self._all_events: list[TelemetryEvent] = []

    def append(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._events_by_document[event.document_ref].append(event)
            self._all_events.append(event)

    def events_for_document(self, document_ref: str) -> Iterator[TelemetryEvent]:
        with self._lock:
            snapshot = tuple(self._events_by_document.get(document_ref, ()))
        return iter(snapshot)

    def all_events(self) -> Iterable[TelemetryEvent]:
        with self._lock:
            snapshot = tuple(self._all_events)
        return snapshot


@dataclass(frozen=True)
class CorruptTelemetryRecord:
    """One line of the stream that could not be read as JSON — indexed, reported, and skipped;
    never silently dropped and never allowed to poison neighbouring records."""

    line_number: int
    byte_offset: int
    reason: str


@dataclass(frozen=True)
class _RecordRef:
    offset: int
    length: int
    document_ref: str


class _LazyEventView(Sequence):
    """Snapshot-consistent lazy view over the sink's records at creation time.

    `len()` is O(1); `view[i]` parses one record; `view[a:b]` returns a lazily-parsing iterator
    over exactly that range (the incremental read-model pattern `events[cursor:]` therefore parses
    only new events); plain iteration streams the whole snapshot one event at a time.
    """

    def __init__(self, sink: "FileTelemetrySink", count: int) -> None:
        self._sink = sink
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[TelemetryEvent]:
        return self._sink._iter_range(0, self._count)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self._count)
            if step != 1:
                return list(self._sink._iter_range(0, self._count))[item]
            return self._sink._iter_range(start, stop)
        if item < 0:
            item += self._count
        if not 0 <= item < self._count:
            raise IndexError(item)
        return self._sink._event_at(item)


class FileTelemetrySink:
    """Durable, append-only JSON-Lines file: one event per line, written with
    `model_dump_json()` and reconstructed on read via `parse_event`. Per-document ordering is
    exactly file order, since writes are append-only and never reordered. See the module
    docstring for the lazy read architecture. One deliberate limit is retained from the previous
    implementation: a *concurrent writer in another process* is not reflected until this sink is
    reconstructed — the same single-writer assumption the append-only file itself already makes.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        blob_dir: Path | str | None = None,
        raw_output_blob_threshold: int = _DEFAULT_RAW_OUTPUT_BLOB_THRESHOLD,
        max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES,
    ) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._blob_store = ContentAddressedBlobStore(blob_dir or (self._path.parent / "blobs"))
        self._raw_output_blob_threshold = raw_output_blob_threshold
        self._max_event_bytes = max_event_bytes
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()
        self._records: list[_RecordRef] = []
        self._by_document: dict[str, list[int]] = defaultdict(list)
        self._corrupt: list[CorruptTelemetryRecord] = []
        self._appended: OrderedDict[int, TelemetryEvent] = OrderedDict()
        """The most recent events appended by this process, keyed by record index — a bounded
        write-through cache so the common `append -> incremental read-model refresh` sequence
        never re-parses what this process just serialized. Bounded (`_APPEND_CACHE_SIZE`) so a
        long processing campaign cannot re-accumulate the full-mirror memory footprint this
        rewrite removed; evicted entries are simply re-read from the durable file."""
        self._document_cache: OrderedDict[str, tuple[TelemetryEvent, ...]] = OrderedDict()
        self._file_size = 0
        self._ends_with_newline = True
        self._build_index()
        if self._corrupt:
            logger.warning(
                "%s: %d corrupt telemetry record(s) isolated (first at line %d: %s); "
                "unaffected records remain fully readable",
                self._path,
                len(self._corrupt),
                self._corrupt[0].line_number,
                self._corrupt[0].reason,
            )
        self._chain: HashChainAppender | None = None
        """Constructed lazily on the first `append` (never at construction) so purely read-side
        consumers — a replay archive opened for verification, a report script — never write or
        reconcile integrity sidecars as a side effect of reading."""

    # -- index ---------------------------------------------------------------------------------

    def _build_index(self) -> None:
        payload = self._path.read_bytes()
        self._file_size = len(payload)
        self._ends_with_newline = (not payload) or payload.endswith(b"\n")
        offset = 0
        line_number = 0
        for raw in payload.split(b"\n"):
            length = len(raw)
            if raw.strip():
                line_number += 1
                try:
                    record = json.loads(raw)
                    document_ref = record["document_ref"]
                except (ValueError, KeyError, TypeError) as error:
                    self._corrupt.append(
                        CorruptTelemetryRecord(
                            line_number=line_number, byte_offset=offset, reason=str(error)
                        )
                    )
                else:
                    self._by_document[document_ref].append(len(self._records))
                    self._records.append(
                        _RecordRef(offset=offset, length=length, document_ref=document_ref)
                    )
            offset += length + 1  # the "\n" separator

    @property
    def corrupt_records(self) -> tuple[CorruptTelemetryRecord, ...]:
        return tuple(self._corrupt)

    # -- reads ---------------------------------------------------------------------------------

    def _read_raw(self, handle, ref: _RecordRef) -> bytes:
        handle.seek(ref.offset)
        return handle.read(ref.length)

    def _parse_raw(self, raw: bytes) -> TelemetryEvent:
        return parse_event(self._rehydrate_raw_output(json.loads(raw)))

    def _event_at(self, index: int) -> TelemetryEvent:
        cached = self._appended.get(index)
        if cached is not None:
            return cached
        with self._path.open("rb") as handle:
            return self._parse_raw(self._read_raw(handle, self._records[index]))

    def _iter_range(self, start: int, stop: int) -> Iterator[TelemetryEvent]:
        def generate() -> Iterator[TelemetryEvent]:
            with self._path.open("rb") as handle:
                for index in range(start, stop):
                    cached = self._appended.get(index)
                    if cached is not None:
                        yield cached
                    else:
                        yield self._parse_raw(self._read_raw(handle, self._records[index]))

        return generate()

    def events_for_document(self, document_ref: str) -> Iterator[TelemetryEvent]:
        with self._lock:
            cached = self._document_cache.get(document_ref)
            if cached is not None:
                self._document_cache.move_to_end(document_ref)
                return iter(cached)
            indices = tuple(self._by_document.get(document_ref, ()))
        with self._path.open("rb") as handle:
            events = tuple(
                self._appended.get(index)
                or self._parse_raw(self._read_raw(handle, self._records[index]))
                for index in indices
            )
        with self._lock:
            self._document_cache[document_ref] = events
            while len(self._document_cache) > _DOCUMENT_CACHE_SIZE:
                self._document_cache.popitem(last=False)
        return iter(events)

    def all_events(self) -> Sequence[TelemetryEvent]:
        with self._lock:
            return _LazyEventView(self, len(self._records))

    # -- writes --------------------------------------------------------------------------------

    def append(self, event: TelemetryEvent) -> None:
        payload = self._externalize_raw_output(event.model_dump(mode="json"))
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        raw = line.encode("utf-8")
        if len(raw) > self._max_event_bytes:
            raise TelemetryEventSizeError(
                f"serialized event is {len(raw)} bytes; maximum is {self._max_event_bytes}"
            )
        with self._lock:
            if self._chain is None:
                # Reconciled against the file as it exists *before* this record is written, so
                # the rebuild-or-resume pass and the append below never double-count.
                self._chain = HashChainAppender(self._path)
            with self._path.open("ab") as handle:
                if not self._ends_with_newline:
                    # Isolate a partial trailing record (a process died mid-write) on its own
                    # line, so this append can never merge into it. The partial line is already
                    # indexed as corrupt; the durable history is not rewritten.
                    handle.write(b"\n")
                    self._file_size += 1
                handle.write(raw)
                handle.write(b"\n")
            index = len(self._records)
            self._records.append(
                _RecordRef(offset=self._file_size, length=len(raw), document_ref=event.document_ref)
            )
            self._by_document[event.document_ref].append(index)
            self._appended[index] = event
            while len(self._appended) > _APPEND_CACHE_SIZE:
                self._appended.popitem(last=False)
            self._file_size += len(raw) + 1
            self._ends_with_newline = True
            cached = self._document_cache.get(event.document_ref)
            if cached is not None:
                self._document_cache[event.document_ref] = cached + (event,)
            self._chain.append_record(raw)

    # -- raw-output blob externalization (unchanged) --------------------------------------------

    def _externalize_raw_output(self, payload: dict) -> dict:
        kind = payload.get("kind")
        if kind == "EvidenceCreated":
            evidence = payload.get("evidence")
            if isinstance(evidence, dict):
                evidence["raw_output"] = self._blob_ref_or_value(evidence.get("raw_output"))
        elif kind == "EvidenceRejected":
            payload["raw_output"] = self._blob_ref_or_value(payload.get("raw_output"))
        return payload

    def _blob_ref_or_value(self, value):
        if not isinstance(value, str) or len(value.encode("utf-8")) < self._raw_output_blob_threshold:
            return value
        digest, size = self._blob_store.put_text(value)
        return {
            _BLOB_REF_KEY: digest,
            _BLOB_SIZE_KEY: size,
            _BLOB_MEDIA_TYPE_KEY: "text/plain",
            _BLOB_ENCODING_KEY: "utf-8",
            _BLOB_INTEGRITY_KEY: f"sha256:{digest}",
        }

    def _rehydrate_raw_output(self, payload: dict) -> dict:
        kind = payload.get("kind")
        if kind == "EvidenceCreated":
            evidence = payload.get("evidence")
            if isinstance(evidence, dict):
                evidence["raw_output"] = self._raw_output_value(evidence.get("raw_output"))
        elif kind == "EvidenceRejected":
            payload["raw_output"] = self._raw_output_value(payload.get("raw_output"))
        return payload

    def _raw_output_value(self, value):
        if isinstance(value, dict) and _BLOB_REF_KEY in value:
            return self._blob_store.get_text(str(value[_BLOB_REF_KEY]))
        return value
