"""Folder Watch (ROADMAP.md §5.13.2) — a production-ready watched-folder Acquisition Source.

Uses real OS-level filesystem notifications (`watchdog` — inotify/FSEvents/ReadDirectoryChangesW)
when the optional `watch` extra is installed; falls back to a documented, honestly-labeled polling
observer when it is not. The choice is probed once, at construction, exactly like
`composition.py`'s `_register_transformers_runtime_factory` chooses between a real
`TransformersRuntime` and `_UnavailableRuntime` based on what is actually importable — never a
silent assumption either way.

A discovered file is only queued once its size is stable across two checks (a debounce), so a file
still being written is never registered mid-write (Article 25: acquisition never registers an
incomplete capture). Temporary/incomplete-write artifacts are ignored by extension and by a
`~$`-prefix check (the common Office lock-file convention) before the stability check even runs.
"""

from __future__ import annotations

import queue
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.acquisition.source import AcquisitionSourceKind, DiscoveredFile, SourceHealth

_DEFAULT_IGNORED_EXTENSIONS = frozenset({".tmp", ".part", ".crdownload", ".download"})
_DEFAULT_STABILITY_INTERVAL_SECONDS = 1.0


class FolderWatchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    recursive: bool = True
    supported_extensions: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ignored_extensions: tuple[str, ...] = tuple(_DEFAULT_IGNORED_EXTENSIONS)
    stability_check_interval_seconds: float = _DEFAULT_STABILITY_INTERVAL_SECONDS


def _is_ignorable(path: Path, config: FolderWatchConfig) -> bool:
    if path.name.startswith("~$"):
        return True
    suffix = path.suffix.lower()
    if suffix in config.ignored_extensions:
        return True
    if config.supported_extensions and suffix not in config.supported_extensions:
        return True
    return False


def _stat_sizes(paths: list[Path]) -> dict[Path, int]:
    sizes: dict[Path, int] = {}
    for path in paths:
        try:
            sizes[path] = path.stat().st_size
        except OSError:
            continue
    return sizes


def _stable_paths(paths: list[Path], interval_seconds: float) -> tuple[list[Path], list[Path]]:
    """Batched stability check (Operational Hardening milestone, Priority 12): one debounce sleep
    for the whole drained batch, not one `time.sleep` per file. A folder drop of N files used to
    cost N * interval_seconds of serial sleeping (measured: ~200s for 200 files at the 1s default);
    this costs one interval regardless of N. Returns (stable, still_writing) — same semantics as
    before (`_is_stable` per file), same debounce interval, no change to what "stable" means.
    """
    if not paths:
        return [], []
    before = _stat_sizes(paths)
    time.sleep(interval_seconds)
    after = _stat_sizes(paths)
    stable, unstable = [], []
    for path in paths:
        if path in before and path in after and before[path] == after[path]:
            stable.append(path)
        else:
            unstable.append(path)
    return stable, unstable


def _to_discovered_file(path: Path) -> DiscoveredFile | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return DiscoveredFile(
        path=path,
        original_filename=path.name,
        byte_size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )


def _watchdog_available() -> bool:
    try:
        import watchdog.observers  # noqa: F401
        import watchdog.events  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class FolderWatchSource:
    """One instance watches one directory tree. `run_scan_cycle()` (via `AcquisitionManager`)
    drains whatever the backing observer has queued since the last cycle — this is not a filesystem
    poll; it is a periodic drain of an event-driven queue (or, only when `watchdog` is unavailable,
    of the honestly-labeled polling fallback below).
    """

    kind = AcquisitionSourceKind.FOLDER_WATCH

    def __init__(self, config: FolderWatchConfig, source_id: str = "folder-watch") -> None:
        self.source_id = source_id
        self.enabled = True
        self.config = config
        self._queue: "queue.Queue[Path]" = queue.Queue()
        self._error_count = 0
        self._last_error: str | None = None
        self._last_scan_at: datetime | None = None
        self._documents_imported = 0
        self._using_native_notifications = _watchdog_available()
        self._observer = None
        self._seen_polled: set[Path] = set()
        self._start()

    # -- lifecycle -----------------------------------------------------------------------------

    def _start(self) -> None:
        self.config.path.mkdir(parents=True, exist_ok=True)
        if self._using_native_notifications:
            self._start_watchdog_observer()
        # Else: the polling fallback has no lifecycle of its own -- see `_poll_once`, run
        # synchronously from `scan()`. No background thread is created for it (a prior revision
        # started one per instance and leaked threads across every short-lived
        # `FolderWatchSource` a test or a Workspace switch created); the periodic "tick" that
        # would otherwise justify a thread already exists one level up, in whatever drives
        # `AcquisitionManager.run_scan_cycle()` (a Qt `QTimer` in the desktop client).

    def _start_watchdog_observer(self) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        source_queue = self._queue

        class _Handler(FileSystemEventHandler):
            def on_created(self, event) -> None:
                if not event.is_directory:
                    source_queue.put(Path(event.src_path))

            def on_moved(self, event) -> None:
                if not event.is_directory:
                    source_queue.put(Path(event.dest_path))

        observer = Observer()
        observer.schedule(_Handler(), str(self.config.path), recursive=self.config.recursive)
        observer.daemon = True
        observer.start()
        self._observer = observer

    def _poll_once(self) -> None:
        """The polling fallback, used only when `watchdog` is not installed: a direct, synchronous
        directory scan invoked from `scan()` itself — honestly reported as "polling" (never claimed
        to be notification-driven) via `health()`/`uses_native_notifications`, and never a
        background thread (see `_start`'s note).
        """
        pattern = "**/*" if self.config.recursive else "*"
        for path in self.config.path.glob(pattern):
            if path.is_file() and path not in self._seen_polled:
                self._seen_polled.add(path)
                self._queue.put(path)

    def shut_down(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)

    # -- AcquisitionSource protocol -------------------------------------------------------------

    def scan(self, known_hashes: frozenset[str]):
        """Drains the queue accumulated by the observer (or, when `watchdog` is unavailable, by one
        synchronous poll performed here) since the last call. Ignores temp/incomplete files and
        waits for size-stability before yielding each candidate.
        """
        if not self._using_native_notifications:
            self._poll_once()

        discovered: list[DiscoveredFile] = []
        drained: list[Path] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break

        candidates = [
            path for path in drained if path.exists() and not _is_ignorable(path, self.config)
        ]
        # One debounce sleep for the whole batch (Priority 12), not one per file — a 200-file
        # drop no longer costs 200 * stability_check_interval_seconds of serial sleeping.
        stable, unstable = _stable_paths(candidates, self.config.stability_check_interval_seconds)
        for path in unstable:
            self._queue.put(path)  # not finished writing yet — try again next cycle

        for path in stable:
            try:
                candidate = _to_discovered_file(path)
            except OSError as exc:
                self._error_count += 1
                self._last_error = str(exc)
                continue
            if candidate is not None:
                discovered.append(candidate)

        self._last_scan_at = datetime.now(timezone.utc)
        self._documents_imported += len(discovered)
        return tuple(discovered)

    def health(self) -> SourceHealth:
        status = "ok" if self._error_count == 0 else "degraded"
        return SourceHealth(
            status=status,
            last_scan_at=self._last_scan_at,
            documents_imported=self._documents_imported,
            error_count=self._error_count,
            last_error=self._last_error,
        )

    @property
    def uses_native_notifications(self) -> bool:
        """True when real OS filesystem notifications (`watchdog`) are backing this source; False
        when the polling fallback is active — surfaced to the Acquisition Manager UI rather than
        left implicit.
        """
        return self._using_native_notifications
