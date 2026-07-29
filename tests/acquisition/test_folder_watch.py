from __future__ import annotations

import time

from archivetrust.acquisition.folder_watch import FolderWatchConfig, FolderWatchSource


def _new_source(path, **overrides) -> FolderWatchSource:
    config = FolderWatchConfig(path=path, stability_check_interval_seconds=0.02, **overrides)
    return FolderWatchSource(config)


def test_uses_polling_fallback_when_watchdog_unavailable(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    source = _new_source(tmp_path / "watched")
    assert source.uses_native_notifications is False


def test_discovers_a_stable_file(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    watched = tmp_path / "watched"
    source = _new_source(watched)
    (watched / "protokoll.pdf").write_bytes(b"stable content")
    time.sleep(0.05)

    discovered = source.scan(frozenset())
    assert len(discovered) == 1
    assert discovered[0].original_filename == "protokoll.pdf"


def test_ignores_temporary_and_unsupported_files(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    watched = tmp_path / "watched"
    source = _new_source(watched)
    (watched / "upload.tmp").write_bytes(b"partial")
    (watched / "~$locked.pdf").write_bytes(b"lock file")
    (watched / "notes.txt").write_bytes(b"unsupported extension")
    (watched / "record.pdf").write_bytes(b"a real record")
    time.sleep(0.05)

    discovered = source.scan(frozenset())
    assert [d.original_filename for d in discovered] == ["record.pdf"]


def test_does_not_register_the_same_file_twice_across_scans(tmp_path, monkeypatch) -> None:
    import archivetrust.acquisition.folder_watch as fw

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    watched = tmp_path / "watched"
    source = _new_source(watched)
    (watched / "record.pdf").write_bytes(b"a real record")
    time.sleep(0.05)

    first_scan = source.scan(frozenset())
    second_scan = source.scan(frozenset())
    assert len(first_scan) == 1
    assert len(second_scan) == 0


def test_no_background_thread_is_created_for_the_polling_fallback(tmp_path, monkeypatch) -> None:
    # Regression test: an earlier revision started a perpetual background polling thread per
    # instance, which leaked across every short-lived FolderWatchSource a test created and crashed
    # the interpreter under load. The fallback must be purely synchronous.
    import archivetrust.acquisition.folder_watch as fw
    import threading

    monkeypatch.setattr(fw, "_watchdog_available", lambda: False)
    before = threading.active_count()
    for _ in range(20):
        _new_source(tmp_path / "watched")
    assert threading.active_count() == before
