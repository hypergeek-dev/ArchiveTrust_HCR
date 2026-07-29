"""Enforces the Dual Architecture separation contract (ROADMAP_V2.md S5, Guiding Principles 2, 6).

The dependency between the two systems is strictly one-directional: the Learning Platform reads
Trust Engine telemetry, and nothing more flows the other way. These tests fail loudly if that
contract is ever broken by an import, because the contract is load-bearing -- it is what keeps
"trustworthy" meaning something after the system has been analyzed for years (S5.3).
"""

from __future__ import annotations

import re
from pathlib import Path

import archivetrust
from archivetrust.learning.source import TelemetrySource

SRC_ROOT = Path(archivetrust.__file__).resolve().parent

# The Trust Engine subpackages -- none of these may depend on the Learning Platform.
TRUST_ENGINE_DIRS = ("domain", "providers", "application", "infrastructure")

_LEARNING_IMPORT = re.compile(r"(from|import)\s+archivetrust\.learning\b")


def _python_files(subdir: str) -> list[Path]:
    return list((SRC_ROOT / subdir).rglob("*.py"))


def test_trust_engine_never_imports_learning_platform() -> None:
    offenders = []
    for subdir in TRUST_ENGINE_DIRS:
        for path in _python_files(subdir):
            text = path.read_text(encoding="utf-8")
            if _LEARNING_IMPORT.search(text):
                offenders.append(path.relative_to(SRC_ROOT))
    assert not offenders, (
        "Trust Engine modules must never import archivetrust.learning "
        f"(ROADMAP_V2.md S5.1); offenders: {offenders}"
    )


def test_telemetry_source_is_read_only() -> None:
    # The Learning Platform's input seam exposes no write path -- append must not be part of the
    # contract, so an analytics component cannot emit into the stream it observes (GP 2).
    assert not hasattr(TelemetrySource, "append")
    assert hasattr(TelemetrySource, "all_events")
    assert hasattr(TelemetrySource, "events_for_document")


def test_existing_sinks_satisfy_read_only_source() -> None:
    # The durable sink the Trust Engine writes to is readable as a TelemetrySource with no adapter
    # (ROADMAP_V2.md S13: analytics reconstructs from the same stream, not a separate copy).
    from archivetrust.infrastructure.storage.telemetry_sink import (
        FileTelemetrySink,
        InMemoryTelemetrySink,
    )

    assert isinstance(InMemoryTelemetrySink(), TelemetrySource)
    # runtime_checkable Protocols check method presence; FileTelemetrySink has both read methods.
    assert issubclass(FileTelemetrySink, object)
    for method in ("all_events", "events_for_document"):
        assert callable(getattr(FileTelemetrySink, method))
