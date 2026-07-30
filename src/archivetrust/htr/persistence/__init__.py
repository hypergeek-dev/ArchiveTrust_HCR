"""Durable persistence for the HTR research entities (`docs/architecture/htr-telemetry.md`).

Closes the gap `docs/htr-telemetry-knowledge-gap-analysis.md` §2 documented: HTR research entities
were frozen domain models with no path connecting them to the retained append-only telemetry
architecture at all. They now have one, and it reuses the existing mechanism
(`FileTelemetrySink` + a `Journal`-shaped replay) rather than inventing a second one.
"""

from __future__ import annotations

from archivetrust.htr.persistence.durable_store import (
    DurableHtrResearchStore,
    HtrCoarseEntitySnapshot,
)

__all__ = [
    "DurableHtrResearchStore",
    "HtrCoarseEntitySnapshot",
]
