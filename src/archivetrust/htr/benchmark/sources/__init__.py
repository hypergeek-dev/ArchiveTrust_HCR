"""Source-format adapters. `ADAPTERS` is the registry inspection and build choose from."""

from __future__ import annotations

from archivetrust.htr.benchmark.sources.base import CandidateLine, Extraction, SourceAdapter
from archivetrust.htr.benchmark.sources.line_pairs import LinePairsAdapter
from archivetrust.htr.benchmark.sources.tabular import TabularAdapter
from archivetrust.htr.benchmark.sources.xml_lines import AltoAdapter, PageXmlAdapter

ADAPTERS: tuple[SourceAdapter, ...] = (PageXmlAdapter(), AltoAdapter(), LinePairsAdapter(), TabularAdapter())


def adapter_by_id(adapter_id: str) -> SourceAdapter:
    for adapter in ADAPTERS:
        if adapter.adapter_id == adapter_id:
            return adapter
    raise KeyError(f"unknown adapter {adapter_id!r}; known: {[a.adapter_id for a in ADAPTERS]}")


__all__ = ["ADAPTERS", "CandidateLine", "Extraction", "SourceAdapter", "adapter_by_id"]
