"""`MappingTable`/`MappingTableEntry` (Constitution Article 28) -- the named, versioned
provider-to-ontology semantic contract. Also proves Phase 5's acceptance criterion directly:
`tesseract_layoutparser`'s `"Text"` -> `PARAGRAPH` mapping (the Semantic Contract Audit's finding,
2026-07-14) is now one queryable table row, not an inline conditional requiring source-reading.
"""

from __future__ import annotations

import pytest

from archivetrust.domain.ontology.mapping import MappingTable, MappingTableEntry
from archivetrust.domain.ontology.types import ObservationType

# The provider-specific parametrized tests that used to live here (Docling/Tesseract+LayoutParser/
# Qwen2.5-VL/Surya/PaddleOCR-VL MAPPING_TABLE fixtures) were deleted along with those provider
# packages (docs/htr-migration-plan.md Stage 5 -- EXECUTED). The generic `MappingTable`/
# `MappingTableEntry` behavior they incidentally also exercised is covered below via a synthetic
# table; a new HTR method importer (SATRN first) should add its own MAPPING_TABLE tests when it
# lands, following the same "Semantic Contract Audit" pattern `test_tesseract_text_label_maps_to_
# paragraph_as_one_named_queryable_entry` used to demonstrate for `tesseract_layoutparser`.


def _table() -> MappingTable:
    return MappingTable(
        provider_id="test-provider",
        table_version=1,
        entries=(
            MappingTableEntry(entry_id="test-provider:A", native_label="A", observation_type=ObservationType.HEADING),
            MappingTableEntry(
                entry_id="test-provider:fallback", native_label=None,
                observation_type=ObservationType.LAYOUT_REGION, fallback=True,
            ),
        ),
    )


def test_entry_for_returns_the_named_entry_for_a_known_label():
    entry = _table().entry_for("A")
    assert entry.entry_id == "test-provider:A"
    assert entry.observation_type == ObservationType.HEADING
    assert entry.fallback is False


def test_entry_for_returns_the_fallback_for_an_unknown_label():
    entry = _table().entry_for("some-undocumented-label")
    assert entry.fallback is True
    assert entry.observation_type == ObservationType.LAYOUT_REGION


def test_entry_for_raises_when_no_entry_and_no_fallback_exist():
    table = MappingTable(
        provider_id="test-provider-no-fallback",
        table_version=1,
        entries=(
            MappingTableEntry(entry_id="x:A", native_label="A", observation_type=ObservationType.HEADING),
        ),
    )
    with pytest.raises(KeyError):
        table.entry_for("unknown")


