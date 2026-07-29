"""Registry of every provider's current `MappingTable` (Constitution Article 28) -- the "currently
registered latest version per provider" the mapping-table-staleness check (`application/journal.py`)
needs. Deliberately trivial, mirroring `ProviderRegistry`'s own "keyed lookup, nothing more"
discipline (`providers/registry.py`); unlike that registry, this one is a static, module-level
collection (every provider's `MAPPING_TABLE` already exists at import time, unlike adapters, which
are constructed per Workspace).

Empty as of docs/htr-migration-plan.md Stage 5 (EXECUTED): the five OCR-era providers that used to
populate this table (Docling, Tesseract+LayoutParser, Qwen2.5-VL, PaddleOCR-VL, Surya) are deleted.
`application/journal.py`'s only consumer, `CURRENT_MAPPING_TABLES.get(...)`, already tolerates a
missing entry (returns `None`, never guessed) -- so an empty table here is a correct, honest state,
not a broken one. Repopulated as each new `HtrMethodAdapter`-era importer is added (SATRN first,
Stage 6).
"""

from __future__ import annotations

from archivetrust.domain.ontology.mapping import MappingTable

CURRENT_MAPPING_TABLES: dict[str, MappingTable] = {}
