"""The Canonical Observation Ontology's payload type enumeration.

Page, Heading, Section, Paragraph, Metadata, Archive Boundary, Table, Caption, Image, Footnote,
Layout Region. Reading Order and Document Hierarchy were retired as payload types (S5.5, S5.6,
the Graph-Reference Rule) and are represented purely as graph properties.

TEXT_LINE, REGION, RAW_TRANSCRIPTION, PARSED_TRANSCRIPTION, and NORMALIZED_TRANSCRIPTION are
additive types landed by docs/htr-migration-plan.md Stage 2, per docs/htr-domain-design.md §2:
"Observation gains new ObservationType values: TEXT_LINE, REGION, RAW_TRANSCRIPTION,
PARSED_TRANSCRIPTION, NORMALIZED_TRANSCRIPTION."

**docs/htr-migration-plan.md Stage 5 (EXECUTED):** `TABLE_CELL`, `NAMED_ENTITY`, `RELATIONSHIP`,
and `HANDWRITTEN_NOTE` are deleted, per docs/htr-repository-cleanup.md's domain-model table --
- `TABLE_CELL` (and `domain/comparison/table_reconciliation.py`, its cell-level reconciliation
  pass): table semantics don't fit line-level HTR research; no in-scope HTR method produces
  structured tables. `TABLE` itself is retained (a distinct, non-cell type).
- `NAMED_ENTITY`/`RELATIONSHIP`: NER-style entity/relationship extraction is out of scope;
  historical-feature comparison (names, dates, places) is implemented as an evaluator, not an
  ontology observation type.
- `HANDWRITTEN_NOTE`: the paragraph-level "is this handwritten" flag Docling used to produce is
  replaced by `TEXT_LINE`/`RAW_TRANSCRIPTION` at line granularity, a first-class recognition
  result rather than a boolean note.

See IMPLEMENTATION_STATUS.md for the historical citation trail of the original ratification, and
docs/htr-repository-cleanup.md for the deletion's own reasoning.
"""

from __future__ import annotations

from enum import Enum


class ObservationType(str, Enum):
    PAGE = "page"
    HEADING = "heading"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    METADATA = "metadata"
    ARCHIVE_BOUNDARY = "archive_boundary"
    TABLE = "table"
    CAPTION = "caption"
    IMAGE = "image"
    FOOTNOTE = "footnote"
    LAYOUT_REGION = "layout_region"
    TEXT_LINE = "text_line"
    REGION = "region"
    RAW_TRANSCRIPTION = "raw_transcription"
    PARSED_TRANSCRIPTION = "parsed_transcription"
    NORMALIZED_TRANSCRIPTION = "normalized_transcription"
