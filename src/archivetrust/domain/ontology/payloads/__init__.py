from archivetrust.domain.ontology.payloads.archive_boundary import ArchiveBoundaryPayload
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.payloads.caption import CaptionPayload
from archivetrust.domain.ontology.payloads.footnote import FootnotePayload
from archivetrust.domain.ontology.payloads.heading import HeadingPayload
from archivetrust.domain.ontology.payloads.image import ImagePayload
from archivetrust.domain.ontology.payloads.layout_region import LayoutRegionPayload
from archivetrust.domain.ontology.payloads.metadata import MetadataPayload
from archivetrust.domain.ontology.payloads.page import PagePayload
from archivetrust.domain.ontology.payloads.paragraph import IllegibleSpan, ParagraphPayload
from archivetrust.domain.ontology.payloads.region import RegionPayload
from archivetrust.domain.ontology.payloads.section import SectionPayload
from archivetrust.domain.ontology.payloads.table import TablePayload
from archivetrust.domain.ontology.payloads.text_line import TextLinePayload
from archivetrust.domain.ontology.payloads.transcription import (
    NormalizedTranscriptionPayload,
    ParsedTranscriptionPayload,
    RawTranscriptionPayload,
)

from archivetrust.domain.ontology.types import ObservationType

# `NamedEntityPayload`/`RelationshipPayload`/`HandwrittenNotePayload`/`TableCellPayload` (and their
# `ObservationType.NAMED_ENTITY`/`RELATIONSHIP`/`HANDWRITTEN_NOTE`/`TABLE_CELL` types) were deleted
# in docs/htr-migration-plan.md Stage 5 (EXECUTED): NER-style extraction and paragraph-level
# handwriting flags are out of scope for line-level HTR research, and table-cell semantics don't
# fit a comparison of HTR recognizers. `TEXT_LINE`/`RAW_TRANSCRIPTION`/`PARSED_TRANSCRIPTION`/
# `NORMALIZED_TRANSCRIPTION` (Phase 5, already additive before this stage) are their replacements.

PAYLOAD_TYPES: tuple[type[ObservationPayload], ...] = (
    PagePayload,
    HeadingPayload,
    SectionPayload,
    ParagraphPayload,
    MetadataPayload,
    ArchiveBoundaryPayload,
    TablePayload,
    CaptionPayload,
    ImagePayload,
    FootnotePayload,
    LayoutRegionPayload,
    TextLinePayload,
    RegionPayload,
    RawTranscriptionPayload,
    ParsedTranscriptionPayload,
    NormalizedTranscriptionPayload,
)

PAYLOAD_TYPE_BY_OBSERVATION_TYPE: dict[ObservationType, type[ObservationPayload]] = {
    payload_cls.observation_type: payload_cls for payload_cls in PAYLOAD_TYPES
}

__all__ = [
    "PAYLOAD_TYPE_BY_OBSERVATION_TYPE",
    "PAYLOAD_TYPES",
    "ArchiveBoundaryPayload",
    "CaptionPayload",
    "FootnotePayload",
    "HeadingPayload",
    "IllegibleSpan",
    "ImagePayload",
    "LayoutRegionPayload",
    "MetadataPayload",
    "NormalizedTranscriptionPayload",
    "ObservationPayload",
    "PagePayload",
    "ParagraphPayload",
    "ParsedTranscriptionPayload",
    "RawTranscriptionPayload",
    "RegionPayload",
    "SectionPayload",
    "TablePayload",
    "TextLinePayload",
]
