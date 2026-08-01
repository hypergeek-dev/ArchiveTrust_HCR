"""Parse-result types for Loghi's own PAGE XML output.

**Deliberately its own type, never merged with `providers.transkribus.parsing_models.ParsedDocument`.**
Both tools export nominally "PAGE XML," but the brief is explicit: "Both methods may produce PAGE XML
with different structures and conventions... do not silently equate fields that have different
semantics." Loghi and Transkribus are independent tools with independent metadata conventions (e.g.
Loghi's `Metadata/Creator` names its own pipeline stages, not a Transkribus job id) -- keeping the
result types separate means a caller can never accidentally read a Loghi-specific field as if it came
from Transkribus, or vice versa, and each type's own docstring can honestly describe only what its
tool actually emits.

The element-level XML walking in `page_xml.py` intentionally mirrors
`providers/transkribus/page_xml.py`'s approach (stdlib `ElementTree`, namespace-tolerant local-name
matching) rather than importing it, keeping this provider package self-contained the same way every
other provider package in this codebase is (SATRN/Florence-2/Swedish-Lion each have their own facade
with no shared code between them).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LoghiParseError(Exception):
    """Raised by `page_xml.parse_loghi_page_xml` on malformed/incomplete input. `category` mirrors
    `FailureRecord.category` from the "PAGE XML parse failure" bucket of the failure taxonomy
    (docs/methods/loghi.md) -- never restricted to a closed enum, so a category a future version adds
    is still carried verbatim."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


class LoghiParsedLine(BaseModel):
    """One text line as Loghi's PAGE XML states it. `confidence`/`polygon`/`baseline` are `None`
    when the file genuinely does not carry that information -- never fabricated."""

    model_config = ConfigDict(frozen=True)

    line_id: str
    text: str
    confidence: float | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    baseline: tuple[tuple[float, float], ...] | None = None
    reading_order_index: int


class LoghiParsedRegion(BaseModel):
    model_config = ConfigDict(frozen=True)

    region_id: str
    region_type: str | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    reading_order_index: int
    lines: tuple[LoghiParsedLine, ...] = ()


class LoghiPageParseResult(BaseModel):
    """The full parse result for one Loghi-produced PAGE XML file, plus the provenance fields the
    brief requires be preserved separately from the parsed content: `source_xml_hash` (the untouched
    file's own content hash) and `page_schema_version` (the PAGE XML namespace version this specific
    file declares, read from the document, never assumed)."""

    model_config = ConfigDict(frozen=True)

    source_xml_hash: str
    page_schema_version: str | None
    page_width: int | None = None
    page_height: int | None = None
    image_filename: str | None = None
    regions: tuple[LoghiParsedRegion, ...] = ()
    creator_metadata: str | None = None
    """Loghi's `Metadata/Creator` text verbatim, when present -- often names the pipeline
    tool/version that produced the file, kept as an unparsed string rather than guessing a structure
    Loghi does not document."""
    processing_date: str | None = None
    warnings: tuple[str, ...] = ()
    omitted_fields: tuple[str, ...] = ()
    """Fields this parser deliberately does not extract even though PAGE XML could in principle carry
    them (e.g. per-glyph geometry) -- recorded so "we don't read that" is a stated fact, not a silent
    gap (brief: "Record: fields omitted during normalization")."""

    def full_text(self) -> str:
        lines_out: list[str] = []
        for region in sorted(self.regions, key=lambda r: r.reading_order_index):
            for line in sorted(region.lines, key=lambda ln: ln.reading_order_index):
                lines_out.append(line.text)
        return "\n".join(lines_out)

    def mean_confidence(self) -> float | None:
        values = [
            line.confidence for region in self.regions for line in region.lines if line.confidence is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)
