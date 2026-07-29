"""Format-neutral parse result types for Transkribus manual imports (docs/htr-migration-plan.md
Stage 8).

`ParsedDocument`/`ParsedRegion`/`ParsedLine` are the common shape every format parser
(`page_xml.py`, `alto_xml.py`, `plain_text.py`) produces, before `adapter.py` maps that shape onto
`RecognitionResult`/`Evidence`. Kept deliberately separate from `htr/corpus/models.py`'s
`Region`/`TextLine` (which are first-class, id-addressed corpus entities produced by a
`SegmentationAdapter`): these are transient parse-time structures scoped to one import file, not
corpus entities themselves. A later association/ingestion step is what would turn a `ParsedLine`
into (or attach it as evidence for) a real `TextLine`, not this module.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TranskribusParseError(Exception):
    """Raised by a format parser on malformed/incomplete input. `category` mirrors
    `FailureRecord.category` -- always one of a known, documented set (see each parser module),
    but never restricted to an enum, so a category a future parser adds is still carried verbatim
    (Constitution Article 6: Full Exposure)."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


class ParsedLine(BaseModel):
    """One text line extracted from an export file. `confidence`/`polygon`/`baseline` are `None`
    when the source format/file genuinely does not carry that information -- never fabricated or
    defaulted to a fake value (e.g. `0.0` would silently look like "certainly wrong", `1.0` would
    silently look like "certainly right"; `None` is the only honest representation of absence)."""

    model_config = ConfigDict(frozen=True)

    line_id: str
    text: str
    confidence: float | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    baseline: tuple[tuple[float, float], ...] | None = None
    reading_order_index: int


class ParsedRegion(BaseModel):
    """One text region extracted from an export file, containing zero or more `ParsedLine`s in
    reading order."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    region_type: str | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    reading_order_index: int
    lines: tuple[ParsedLine, ...] = ()


class ParsedDocument(BaseModel):
    """The full parse result for one export file -- one page's worth of regions/lines, plus
    whatever page-level and vendor metadata the file itself declared. Every optional field is
    `None` (never a fabricated default) when the file does not state it."""

    model_config = ConfigDict(frozen=True)

    export_format: str
    """`"page_xml" | "alto_xml" | "plain_text"` -- kept as a plain string (not an enum) here so
    this module never needs to import `ExternalImportFormat` and the two id spaces can evolve
    independently; `adapter.py` is the mapping boundary between them."""
    page_width: int | None = None
    page_height: int | None = None
    image_filename: str | None = None
    regions: tuple[ParsedRegion, ...] = ()
    processing_date: str | None = None
    """From the export file's own metadata (PAGE `Created`/`LastChange`, ALTO
    `processingDateTime`) when present -- `None` otherwise. Callers must not substitute the import
    timestamp for this field; that distinction is made explicitly one layer up, in
    `adapter.py::recognize`'s `raw_response["processing_date_source"]`."""
    vendor_reported_accuracy: float | None = None
    """A vendor-reported accuracy/CER figure, when the export file states one (e.g. a PAGE XML
    `MetadataItem name="vendorReportedAccuracy"`) -- kept as a distinct field for exactly one
    reason: it must never be merged into or compared directly against ArchiveTrust's own CER/WER
    metrics (evaluation/metrics.py). See `providers/transkribus/README.md`."""
    model_version_hint: str | None = None
    """A model/checkpoint version string, only if the export file states one (e.g. a
    `MetadataItem name="modelVersion"`). Transkribus manual exports frequently state nothing here
    -- this is `None` far more often than not, and that is the honest, expected case, not a
    parsing failure."""
    job_id_hint: str | None = None
    """A Transkribus job identifier, only if the export file states one."""
    transkribus_document_id_hint: str | None = None
    """Transkribus's own document/collection id, only if the export file states one -- maps onto
    `ExternalImport.transkribus_document_id` at the ingestion boundary."""
    warnings: tuple[str, ...] = ()
    """Non-fatal parse observations (e.g. an unparseable `conf`/`WC` attribute that was ignored
    rather than raising) -- surfaced, never silently swallowed."""

    def full_text(self) -> str:
        """The linearized transcription: every region's lines, in reading order, region by
        region, joined with newlines. This is the adapter's "parsed transcription" -- the
        structural extraction from the file's regions/lines, before any normalization."""
        lines_out: list[str] = []
        for region in sorted(self.regions, key=lambda r: r.reading_order_index):
            for line in sorted(region.lines, key=lambda ln: ln.reading_order_index):
                lines_out.append(line.text)
        return "\n".join(lines_out)

    def mean_confidence(self) -> float | None:
        """The mean of every line's confidence that is actually present. `None` when *no* line in
        the file carries a confidence value -- never `0.0`/`1.0` as a fabricated stand-in for
        "the file doesn't say"."""
        values = [line.confidence for region in self.regions for line in region.lines if line.confidence is not None]
        if not values:
            return None
        return sum(values) / len(values)
