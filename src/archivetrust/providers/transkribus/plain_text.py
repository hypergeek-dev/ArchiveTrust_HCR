"""Plain-text manual import -- the trivial case. No geometry, no confidence, no reading-order
metadata exists in a bare `.txt` export; every line of the file becomes one `ParsedLine` with only
`text` and `reading_order_index` populated, everything else honestly `None`. Still goes through
the same `ParsedDocument`/`ExternalImport` provenance wrapper as PAGE/ALTO -- no special-cased
shortcut around content-addressing or evidence construction.
"""

from __future__ import annotations

from archivetrust.providers.transkribus.parsing_models import (
    ParsedDocument,
    ParsedLine,
    ParsedRegion,
    TranskribusParseError,
)


def parse_plain_text(text: str) -> ParsedDocument:
    if text is None or text.strip() == "":
        raise TranskribusParseError("empty_file", "Plain text import file is empty")

    raw_lines = text.splitlines()
    parsed_lines = tuple(
        ParsedLine(
            line_id=f"line_{index}",
            text=line,
            confidence=None,
            polygon=None,
            baseline=None,
            reading_order_index=index,
        )
        for index, line in enumerate(raw_lines)
        if line.strip() != ""
    )

    if not parsed_lines:
        # Every line was whitespace-only -- not the same failure category as a truly empty file
        # (the file has bytes, just no recoverable text), so no region is fabricated either.
        return ParsedDocument(
            export_format="plain_text",
            regions=(),
            warnings=("Plain text import file contained no non-blank lines",),
        )

    region = ParsedRegion(
        region_id="region_0",
        region_type=None,
        polygon=None,
        reading_order_index=0,
        lines=parsed_lines,
    )
    return ParsedDocument(export_format="plain_text", regions=(region,))
