"""The single entry point used by `adapter.py`: reads a pre-existing export file from local disk
and dispatches to the correct format parser -- **never** fetches anything over the network. There
is no code path in this module (or anywhere in this package) that makes an HTTP/API call to
Transkribus; `source_file_path` must already exist on disk, placed there by a human researcher's
own manual export/download action outside this application.
"""

from __future__ import annotations

import re

from pathlib import Path

from archivetrust.providers.transkribus.alto_xml import parse_alto_xml
from archivetrust.providers.transkribus.page_xml import parse_page_xml
from archivetrust.providers.transkribus.parsing_models import ParsedDocument, TranskribusParseError
from archivetrust.providers.transkribus.plain_text import parse_plain_text

_FIRST_TAG_RE = re.compile(r"<(?!\?|!)([A-Za-z_][\w:.-]*)")


def _localname(tag: str) -> str:
    return tag.rsplit(":", 1)[-1] if ":" in tag else tag


def sniff_format(text: str) -> str:
    """Determines which parser to use from the file's own content -- not from its extension alone
    (a `.xml` file could be either PAGE or ALTO). Returns `"page_xml"`, `"alto_xml"`,
    `"plain_text"`, or `"unknown_xml"` (looks like XML but neither recognized root element).

    Deliberately a lightweight *first-tag-name* scan (regex, skipping `<?xml ...?>` declarations
    and `<!-- ... -->` comments), not a full parse: a **malformed** PAGE/ALTO file (this parser's
    own `malformed_xml` failure category) must still be routed to the correct format parser so
    that parser reports the honest, specific `malformed_xml` category -- sniffing via a full parse
    would instead swallow the real parse error here and misreport it as `unsupported_format`,
    which would hide the actual problem from a caller (Constitution Article 6: Full Exposure)."""
    stripped = text.lstrip()
    if not stripped.startswith("<"):
        return "plain_text"
    match = _FIRST_TAG_RE.search(stripped)
    if match is None:
        return "unknown_xml"
    local = _localname(match.group(1))
    if local == "PcGts":
        return "page_xml"
    if local == "alto":
        return "alto_xml"
    return "unknown_xml"


def parse_export_file(source_file_path: str, *, export_format: str | None = None) -> tuple[ParsedDocument, str]:
    """Reads `source_file_path` from local disk (the only I/O this function performs -- no
    network call, ever) and parses it. Returns `(parsed_document, raw_file_text)` -- the raw text
    is returned alongside so the caller can content-address it as `Evidence.raw_output` unmodified
    (docs/htr-migration-plan.md Stage 8: "original export file ... preserved, content-addressed
    like other raw evidence").

    `export_format`, if given, skips content-sniffing (useful when the caller already knows the
    format, e.g. from `ExternalImportFormat`); otherwise the format is sniffed from content.
    """
    path = Path(source_file_path)
    if not path.exists():
        raise TranskribusParseError("file_not_found", f"Import file not found: {source_file_path}")
    if not path.is_file():
        raise TranskribusParseError("file_not_found", f"Import path is not a file: {source_file_path}")

    raw_text = path.read_text(encoding="utf-8")

    resolved_format = export_format or sniff_format(raw_text)

    if resolved_format == "page_xml":
        return parse_page_xml(raw_text), raw_text
    if resolved_format == "alto_xml":
        return parse_alto_xml(raw_text), raw_text
    if resolved_format == "plain_text":
        return parse_plain_text(raw_text), raw_text

    raise TranskribusParseError(
        "unsupported_format",
        f"Could not determine a supported export format for {source_file_path!r} "
        f"(sniffed: {resolved_format!r}; supported: page_xml, alto_xml, plain_text)",
    )
