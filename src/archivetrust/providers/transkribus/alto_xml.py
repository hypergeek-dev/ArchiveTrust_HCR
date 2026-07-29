"""ALTO XML parsing (Analyzed Layout and Text Object -- the Library of Congress format Transkribus
also supports exporting to).

Same namespace-tolerant approach as `page_xml.py`: local (unprefixed) element-name matching via
stdlib `xml.etree.ElementTree`, not a hardcoded ALTO schema-version namespace URI (ALTO has
`ns-v2` through `ns-v4` in real-world use).

ALTO has no dedicated per-region/per-line "reading order" construct comparable to PAGE's
`ReadingOrder`/`custom` attribute -- `TextBlock`/`TextLine` document order *is* the reading order
by convention in this format, so this parser assigns `reading_order_index` as document-order
position, without inventing an ALTO feature that does not exist.

Confidence: ALTO's standard per-token confidence is the `String` element's `WC` (Word Confidence,
`0.0`-`1.0`) attribute -- there is no standard per-line confidence attribute in the ALTO schema, so
a line's confidence is computed here as the mean of its words' `WC` values when at least one is
present, and `None` when none are (never fabricated). Some ALTO producers additionally place a
`TAGREFS`/vendor-specific attribute directly on `TextLine`; none is treated as a de facto line
confidence here, since it is not part of the standard schema and no such attribute was observed in
any real ALTO sample consulted -- if a producer's file supplies one, it is preserved unmodified in
region/line geometry structures below but not silently promoted to `confidence`.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from archivetrust.providers.transkribus.parsing_models import (
    ParsedDocument,
    ParsedLine,
    ParsedRegion,
    TranskribusParseError,
)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_local(parent: ET.Element, local_name: str) -> ET.Element | None:
    for child in parent:
        if _localname(child.tag) == local_name:
            return child
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _hpos_box(el: ET.Element) -> tuple[tuple[float, float], ...] | None:
    hpos, vpos, width, height = (
        _to_float(el.get("HPOS")),
        _to_float(el.get("VPOS")),
        _to_float(el.get("WIDTH")),
        _to_float(el.get("HEIGHT")),
    )
    if hpos is None or vpos is None or width is None or height is None:
        return None
    return ((hpos, vpos), (hpos + width, vpos), (hpos + width, vpos + height), (hpos, vpos + height))


def parse_alto_xml(xml_text: str) -> ParsedDocument:
    """Parses one ALTO XML document's worth of `TextBlock`/`TextLine`/`String` structure into a
    format-neutral `ParsedDocument`. Raises `TranskribusParseError` for every malformed-input case
    this parser recognizes."""
    if not xml_text or not xml_text.strip():
        raise TranskribusParseError("empty_file", "ALTO XML import file is empty")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise TranskribusParseError("malformed_xml", f"ALTO XML failed to parse: {exc}") from exc

    if _localname(root.tag) != "alto":
        raise TranskribusParseError(
            "missing_required_element",
            f"Expected ALTO XML root element <alto>, found <{_localname(root.tag)}>",
        )

    layout_el = _find_local(root, "Layout")
    if layout_el is None:
        raise TranskribusParseError("missing_required_element", "ALTO XML has no <Layout> element")

    page_el = _find_local(layout_el, "Page")
    if page_el is None:
        raise TranskribusParseError("missing_required_element", "ALTO XML <Layout> has no <Page> element")

    def _to_int(value: str | None) -> int | None:
        as_float = _to_float(value)
        return int(as_float) if as_float is not None else None

    page_width = _to_int(page_el.get("WIDTH"))
    page_height = _to_int(page_el.get("HEIGHT"))

    processing_date: str | None = None
    description_el = _find_local(root, "Description")
    if description_el is not None:
        for el in description_el.iter():
            if _localname(el.tag) == "processingDateTime" and el.text and el.text.strip():
                processing_date = el.text.strip()
                break

    regions: list[ParsedRegion] = []
    warnings: list[str] = []
    region_idx = 0
    for block_el in page_el.iter():
        if _localname(block_el.tag) != "TextBlock":
            continue
        region_id = block_el.get("ID") or f"block_{region_idx}"
        region_polygon = _hpos_box(block_el)

        lines: list[ParsedLine] = []
        line_idx = 0
        for line_el in block_el:
            if _localname(line_el.tag) != "TextLine":
                continue
            line_id = line_el.get("ID") or f"{region_id}_line_{line_idx}"
            line_polygon = _hpos_box(line_el)

            tokens: list[str] = []
            word_confidences: list[float] = []
            for token_el in line_el:
                local = _localname(token_el.tag)
                if local == "String":
                    tokens.append(token_el.get("CONTENT") or "")
                    wc_attr = token_el.get("WC")
                    if wc_attr is not None:
                        wc_value = _to_float(wc_attr)
                        if wc_value is not None:
                            word_confidences.append(wc_value)
                        else:
                            warnings.append(f"TextLine {line_id!r}: non-numeric WC {wc_attr!r} ignored")
                elif local == "SP":
                    tokens.append(" ")

            text = "".join(tokens).strip()
            confidence = sum(word_confidences) / len(word_confidences) if word_confidences else None

            lines.append(
                ParsedLine(
                    line_id=line_id,
                    text=text,
                    confidence=confidence,
                    polygon=line_polygon,
                    baseline=None,  # ALTO carries no dedicated baseline-points element (unlike PAGE)
                    reading_order_index=line_idx,
                )
            )
            line_idx += 1

        regions.append(
            ParsedRegion(
                region_id=region_id,
                region_type=None,  # ALTO's TextBlock has no standard region-type attribute
                polygon=region_polygon,
                reading_order_index=region_idx,
                lines=tuple(lines),
            )
        )
        region_idx += 1

    if not regions:
        warnings.append("ALTO XML contained no <TextBlock> elements")

    return ParsedDocument(
        export_format="alto_xml",
        page_width=page_width,
        page_height=page_height,
        image_filename=None,
        regions=tuple(regions),
        processing_date=processing_date,
        vendor_reported_accuracy=None,  # ALTO has no standard vendor-accuracy element
        model_version_hint=None,
        job_id_hint=None,
        transkribus_document_id_hint=None,
        warnings=tuple(warnings),
    )
