"""PAGE XML parsing for Loghi pipeline output.

Uses stdlib `xml.etree.ElementTree` with namespace-tolerant local-name matching, same technique as
`providers/transkribus/page_xml.py` (see `parsing_models.py`'s module docstring for why this is a
parallel implementation rather than a shared import). `source_xml_hash` is computed over the exact
input text handed in, so the untouched export is always independently verifiable against the parsed
result.
"""

from __future__ import annotations

import hashlib
import re
from xml.etree import ElementTree as ET

from archivetrust.providers.loghi.parsing_models import (
    LoghiParsedLine,
    LoghiParsedRegion,
    LoghiPageParseResult,
    LoghiParseError,
)

_READING_ORDER_RE = re.compile(r"readingOrder\s*\{\s*index\s*:\s*(\d+)\s*;?\s*\}")
_PAGE_NAMESPACE_RE = re.compile(r"\{(http://schema\.primaresearch\.org/PAGE/gts/pagecontent/[\d-]+)\}")

_OMITTED_FIELDS = ("per_glyph_geometry", "text_style", "reading_order_alternatives")


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_local(parent: ET.Element, local_name: str) -> ET.Element | None:
    for child in parent:
        if _localname(child.tag) == local_name:
            return child
    return None


def _parse_points(points_attr: str | None) -> tuple[tuple[float, float], ...] | None:
    if not points_attr or not points_attr.strip():
        return None
    coords: list[tuple[float, float]] = []
    for pair in points_attr.strip().split():
        if "," not in pair:
            continue
        x_str, _, y_str = pair.partition(",")
        try:
            coords.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return tuple(coords) if coords else None


def _custom_reading_order_index(custom_attr: str | None) -> int | None:
    if not custom_attr:
        return None
    match = _READING_ORDER_RE.search(custom_attr)
    return int(match.group(1)) if match else None


def _schema_version(root_tag: str) -> str | None:
    match = _PAGE_NAMESPACE_RE.search(root_tag)
    return match.group(1) if match else None


def parse_loghi_page_xml(xml_text: str) -> LoghiPageParseResult:
    """Parses one Loghi-produced PAGE XML document. Raises `LoghiParseError` (never a bare stdlib
    exception) for every malformed-input case this parser recognizes."""
    if not xml_text or not xml_text.strip():
        raise LoghiParseError("empty_file", "Loghi PAGE XML output is empty")

    source_xml_hash = hashlib.sha256(xml_text.encode("utf-8")).hexdigest()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise LoghiParseError("malformed_xml", f"Loghi PAGE XML failed to parse: {exc}") from exc

    if _localname(root.tag) != "PcGts":
        raise LoghiParseError(
            "missing_required_element",
            f"Expected PAGE XML root element <PcGts>, found <{_localname(root.tag)}>",
        )

    page_schema_version = _schema_version(root.tag)

    page_el = None
    for el in root.iter():
        if _localname(el.tag) == "Page":
            page_el = el
            break
    if page_el is None:
        raise LoghiParseError("missing_required_element", "Loghi PAGE XML has no <Page> element")

    def _to_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    page_width = _to_int(page_el.get("imageWidth"))
    page_height = _to_int(page_el.get("imageHeight"))
    image_filename = page_el.get("imageFilename")

    explicit_region_order: dict[str, int] = {}
    for reading_order_el in root.iter():
        if _localname(reading_order_el.tag) == "ReadingOrder":
            fallback_idx = 0
            for ref_el in reading_order_el.iter():
                if _localname(ref_el.tag) == "RegionRefIndexed":
                    region_ref = ref_el.get("regionRef")
                    index_attr = _to_int(ref_el.get("index"))
                    if region_ref is not None:
                        explicit_region_order[region_ref] = (
                            index_attr if index_attr is not None else fallback_idx
                        )
                    fallback_idx += 1
            break

    creator_metadata: str | None = None
    processing_date: str | None = None
    metadata_el = _find_local(root, "Metadata")
    if metadata_el is not None:
        creator_el = _find_local(metadata_el, "Creator")
        if creator_el is not None and creator_el.text:
            creator_metadata = creator_el.text.strip()
        for child in metadata_el:
            if _localname(child.tag) in ("LastChange", "Created") and child.text and child.text.strip():
                processing_date = child.text.strip()

    regions: list[LoghiParsedRegion] = []
    warnings: list[str] = []
    region_fallback_idx = 0
    for region_el in page_el:
        if _localname(region_el.tag) != "TextRegion":
            continue
        region_id = region_el.get("id") or f"region_{region_fallback_idx}"

        region_polygon = None
        coords_el = _find_local(region_el, "Coords")
        if coords_el is not None:
            region_polygon = _parse_points(coords_el.get("points"))

        lines: list[LoghiParsedLine] = []
        line_fallback_idx = 0
        for line_el in region_el:
            if _localname(line_el.tag) != "TextLine":
                continue
            line_id = line_el.get("id") or f"{region_id}_line_{line_fallback_idx}"

            line_polygon = None
            baseline = None
            text = ""
            confidence = None
            for line_child in line_el:
                local = _localname(line_child.tag)
                if local == "Coords":
                    line_polygon = _parse_points(line_child.get("points"))
                elif local == "Baseline":
                    baseline = _parse_points(line_child.get("points"))
                elif local == "TextEquiv":
                    conf_attr = line_child.get("conf")
                    if conf_attr is not None:
                        try:
                            confidence = float(conf_attr)
                        except ValueError:
                            warnings.append(f"TextLine {line_id!r}: non-numeric conf {conf_attr!r} ignored")
                    unicode_el = _find_local(line_child, "Unicode")
                    if unicode_el is not None:
                        text = unicode_el.text or ""

            custom_line_idx = _custom_reading_order_index(line_el.get("custom"))
            reading_order_index = custom_line_idx if custom_line_idx is not None else line_fallback_idx
            lines.append(
                LoghiParsedLine(
                    line_id=line_id,
                    text=text,
                    confidence=confidence,
                    polygon=line_polygon,
                    baseline=baseline,
                    reading_order_index=reading_order_index,
                )
            )
            line_fallback_idx += 1

        if region_id in explicit_region_order:
            region_reading_order_index = explicit_region_order[region_id]
        else:
            custom_region_idx = _custom_reading_order_index(region_el.get("custom"))
            region_reading_order_index = (
                custom_region_idx if custom_region_idx is not None else region_fallback_idx
            )

        regions.append(
            LoghiParsedRegion(
                region_id=region_id,
                region_type=region_el.get("type"),
                polygon=region_polygon,
                reading_order_index=region_reading_order_index,
                lines=tuple(lines),
            )
        )
        region_fallback_idx += 1

    if not regions:
        warnings.append("Loghi PAGE XML contained no <TextRegion> elements")

    return LoghiPageParseResult(
        source_xml_hash=source_xml_hash,
        page_schema_version=page_schema_version,
        page_width=page_width,
        page_height=page_height,
        image_filename=image_filename,
        regions=tuple(regions),
        creator_metadata=creator_metadata,
        processing_date=processing_date,
        warnings=tuple(warnings),
        omitted_fields=_OMITTED_FIELDS,
    )
