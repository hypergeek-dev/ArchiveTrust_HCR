"""PAGE XML parsing (the PRImA Page format Transkribus exports as one of its standard formats).

Uses the Python standard library's `xml.etree.ElementTree` -- a real, namespace-aware parser, not
a hand-rolled regex/string scan. Namespace handling is done by comparing local (unprefixed) tag
names rather than hardcoding one PAGE schema namespace URI: PAGE XML has been revised many times
(`2010-03-19` through `2019-07-15` and beyond) and Transkribus exports may use any of them; the
element/attribute *names* this parser depends on (`PcGts`, `Page`, `TextRegion`, `TextLine`,
`Coords`, `Baseline`, `TextEquiv`, `Unicode`, `Metadata`, `MetadataItem`, `ReadingOrder`,
`OrderedGroup`, `RegionRefIndexed`) have been stable across those revisions, so this is
deliberately schema-version-tolerant rather than pinned to one xsd. `lxml` was considered and
rejected: nothing here needs XPath 2.0, XInclude, or schema validation against the actual PRImA
`.xsd` (which would require vendoring or fetching that schema, both undesirable for a small,
well-understood XML shape) -- stdlib `ElementTree` is sufficient and keeps this adapter
dependency-free.

Reading order is resolved in priority order: (1) an explicit `<ReadingOrder><OrderedGroup>
<RegionRefIndexed regionRef="..." index="N"/>` block, the schema's dedicated mechanism for this;
(2) a `custom="readingOrder {index:N;}"` attribute (Transkribus's own convention, also used by
many PAGE XML tools); (3) document order as a last-resort fallback, applied separately at both
region and line granularity.
"""

from __future__ import annotations

import re

from xml.etree import ElementTree as ET

from archivetrust.providers.transkribus.parsing_models import (
    ParsedDocument,
    ParsedLine,
    ParsedRegion,
    TranskribusParseError,
)

_READING_ORDER_RE = re.compile(r"readingOrder\s*\{\s*index\s*:\s*(\d+)\s*;?\s*\}")


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


def parse_page_xml(xml_text: str) -> ParsedDocument:
    """Parses one PAGE XML document's worth of regions/lines/geometry/confidence/reading order.
    Raises `TranskribusParseError` (never a bare stdlib exception) for every malformed-input case
    this parser recognizes, each carrying a `category` a caller can map to a `FailureRecord`."""
    if not xml_text or not xml_text.strip():
        raise TranskribusParseError("empty_file", "PAGE XML import file is empty")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise TranskribusParseError("malformed_xml", f"PAGE XML failed to parse: {exc}") from exc

    if _localname(root.tag) != "PcGts":
        raise TranskribusParseError(
            "missing_required_element",
            f"Expected PAGE XML root element <PcGts>, found <{_localname(root.tag)}>",
        )

    page_el = None
    for el in root.iter():
        if _localname(el.tag) == "Page":
            page_el = el
            break
    if page_el is None:
        raise TranskribusParseError("missing_required_element", "PAGE XML has no <Page> element")

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

    # Explicit reading-order block, if present (schema's dedicated mechanism -- highest priority).
    explicit_region_order: dict[str, int] = {}
    for reading_order_el in root.iter():
        if _localname(reading_order_el.tag) == "ReadingOrder":
            fallback_idx = 0
            for ref_el in reading_order_el.iter():
                if _localname(ref_el.tag) == "RegionRefIndexed":
                    region_ref = ref_el.get("regionRef")
                    index_attr = _to_int(ref_el.get("index"))
                    if region_ref is not None:
                        explicit_region_order[region_ref] = index_attr if index_attr is not None else fallback_idx
                    fallback_idx += 1
            break

    metadata_el = _find_local(root, "Metadata")
    processing_date: str | None = None
    vendor_reported_accuracy: float | None = None
    model_version_hint: str | None = None
    job_id_hint: str | None = None
    transkribus_document_id_hint: str | None = None
    if metadata_el is not None:
        for child in metadata_el:
            local = _localname(child.tag)
            if local in ("LastChange", "Created") and child.text and child.text.strip():
                processing_date = child.text.strip()
            elif local == "MetadataItem":
                name = child.get("name")
                value = child.get("value")
                if name == "vendorReportedAccuracy" and value is not None:
                    try:
                        vendor_reported_accuracy = float(value)
                    except ValueError:
                        pass
                elif name == "modelVersion" and value is not None:
                    model_version_hint = value
                elif name == "jobId" and value is not None:
                    job_id_hint = value
                elif name == "transkribusDocId" and value is not None:
                    transkribus_document_id_hint = value

    regions: list[ParsedRegion] = []
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

        lines: list[ParsedLine] = []
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
                ParsedLine(
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
            region_reading_order_index = custom_region_idx if custom_region_idx is not None else region_fallback_idx

        regions.append(
            ParsedRegion(
                region_id=region_id,
                region_type=region_el.get("type"),
                polygon=region_polygon,
                reading_order_index=region_reading_order_index,
                lines=tuple(lines),
            )
        )
        region_fallback_idx += 1

    if not regions:
        warnings.append("PAGE XML contained no <TextRegion> elements")

    return ParsedDocument(
        export_format="page_xml",
        page_width=page_width,
        page_height=page_height,
        image_filename=image_filename,
        regions=tuple(regions),
        processing_date=processing_date,
        vendor_reported_accuracy=vendor_reported_accuracy,
        model_version_hint=model_version_hint,
        job_id_hint=job_id_hint,
        transkribus_document_id_hint=transkribus_document_id_hint,
        warnings=tuple(warnings),
    )
