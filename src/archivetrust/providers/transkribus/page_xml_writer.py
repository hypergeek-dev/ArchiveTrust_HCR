"""PAGE XML *writing* -- the inverse of `page_xml.py`, for handing pre-detected layout to
Transkribus.

`page_xml.py` reads what Transkribus produced. This module writes what Transkribus is *given*: a
PAGE XML file carrying region and line geometry with **empty `TextEquiv`**, so a researcher can run
Transkribus's "recognition only / skip layout analysis" workflow against layout ArchiveTrust already
detected, instead of letting Transkribus re-segment the page.

**This module still uploads nothing.** It writes local files, exactly like
`htr/preprocessing/export_package.py`. `providers/transkribus/README.md`'s "manual import only,
never automatic" constraint is unchanged and unchallenged by anything here.

## Why empty `TextEquiv` elements are written at all, rather than omitted

An absent `TextEquiv` and an empty one are different claims. Writing
`<TextEquiv><Unicode></Unicode></TextEquiv>` states positively that this line's text is *not yet
known* and is the slot a recognizer fills; omitting the element entirely would leave a reader unable
to distinguish "no text yet" from "this file does not carry text at all". `page_xml.py` parses an
empty `Unicode` to `text=""`, so a round-trip through this repository's own parser reproduces
exactly the emptiness that was written -- asserted in the tests rather than assumed.

## Namespace

Written in the `2013-07-15` PAGE schema namespace, the revision Transkribus itself emits and the one
most broadly accepted by its importer. `page_xml.py` compares *local* tag names and is deliberately
schema-version-tolerant, so it parses this output regardless; the pinned namespace is for
Transkribus's benefit, not this repository's.

## Geometry

`TextLine/Coords` is written as the four corners of the detected axis-aligned box, and `Baseline` as
the horizontal segment at the box's bottom edge. Both are honest renderings of what the detector
actually produced: `htr/segmentation/` emits axis-aligned rectangles, not polygons, so no polygon
detail is invented here, and the baseline is derived from the box rather than measured -- which is
recorded in the file's own `MetadataItem` rather than left for a reader to discover.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
_SCHEMA_LOCATION = (
    f"{PAGE_XML_NAMESPACE} "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
)

BASELINE_DERIVATION_NOTE = (
    "Baselines are derived from the bottom edge of each detected axis-aligned line box, not "
    "measured from ink. The detector (htr/segmentation, Florence-2 <OD>) emits rectangles, not "
    "polygons or baselines."
)

TEXT_EQUIV_EMPTY_NOTE = (
    "Every TextEquiv/Unicode is deliberately empty: this file carries layout only, for a "
    "recognition-only Transkribus run. Any text present after processing was produced by "
    "Transkribus, not by ArchiveTrust."
)


@dataclass(frozen=True)
class PageXmlLine:
    """One line to write: its box, its order, and the `InputCrop` it corresponds to.

    `input_crop_id`/`input_crop_hash` are carried into the XML as `MetadataItem`s *and* into the
    manifest, so the correspondence survives even if the two files are separated. They are recorded,
    never relied on for identity: Transkribus does not preserve or return them (see
    `export_package.py`'s correspondence caveat).
    """

    line_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    reading_order_index: int
    input_crop_id: str | None = None
    input_crop_hash: str | None = None


@dataclass(frozen=True)
class PageXmlRegion:
    """One region to write, with its lines in reading order."""

    region_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    reading_order_index: int
    lines: tuple[PageXmlLine, ...]
    region_type: str | None = None


def _points(x0: float, y0: float, x1: float, y1: float) -> str:
    """The four corners of an axis-aligned box, clockwise from top-left, as PAGE `points`."""
    left, top = int(round(x0)), int(round(y0))
    right, bottom = int(round(x1)), int(round(y1))
    return f"{left},{top} {right},{top} {right},{bottom} {left},{bottom}"


def _baseline_points(x0: float, y1: float, x1: float) -> str:
    bottom = int(round(y1))
    return f"{int(round(x0))},{bottom} {int(round(x1))},{bottom}"


def build_layout_only_page_xml(
    *,
    image_filename: str,
    image_width: int,
    image_height: int,
    regions: tuple[PageXmlRegion, ...],
    created_at: str,
    creator: str = "ArchiveTrust htr/segmentation",
    segmentation_adapter_name: str | None = None,
    extra_metadata: tuple[tuple[str, str], ...] = (),
) -> str:
    """Serializes detected layout as a PAGE XML document with no transcribed text.

    Raises `ValueError` for an empty region set: a layout-only file with no layout in it would claim
    a handover that carries nothing, the same reasoning `ExportPackageManifest` rejects empty
    `entries` for.
    """
    if not regions:
        raise ValueError(
            "build_layout_only_page_xml requires at least one region -- a layout-only PAGE XML "
            "with no regions would tell Transkribus nothing"
        )

    ET.register_namespace("", PAGE_XML_NAMESPACE)
    root = ET.Element(
        f"{{{PAGE_XML_NAMESPACE}}}PcGts",
        {
            "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation": _SCHEMA_LOCATION,
        },
    )

    metadata = ET.SubElement(root, f"{{{PAGE_XML_NAMESPACE}}}Metadata")
    ET.SubElement(metadata, f"{{{PAGE_XML_NAMESPACE}}}Creator").text = creator
    ET.SubElement(metadata, f"{{{PAGE_XML_NAMESPACE}}}Created").text = created_at
    ET.SubElement(metadata, f"{{{PAGE_XML_NAMESPACE}}}LastChange").text = created_at

    items: list[tuple[str, str]] = [
        ("layoutSource", "ArchiveTrust htr/segmentation (pre-detected; do not re-segment)"),
        ("textEquivPolicy", TEXT_EQUIV_EMPTY_NOTE),
        ("baselineDerivation", BASELINE_DERIVATION_NOTE),
    ]
    if segmentation_adapter_name:
        items.append(("segmentationAdapter", segmentation_adapter_name))
    items.extend(extra_metadata)
    for name, value in items:
        ET.SubElement(
            metadata,
            f"{{{PAGE_XML_NAMESPACE}}}MetadataItem",
            {"type": "processingStep", "name": name, "value": value},
        )

    page = ET.SubElement(
        root,
        f"{{{PAGE_XML_NAMESPACE}}}Page",
        {
            "imageFilename": image_filename,
            "imageWidth": str(image_width),
            "imageHeight": str(image_height),
        },
    )

    # The schema's dedicated reading-order mechanism, which page_xml.py resolves at highest
    # priority -- written explicitly rather than relying on document order.
    reading_order = ET.SubElement(page, f"{{{PAGE_XML_NAMESPACE}}}ReadingOrder")
    ordered_group = ET.SubElement(
        reading_order, f"{{{PAGE_XML_NAMESPACE}}}OrderedGroup", {"id": "reading_order_root"}
    )
    for region in sorted(regions, key=lambda r: r.reading_order_index):
        ET.SubElement(
            ordered_group,
            f"{{{PAGE_XML_NAMESPACE}}}RegionRefIndexed",
            {"regionRef": region.region_id, "index": str(region.reading_order_index)},
        )

    for region in sorted(regions, key=lambda r: r.reading_order_index):
        attrs = {
            "id": region.region_id,
            "custom": f"readingOrder {{index:{region.reading_order_index};}}",
        }
        if region.region_type:
            attrs["type"] = region.region_type
        region_el = ET.SubElement(page, f"{{{PAGE_XML_NAMESPACE}}}TextRegion", attrs)
        ET.SubElement(
            region_el,
            f"{{{PAGE_XML_NAMESPACE}}}Coords",
            {"points": _points(region.x0, region.y0, region.x1, region.y1)},
        )

        for line in sorted(region.lines, key=lambda line: line.reading_order_index):
            line_attrs = {
                "id": line.line_id,
                "custom": f"readingOrder {{index:{line.reading_order_index};}}",
            }
            line_el = ET.SubElement(region_el, f"{{{PAGE_XML_NAMESPACE}}}TextLine", line_attrs)
            ET.SubElement(
                line_el,
                f"{{{PAGE_XML_NAMESPACE}}}Coords",
                {"points": _points(line.x0, line.y0, line.x1, line.y1)},
            )
            ET.SubElement(
                line_el,
                f"{{{PAGE_XML_NAMESPACE}}}Baseline",
                {"points": _baseline_points(line.x0, line.y1, line.x1)},
            )
            text_equiv = ET.SubElement(line_el, f"{{{PAGE_XML_NAMESPACE}}}TextEquiv")
            # Empty, on purpose -- see module docstring. `.text = ""` (not None) so the element
            # serializes as <Unicode></Unicode> rather than a self-closing <Unicode/>, which is what
            # Transkribus's importer and this repository's own parser both read as "empty string".
            ET.SubElement(text_equiv, f"{{{PAGE_XML_NAMESPACE}}}Unicode").text = ""

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
