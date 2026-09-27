"""PAGE XML and ALTO XML line readers for ground-truth import.

Deliberately *not* `providers.transkribus.parse_page_xml` / `parse_alto_xml`: those serve the
Transkribus-export provider and, for that purpose, walk only top-level `TextRegion`s (lines in table
cells or nested regions are skipped), keep the last of several `TextEquiv`s, and strip ALTO line
text. For ground truth each of those would silently drop or alter reference text, so these readers
visit every `TextLine`, refuse to choose between competing transcriptions, and never trim.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path, PurePosixPath

from archivetrust.htr.benchmark.findings import Finding
from archivetrust.htr.benchmark.sources.base import IMAGE_EXTENSIONS, CandidateLine, Extraction

_UNORDERED = 10**9


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _root_local(root: Path, name: str) -> str | None:
    try:
        for _event, element in ET.iterparse(root / name, events=("start",)):
            return _local(element.tag)
    except ET.ParseError:
        return None
    return None


def _points(value: str | None) -> tuple[tuple[float, float], ...] | None:
    if not value or not value.strip():
        return None
    out = []
    for pair in value.split():
        x, y = pair.split(",")
        out.append((float(x), float(y)))
    return tuple(out)


def _int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def resolve_image(root: Path, xml_rel: str, image_filename: str | None, names: set[str]) -> str | None:
    """Finds the page image an XML file refers to: its declared filename next to the XML or one level
    up (Transkribus `page/` layout), then any image with the XML's stem in those two places."""
    xml_dir = PurePosixPath(xml_rel).parent
    candidates: list[PurePosixPath] = []
    if image_filename:
        base = PurePosixPath(image_filename.replace("\\", "/")).name
        candidates += [xml_dir / base, xml_dir.parent / base]
    stem = PurePosixPath(xml_rel).stem
    for directory in (xml_dir, xml_dir.parent):
        candidates += [directory / f"{stem}{ext}" for ext in sorted(IMAGE_EXTENSIONS)]
    for candidate in candidates:
        if str(candidate) in names:
            return str(candidate)
    return None


def _document_and_page(image_rel: str | None, xml_rel: str) -> tuple[str, str]:
    anchor = PurePosixPath(image_rel or xml_rel)
    parent = str(anchor.parent)
    return ("root" if parent == "." else parent), anchor.stem


class PageXmlAdapter:
    adapter_id = "page_xml"
    version = "1"
    _root_name = "PcGts"

    def _xml_files(self, root: Path, files: list[str]) -> list[str]:
        return [f for f in files if f.lower().endswith(".xml") and _root_local(root, f) == self._root_name]

    def detect(self, root: Path, files: list[str]) -> float:
        xml = [f for f in files if f.lower().endswith(".xml")]
        return len(self._xml_files(root, files)) / len(xml) if xml else 0.0

    def extract(self, root: Path, files: list[str]) -> Extraction:
        out = Extraction(adapter_id=self.adapter_id, adapter_version=self.version)
        names = set(files)
        for xml_rel in self._xml_files(root, files):
            out.consumed_files.add(xml_rel)
            try:
                tree = ET.parse(root / xml_rel)
            except ET.ParseError as exc:
                out.findings.append(Finding(severity="blocker", code="xml.malformed", path=xml_rel, message=str(exc)))
                continue
            self._extract_file(tree.getroot(), root, xml_rel, names, out)
        return out

    def _extract_file(self, xml_root: ET.Element, root: Path, xml_rel: str, names: set[str], out: Extraction) -> None:
        page = next((el for el in xml_root.iter() if _local(el.tag) == "Page"), None)
        if page is None:
            out.findings.append(Finding(severity="blocker", code="xml.no_page", path=xml_rel, message="PAGE XML has no <Page>"))
            return
        image_rel = resolve_image(root, xml_rel, page.get("imageFilename"), names)
        if image_rel is None:
            out.findings.append(Finding(severity="blocker", code="page.image_missing", path=xml_rel,
                                        message=f"page image {page.get('imageFilename')!r} not found next to the XML or one level up"))
        else:
            out.consumed_files.add(image_rel)
        width, height = _int(page.get("imageWidth")), _int(page.get("imageHeight"))
        document, page_name = _document_and_page(image_rel, xml_rel)

        region_rank: dict[str, int] = {}
        for element in page.iter():
            ref = element.get("regionRef")
            if _local(element.tag) == "RegionRefIndexed" and ref:
                region_rank[ref] = _int(element.get("index")) or 0
        if not region_rank:
            out.findings.append(Finding(severity="info", code="layout.no_reading_order", path=xml_rel,
                                        message="no <ReadingOrder>; document order is used"))

        parents = {child: parent for parent in page.iter() for child in parent}
        rows: list[tuple[tuple, CandidateLine]] = []
        seen_ids: set[str] = set()
        for doc_index, line in enumerate(el for el in page.iter() if _local(el.tag) == "TextLine"):
            region, ancestor = None, parents.get(line)
            while ancestor is not None and region is None:
                if _local(ancestor.tag).endswith("Region"):
                    region = ancestor
                ancestor = parents.get(ancestor)
            region_id = region.get("id") if region is not None else None
            line_id = line.get("id")
            if not line_id:
                line_id = f"line{doc_index:04d}"
                out.findings.append(Finding(severity="warning", code="layout.generated_line_id", path=xml_rel,
                                            message=f"TextLine #{doc_index} has no id; using {line_id}"))
            if line_id in seen_ids:
                out.findings.append(Finding(severity="blocker", code="layout.duplicate_line_id", path=xml_rel,
                                            message=f"TextLine id {line_id!r} occurs more than once"))
                continue
            seen_ids.add(line_id)

            polygon = baseline = None
            equivs: list[str] = []
            for child in line:
                local = _local(child.tag)
                try:
                    if local == "Coords":
                        polygon = _points(child.get("points"))
                    elif local == "Baseline":
                        baseline = _points(child.get("points"))
                except ValueError:
                    out.findings.append(Finding(severity="needs_review", code="layout.malformed_points", path=xml_rel,
                                                message=f"TextLine {line_id!r}: unparseable {local} points"))
                if local == "TextEquiv":
                    unicode_el = next((u for u in child if _local(u.tag) == "Unicode"), None)
                    equivs.append(unicode_el.text or "" if unicode_el is not None else "")

            text: str | None = None
            key = f"{document}/{page_name}/{line_id}"
            if len(equivs) == 1:
                text = equivs[0]
            elif len(equivs) > 1:
                out.findings.append(Finding(severity="needs_review", code="gt.multiple_textequiv", path=xml_rel, line_id=key,
                                            message=f"TextLine {line_id!r} has {len(equivs)} TextEquiv; which is ground truth must be decided",
                                            detail={"alternatives": equivs}))
            else:
                has_words = any(_local(el.tag) == "TextEquiv" for w in line if _local(w.tag) == "Word" for el in w)
                out.findings.append(Finding(
                    severity="needs_review", code="gt.word_level_only" if has_words else "gt.missing", path=xml_rel, line_id=key,
                    message=f"TextLine {line_id!r} has no line-level TextEquiv" + (" (only word-level text)" if has_words else ""),
                ))
            custom = line.get("custom") or ""
            line_rank = _UNORDERED
            if "readingOrder" in custom and "index:" in custom:
                line_rank = _int(custom.split("readingOrder", 1)[1].split("index:", 1)[1].split(";", 1)[0]) or 0
            sort = (region_rank.get(region_id or "", _UNORDERED), line_rank, doc_index)
            rows.append((sort, CandidateLine(
                document_raw=document, page_raw=page_name, line_raw=line_id, line_order=0, image_kind="page",
                image_path=image_rel or "", gt_path=xml_rel, gt_source=text, polygon=polygon, baseline=baseline,
                declared_page_size=(width, height) if width and height else None, source_line_ref=line_id,
                metadata={"region_id": region_id, "region_type": _local(region.tag) if region is not None else None,
                          "custom": custom or None},
            )))
        if not rows:
            out.findings.append(Finding(severity="warning", code="layout.no_lines", path=xml_rel, message="page has no TextLine"))
        for order, (_sort, candidate) in enumerate(sorted(rows, key=lambda r: r[0])):
            out.lines.append(replace(candidate, line_order=order))


class AltoAdapter(PageXmlAdapter):
    """ALTO: `String/@CONTENT` tokens joined with `SP` as a space and `HYP/@CONTENT` kept verbatim.
    Consecutive `String`s with no `SP` between them are joined by one space (recorded per line as
    `alto_implicit_space`). Only pixel `MeasurementUnit` coordinates can be cropped."""

    adapter_id = "alto_xml"
    version = "1"
    _root_name = "alto"

    def _extract_file(self, xml_root: ET.Element, root: Path, xml_rel: str, names: set[str], out: Extraction) -> None:
        unit_el = next((el for el in xml_root.iter() if _local(el.tag) == "MeasurementUnit"), None)
        unit = (unit_el.text or "").strip() if unit_el is not None else None
        if unit != "pixel":
            out.findings.append(Finding(severity="blocker", code="alto.non_pixel_units", path=xml_rel,
                                        message=f"MeasurementUnit is {unit!r}; only 'pixel' coordinates can be cropped without a DPI decision"))
        file_el = next((el for el in xml_root.iter() if _local(el.tag) == "fileName"), None)
        page = next((el for el in xml_root.iter() if _local(el.tag) == "Page"), None)
        image_rel = resolve_image(root, xml_rel, file_el.text.strip() if file_el is not None and file_el.text else None, names)
        if image_rel is None:
            out.findings.append(Finding(severity="blocker", code="page.image_missing", path=xml_rel,
                                        message="page image not found next to the XML or one level up"))
        else:
            out.consumed_files.add(image_rel)
        width = _int(page.get("WIDTH")) if page is not None else None
        height = _int(page.get("HEIGHT")) if page is not None else None
        document, page_name = _document_and_page(image_rel, xml_rel)
        order = 0
        seen: set[str] = set()
        for doc_index, line in enumerate(el for el in xml_root.iter() if _local(el.tag) == "TextLine"):
            line_id = line.get("ID") or f"line{doc_index:04d}"
            if line_id in seen:
                out.findings.append(Finding(severity="blocker", code="layout.duplicate_line_id", path=xml_rel,
                                            message=f"TextLine ID {line_id!r} occurs more than once"))
                continue
            seen.add(line_id)
            parts: list[str] = []
            implicit_space = False
            previous = None
            for child in line:
                local = _local(child.tag)
                if local == "String":
                    if previous == "String":
                        parts.append(" ")
                        implicit_space = True
                    parts.append(child.get("CONTENT") or "")
                elif local == "SP":
                    parts.append(" ")
                elif local == "HYP":
                    parts.append(child.get("CONTENT") or "")
                if local in ("String", "SP", "HYP"):
                    previous = local
            polygon = None
            shape = next((el for el in line.iter() if _local(el.tag) == "Polygon"), None)
            shape_points = shape.get("POINTS") if shape is not None else None
            if shape_points:
                raw = shape_points.replace(",", " ").split()
                polygon = tuple((float(raw[i]), float(raw[i + 1])) for i in range(0, len(raw) - 1, 2))
            else:
                x, y, w, h = (_int(line.get(k)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
                if x is not None and y is not None and w is not None and h is not None:
                    polygon = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
            out.lines.append(CandidateLine(
                document_raw=document, page_raw=page_name, line_raw=line_id, line_order=order, image_kind="page",
                image_path=image_rel or "", gt_path=xml_rel, gt_source="".join(parts) if parts else None,
                polygon=polygon, declared_page_size=(width, height) if width and height else None, source_line_ref=line_id,
                metadata={"alto_implicit_space": implicit_space, "alto_baseline": line.get("BASELINE")},
            ))
            if not parts:
                out.findings.append(Finding(severity="needs_review", code="gt.missing", path=xml_rel,
                                            line_id=f"{document}/{page_name}/{line_id}", message=f"TextLine {line_id!r} has no String content"))
            order += 1
