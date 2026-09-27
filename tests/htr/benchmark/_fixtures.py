"""Tiny synthetic deliveries for benchmark tests (no real archive data)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageDraw


def line_png(text: str = "x", size: tuple[int, int] = (240, 40), seed: int = 0) -> bytes:
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    for i, _ in enumerate(text):
        x = 6 + (i * 13 + seed * 7) % (size[0] - 12)
        draw.rectangle((x, 10 + seed % 5, x + 6, 28), fill=20)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def page_jpg(size: tuple[int, int] = (800, 1000)) -> bytes:
    image = Image.new("RGB", size, (240, 235, 220))
    draw = ImageDraw.Draw(image)
    for row in range(5):
        draw.rectangle((50, 100 + row * 120, 700, 140 + row * 120), fill=(40, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def write(root: Path, relative: str, data: bytes | str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def page_xml(image_filename: str, lines: list[tuple[str, str | list[str] | None, str | None]], *, width=800, height=1000,
             reading_order: list[str] | None = None, table_line: tuple[str, str, str] | None = None) -> str:
    """`lines`: (line_id, text | [alternatives] | None, points)."""
    body = []
    for line_id, text, points in lines:
        coords = f'<Coords points="{points}"/>' if points else ""
        if isinstance(text, list):
            equiv = "".join(f"<TextEquiv><Unicode>{t}</Unicode></TextEquiv>" for t in text)
        elif text is None:
            equiv = ""
        else:
            equiv = f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv>"
        body.append(f'<TextLine id="{line_id}">{coords}{equiv}</TextLine>')
    order = ""
    if reading_order:
        refs = "".join(f'<RegionRefIndexed index="{i}" regionRef="{r}"/>' for i, r in enumerate(reading_order))
        order = f'<ReadingOrder><OrderedGroup id="ro">{refs}</OrderedGroup></ReadingOrder>'
    table = ""
    if table_line:
        line_id, text, points = table_line
        table = (f'<TableRegion id="t1"><TableCell id="c1"><TextRegion id="r_cell"><TextLine id="{line_id}">'
                 f'<Coords points="{points}"/><TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine></TextRegion></TableCell></TableRegion>')
    return (f'<?xml version="1.0" encoding="UTF-8"?><PcGts xmlns="{PAGE_NS}"><Page imageFilename="{image_filename}" '
            f'imageWidth="{width}" imageHeight="{height}">{order}<TextRegion id="r1">{"".join(body)}</TextRegion>{table}</Page></PcGts>')


def box(x0: int, y0: int, x1: int, y1: int) -> str:
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"
