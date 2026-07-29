"""Writes a minimal, hand-built (no third-party PDF-writing dependency), valid multi-page PDF for
rendering tests -- blank pages of a fixed size, enough for `pypdfium2` to open and rasterize.
"""

from __future__ import annotations

from pathlib import Path


def write_minimal_pdf(path: Path, *, num_pages: int = 1, width: float = 200, height: float = 300) -> None:
    objects: list[bytes] = [b"<</Type/Catalog/Pages 2 0 R>>"]
    kids = " ".join(f"{3 + i} 0 R" for i in range(num_pages))
    objects.append(f"<</Type/Pages/Kids[{kids}]/Count {num_pages}>>".encode())
    for _ in range(num_pages):
        objects.append(
            f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {width} {height}]/Resources<<>>>>".encode()
        )

    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{idx} 0 obj".encode())
        body.extend(obj)
        body.extend(b"endobj\n")

    xref_offset = len(body)
    count = len(objects) + 1
    body.extend(f"xref\n0 {count}\n".encode())
    body.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.extend(f"{off:010d} 00000 n \n".encode())
    body.extend(f"trailer<</Size {count}/Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF".encode())

    path.write_bytes(bytes(body))
