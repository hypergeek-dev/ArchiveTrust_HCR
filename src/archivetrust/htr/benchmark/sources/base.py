"""Source adapters: turn one incoming delivery into format-neutral `CandidateLine`s plus findings.

An adapter only *reads* and *describes*. It never fixes GT, never guesses an alignment, and never
writes anywhere; everything it is unsure about becomes a `Finding` so the review queue can carry it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from archivetrust.htr.benchmark.findings import Finding

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jp2", ".gif"})
IGNORED_NAMES = frozenset({"thumbs.db", ".ds_store", "desktop.ini"})
IGNORED_DIRS = frozenset({"__macosx", ".git"})


@dataclass(frozen=True)
class CandidateLine:
    document_raw: str
    page_raw: str
    line_raw: str
    line_order: int
    image_kind: Literal["line", "page"]
    image_path: str
    """Relative (forward slashes) to the source root. For embedded images (parquet) a pseudo-path
    `<file>#row=<n>` and the bytes are in `image_bytes`."""
    gt_path: str
    gt_source: str | None
    """None means no transcription was found for this line (always a finding)."""
    gt_from_line_file: bool = False
    polygon: tuple[tuple[float, float], ...] | None = None
    baseline: tuple[tuple[float, float], ...] | None = None
    declared_page_size: tuple[int, int] | None = None
    source_line_ref: str | None = None
    collection: str | None = None
    writer_id: str | None = None
    metadata: dict = field(default_factory=dict)
    image_bytes: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> str:
        return f"{self.document_raw}|{self.page_raw}|{self.line_raw}"


@dataclass
class Extraction:
    adapter_id: str
    adapter_version: str
    lines: list[CandidateLine] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    consumed_files: set[str] = field(default_factory=set)
    """Relative paths this adapter accounted for; everything else in the delivery is reported as
    unused so nothing is silently ignored."""


class SourceAdapter(Protocol):
    adapter_id: str
    version: str

    def detect(self, root: Path, files: list[str]) -> float:
        """0.0 (not this format) .. 1.0 (certainly this format)."""
        ...

    def extract(self, root: Path, files: list[str]) -> Extraction: ...


def rel(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root).as_posix()).as_posix()


def list_files(root: Path) -> list[str]:
    """Every regular file under `root`, relative, sorted, excluding OS junk (which inspection reports
    separately)."""
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = [p.lower() for p in path.relative_to(root).parts]
        if parts[-1] in IGNORED_NAMES or any(p in IGNORED_DIRS for p in parts[:-1]) or parts[-1].startswith("._"):
            continue
        out.append(rel(path, root))
    return out


def is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS


def read_text_file(root: Path, relative: str) -> tuple[str | None, list[Finding]]:
    """Strict UTF-8 (a BOM is accepted and reported). Never guesses another encoding: a non-UTF-8
    transcription is a blocker, with a hint if it decodes as cp1252."""
    data = (root / relative).read_bytes()
    findings: list[Finding] = []
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
        findings.append(Finding(severity="warning", code="text.utf8_bom", path=relative,
                                message="UTF-8 byte-order mark present (removed when decoding; file encoding, not text)"))
    try:
        return data.decode("utf-8"), findings
    except UnicodeDecodeError as exc:
        hint = ""
        try:
            data.decode("cp1252")
            hint = " (decodes as cp1252 -- re-export as UTF-8 or confirm the encoding)"
        except UnicodeDecodeError:
            pass
        findings.append(Finding(severity="blocker", code="text.not_utf8", path=relative,
                                message=f"not valid UTF-8 at byte {exc.start}{hint}"))
        return None, findings
