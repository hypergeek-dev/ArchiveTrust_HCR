"""Hostile-input admission policy for the supported municipal workstation profile."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class HostileInputRejected(ValueError):
    """The source is not admitted to the immutable archive."""


class InputSecurityPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_version: int = 1
    max_file_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    max_pdf_pages: int = Field(default=1_000, ge=1)
    max_page_dimension_points: float = Field(default=20_000.0, gt=0)
    allowed_extensions: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".html", ".htm")
    reject_pdf_embedded_files: bool = True


_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
)
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._()\- ]+")


def normalized_filename(name: str) -> str:
    """Return a basename safe for archive storage; never preserve path components."""

    basename = Path(name.replace("\\", "/")).name.strip().rstrip(". ")
    basename = _UNSAFE_FILENAME.sub("_", basename)
    if not basename or basename in {".", ".."}:
        raise HostileInputRejected("source filename is empty or unsafe")
    if Path(basename).stem.upper() in _WINDOWS_RESERVED:
        raise HostileInputRejected("source filename is reserved by Windows")
    return basename[:240]


def _signature_ok(path: Path, suffix: str) -> bool:
    with path.open("rb") as handle:
        head = handle.read(16)
    if suffix == ".pdf":
        return head.startswith(b"%PDF-")
    if suffix == ".png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return head.startswith(b"\xff\xd8\xff")
    if suffix in {".tif", ".tiff"}:
        return head.startswith((b"II*\x00", b"MM\x00*"))
    if suffix in {".html", ".htm"}:
        normalized = head.lstrip().lower()
        return normalized.startswith((b"<!doctype html", b"<html"))
    return False


def _pdf_has_embedded_file_marker(path: Path) -> bool:
    marker = b"/EmbeddedFile"
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            candidate = overlap + chunk
            if marker in candidate:
                return True
            overlap = candidate[-len(marker) :]
    return False


def validate_source_file(path: Path, *, original_filename: str, policy: InputSecurityPolicy) -> str:
    """Validate metadata and magic bytes before hashing/copying; return the safe filename."""

    if path.is_symlink():
        raise HostileInputRejected("symbolic-link inputs are not accepted")
    if not path.is_file():
        raise HostileInputRejected("source is not a regular file")
    safe_name = normalized_filename(original_filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in policy.allowed_extensions:
        raise HostileInputRejected(f"file type {suffix or '[none]'} is not allowed")
    size = path.stat().st_size
    if size <= 0 or size > policy.max_file_bytes:
        raise HostileInputRejected(f"source size {size} is outside the allowed range")
    if not _signature_ok(path, suffix):
        raise HostileInputRejected("file content does not match its declared type")
    if suffix == ".pdf" and policy.reject_pdf_embedded_files and _pdf_has_embedded_file_marker(path):
        raise HostileInputRejected("PDF embedded files are not accepted")
    return safe_name
