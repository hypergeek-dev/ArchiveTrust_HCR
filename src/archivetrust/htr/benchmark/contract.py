"""The canonical benchmark line contract (`benchmark-line/1`) and its JSONL manifest.

Every source format is converted into exactly this shape before any model sees it, and both models
are run from the same manifest, so "the same input" is checkable: one line = one `line_id` = one
image file whose SHA-256 is recorded here = one canonical reference transcription.

Manifest file rules (so the manifest's own SHA-256 is a stable identity for the dataset):
- UTF-8, one JSON object per line, `\\n` line endings, keys sorted, no ASCII escaping.
- Rows sorted by (document_id, page_id, line_order, line_id).

Hashes are plain lowercase SHA-256 hex digests of the exact bytes (image file bytes; transcription
encoded as UTF-8), with no prefix.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from archivetrust.htr.benchmark.findings import Finding
from archivetrust.htr.benchmark.normalization import PROTOCOL_ID, RULES

LINE_SCHEMA = "benchmark-line/1"

CropPolicy = Literal["supplied_line_image", "bbox_v1", "polygon_mask_v1"]
"""How the line image was produced.

- `supplied_line_image`: the dataset shipped line images; copied byte-identical, never re-cropped.
- `bbox_v1`: axis-aligned bounding box of the line polygon, cut from the full-resolution page decoded
  as RGB, clamped to the raster, saved as PNG with the project's pinned encoder settings -- the same
  policy `florence2_line_detector.crop_lines` uses.
- `polygon_mask_v1`: as `bbox_v1`, with pixels outside the polygon filled white. Only for a recorded
  sensitivity run; never mixed with `bbox_v1` inside one manifest.
"""

_ID_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_id_component(raw: str) -> str:
    """Maps an arbitrary source identifier (file stem, XML id) to a filesystem- and URL-safe ID
    component. Identity is preserved when `raw` is already safe; otherwise unsafe characters become
    `_` and an 8-hex suffix of the raw value's hash is appended, so two different raw IDs can never
    collapse onto the same component. The raw value is always kept in the line's `source_line_ref`
    or `source_metadata`."""
    if _ID_COMPONENT.match(raw):
        return raw
    ascii_ish = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_ish).strip("._-")[:100] or "id"
    if not cleaned[0].isalnum():
        cleaned = "x" + cleaned
    return f"{cleaned}-{sha256_text(raw)[:8]}"


def make_line_id(document_id: str, page_id: str, line_key: str) -> str:
    return f"{document_id}/{page_id}/{line_key}"


PATH_COMPONENT_MAX = 48
"""Line image files are stored under shortened components so benchmark trees stay well inside the
Windows 260-character path limit. IDs themselves are never shortened; the manifest maps them."""


def _path_component(component: str) -> str:
    if len(component) <= PATH_COMPONENT_MAX:
        return component
    return f"{component[:PATH_COMPONENT_MAX - 9]}-{sha256_text(component)[:8]}"


def line_image_relpath(document_id: str, page_id: str, line_key: str, ext: str) -> str:
    return f"lines/{_path_component(document_id)}/{_path_component(page_id)}/{_path_component(line_key)}{ext}"


class CropRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy: CropPolicy
    source_image_relative_path: str | None = None
    """Page image the crop was cut from, relative to the incoming source root (forward slashes).
    None for `supplied_line_image`."""
    source_image_sha256: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    """(x0, y0, x1, y1) in page pixel coordinates after clamping; x1/y1 exclusive."""
    polygon: tuple[tuple[int, int], ...] | None = None
    baseline: tuple[tuple[int, int], ...] | None = None

    @model_validator(mode="after")
    def _check(self) -> CropRecord:
        if self.policy == "supplied_line_image":
            if self.bbox is not None:
                raise ValueError("supplied_line_image crops carry no bbox (the image was not cut here)")
        else:
            if self.bbox is None or self.source_image_relative_path is None or self.source_image_sha256 is None:
                raise ValueError(f"{self.policy} crops need bbox, source_image_relative_path and source_image_sha256")
            x0, y0, x1, y1 = self.bbox
            if not (0 <= x0 < x1 and 0 <= y0 < y1):
                raise ValueError(f"degenerate bbox {self.bbox}")
            if self.policy == "polygon_mask_v1" and not self.polygon:
                raise ValueError("polygon_mask_v1 crops need a polygon")
        return self


class BenchmarkLine(BaseModel):
    """One benchmark line. Frozen; every field is either copied from the source or derived by a
    recorded, deterministic rule."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_id: Literal["benchmark-line/1"] = Field(default=LINE_SCHEMA, alias="schema")

    # --- identity -------------------------------------------------------------------------------
    dataset_id: str
    source_id: str
    """The `incoming/<source_id>` delivery this line came from."""
    document_id: str
    page_id: str
    line_key: str
    line_id: str
    """`{document_id}/{page_id}/{line_key}` -- unique within the dataset."""
    line_order: int = Field(ge=0)
    """Reading-order position within the page (0-based)."""
    collection: str | None = None
    writer_id: str | None = None

    # --- provenance -----------------------------------------------------------------------------
    source_relative_path: str
    """File in the incoming delivery the image came from (line image, or page image)."""
    original_filename: str
    gt_source_relative_path: str
    """File the transcription came from (a .txt, the XML, the CSV...)."""
    source_line_ref: str | None = None
    """Line identifier inside the source file (XML id, CSV row number), verbatim."""
    source_metadata: dict = Field(default_factory=dict)
    adapter_id: str
    adapter_version: str

    # --- image ----------------------------------------------------------------------------------
    line_image_path: str
    """Relative to the benchmark (or candidate) directory, forward slashes."""
    image_sha256: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    crop: CropRecord
    segmentation_source: str
    """`supplied_line_image` | `source_polygon` | `segmenter:<id>@<version>`."""

    # --- ground truth ---------------------------------------------------------------------------
    gt_source: str
    """Exactly as supplied (after decoding the file as UTF-8)."""
    gt_canonical: str
    """What predictions are scored against."""
    gt_source_sha256: str
    gt_canonical_sha256: str
    normalization_protocol: str = PROTOCOL_ID
    normalization_applied: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> BenchmarkLine:
        for name in ("document_id", "page_id", "line_key", "dataset_id", "source_id"):
            value = getattr(self, name)
            if not _ID_COMPONENT.match(value):
                raise ValueError(f"{name}={value!r} is not a safe ID component (use safe_id_component)")
        if self.line_id != make_line_id(self.document_id, self.page_id, self.line_key):
            raise ValueError(f"line_id {self.line_id!r} != document_id/page_id/line_key")
        for name in ("image_sha256", "gt_source_sha256", "gt_canonical_sha256"):
            if not _SHA256.match(getattr(self, name)):
                raise ValueError(f"{name} is not a lowercase sha256 hex digest")
        if self.gt_source_sha256 != sha256_text(self.gt_source):
            raise ValueError("gt_source_sha256 does not match gt_source")
        if self.gt_canonical_sha256 != sha256_text(self.gt_canonical):
            raise ValueError("gt_canonical_sha256 does not match gt_canonical")
        if unicodedata.normalize("NFC", self.gt_canonical) != self.gt_canonical:
            raise ValueError("gt_canonical is not NFC")
        unknown = [r for r in self.normalization_applied if r not in RULES]
        if unknown:
            raise ValueError(f"unknown normalization rules {unknown}")
        if bool(self.normalization_applied) != (self.gt_source != self.gt_canonical):
            raise ValueError("normalization_applied must be non-empty exactly when gt_canonical differs from gt_source")
        for name in ("line_image_path", "source_relative_path", "gt_source_relative_path"):
            value = getattr(self, name)
            if "\\" in value or value.startswith("/") or ".." in value.split("/"):
                raise ValueError(f"{name}={value!r} must be a relative forward-slash path inside its root")
        if (self.crop.policy == "supplied_line_image") != (self.segmentation_source == "supplied_line_image"):
            raise ValueError("crop.policy and segmentation_source disagree about supplied line images")
        return self


def sort_key(line: BenchmarkLine) -> tuple:
    return (line.document_id, line.page_id, line.line_order, line.line_id)


def serialize_line(line: BenchmarkLine) -> str:
    return json.dumps(line.model_dump(mode="json", by_alias=True), ensure_ascii=False, sort_keys=True)


def manifest_bytes(lines: Iterable[BenchmarkLine]) -> bytes:
    return "".join(serialize_line(line) + "\n" for line in sorted(lines, key=sort_key)).encode("utf-8")


def write_manifest(path: Path, lines: Iterable[BenchmarkLine]) -> str:
    """Writes the canonical manifest and returns its SHA-256."""
    data = manifest_bytes(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def read_manifest(path: Path) -> list[BenchmarkLine]:
    lines: list[BenchmarkLine] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                lines.append(BenchmarkLine.model_validate(json.loads(raw)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ManifestError(f"{path}:{number}: {exc}") from exc
    return lines


def validate_manifest(lines: Sequence[BenchmarkLine], *, root: Path | None = None) -> list[Finding]:
    """Cross-line checks the per-line model cannot do. With `root`, also verifies every image file
    exists and still has its recorded hash."""
    findings: list[Finding] = []
    if not lines:
        return [Finding(severity="blocker", code="manifest.empty", message="manifest has no lines")]

    for field, code in (("dataset_id", "manifest.mixed_dataset_id"), ("normalization_protocol", "manifest.mixed_normalization")):
        values = sorted({getattr(line, field) for line in lines})
        if len(values) > 1:
            findings.append(Finding(severity="blocker", code=code, message=f"{field} differs across lines: {values}"))
    policies = sorted({line.crop.policy for line in lines if line.crop.policy != "supplied_line_image"})
    if len(policies) > 1:
        findings.append(Finding(severity="blocker", code="manifest.mixed_crop_policy",
                                message=f"one manifest mixes crop policies {policies}; run a sensitivity variant as its own benchmark"))

    for field, code in (("line_id", "manifest.duplicate_line_id"), ("line_image_path", "manifest.duplicate_image_path")):
        for value, count in Counter(getattr(line, field) for line in lines).items():
            if count > 1:
                findings.append(Finding(severity="blocker", code=code, message=f"{field} {value!r} occurs {count} times"))

    by_image: dict[str, list[str]] = {}
    for line in lines:
        by_image.setdefault(line.image_sha256, []).append(line.line_id)
    for digest, ids in by_image.items():
        if len(ids) > 1:
            findings.append(Finding(severity="warning", code="manifest.duplicate_image_content",
                                    message=f"{len(ids)} lines share identical image bytes (reviewed at build time)",
                                    detail={"sha256": digest, "line_ids": ids}))

    for line in lines:
        if not line.gt_canonical.strip():
            findings.append(Finding(severity="blocker", code="gt.empty", line_id=line.line_id,
                                    message="empty reference transcription (exclude the line or supply GT)"))

    if root is not None:
        for line in lines:
            image = root / line.line_image_path
            if not image.is_file():
                findings.append(Finding(severity="blocker", code="image.missing", line_id=line.line_id, path=line.line_image_path,
                                        message="line image file is missing"))
            elif sha256_file(image) != line.image_sha256:
                findings.append(Finding(severity="blocker", code="image.hash_mismatch", line_id=line.line_id, path=line.line_image_path,
                                        message="line image bytes changed since the manifest was written"))
    return findings
