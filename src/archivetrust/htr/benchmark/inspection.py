"""Non-destructive inspection of one incoming delivery.

Reads `incoming/<source_id>/` and reports -- without changing a byte of it -- what format it is, what
is wrong, what needs a human, what the build would normalize automatically, whether segmentation is
needed, whether GT aligns to images, duplicates, and an overall readiness verdict.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from archivetrust.htr.benchmark.findings import Finding, findings_to_jsonl, findings_to_markdown, summarize
from archivetrust.htr.benchmark.imaging import ImageProbe, near_duplicate_pairs, probe_image, probe_image_bytes
from archivetrust.htr.benchmark.normalization import canonicalize_gt, protocol_record
from archivetrust.htr.benchmark.sources import ADAPTERS, CandidateLine, Extraction, SourceAdapter, adapter_by_id
from archivetrust.htr.benchmark.sources.base import IGNORED_DIRS, IGNORED_NAMES, is_image, list_files

INSPECTION_VERSION = "1"

PAGE_LIKE_MIN_HEIGHT = 600
PAGE_LIKE_MAX_ASPECT = 2.0
TINY_SIDE = 10
BOUNDS_TOLERANCE_PX = 2
NEAR_DUPLICATE_BITS = 4
IDENTICAL_TEXT_MIN_CHARS = 20

_MOJIBAKE = re.compile("Ã[\u0080-¿]|Â[ -¿]|â€")
_MARKUP = {
    "square_brackets": re.compile(r"\[[^\]]*\]"),
    "angle_brackets": re.compile(r"<[^>]*>"),
    "curly_braces": re.compile(r"\{[^}]*\}"),
}
_ARCHIVES = (".zip", ".7z", ".rar", ".tar", ".gz", ".tgz")


@dataclass
class InspectionResult:
    source_dir: Path
    adapter_scores: dict[str, float]
    adapter_id: str | None
    extraction: Extraction | None
    probes: dict[str, ImageProbe]
    findings: list[Finding]
    stats: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        summary = summarize(self.findings)
        if summary["blocking"]:
            return "blocked"
        if summary["counts_by_severity"]["needs_review"]:
            return "needs_review"
        return "ready_to_build"


def text_findings(line_key: str, path: str, gt: str, *, from_line_file: bool) -> list[Finding]:
    """Per-transcription checks. Only NFC and a file's own line terminator are ever fixed
    automatically (reported as warnings); everything else goes to a human."""
    out: list[Finding] = []

    def add(severity, code, message, **detail):
        out.append(Finding(severity=severity, code=code, path=path, line_id=line_key, message=message, detail=detail))

    canonical, applied = canonicalize_gt(gt, from_line_file=from_line_file)
    for rule in applied:
        add("warning", f"gt.auto_{rule}", f"auto-normalized: {rule}")
    if canonical == "":
        add("needs_review", "gt.empty", "empty transcription")
        return out
    if not canonical.strip():
        add("needs_review", "gt.whitespace_only", "transcription is only whitespace")
        return out
    if "\n" in canonical or "\r" in canonical:
        add("needs_review", "gt.embedded_newline", "transcription contains a line break", lines=canonical.count("\n") + 1)
    if canonical != canonical.strip(" \t"):
        add("needs_review", "gt.outer_whitespace", "leading/trailing whitespace (kept verbatim unless a human decides)")
    if "\t" in canonical:
        add("needs_review", "gt.tab", "transcription contains a tab")
    if "  " in canonical:
        add("info", "gt.multiple_spaces", "consecutive spaces (kept verbatim)")
    if "�" in canonical:
        add("blocker", "gt.replacement_char", "U+FFFD replacement character: text was damaged by a failed decode")
    if _MOJIBAKE.search(canonical):
        add("needs_review", "gt.mojibake", "looks like UTF-8 decoded as Latin-1/cp1252 (e.g. 'Ã¥' for 'å')")
    invisible = sorted({f"U+{ord(c):04X}" for c in canonical if unicodedata.category(c) in ("Cc", "Cf") and c not in "\n\r\t"})
    if invisible:
        add("needs_review", "gt.invisible_chars", f"invisible/control characters {invisible}", codepoints=invisible)
    private = sorted({f"U+{ord(c):04X}" for c in canonical if unicodedata.category(c) == "Co"})
    if private:
        add("warning", "gt.private_use_chars", f"private-use characters (e.g. MUFI) {private}; no public model can emit these",
            codepoints=private)
    markup = {name: pattern.findall(canonical) for name, pattern in _MARKUP.items()}
    markup = {k: v for k, v in markup.items() if v}
    if markup:
        add("warning", "gt.possible_editorial_markup",
            "bracketed spans: confirm whether they are written on the page or editorial conventions", spans=markup)
    return out


def _filesystem_findings(root: Path) -> list[Finding]:
    out: list[Finding] = []
    lowered: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = [p.lower() for p in path.relative_to(root).parts]
        if path.is_file() and (parts[-1] in IGNORED_NAMES or parts[-1].startswith("._") or any(p in IGNORED_DIRS for p in parts[:-1])):
            out.append(Finding(severity="info", code="fs.os_junk", path=relative, message="OS metadata file, ignored"))
            continue
        if unicodedata.normalize("NFC", relative) != relative:
            out.append(Finding(severity="warning", code="fs.non_nfc_filename", path=relative,
                               message="filename is not NFC (macOS-style); references to it may not match"))
        if relative.lower() in lowered and lowered[relative.lower()] != relative:
            out.append(Finding(severity="blocker", code="fs.case_collision", path=relative,
                               message=f"differs only in case from {lowered[relative.lower()]!r} (breaks on Windows)"))
        lowered[relative.lower()] = relative
        if len(str(path)) > 240:
            out.append(Finding(severity="warning", code="fs.long_path", path=relative, message="path longer than 240 characters"))
        if path.is_file():
            if path.stat().st_size == 0:
                out.append(Finding(severity="blocker", code="fs.empty_file", path=relative, message="zero-byte file"))
            if relative.lower().endswith(_ARCHIVES):
                out.append(Finding(severity="needs_review", code="fs.archive_not_extracted", path=relative,
                                   message="archive inside the delivery; extract it into a new incoming/<source_id> (incoming is never modified)"))
    return out


def _image_findings(relative: str, probe: ImageProbe, kind: str) -> list[Finding]:
    out: list[Finding] = []

    def add(severity, code, message):
        out.append(Finding(severity=severity, code=code, path=relative, message=message))

    if not probe.ok:
        add("needs_review" if (probe.error or "").startswith("too_large") else "blocker",
            "image.too_large" if (probe.error or "").startswith("too_large") else "image.unreadable", probe.error or "")
        return out
    if probe.exif_orientation not in (None, 1):
        add("needs_review", "image.exif_orientation",
            f"EXIF orientation {probe.exif_orientation}: stored raster is rotated/mirrored relative to display; "
            "neither model applies EXIF, so a human must confirm which raster the GT/coordinates describe")
    if probe.has_alpha and not probe.alpha_all_opaque:
        add("needs_review", "image.transparency", "non-opaque alpha channel; the two model stacks would flatten it differently")
    if probe.frames > 1:
        add("needs_review", "image.multi_frame", f"{probe.frames} frames; only the first would be read")
    if probe.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        add("needs_review" if kind == "line" else "warning", "image.unusual_mode", f"mode {probe.mode}")
    if probe.blank:
        add("needs_review", "image.blank", f"near-uniform image (grayscale stddev {probe.gray_stddev})")
    if kind == "line" and probe.width and probe.height:
        if min(probe.width, probe.height) < TINY_SIDE:
            add("needs_review", "image.tiny", f"{probe.width}x{probe.height}")
        if probe.height >= PAGE_LIKE_MIN_HEIGHT and probe.width / probe.height < PAGE_LIKE_MAX_ASPECT:
            add("needs_review", "image.page_like", f"{probe.width}x{probe.height} looks like a page or block, not a line")
    return out


def _page_line_findings(line: CandidateLine, probe: ImageProbe | None) -> list[Finding]:
    out: list[Finding] = []
    key = line.key.replace("|", "/")
    if line.polygon is None:
        out.append(Finding(severity="needs_review", code="layout.line_without_coords", path=line.gt_path, line_id=key,
                           message="line has no coordinates; it cannot be cropped"))
        return out
    if probe is None or not probe.ok or probe.width is None or probe.height is None:
        return out
    xs, ys = [p[0] for p in line.polygon], [p[1] for p in line.polygon]
    if min(xs) < -BOUNDS_TOLERANCE_PX or min(ys) < -BOUNDS_TOLERANCE_PX or max(xs) > probe.width + BOUNDS_TOLERANCE_PX \
            or max(ys) > probe.height + BOUNDS_TOLERANCE_PX:
        out.append(Finding(severity="warning", code="layout.polygon_out_of_bounds", path=line.gt_path, line_id=key,
                           message="polygon extends beyond the raster; the crop is clamped"))
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width < TINY_SIDE or height < TINY_SIDE:
        out.append(Finding(severity="needs_review", code="layout.tiny_line", path=line.gt_path, line_id=key,
                           message=f"line box {width:.0f}x{height:.0f} px"))
    return out


def _declared_size_findings(extraction: Extraction, probes: dict[str, ImageProbe]) -> list[Finding]:
    out: list[Finding] = []
    seen: set[str] = set()
    for line in extraction.lines:
        if line.image_kind != "page" or line.image_path in seen or line.declared_page_size is None:
            continue
        seen.add(line.image_path)
        probe = probes.get(line.image_path)
        if probe is None or not probe.ok:
            continue
        declared, actual = line.declared_page_size, (probe.width, probe.height)
        if declared == actual:
            continue
        if declared == (actual[1], actual[0]):
            out.append(Finding(severity="blocker", code="page.size_swapped", path=line.gt_path,
                               message=f"XML declares {declared}, raster is {actual}: coordinates describe a rotated page"))
        else:
            out.append(Finding(severity="blocker", code="page.size_mismatch", path=line.gt_path,
                               message=f"XML declares {declared}, raster is {actual}: coordinates may not fit this image",
                               detail={"scale_x": actual[0] / declared[0], "scale_y": actual[1] / declared[1]}))
    return out


def _duplicate_findings(lines: list[CandidateLine], probes: dict[str, ImageProbe]) -> list[Finding]:
    out: list[Finding] = []
    for key, count in Counter(line.key for line in lines).items():
        if count > 1:
            out.append(Finding(severity="blocker", code="dup.line_key", line_id=key.replace("|", "/"),
                               message=f"document/page/line identifier occurs {count} times"))
    line_images = [line for line in lines if line.image_kind == "line" and line.image_path in probes and probes[line.image_path].ok]
    by_hash: dict[str, list[CandidateLine]] = {}
    for line in line_images:
        by_hash.setdefault(probes[line.image_path].sha256, []).append(line)
    for digest, group in by_hash.items():
        if len({g.image_path for g in group}) > 1:
            same_text = len({g.gt_source for g in group}) == 1
            out.append(Finding(severity="needs_review", code="dup.identical_image",
                               message=f"{len(group)} line images have identical bytes" + (" and identical GT" if same_text else " but different GT"),
                               detail={"sha256": digest, "images": sorted({g.image_path for g in group})}))
    unique = {line.image_path: probes[line.image_path] for line in line_images}
    pairs = near_duplicate_pairs([(path, p.dhash) for path, p in unique.items() if p.dhash], max_distance=NEAR_DUPLICATE_BITS)
    for a, b, distance in pairs:
        if unique[a].sha256 != unique[b].sha256:
            out.append(Finding(severity="warning", code="dup.near_identical_image", path=a,
                               message=f"perceptually near-identical to {b} (dHash distance {distance})", detail={"other": b}))
    by_text: dict[str, set[str]] = {}
    for line in lines:
        if line.gt_source and len(line.gt_source.strip()) >= IDENTICAL_TEXT_MIN_CHARS:
            by_text.setdefault(line.gt_source.strip(), set()).add(line.key.replace("|", "/"))
    for text, keys in by_text.items():
        if len(keys) > 1:
            out.append(Finding(severity="warning", code="dup.identical_text",
                               message=f"{len(keys)} lines share the transcription {text[:60]!r}", detail={"line_ids": sorted(keys)}))
    return out


def _segmentation_status(extraction: Extraction, files: list[str]) -> dict:
    kinds = Counter(line.image_kind for line in extraction.lines)
    with_coords = sum(1 for line in extraction.lines if line.image_kind == "page" and line.polygon)
    consumed_images = {f for f in extraction.consumed_files if is_image(f)}
    unassigned_images = [f for f in files if is_image(f) and f not in consumed_images]
    if kinds.get("line") and not kinds.get("page"):
        mode = "not_needed:supplied_line_images"
    elif kinds.get("page") and with_coords == kinds["page"]:
        mode = "crop_from_source_polygons"
    elif kinds.get("page"):
        mode = "partial:some_lines_lack_coordinates"
    else:
        mode = "segmentation_required" if unassigned_images else "no_lines_found"
    return {"mode": mode, "lines_by_image_kind": dict(kinds), "page_lines_with_coords": with_coords,
            "images_without_line_gt": len(unassigned_images)}


def choose_adapter(root: Path, files: list[str], adapter_id: str | None) -> tuple[SourceAdapter | None, dict[str, float]]:
    scores = {adapter.adapter_id: round(adapter.detect(root, files), 3) for adapter in ADAPTERS}
    if adapter_id:
        return adapter_by_id(adapter_id), scores
    best = max(ADAPTERS, key=lambda a: scores[a.adapter_id])  # ties: registry order (PAGE before ALTO)
    return (best if scores[best.adapter_id] > 0 else None), scores


def inspect_source(source_dir: Path, *, adapter_id: str | None = None, charset: set[str] | None = None) -> InspectionResult:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    files = list_files(source_dir)
    findings = _filesystem_findings(source_dir)
    adapter, scores = choose_adapter(source_dir, files, adapter_id)
    detected = [name for name, score in scores.items() if score > 0]
    findings.append(Finding(severity="info", code="format.scores", message=f"format detection scores {scores}", detail=scores))
    if len([n for n in detected if scores[n] >= 0.5]) > 1:
        findings.append(Finding(severity="warning", code="format.multiple",
                                message=f"several formats present {detected}; using {adapter.adapter_id if adapter else None} (override with --adapter)"))
    if adapter is None:
        images = [f for f in files if is_image(f)]
        text_like = [f for f in files if f.lower().endswith((".txt", ".xml", ".csv", ".tsv", ".jsonl", ".json", ".parquet"))]
        findings.append(Finding(severity="blocker", code="format.unrecognized",
                                message="no supported format recognized (PAGE XML, ALTO, line image + .txt, CSV/TSV/JSONL/Parquet); "
                                        "page images with page-level text need manual line alignment"))
        if images and not text_like:
            findings.append(Finding(severity="blocker", code="gt.none_found",
                                    message=f"{len(images)} images and no transcription files: there is no ground truth to benchmark against"))
        stats = {"files": len(files), "images": len(images), "transcription_like_files": len(text_like),
                 "files_by_extension": dict(sorted(Counter(PurePosixPath(f).suffix.lower() or "(none)" for f in files).items())),
                 "segmentation": {"mode": "undetermined:no_ground_truth" if images else "no_lines_found",
                                  "images_without_line_gt": len(images)}}
        return InspectionResult(source_dir, scores, None, None, {}, findings, stats)

    extraction = adapter.extract(source_dir, files)
    findings.extend(extraction.findings)

    probes: dict[str, ImageProbe] = {}
    kinds = {line.image_path: line.image_kind for line in extraction.lines if line.image_path}
    for image_rel, kind in sorted(kinds.items()):
        if "#" in image_rel:  # embedded (parquet) image
            continue
        probes[image_rel] = probe_image(source_dir / image_rel)
        findings.extend(_image_findings(image_rel, probes[image_rel], kind))
    for line in extraction.lines:
        if line.image_bytes is not None and line.image_path not in probes:
            probes[line.image_path] = probe_image_bytes(line.image_bytes)
            findings.extend(_image_findings(line.image_path, probes[line.image_path], line.image_kind))

    for line in extraction.lines:
        key = line.key.replace("|", "/")
        if line.gt_source is not None:
            line_findings = text_findings(key, line.gt_path, line.gt_source, from_line_file=line.gt_from_line_file)
            probe = probes.get(line.image_path)
            if any(f.code == "gt.embedded_newline" for f in line_findings) and probe and probe.ok and probe.height \
                    and probe.height >= PAGE_LIKE_MIN_HEIGHT:
                line_findings.append(Finding(severity="needs_review", code="gt.page_level_unaligned", path=line.gt_path, line_id=key,
                                             message="multi-line text for a page-sized image: page-level GT cannot be aligned to lines automatically"))
            findings.extend(line_findings)
        if line.image_kind == "page":
            findings.extend(_page_line_findings(line, probes.get(line.image_path)))
    findings.extend(_declared_size_findings(extraction, probes))
    findings.extend(_duplicate_findings(extraction.lines, probes))

    unused = [f for f in files if f not in extraction.consumed_files]
    for name in unused[:200]:
        findings.append(Finding(severity="warning", code="fs.unused_file", path=name, message=f"not used by the {adapter.adapter_id} adapter"))
    if len(unused) > 200:
        findings.append(Finding(severity="warning", code="fs.unused_file", message=f"... and {len(unused) - 200} more unused files"))

    stats = _stats(files, extraction, probes, charset)
    stats["segmentation"] = _segmentation_status(extraction, files)
    stats["unused_files"] = len(unused)
    return InspectionResult(source_dir, scores, adapter.adapter_id, extraction, probes, findings, stats)


def _stats(files: list[str], extraction: Extraction, probes: dict[str, ImageProbe], charset: set[str] | None) -> dict:
    lines = extraction.lines
    texts = [canonicalize_gt(line.gt_source, from_line_file=line.gt_from_line_file)[0] for line in lines if line.gt_source is not None]
    chars = Counter(c for text in texts for c in text)
    line_heights = [probes[line.image_path].height for line in lines
                    if line.image_kind == "line" and line.image_path in probes and probes[line.image_path].ok]
    stats = {
        "files": len(files),
        "files_by_extension": dict(sorted(Counter(PurePosixPath(f).suffix.lower() or "(none)" for f in files).items())),
        "lines": len(lines),
        "lines_with_gt": len(texts),
        "documents": len({line.document_raw for line in lines}),
        "pages": len({(line.document_raw, line.page_raw) for line in lines}),
        "collections": dict(Counter(line.collection for line in lines if line.collection)),
        "gt_characters": sum(len(t) for t in texts),
        "gt_length": _describe([len(t) for t in texts]),
        "line_image_height_px": _describe([h for h in line_heights if h is not None]),
        "character_inventory": {
            c: {"count": n, "name": unicodedata.name(c, f"U+{ord(c):04X}")} for c, n in sorted(chars.items(), key=lambda kv: (-kv[1], kv[0]))
        },
    }
    if charset is not None:
        oov = {c: n for c, n in chars.items() if c not in charset}
        total = sum(chars.values()) or 1
        stats["loghi_out_of_vocabulary"] = {
            "characters": {c: {"count": n, "name": unicodedata.name(c, f"U+{ord(c):04X}")} for c, n in sorted(oov.items(), key=lambda kv: -kv[1])},
            "occurrences": sum(oov.values()),
            "rate": sum(oov.values()) / total,
            "lines_affected": sum(1 for t in texts if any(c in oov for c in t)),
        }
    return stats


def _describe(values: list[int]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values), "min": ordered[0], "median": statistics.median(ordered), "max": ordered[-1],
            "p05": ordered[int(0.05 * (len(ordered) - 1))], "p95": ordered[int(0.95 * (len(ordered) - 1))]}


def write_inspection(result: InspectionResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(result.findings)
    payload = {
        "inspection_version": INSPECTION_VERSION,
        "source_dir": str(result.source_dir),
        "verdict": result.verdict,
        "adapter": {"chosen": result.adapter_id, "scores": result.adapter_scores,
                    "version": result.extraction.adapter_version if result.extraction else None},
        "normalization_protocol": protocol_record(),
        "findings_summary": summary,
        "stats": result.stats,
    }
    (out_dir / "inspection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "findings.jsonl").write_text(findings_to_jsonl(result.findings), encoding="utf-8")
    (out_dir / "inspection.md").write_text(render_markdown(result, summary), encoding="utf-8")


def render_markdown(result: InspectionResult, summary: dict) -> str:
    s = result.stats
    seg = s.get("segmentation", {})
    codes = summary["counts_by_code"]
    auto = {c: n for c, n in codes.items() if c.startswith("gt.auto_")}
    review = sorted({f.code for f in result.findings if f.severity == "needs_review"})
    oov = s.get("loghi_out_of_vocabulary")
    oov_text = "not checked (no charset given)" if oov is None else (
        f"{oov['occurrences']} chars ({oov['rate']:.3%}), {oov['lines_affected']} lines: {''.join(oov['characters'])}")
    lines = [
        f"# Inspection: `{result.source_dir.name}`",
        "",
        f"**Verdict: {result.verdict.upper()}**  ",
        f"Blockers {summary['counts_by_severity']['blocker']} | needs review {summary['counts_by_severity']['needs_review']} | "
        f"warnings {summary['counts_by_severity']['warning']}",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| Format | {result.adapter_id or 'unrecognized'} (scores {result.adapter_scores}) |",
        f"| Size | {s.get('files', 0)} files, {s.get('lines', 0)} lines in {s.get('pages', 0)} pages / {s.get('documents', 0)} documents |",
        f"| GT present | {s.get('lines_with_gt', 0)} of {s.get('lines', 0)} lines |",
        f"| Segmentation | {seg.get('mode', 'n/a')} |",
        f"| Auto-normalizable | {auto or 'nothing'} |",
        f"| Needs human review | {review or 'nothing'} |",
        f"| Duplicates | {({c: n for c, n in codes.items() if c.startswith('dup.')}) or 'none found'} |",
        f"| Loghi out-of-vocabulary | {oov_text} |",
        "| Overlap with training data | not checked here -- run `overlap` (absence of a match is not proof of no overlap) |",
        f"| Ready to build/freeze | {'yes' if result.verdict == 'ready_to_build' else 'no -- resolve the items below'} |",
        "",
        "## Findings",
        "",
        findings_to_markdown(result.findings),
    ]
    return "\n".join(lines)
