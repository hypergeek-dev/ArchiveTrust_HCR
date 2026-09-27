"""Candidate build, human review decisions, and the freeze.

    build   incoming/<source> -> work/<source>/candidate/   (lines/, manifest.jsonl, review_queue.jsonl,
                                                             excluded.jsonl, build.json)
    freeze  candidate -> benchmark/<benchmark_id>/          (immutable: lines/, manifest.jsonl, FROZEN.json)

A line enters the candidate manifest only when every blocker / needs-review finding attached to it
is resolved by a recorded decision in `work/<source>/decisions.jsonl`. Decisions are hand-written
JSON lines:

    {"target": "docA/lines/l3", "action": "exclude", "reason": "illegible, torn", "reviewer": "DJ"}
    {"target": "docA/lines/l7", "action": "accept", "codes": ["gt.possible_editorial_markup"], "reason": "..."}
    {"target": "docA/lines/l9", "action": "set_gt", "gt": "Anno 1723", "reason": "GT typo 1732"}
    {"target": "file:d/scan_004.jpg", "action": "exclude", "reason": "duplicate scan"}

`target` is the line key printed in the review queue, or `file:<relative path>`. Decisions must be
made before any model has been run on the data; the freeze records their hash, so they cannot change
afterwards without producing a different benchmark.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from archivetrust.htr.benchmark.contract import (
    BenchmarkLine,
    CropPolicy,
    CropRecord,
    line_image_relpath,
    make_line_id,
    manifest_bytes,
    read_manifest,
    safe_id_component,
    sha256_bytes,
    sha256_file,
    sha256_text,
    validate_manifest,
)
from archivetrust.htr.benchmark.findings import Finding, findings_to_jsonl
from archivetrust.htr.benchmark.imaging import crop_line, open_page_rgb, polygon_bbox, prepare_supplied_line_image, probe_image_bytes
from archivetrust.htr.benchmark.inspection import BOUNDS_TOLERANCE_PX, InspectionResult, inspect_source
from archivetrust.htr.benchmark.layout import assert_outside
from archivetrust.htr.benchmark.normalization import PROTOCOL_ID, RULE_HUMAN_CORRECTION, canonicalize_gt, protocol_record
from archivetrust.htr.benchmark.provenance import provenance, utc_now
from archivetrust.htr.benchmark.sources import CandidateLine

BUILD_VERSION = "1"
FROZEN_SCHEMA = "benchmark-frozen/1"
_GT_CODES = ("gt.", "text.")
NOT_ACCEPTABLE = frozenset({"gt.empty", "gt.whitespace_only", "gt.outer_whitespace", "gt.missing"})
"""Findings `accept` cannot resolve: an empty reference cannot be scored, and predictions are
compared with outer whitespace stripped, so reference outer whitespace could never be matched.
Resolve these with `set_gt` or `exclude`."""


class BuildError(RuntimeError):
    pass


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    action: Literal["exclude", "accept", "set_gt"]
    reason: str
    reviewer: str | None = None
    codes: tuple[str, ...] | None = None
    gt: str | None = None

    @property
    def decision_id(self) -> str:
        return sha256_text(json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))[:12]

    def resolves(self, finding: Finding) -> bool:
        if self.action == "exclude":
            return True
        if self.action == "set_gt":
            return finding.code.startswith(_GT_CODES)
        return (finding.severity == "needs_review" and finding.code not in NOT_ACCEPTABLE
                and (self.codes is None or finding.code in self.codes))


def load_decisions(path: Path | None) -> tuple[list[Decision], str | None]:
    if path is None or not path.is_file():
        return [], None
    decisions: list[Decision] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            decision = Decision.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise BuildError(f"{path}:{number}: invalid decision: {exc}") from exc
        if decision.action == "set_gt" and decision.gt is None:
            raise BuildError(f"{path}:{number}: set_gt needs a 'gt' value")
        decisions.append(decision)
    return decisions, sha256_file(path)


def line_target(line: CandidateLine) -> str:
    return line.key.replace("|", "/")


@dataclass
class BuildSummary:
    candidate_dir: Path
    included: int
    review: int
    excluded: int
    unresolved_dataset_findings: list[Finding] = field(default_factory=list)
    manifest_sha256: str | None = None


def _attach(result: InspectionResult) -> tuple[dict[str, list[Finding]], list[Finding]]:
    """Maps each blocking/review finding to the lines it concerns; findings that concern no line
    (a skipped table row, an unused archive...) are returned separately."""
    assert result.extraction is not None
    by_key: dict[str, list[Finding]] = defaultdict(list)
    by_path: dict[str, list[str]] = defaultdict(list)
    keys = set()
    for line in result.extraction.lines:
        key = line_target(line)
        keys.add(key)
        for path in {line.image_path, line.gt_path}:
            if path:
                by_path[path].append(key)
    unattached: list[Finding] = []
    for finding in result.findings:
        if finding.severity not in ("blocker", "needs_review"):
            continue
        targets: set[str] = set()
        if finding.line_id in keys:
            targets.add(finding.line_id)
        elif finding.path in by_path:
            targets.update(by_path[finding.path])
        for image in finding.detail.get("images", []) if isinstance(finding.detail, dict) else []:
            targets.update(by_path.get(image, []))
        if not targets:
            unattached.append(finding)
        for key in targets:
            by_key[key].append(finding)
    return by_key, unattached


def build_candidate(source_dir: Path, candidate_dir: Path, *, dataset_id: str, source_id: str, adapter_id: str | None = None,
                    crop_policy: CropPolicy = "bbox_v1", decisions_path: Path | None = None, overwrite: bool = False) -> BuildSummary:
    source_dir, candidate_dir = Path(source_dir), Path(candidate_dir)
    assert_outside(candidate_dir, source_dir)
    if crop_policy == "supplied_line_image":
        raise BuildError("crop_policy is for page-based sources: bbox_v1 or polygon_mask_v1")
    if candidate_dir.exists():
        if not overwrite:
            raise BuildError(f"{candidate_dir} exists; pass overwrite=True to rebuild it")
        if any(candidate_dir.iterdir()) and not (candidate_dir / "build.json").is_file():
            raise BuildError(f"{candidate_dir} is not a candidate directory (no build.json); refusing to remove it")
        shutil.rmtree(candidate_dir)
    for value in (dataset_id, source_id):
        if safe_id_component(value) != value:
            raise BuildError(f"{value!r} is not a safe identifier")

    result = inspect_source(source_dir, adapter_id=adapter_id)
    if result.extraction is None:
        raise BuildError("no supported format recognized; see the inspection report")
    decisions, decisions_sha = load_decisions(decisions_path)
    line_decisions: dict[str, list[Decision]] = defaultdict(list)
    for decision in decisions:
        line_decisions[decision.target].append(decision)
    attached, unattached = _attach(result)
    unresolved_dataset = [f for f in unattached
                          if not any(d.resolves(f) for d in line_decisions.get(f"file:{f.path}", []))]

    included: list[BenchmarkLine] = []
    review: list[dict] = []
    excluded: list[dict] = []
    page_cache: dict[str, tuple[bytes, object]] = {}
    lines_dir = candidate_dir / "lines"
    lines_dir.mkdir(parents=True)

    ordered = sorted(result.extraction.lines, key=lambda c: (c.image_path, c.document_raw, c.page_raw, c.line_order))
    for line in ordered:
        key = line_target(line)
        mine = line_decisions.get(key, []) + line_decisions.get(f"file:{line.image_path}", []) + line_decisions.get(f"file:{line.gt_path}", [])
        exclusion = next((d for d in mine if d.action == "exclude"), None)
        if exclusion is not None:
            excluded.append({"target": key, "decision_id": exclusion.decision_id, "reason": exclusion.reason})
            continue
        set_gt = next((d for d in mine if d.action == "set_gt"), None)
        open_findings = [f for f in attached.get(key, []) if not any(d.resolves(f) for d in mine)]
        if line.gt_source is None and set_gt is None and not any(f.code.startswith(_GT_CODES) for f in open_findings):
            open_findings.append(Finding(severity="needs_review", code="gt.missing", line_id=key, message="no transcription"))
        if open_findings:
            review.append(_review_entry(line, open_findings))
            continue
        try:
            included.append(_materialize(line, source_dir, lines_dir, dataset_id=dataset_id, source_id=source_id,
                                         adapter=(result.extraction.adapter_id, result.extraction.adapter_version),
                                         crop_policy=crop_policy, set_gt=set_gt, page_cache=page_cache))
        except _MaterializeError as exc:
            review.append(_review_entry(line, [Finding(severity="needs_review", code=exc.code, line_id=key, message=str(exc))]))

    manifest_findings = validate_manifest(included, root=candidate_dir) if included else []
    data = manifest_bytes(included)
    (candidate_dir / "manifest.jsonl").write_bytes(data)
    _write_jsonl(candidate_dir / "review_queue.jsonl", review)
    _write_jsonl(candidate_dir / "excluded.jsonl", excluded)
    (candidate_dir / "unresolved_dataset_findings.jsonl").write_text(findings_to_jsonl(unresolved_dataset), encoding="utf-8")
    (candidate_dir / "manifest_findings.jsonl").write_text(findings_to_jsonl(manifest_findings), encoding="utf-8")
    build_record = {
        "build_version": BUILD_VERSION,
        "dataset_id": dataset_id,
        "source_id": source_id,
        "source_dir": str(source_dir),
        "source_tree": _tree_digest(source_dir),
        "adapter": {"id": result.extraction.adapter_id, "version": result.extraction.adapter_version, "scores": result.adapter_scores},
        "crop_policy": crop_policy,
        "normalization_protocol": protocol_record(),
        "decisions": {"path": str(decisions_path) if decisions_path else None, "sha256": decisions_sha, "count": len(decisions)},
        "counts": {"source_lines": len(result.extraction.lines), "included": len(included), "review": len(review), "excluded": len(excluded),
                   "unresolved_dataset_findings": len(unresolved_dataset)},
        "inspection_verdict": result.verdict,
        "manifest_sha256": sha256_bytes(data),
        "provenance": provenance(official=False),
        "built_at_utc": utc_now(),
    }
    (candidate_dir / "build.json").write_text(json.dumps(build_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BuildSummary(candidate_dir, len(included), len(review), len(excluded), unresolved_dataset, sha256_bytes(data))


def _review_entry(line: CandidateLine, findings: list[Finding]) -> dict:
    return {
        "target": line_target(line),
        "image_path": line.image_path,
        "gt_path": line.gt_path,
        "gt_source": line.gt_source,
        "findings": [f.model_dump(mode="json") for f in findings],
        "resolve_with": "exclude | accept (needs_review only) | set_gt (gt./text. findings) in decisions.jsonl",
    }


class _MaterializeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _materialize(line: CandidateLine, source_dir: Path, lines_dir: Path, *, dataset_id: str, source_id: str,
                 adapter: tuple[str, str], crop_policy: CropPolicy, set_gt: Decision | None,
                 page_cache: dict[str, tuple[bytes, object]]) -> BenchmarkLine:
    document_id, page_id, line_key = (safe_id_component(v) for v in (line.document_raw, line.page_raw, line.line_raw))
    metadata = dict(line.metadata)
    ids_raw = {"document": line.document_raw, "page": line.page_raw, "line": line.line_raw}
    if any(safe_id_component(v) != v for v in ids_raw.values()):
        metadata["raw_ids"] = ids_raw

    if line.image_kind == "line":
        data = line.image_bytes if line.image_bytes is not None else (source_dir / line.image_path).read_bytes()
        try:
            out_bytes, ext, conversion = prepare_supplied_line_image(data, probe_image_bytes(data))
        except ValueError as exc:
            raise _MaterializeError("build.image_not_convertible", str(exc)) from exc
        if conversion:
            metadata["image_conversion"] = conversion
        crop = CropRecord(policy="supplied_line_image")
        segmentation_source = "supplied_line_image"
    else:
        if not line.image_path:
            raise _MaterializeError("page.image_missing", "no page image")
        if line.image_path not in page_cache:
            page_cache.clear()  # lines are processed grouped by page: keep one decoded page in memory
            page_bytes = (source_dir / line.image_path).read_bytes()
            page_cache[line.image_path] = (page_bytes, open_page_rgb(page_bytes))
        page_bytes, page = page_cache[line.image_path]
        width, height = page.size  # type: ignore[attr-defined]
        bbox = polygon_bbox(line.polygon or (), width=width, height=height)
        if bbox is None:
            raise _MaterializeError("build.empty_crop", "line polygon is empty after clamping to the page raster")
        if line.polygon and _out_of_bounds(line.polygon, width=width, height=height):
            metadata["crop_clamped"] = True  # same test as inspection's layout.polygon_out_of_bounds
        polygon = tuple((int(round(x)), int(round(y))) for x, y in line.polygon or ())
        out_bytes = crop_line(page, bbox, polygon=polygon, mask_polygon=crop_policy == "polygon_mask_v1")  # type: ignore[arg-type]
        ext = ".png"
        crop = CropRecord(policy=crop_policy, source_image_relative_path=line.image_path, source_image_sha256=sha256_bytes(page_bytes),
                          bbox=bbox, polygon=polygon or None,
                          baseline=tuple((int(round(x)), int(round(y))) for x, y in line.baseline) if line.baseline else None)
        segmentation_source = "source_polygon"

    relative = line_image_relpath(document_id, page_id, line_key, ext)
    target = lines_dir.parent / relative
    if target.exists():
        raise _MaterializeError("build.id_collision", f"{relative} already written by another line")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(out_bytes)
    probe = probe_image_bytes(out_bytes)

    gt_source = line.gt_source if line.gt_source is not None else ""
    if line.gt_source is None:
        metadata["gt_source_missing"] = True
    canonical, applied = canonicalize_gt(gt_source, from_line_file=line.gt_from_line_file)
    if set_gt is not None:
        corrected, _ = canonicalize_gt(set_gt.gt or "")
        if corrected != canonical:
            canonical, applied = corrected, (RULE_HUMAN_CORRECTION,)
        metadata["decision_id"] = set_gt.decision_id
    if canonical == gt_source:
        applied = ()

    return BenchmarkLine(
        dataset_id=dataset_id, source_id=source_id, document_id=document_id, page_id=page_id, line_key=line_key,
        line_id=make_line_id(document_id, page_id, line_key), line_order=line.line_order, collection=line.collection,
        writer_id=line.writer_id, source_relative_path=line.image_path, original_filename=_original_filename(line),
        gt_source_relative_path=line.gt_path or line.image_path, source_line_ref=line.source_line_ref, source_metadata=metadata,
        adapter_id=adapter[0], adapter_version=adapter[1], line_image_path=relative, image_sha256=sha256_bytes(out_bytes),
        image_width=probe.width or 1, image_height=probe.height or 1, crop=crop, segmentation_source=segmentation_source,
        gt_source=gt_source, gt_canonical=canonical, gt_source_sha256=sha256_text(gt_source),
        gt_canonical_sha256=sha256_text(canonical), normalization_protocol=PROTOCOL_ID, normalization_applied=applied,
    )


def _out_of_bounds(polygon, *, width: int, height: int) -> bool:
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    return (min(xs) < -BOUNDS_TOLERANCE_PX or min(ys) < -BOUNDS_TOLERANCE_PX
            or max(xs) > width + BOUNDS_TOLERANCE_PX or max(ys) > height + BOUNDS_TOLERANCE_PX)


def _original_filename(line: CandidateLine) -> str:
    original = line.metadata.get("original_image_name") or line.image_path
    return str(original).replace("\\", "/").rsplit("/", 1)[-1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _tree_digest(root: Path) -> dict:
    """SHA-256 over the sorted (relative path, file SHA-256) list: one identity for the delivery as received."""
    entries = [f"{p.relative_to(root).as_posix()}\t{sha256_file(p)}" for p in sorted(root.rglob("*")) if p.is_file()]
    return {"files": len(entries), "sha256": sha256_text("\n".join(entries))}


# --- freeze --------------------------------------------------------------------------------------


def freeze(candidate_dir: Path, frozen_dir: Path, *, benchmark_id: str, exclude_unresolved: bool = False,
           official: bool = False, dataset_card: dict | None = None) -> dict:
    """dataset_card: model-independent facts about the reference (provenance status, caveats, delivery-specific
    eligibility rules), embedded verbatim in FROZEN.json. The decisions file the candidate was built from is copied
    next to the manifest so the decision state can be reconstructed from the frozen benchmark alone."""
    from archivetrust.htr.benchmark.scoring import scoring_record  # noqa: PLC0415 -- scoring -> inference -> build

    candidate_dir, frozen_dir = Path(candidate_dir), Path(frozen_dir)
    if safe_id_component(benchmark_id) != benchmark_id:
        raise BuildError(f"{benchmark_id!r} is not a safe identifier")
    if frozen_dir.exists():
        raise BuildError(f"{frozen_dir} exists; a frozen benchmark is never overwritten -- choose a new benchmark_id")
    build = json.loads((candidate_dir / "build.json").read_text(encoding="utf-8"))
    if build["counts"]["unresolved_dataset_findings"]:
        raise BuildError(f"{build['counts']['unresolved_dataset_findings']} unresolved dataset-level findings "
                         "(unresolved_dataset_findings.jsonl); resolve them with file: decisions and rebuild")
    review = [json.loads(r) for r in (candidate_dir / "review_queue.jsonl").read_text(encoding="utf-8").splitlines() if r.strip()]
    if review and not exclude_unresolved:
        raise BuildError(f"{len(review)} lines still in review_queue.jsonl; decide them, or pass exclude_unresolved "
                         "to freeze without them (they are recorded as excluded, which can bias the benchmark)")
    lines = read_manifest(candidate_dir / "manifest.jsonl")
    blockers = [f for f in validate_manifest(lines, root=candidate_dir) if f.severity == "blocker"]
    if blockers:
        raise BuildError(f"manifest has {len(blockers)} blockers, e.g. {blockers[0].code}: {blockers[0].message}")
    record_provenance = provenance(official=official)

    partial = frozen_dir.with_name(frozen_dir.name + ".partial")
    if partial.exists():
        raise BuildError(f"{partial} exists from an interrupted freeze; inspect and remove it first")
    partial.mkdir(parents=True)
    for line in lines:
        target = partial / line.line_image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_dir / line.line_image_path, target)
        if sha256_file(target) != line.image_sha256:
            raise BuildError(f"copy of {line.line_image_path} does not match its manifest hash")
    data = manifest_bytes(lines)
    (partial / "manifest.jsonl").write_bytes(data)
    unresolved = [{"target": r["target"], "reason": "unresolved at freeze",
                   "codes": sorted({f["code"] for f in r["findings"]})} for r in review]
    _write_jsonl(partial / "excluded_at_freeze.jsonl", unresolved)
    shutil.copyfile(candidate_dir / "build.json", partial / "build.json")
    shutil.copyfile(candidate_dir / "excluded.jsonl", partial / "excluded_by_decision.jsonl")
    if build["decisions"]["path"]:
        shutil.copyfile(build["decisions"]["path"], partial / "decisions.jsonl")
        if sha256_file(partial / "decisions.jsonl") != build["decisions"]["sha256"]:
            raise BuildError(f"{build['decisions']['path']} changed since the candidate was built; rebuild first")
    record = {
        "schema": FROZEN_SCHEMA,
        "benchmark_id": benchmark_id,
        "dataset_id": lines[0].dataset_id if lines else None,
        "manifest_sha256": sha256_bytes(data),
        "lines": len(lines),
        "documents": len({line.document_id for line in lines}),
        "pages": len({(line.document_id, line.page_id) for line in lines}),
        "reference_characters": sum(len(line.gt_canonical) for line in lines),
        "reference_words": sum(len(line.gt_canonical.split()) for line in lines),
        "crop_policies": sorted({line.crop.policy for line in lines}),
        "normalization_protocol": PROTOCOL_ID,
        "scoring": scoring_record(),
        "build_sha256": sha256_file(candidate_dir / "build.json"),
        "decisions_sha256": build["decisions"]["sha256"],
        "source_tree": build["source_tree"],
        "excluded_by_decision": build["counts"]["excluded"],
        "excluded_unresolved_at_freeze": len(unresolved),
        "frozen_at_utc": utc_now(),
        "provenance": record_provenance,
        "rule": "Frozen before any model output was seen. Results on this benchmark are pre-adaptation results.",
    }
    if dataset_card is not None:
        record["dataset_card"] = dataset_card
    (partial / "FROZEN.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.rename(frozen_dir)
    for path in frozen_dir.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return record


def verify_frozen(frozen_dir: Path) -> tuple[dict, list[BenchmarkLine], list[Finding]]:
    """Re-hashes a frozen benchmark: manifest bytes against FROZEN.json, every image against the manifest."""
    frozen_dir = Path(frozen_dir)
    record = json.loads((frozen_dir / "FROZEN.json").read_text(encoding="utf-8"))
    findings: list[Finding] = []
    actual = sha256_file(frozen_dir / "manifest.jsonl")
    if actual != record["manifest_sha256"]:
        findings.append(Finding(severity="blocker", code="frozen.manifest_changed",
                                message=f"manifest sha256 {actual} != recorded {record['manifest_sha256']}"))
    decisions = frozen_dir / "decisions.jsonl"
    if decisions.exists() and sha256_file(decisions) != record["decisions_sha256"]:
        findings.append(Finding(severity="blocker", code="frozen.decisions_changed",
                                message=f"decisions sha256 {sha256_file(decisions)} != recorded {record['decisions_sha256']}"))
    lines = read_manifest(frozen_dir / "manifest.jsonl")
    if len(lines) != record["lines"]:
        findings.append(Finding(severity="blocker", code="frozen.line_count", message=f"{len(lines)} lines != recorded {record['lines']}"))
    findings.extend(f for f in validate_manifest(lines, root=frozen_dir) if f.severity == "blocker")
    return record, lines, findings
