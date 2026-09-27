import io
import json

import pytest
from PIL import Image

from archivetrust.htr.benchmark.build import BuildError, build_candidate, freeze, verify_frozen
from archivetrust.htr.benchmark.contract import read_manifest, sha256_file
from tests.htr.benchmark._fixtures import box, line_png, page_jpg, page_xml, tree_digest, write


def _line_source(root):
    write(root, "docA/l1.png", line_png("abc", seed=1))
    write(root, "docA/l1.gt.txt", "Anno 1723\n")
    write(root, "docA/l2.png", line_png("abcd", seed=2))
    write(root, "docA/l2.gt.txt", " kantblanksteg")
    write(root, "docA/l3.png", line_png("abcde", seed=3))
    return root


def _decide(path, *rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_build_routes_problems_to_review_and_supplied_images_are_byte_identical(tmp_path):
    src = _line_source(tmp_path / "incoming" / "s1")
    before = tree_digest(src)
    summary = build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="s1")
    assert tree_digest(src) == before
    assert (summary.included, summary.review, summary.excluded) == (1, 2, 0)
    queue = [json.loads(r) for r in (tmp_path / "cand/review_queue.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {q["target"]: {f["code"] for f in q["findings"]} for q in queue} == {
        "docA/lines/l2": {"gt.outer_whitespace"}, "docA/lines/l3": {"gt.missing"}}
    [line] = read_manifest(tmp_path / "cand/manifest.jsonl")
    assert line.line_id == "docA/lines/l1" and line.gt_source == "Anno 1723\n" and line.gt_canonical == "Anno 1723"
    assert line.normalization_applied == ("strip_file_line_terminator",)
    assert sha256_file(tmp_path / "cand" / line.line_image_path) == sha256_file(src / "docA/l1.png")
    with pytest.raises(BuildError, match="exists"):
        build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="s1")


def test_decisions_resolve_review_items_and_freeze_is_immutable(tmp_path):
    src = _line_source(tmp_path / "incoming" / "s1")
    decisions = _decide(tmp_path / "decisions.jsonl",
                        {"target": "docA/lines/l2", "action": "set_gt", "gt": "kantblanksteg", "reason": "stray leading space"},
                        {"target": "docA/lines/l3", "action": "set_gt", "gt": "Maj", "reason": "transcribed by reviewer"})
    summary = build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="s1", decisions_path=decisions)
    assert (summary.included, summary.review) == (3, 0)
    by_key = {line.line_key: line for line in read_manifest(tmp_path / "cand/manifest.jsonl")}
    assert by_key["l2"].gt_source == " kantblanksteg" and by_key["l2"].gt_canonical == "kantblanksteg"
    assert by_key["l2"].normalization_applied == ("human_correction",)
    assert by_key["l3"].gt_canonical == "Maj" and by_key["l3"].normalization_applied == ("human_correction",)
    assert by_key["l3"].source_metadata["gt_source_missing"] is True

    record = freeze(tmp_path / "cand", tmp_path / "bench" / "b1", benchmark_id="b1")
    assert record["lines"] == 3 and record["decisions_sha256"] == sha256_file(decisions)
    assert record["provenance"]["official"] is False
    assert record["scoring"]["primary"] == "raw" and "line_end_hyphen_harmonized" in record["scoring"]["sensitivity"]
    _, lines, findings = verify_frozen(tmp_path / "bench" / "b1")
    assert findings == [] and len(lines) == 3
    with pytest.raises(BuildError, match="never overwritten"):
        freeze(tmp_path / "cand", tmp_path / "bench" / "b1", benchmark_id="b1")


def test_accept_cannot_resolve_unscorable_gt(tmp_path):
    src = _line_source(tmp_path / "incoming" / "s1")
    decisions = _decide(tmp_path / "d.jsonl", {"target": "docA/lines/l2", "action": "accept", "reason": "x"})
    summary = build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="s1", decisions_path=decisions)
    assert summary.review == 2


def test_freeze_refuses_pending_review_unless_explicitly_excluded(tmp_path):
    src = _line_source(tmp_path / "incoming" / "s1")
    build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="s1")
    with pytest.raises(BuildError, match="review_queue"):
        freeze(tmp_path / "cand", tmp_path / "bench" / "b1", benchmark_id="b1")
    record = freeze(tmp_path / "cand", tmp_path / "bench" / "b1", benchmark_id="b1", exclude_unresolved=True)
    assert record["excluded_unresolved_at_freeze"] == 2 and record["lines"] == 1


def test_verify_detects_tampering(tmp_path):
    src = _line_source(tmp_path / "incoming" / "s1")
    build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="s1")
    frozen = tmp_path / "bench" / "b1"
    freeze(tmp_path / "cand", frozen, benchmark_id="b1", exclude_unresolved=True)
    image = next((frozen / "lines").rglob("*.png"))
    image.chmod(0o666)
    image.write_bytes(line_png("changed", seed=9))
    _, _, findings = verify_frozen(frozen)
    assert [f.code for f in findings] == ["image.hash_mismatch"]


def test_page_xml_build_crops_from_polygons(tmp_path):
    src = tmp_path / "incoming" / "px"
    write(src, "doc/0001.jpg", page_jpg((800, 1000)))
    write(src, "doc/page/0001.xml", page_xml("0001.jpg", [("l1", "Anno", box(50, 100, 700, 140)),
                                                            ("l2", "Maj", box(50, 220, 700, 260))]))
    summary = build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="px")
    assert (summary.included, summary.review) == (2, 0)
    lines = read_manifest(tmp_path / "cand/manifest.jsonl")
    assert [line.line_id for line in lines] == ["doc/0001/l1", "doc/0001/l2"]
    assert lines[0].crop.policy == "bbox_v1" and lines[0].crop.bbox == (50, 100, 700, 140)
    assert lines[0].crop.source_image_sha256 == sha256_file(src / "doc/0001.jpg")
    assert Image.open(io.BytesIO((tmp_path / "cand" / lines[0].line_image_path).read_bytes())).size == (650, 40)
    again = build_candidate(src, tmp_path / "cand2", dataset_id="ds1", source_id="px")
    assert again.manifest_sha256 == summary.manifest_sha256  # deterministic
    assert "crop_clamped" not in lines[0].source_metadata


def test_page_xml_build_flags_clamped_crops(tmp_path):
    src = tmp_path / "incoming" / "px"
    write(src, "doc/0001.jpg", page_jpg((800, 1000)))
    write(src, "doc/page/0001.xml", page_xml("0001.jpg", [("l1", "Anno", box(50, 100, 700, 140)),
                                                            ("l2", "Maj", box(50, 960, 900, 1040))]))
    build_candidate(src, tmp_path / "cand", dataset_id="ds1", source_id="px")
    by_key = {line.line_key: line for line in read_manifest(tmp_path / "cand/manifest.jsonl")}
    assert "crop_clamped" not in by_key["l1"].source_metadata
    assert by_key["l2"].source_metadata["crop_clamped"] is True and by_key["l2"].crop.bbox == (50, 960, 800, 1000)


def test_candidate_cannot_be_written_inside_incoming(tmp_path):
    src = _line_source(tmp_path / "incoming" / "s1")
    with pytest.raises(Exception, match="protected"):
        build_candidate(src, src / "cand", dataset_id="ds1", source_id="s1")
