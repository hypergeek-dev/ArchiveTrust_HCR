import unicodedata

import pytest
from pydantic import ValidationError

from archivetrust.htr.benchmark.contract import (
    CropRecord,
    ManifestError,
    manifest_bytes,
    read_manifest,
    safe_id_component,
    sha256_bytes,
    sha256_text,
    validate_manifest,
    write_manifest,
)
from tests.htr.benchmark._builders import make_line


def test_valid_line_round_trips_through_manifest(tmp_path):
    lines = [make_line(line_key="l002", line_order=1), make_line()]
    digest = write_manifest(tmp_path / "manifest.jsonl", lines)
    assert digest == sha256_bytes((tmp_path / "manifest.jsonl").read_bytes())
    loaded = read_manifest(tmp_path / "manifest.jsonl")
    assert [line.line_key for line in loaded] == ["l001", "l002"]
    assert loaded == sorted(lines, key=lambda line: line.line_order)


def test_manifest_hash_is_independent_of_input_order():
    a, b = make_line(), make_line(line_key="l002", line_order=1)
    assert manifest_bytes([a, b]) == manifest_bytes([b, a])


def test_manifest_keeps_non_ascii_verbatim():
    data = manifest_bytes([make_line(gt="Stockholms slott å ö ſ")]).decode("utf-8")
    assert "Stockholms slott å ö ſ" in data and '"schema": "benchmark-line/1"' in data


def test_source_and_canonical_gt_both_preserved():
    line = make_line(gt=unicodedata.normalize("NFD", "Åbo"))
    assert line.gt_source != line.gt_canonical == "Åbo"
    assert line.normalization_applied == ("unicode_nfc",)


@pytest.mark.parametrize(
    "override",
    [
        {"line_id": "doc1/p001/other"},
        {"gt_canonical_sha256": "0" * 64},
        {"gt_source_sha256": "0" * 64},
        {"document_id": "has space"},
        {"line_image_path": "../escape.png"},
        {"line_image_path": "lines\\x.png"},
        {"image_sha256": "ABC"},
        {"normalization_applied": ("lowercase",)},
        {"segmentation_source": "source_polygon"},
        {"unexpected_field": 1},
    ],
)
def test_invalid_lines_are_rejected(override):
    with pytest.raises(ValidationError):
        make_line(**override)


def test_canonical_gt_must_be_nfc_and_rules_must_explain_differences():
    nfd = unicodedata.normalize("NFD", "Åbo")
    with pytest.raises(ValidationError):
        make_line(gt_canonical=nfd, gt_canonical_sha256=sha256_text(nfd), normalization_applied=())
    with pytest.raises(ValidationError):  # differs from source but no rule recorded
        make_line(gt="abc", gt_canonical="abd", gt_canonical_sha256=sha256_text("abd"), normalization_applied=())


def test_bbox_crop_requires_source_image_and_valid_box():
    with pytest.raises(ValidationError):
        CropRecord(policy="bbox_v1", bbox=(0, 0, 10, 10))
    with pytest.raises(ValidationError):
        CropRecord(policy="bbox_v1", bbox=(5, 0, 5, 10), source_image_relative_path="p.jpg", source_image_sha256="a" * 64)
    CropRecord(policy="bbox_v1", bbox=(0, 0, 10, 10), source_image_relative_path="p.jpg", source_image_sha256="a" * 64)


def test_safe_id_component_is_identity_for_safe_ids_and_collision_free_otherwise():
    assert safe_id_component("RA_0042-p.3") == "RA_0042-p.3"
    a, b = safe_id_component("Göteborg 1"), safe_id_component("Göteborg/1")
    assert a != b and a.startswith("Goteborg_1-")
    assert safe_id_component(a) == a


def test_validate_manifest_flags_duplicates_and_empty_gt():
    lines = [
        make_line(),
        make_line(line_key="l002", line_order=1, gt=" "),
        make_line(dataset_id="other", line_key="l003", line_order=2, image_sha256="b" * 64),
    ]
    codes = {f.code: f.severity for f in validate_manifest(lines)}
    assert codes["manifest.duplicate_image_content"] == "warning"
    assert codes["gt.empty"] == "blocker"
    assert codes["manifest.mixed_dataset_id"] == "blocker"
    dup = validate_manifest([make_line(), make_line()])
    assert {f.code for f in dup} >= {"manifest.duplicate_line_id", "manifest.duplicate_image_path"}


def test_validate_manifest_checks_image_files_against_hashes(tmp_path):
    image = tmp_path / "lines/doc1/p001/l001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png-bytes")
    good = make_line(image_sha256=sha256_bytes(b"png-bytes"))
    assert validate_manifest([good], root=tmp_path) == []
    image.write_bytes(b"changed")
    assert [f.code for f in validate_manifest([good], root=tmp_path)] == ["image.hash_mismatch"]
    image.unlink()
    assert [f.code for f in validate_manifest([good], root=tmp_path)] == ["image.missing"]


def test_empty_manifest_is_a_blocker():
    assert [f.code for f in validate_manifest([])] == ["manifest.empty"]


def test_read_manifest_reports_line_number(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"schema": "benchmark-line/1"}\n', encoding="utf-8")
    with pytest.raises(ManifestError, match=r"m.jsonl:1"):
        read_manifest(path)


def test_line_image_paths_stay_short_and_distinct_for_long_ids():
    from archivetrust.htr.benchmark.contract import PATH_COMPONENT_MAX, line_image_relpath

    long_doc = "Gota_hovratt__Fragment_-_Handlingar_rorande_vidskepelse__signerier_och_trolldom__1669-1728"
    a = line_image_relpath(long_doc, long_doc + "__A0060209_00025", "l1", ".png")
    b = line_image_relpath(long_doc, long_doc + "__A0060209_00026", "l1", ".png")
    assert a != b and all(len(part) <= PATH_COMPONENT_MAX + 4 for part in a.split("/"))
    assert line_image_relpath("d", "p", "l1", ".png") == "lines/d/p/l1.png"
