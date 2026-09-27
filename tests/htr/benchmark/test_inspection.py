import json
import unicodedata

from archivetrust.htr.benchmark.inspection import inspect_source, text_findings, write_inspection
from tests.htr.benchmark._fixtures import box, line_png, page_jpg, page_xml, tree_digest, write


def _codes(findings):
    return {f.code for f in findings}


def test_text_findings_flags_but_never_fixes():
    assert _codes(text_findings("k", "p", "Anno 1723", from_line_file=False)) == set()
    codes = _codes(text_findings("k", "p", " lead​ [illeg.] Ã¥r�", from_line_file=False))
    assert {"gt.outer_whitespace", "gt.invisible_chars", "gt.possible_editorial_markup", "gt.mojibake", "gt.replacement_char"} <= codes
    assert _codes(text_findings("k", "p", unicodedata.normalize("NFD", "å"), from_line_file=False)) == {"gt.auto_unicode_nfc"}
    assert _codes(text_findings("k", "p", "text\n", from_line_file=True)) == {"gt.auto_strip_file_line_terminator"}
    assert "gt.embedded_newline" in _codes(text_findings("k", "p", "a\nb", from_line_file=True))


def test_clean_line_pairs_are_ready_and_incoming_is_untouched(tmp_path):
    src = tmp_path / "incoming" / "friend"
    for n in range(3):
        write(src, f"docA/l{n}.png", line_png("abcdefgh"[: n + 3], seed=n * 11))
        write(src, f"docA/l{n}.gt.txt", f"Rad {n} i protokollet\n")
    before = tree_digest(src)
    result = inspect_source(src, charset=set("Rad ipotkl0123"))
    out = tmp_path / "work" / "friend" / "inspection"
    write_inspection(result, out)
    assert tree_digest(src) == before
    assert result.adapter_id == "line_pairs"
    assert result.verdict == "ready_to_build", [f for f in result.findings if f.severity in ("blocker", "needs_review")]
    payload = json.loads((out / "inspection.json").read_text(encoding="utf-8"))
    assert payload["stats"]["segmentation"]["mode"] == "not_needed:supplied_line_images"
    assert payload["stats"]["loghi_out_of_vocabulary"]["characters"].keys() == {"e", "r"}
    assert "READY_TO_BUILD" in (out / "inspection.md").read_text(encoding="utf-8")


def test_problems_are_found(tmp_path):
    src = tmp_path / "src"
    write(src, "d/a.png", line_png(seed=1))
    write(src, "d/a.gt.txt", "text")
    write(src, "d/copy.png", line_png(seed=1))  # identical bytes to a.png
    write(src, "d/copy.gt.txt", "annan text")
    write(src, "d/bad.png", b"\x89PNG broken")
    write(src, "d/bad.gt.txt", "x")
    write(src, "d/latin.png", line_png(seed=5))
    write(src, "d/latin.gt.txt", "sm\xe5".encode("cp1252"))
    write(src, "d/page.png", line_png(size=(900, 1200), seed=3))
    write(src, "d/page.gt.txt", "rad ett\nrad två\nrad tre")
    write(src, "Thumbs.db", b"junk")
    result = inspect_source(src)
    codes = _codes(result.findings)
    assert {"dup.identical_image", "image.unreadable", "text.not_utf8", "gt.page_level_unaligned", "image.page_like",
            "fs.os_junk"} <= codes
    assert result.verdict == "blocked"


def test_page_xml_delivery_checks_geometry(tmp_path):
    src = tmp_path / "src"
    write(src, "doc/0001.jpg", page_jpg((800, 1000)))
    write(src, "doc/page/0001.xml", page_xml("0001.jpg", [("l1", "Anno", box(50, 100, 700, 140)),
                                                            ("l2", "utan koord", None),
                                                            ("l3", "utanför", box(700, 900, 820, 960))]))
    write(src, "doc/0002.jpg", page_jpg((800, 1000)))
    write(src, "doc/page/0002.xml", page_xml("0002.jpg", [("l1", "roterad", box(10, 10, 300, 50))], width=1000, height=800))
    result = inspect_source(src)
    assert result.adapter_id == "page_xml"
    assert {"layout.line_without_coords", "layout.polygon_out_of_bounds", "page.size_swapped"} <= _codes(result.findings)
    assert result.stats["segmentation"]["mode"] == "partial:some_lines_lack_coordinates"


def test_unrecognized_delivery_is_blocked(tmp_path):
    write(tmp_path, "scan.jpg", page_jpg())
    result = inspect_source(tmp_path)
    assert result.verdict == "blocked" and "format.unrecognized" in _codes(result.findings)
