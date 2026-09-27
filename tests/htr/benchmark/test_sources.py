import json

import pytest

from archivetrust.htr.benchmark.sources import LinePairsAdapter, PageXmlAdapter, TabularAdapter
from archivetrust.htr.benchmark.sources.base import list_files
from archivetrust.htr.benchmark.sources.xml_lines import AltoAdapter
from tests.htr.benchmark._fixtures import box, line_png, page_jpg, page_xml, write


def _extract(adapter, root):
    files = list_files(root)
    return adapter.detect(root, files), adapter.extract(root, files)


def test_line_pairs_reads_sidecars_in_natural_order(tmp_path):
    for n, text in ((10, "tio\n"), (2, "två")):
        write(tmp_path, f"vol1/line{n}.png", line_png(text, seed=n))
        write(tmp_path, f"vol1/line{n}.gt.txt", text)
    write(tmp_path, "vol1/line3.png", line_png(seed=3))
    write(tmp_path, "notes.txt", "readme")
    score, ext = _extract(LinePairsAdapter(), tmp_path)
    assert score == pytest.approx(2 / 3)
    assert [(c.line_raw, c.line_order, c.gt_source) for c in ext.lines] == [
        ("line2", 0, "två"), ("line3", 1, None), ("line10", 2, "tio\n")]
    assert {c.document_raw for c in ext.lines} == {"vol1"}
    codes = {f.code for f in ext.findings}
    assert {"gt.missing", "gt.orphan_transcription", "layout.page_unknown"} <= codes


def test_page_xml_reads_nested_table_lines_and_reading_order(tmp_path):
    xml = page_xml("0001.jpg", [("l1", "andra", box(50, 220, 700, 260)), ("l2", "första", box(50, 100, 700, 140))],
                   reading_order=["r_cell", "r1"], table_line=("t1l1", "tabell", box(50, 580, 700, 620)))
    write(tmp_path, "doc/page/0001.xml", xml)
    write(tmp_path, "doc/0001.jpg", page_jpg())
    score, ext = _extract(PageXmlAdapter(), tmp_path)
    assert score == 1.0
    assert [(c.line_raw, c.gt_source) for c in ext.lines] == [("t1l1", "tabell"), ("l1", "andra"), ("l2", "första")]
    assert ext.lines[0].image_path == "doc/0001.jpg" and ext.lines[0].document_raw == "doc" and ext.lines[0].page_raw == "0001"
    assert ext.lines[0].declared_page_size == (800, 1000)


def test_page_xml_refuses_to_choose_between_textequivs_and_reports_missing_image(tmp_path):
    write(tmp_path, "p.xml", page_xml("gone.jpg", [("l1", ["GT", "OCR"], box(0, 0, 10, 10)), ("l2", None, None)]))
    _, ext = _extract(PageXmlAdapter(), tmp_path)
    assert [c.gt_source for c in ext.lines] == [None, None]
    assert {f.code for f in ext.findings} >= {"gt.multiple_textequiv", "gt.missing", "page.image_missing"}


def test_page_xml_keeps_text_verbatim(tmp_path):
    write(tmp_path, "p.xml", page_xml("p.jpg", [("l1", "  Anno  1723 ", box(0, 0, 10, 10))]))
    write(tmp_path, "p.jpg", page_jpg())
    _, ext = _extract(PageXmlAdapter(), tmp_path)
    assert ext.lines[0].gt_source == "  Anno  1723 "


def test_alto_joins_tokens_and_blocks_non_pixel_units(tmp_path):
    alto = """<?xml version="1.0"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Description>
<MeasurementUnit>pixel</MeasurementUnit><sourceImageInformation><fileName>a.jpg</fileName></sourceImageInformation></Description>
<Layout><Page WIDTH="800" HEIGHT="1000"><PrintSpace><TextBlock ID="b1">
<TextLine ID="t1" HPOS="10" VPOS="20" WIDTH="300" HEIGHT="40"><String CONTENT="Kongl."/><SP/><String CONTENT="Maj"/><HYP CONTENT="-"/></TextLine>
<TextLine ID="t2" HPOS="10" VPOS="80" WIDTH="300" HEIGHT="40"><String CONTENT="utan"/><String CONTENT="mellanslag"/></TextLine>
</TextBlock></PrintSpace></Page></Layout></alto>"""
    write(tmp_path, "a.xml", alto)
    write(tmp_path, "a.jpg", page_jpg())
    _, ext = _extract(AltoAdapter(), tmp_path)
    assert [c.gt_source for c in ext.lines] == ["Kongl. Maj-", "utan mellanslag"]
    assert [c.metadata["alto_implicit_space"] for c in ext.lines] == [False, True]
    assert ext.lines[0].polygon == ((10, 20), (310, 20), (310, 60), (10, 60))
    write(tmp_path, "a.xml", alto.replace(">pixel<", ">mm10<"))
    _, ext = _extract(AltoAdapter(), tmp_path)
    assert "alto.non_pixel_units" in {f.code for f in ext.findings}


def test_tabular_csv_headerless_tsv_and_unresolved_columns(tmp_path):
    write(tmp_path, "img/a.png", line_png(seed=1))
    write(tmp_path, "img/b.png", line_png(seed=2))
    write(tmp_path, "gt.csv", 'file,transcription,writer\nimg/a.png,"Anno, 1723",W1\nimg/b.png,Maj,W2\n')
    score, ext = _extract(TabularAdapter(), tmp_path)
    assert score == 0.9
    assert [(c.line_raw, c.gt_source, c.writer_id, c.document_raw) for c in ext.lines] == [
        ("a", "Anno, 1723", "W1", "img"), ("b", "Maj", "W2", "img")]

    other = tmp_path / "loghi"
    write(other, "x.png", line_png())
    write(other, "list.tsv", "x.png\tmed\ttab\n")
    _, ext = _extract(TabularAdapter(), other)
    assert ext.lines[0].gt_source == "med\ttab"
    assert "tabular.headerless" in {f.code for f in ext.findings}

    bad = tmp_path / "bad"
    write(bad, "t.jsonl", json.dumps({"image": "x.png", "text": "a", "label": "b"}) + "\n")
    _, ext = _extract(TabularAdapter(), bad)
    assert "tabular.columns_unresolved" in {f.code for f in ext.findings}


def test_tabular_parquet_with_embedded_images(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({
        "image": [{"bytes": line_png(seed=1), "path": "rad_1.png"}, {"bytes": line_png(seed=2), "path": "rad_2.png"}],
        "transcription": ["ett", "två"],
        "collection": ["C1", "C1"],
    })
    pq.write_table(table, tmp_path / "test.parquet")
    _, ext = _extract(TabularAdapter(), tmp_path)
    assert [(c.line_raw, c.gt_source, c.collection, c.image_path) for c in ext.lines] == [
        ("rad_1", "ett", "C1", "test.parquet#row=0"), ("rad_2", "två", "C1", "test.parquet#row=1")]
    assert ext.lines[0].image_bytes == line_png(seed=1)
