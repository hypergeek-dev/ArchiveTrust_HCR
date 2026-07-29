"""PAGE XML parser tests -- valid parsing (geometry, confidence, reading order, vendor metadata)
and malformed-input failure modes. Fast, no network, no GPU -- always-on."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.providers.transkribus.page_xml import parse_page_xml
from archivetrust.providers.transkribus.parsing_models import TranskribusParseError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "transkribus"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parses_regions_lines_geometry_confidence_and_reading_order():
    doc = parse_page_xml(_read("sample_page.xml"))
    assert doc.export_format == "page_xml"
    assert doc.page_width == 2480
    assert doc.page_height == 3508
    assert doc.image_filename == "volume_12_folio_003r.jpg"
    assert len(doc.regions) == 2

    r1 = next(r for r in doc.regions if r.region_id == "r1")
    assert r1.region_type == "paragraph"
    assert r1.reading_order_index == 0  # from explicit <ReadingOrder>
    assert r1.polygon == ((120.0, 140.0), (2200.0, 140.0), (2200.0, 900.0), (120.0, 900.0))
    assert len(r1.lines) == 2

    l1 = r1.lines[0]
    assert l1.text == "Anno 1712 den 3 Januarii holltes ting"
    assert l1.confidence == pytest.approx(0.93)
    assert l1.reading_order_index == 0
    assert l1.polygon is not None
    assert l1.baseline == ((130.0, 210.0), (2000.0, 210.0))

    r2 = next(r for r in doc.regions if r.region_id == "r2")
    assert r2.reading_order_index == 1  # from explicit <ReadingOrder>
    assert r2.region_type == "marginalia"


def test_full_text_is_linearized_in_reading_order():
    doc = parse_page_xml(_read("sample_page.xml"))
    assert doc.full_text() == (
        "Anno 1712 den 3 Januarii holltes ting\n"
        "medh allmogen aff Sochnen\n"
        "NB dombook"
    )


def test_mean_confidence_averages_present_line_confidences():
    doc = parse_page_xml(_read("sample_page.xml"))
    assert doc.mean_confidence() == pytest.approx((0.93 + 0.87 + 0.79) / 3)


def test_vendor_metadata_extracted_as_separate_fields():
    doc = parse_page_xml(_read("sample_page.xml"))
    assert doc.vendor_reported_accuracy == pytest.approx(0.912)
    assert doc.model_version_hint == "Swedish Lion I - v3"
    assert doc.job_id_hint == "job_884215"
    assert doc.transkribus_document_id_hint == "doc_55201"
    assert doc.processing_date == "2026-06-12T09:44:17"  # LastChange wins over Created (later in file)


def test_confidence_absent_is_none_not_fabricated():
    doc = parse_page_xml(_read("no_confidence_page.xml"))
    assert len(doc.regions) == 1
    line = doc.regions[0].lines[0]
    assert line.confidence is None
    assert doc.mean_confidence() is None


# --- Malformed-input failure modes ----------------------------------------------------------


def test_empty_file_raises_typed_error():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_page_xml("")
    assert excinfo.value.category == "empty_file"


def test_whitespace_only_file_raises_empty_file_error():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_page_xml("   \n  \n")
    assert excinfo.value.category == "empty_file"


def test_malformed_xml_raises_typed_error_not_a_bare_parse_error():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_page_xml(_read("malformed_page.xml"))
    assert excinfo.value.category == "malformed_xml"


def test_wrong_root_element_raises_missing_required_element():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_page_xml("<NotPcGts></NotPcGts>")
    assert excinfo.value.category == "missing_required_element"


def test_missing_page_element_raises_missing_required_element():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_page_xml(_read("missing_page_element.xml"))
    assert excinfo.value.category == "missing_required_element"


def test_no_text_regions_produces_empty_document_with_warning_not_a_crash():
    xml_text = (
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageWidth="10" imageHeight="10"></Page></PcGts>'
    )
    doc = parse_page_xml(xml_text)
    assert doc.regions == ()
    assert any("no <TextRegion>" in w for w in doc.warnings)
