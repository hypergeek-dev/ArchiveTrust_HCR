"""ALTO XML parser tests -- valid parsing (geometry, word confidence, reading order) and
malformed-input failure modes. Fast, no network, no GPU -- always-on."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.providers.transkribus.alto_xml import parse_alto_xml
from archivetrust.providers.transkribus.parsing_models import TranskribusParseError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "transkribus"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parses_blocks_lines_geometry_word_confidence_and_reading_order():
    doc = parse_alto_xml(_read("sample_alto.xml"))
    assert doc.export_format == "alto_xml"
    assert doc.page_width == 2480
    assert doc.page_height == 3508
    assert len(doc.regions) == 2

    block1 = doc.regions[0]
    assert block1.region_id == "block_1"
    assert block1.reading_order_index == 0  # ALTO document order IS reading order
    assert block1.polygon == ((120.0, 140.0), (2200.0, 140.0), (2200.0, 900.0), (120.0, 900.0))
    assert len(block1.lines) == 2

    line1 = block1.lines[0]
    assert line1.text == "Anno 1712 den"
    assert line1.confidence == pytest.approx((0.95 + 0.91 + 0.88) / 3)
    assert line1.reading_order_index == 0
    assert line1.baseline is None  # ALTO has no dedicated baseline element


def test_full_text_is_linearized_in_document_order():
    doc = parse_alto_xml(_read("sample_alto.xml"))
    assert doc.full_text() == "Anno 1712 den\nmedh allmogen\nNB dombook"


def test_processing_date_extracted_from_description_block():
    doc = parse_alto_xml(_read("sample_alto.xml"))
    assert doc.processing_date == "2026-06-12T09:44:17"


def test_alto_has_no_vendor_reported_accuracy_field():
    doc = parse_alto_xml(_read("sample_alto.xml"))
    assert doc.vendor_reported_accuracy is None


# --- Malformed-input failure modes ----------------------------------------------------------


def test_empty_file_raises_typed_error():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_alto_xml("")
    assert excinfo.value.category == "empty_file"


def test_malformed_xml_raises_typed_error():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_alto_xml(_read("malformed_alto.xml"))
    assert excinfo.value.category == "malformed_xml"


def test_wrong_root_element_raises_missing_required_element():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_alto_xml("<notAlto></notAlto>")
    assert excinfo.value.category == "missing_required_element"


def test_missing_layout_element_raises_missing_required_element():
    xml_text = '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"></alto>'
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_alto_xml(xml_text)
    assert excinfo.value.category == "missing_required_element"


def test_missing_page_element_raises_missing_required_element():
    xml_text = (
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout></Layout></alto>'
    )
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_alto_xml(xml_text)
    assert excinfo.value.category == "missing_required_element"


def test_no_text_blocks_produces_empty_document_with_warning():
    xml_text = (
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">'
        "<Layout><Page ID=\"p1\" WIDTH=\"10\" HEIGHT=\"10\">"
        "<PrintSpace></PrintSpace></Page></Layout></alto>"
    )
    doc = parse_alto_xml(xml_text)
    assert doc.regions == ()
    assert any("no <TextBlock>" in w for w in doc.warnings)
