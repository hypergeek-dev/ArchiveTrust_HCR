"""`page_xml.py::parse_loghi_page_xml` -- valid parsing, malformed-input failure, and the provenance
fields (source hash, schema version, omitted fields) the brief requires be preserved."""

from __future__ import annotations

import hashlib

import pytest

from archivetrust.providers.loghi.page_xml import parse_loghi_page_xml
from archivetrust.providers.loghi.parsing_models import LoghiParseError

_VALID_XML = """<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Metadata><Creator>loghi-htr v1.2</Creator><Created>2026-08-01T00:00:00Z</Created></Metadata>
  <Page imageWidth="1000" imageHeight="2000" imageFilename="page1.jpg">
    <ReadingOrder>
      <OrderedGroup id="ro1">
        <RegionRefIndexed index="0" regionRef="r2"/>
        <RegionRefIndexed index="1" regionRef="r1"/>
      </OrderedGroup>
    </ReadingOrder>
    <TextRegion id="r1">
      <Coords points="0,0 100,0 100,100 0,100"/>
      <TextLine id="l1">
        <Coords points="0,0 100,10 0,10"/>
        <Baseline points="0,10 100,10"/>
        <TextEquiv conf="0.95"><Unicode>first line</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
    <TextRegion id="r2">
      <TextLine id="l2">
        <TextEquiv><Unicode>second line, no confidence</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>"""


def test_parses_regions_lines_and_reading_order() -> None:
    parsed = parse_loghi_page_xml(_VALID_XML)
    assert parsed.page_width == 1000
    assert parsed.page_height == 2000
    assert parsed.image_filename == "page1.jpg"
    assert len(parsed.regions) == 2
    # explicit ReadingOrder puts r2 (index 0) before r1 (index 1)
    assert parsed.full_text() == "second line, no confidence\nfirst line"


def test_computes_source_xml_hash_over_the_exact_input_text() -> None:
    parsed = parse_loghi_page_xml(_VALID_XML)
    expected = hashlib.sha256(_VALID_XML.encode("utf-8")).hexdigest()
    assert parsed.source_xml_hash == expected


def test_reads_page_schema_version_from_namespace() -> None:
    parsed = parse_loghi_page_xml(_VALID_XML)
    assert parsed.page_schema_version == "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def test_reads_creator_metadata_verbatim() -> None:
    parsed = parse_loghi_page_xml(_VALID_XML)
    assert parsed.creator_metadata == "loghi-htr v1.2"


def test_mean_confidence_ignores_lines_with_no_confidence() -> None:
    parsed = parse_loghi_page_xml(_VALID_XML)
    assert parsed.mean_confidence() == 0.95


def test_omitted_fields_are_declared() -> None:
    parsed = parse_loghi_page_xml(_VALID_XML)
    assert "per_glyph_geometry" in parsed.omitted_fields


def test_geometry_is_preserved() -> None:
    parsed = parse_loghi_page_xml(_VALID_XML)
    region = next(r for r in parsed.regions if r.region_id == "r1")
    assert region.polygon == ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    line = region.lines[0]
    assert line.baseline == ((0.0, 10.0), (100.0, 10.0))


def test_empty_file_raises_loghi_parse_error() -> None:
    with pytest.raises(LoghiParseError) as exc_info:
        parse_loghi_page_xml("")
    assert exc_info.value.category == "empty_file"


def test_malformed_xml_raises_loghi_parse_error() -> None:
    with pytest.raises(LoghiParseError) as exc_info:
        parse_loghi_page_xml("<PcGts><Page unterminated")
    assert exc_info.value.category == "malformed_xml"


def test_wrong_root_element_raises_loghi_parse_error() -> None:
    with pytest.raises(LoghiParseError) as exc_info:
        parse_loghi_page_xml("<NotPcGts></NotPcGts>")
    assert exc_info.value.category == "missing_required_element"


def test_missing_page_element_raises_loghi_parse_error() -> None:
    with pytest.raises(LoghiParseError) as exc_info:
        parse_loghi_page_xml(
            '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"></PcGts>'
        )
    assert exc_info.value.category == "missing_required_element"


def test_no_regions_produces_a_warning_not_an_error() -> None:
    xml = """<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
      <Page imageWidth="10" imageHeight="10"></Page>
    </PcGts>"""
    parsed = parse_loghi_page_xml(xml)
    assert parsed.regions == ()
    assert any("no <TextRegion>" in w for w in parsed.warnings)


def test_never_confused_with_transkribus_parse_result_type() -> None:
    """Structural guard for the brief's "never merge" instruction -- Loghi's parse result is its own
    type, not `providers.transkribus.parsing_models.ParsedDocument`."""
    from archivetrust.providers.transkribus.parsing_models import ParsedDocument

    parsed = parse_loghi_page_xml(_VALID_XML)
    assert not isinstance(parsed, ParsedDocument)
