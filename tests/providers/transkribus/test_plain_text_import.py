"""Plain-text manual import -- the trivial case, still going through the same ParsedDocument
provenance wrapper as PAGE/ALTO."""

from __future__ import annotations

import pytest

from archivetrust.providers.transkribus.parsing_models import TranskribusParseError
from archivetrust.providers.transkribus.plain_text import parse_plain_text


def test_parses_each_nonblank_line_as_a_line_with_no_geometry_or_confidence():
    doc = parse_plain_text("Anno 1712 den 3 Januarii\nmedh allmogen\n\nNB dombook\n")
    assert doc.export_format == "plain_text"
    assert len(doc.regions) == 1
    lines = doc.regions[0].lines
    # The blank line is skipped, but reading_order_index reflects original file line position.
    assert [line.text for line in lines] == ["Anno 1712 den 3 Januarii", "medh allmogen", "NB dombook"]
    assert [line.reading_order_index for line in lines] == [0, 1, 3]
    for line in lines:
        assert line.confidence is None
        assert line.polygon is None
        assert line.baseline is None


def test_full_text_joins_lines_with_newlines():
    doc = parse_plain_text("line one\nline two")
    assert doc.full_text() == "line one\nline two"


def test_empty_string_raises_empty_file_error():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_plain_text("")
    assert excinfo.value.category == "empty_file"


def test_whitespace_only_raises_empty_file_error():
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_plain_text("   \n\n  ")
    assert excinfo.value.category == "empty_file"


def test_all_whitespace_input_raises_empty_file_not_a_crash():
    # Every line individually blank -> the overall text.strip() guard catches this as empty_file
    # before any per-line logic runs; asserted here so that guard's behavior stays pinned.
    with pytest.raises(TranskribusParseError) as excinfo:
        parse_plain_text("\n\n\n \n")
    assert excinfo.value.category == "empty_file"
