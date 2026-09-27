import unicodedata

from archivetrust.htr.benchmark.normalization import (
    RULE_NFC,
    RULE_STRIP_FILE_LINE_TERMINATOR,
    canonicalize_gt,
    canonicalize_prediction,
)


def test_nfc_composes_and_is_recorded():
    decomposed = unicodedata.normalize("NFD", "Åhus")
    canonical, applied = canonicalize_gt(decomposed)
    assert canonical == "Åhus" and applied == (RULE_NFC,)


def test_already_canonical_text_records_no_rule():
    assert canonicalize_gt("Anno 1723") == ("Anno 1723", ())


def test_file_terminator_stripped_only_for_line_files_and_only_once():
    assert canonicalize_gt("text\r\n", from_line_file=True) == ("text", (RULE_STRIP_FILE_LINE_TERMINATOR,))
    assert canonicalize_gt("text\n\n", from_line_file=True) == ("text\n", (RULE_STRIP_FILE_LINE_TERMINATOR,))
    assert canonicalize_gt("text\n") == ("text\n", ())


def test_nothing_lexical_is_changed():
    for text in ("  lead", "trail ", "a  b", "ſtor", "Ano", "Kongl. M:tt", "ÆØ", "d. 4:de"):
        assert canonicalize_gt(text) == (text, ())


def test_prediction_canonicalization_is_nfc_and_outer_strip_only():
    assert canonicalize_prediction(unicodedata.normalize("NFD", " Öl  och ")) == "Öl  och"
