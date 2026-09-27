from types import SimpleNamespace

import pytest

from archivetrust.htr.benchmark.scoring import aggregate, harmonize_line_end_hyphen, score_line, scoring_record


@pytest.mark.parametrize("text, expected", [
    ("hwar¬", "hwar-"),
    ("hwar-", "hwar-"),
    ("a¬b", "a¬b"),  # internal: untouched
    ("¬ß;", "¬ß;"),  # nothing but a line-final ¬ changes
    ("", ""),
])
def test_harmonize_line_end_hyphen_only_touches_the_final_character(text, expected):
    assert harmonize_line_end_hyphen(text) == expected


def _score(ref: str, hyp: str):
    line = SimpleNamespace(line_id="d/p/l", document_id="d", page_id="p", collection=None, gt_canonical=ref)
    return score_line(line, SimpleNamespace(prediction=hyp, status="ok"))


def test_sensitivity_score_equates_line_final_hyphen_conventions_and_leaves_raw_alone():
    final = _score("anled-", "anled¬")
    assert final.char_edits == 1 and final.hyph_char_edits == 0
    assert final.word_edits == 1 and final.hyph_word_edits == 0
    internal = _score("Jöns-son", "Jöns¬son")
    assert internal.char_edits == 1 and internal.hyph_char_edits == 1
    other = _score("hwarß;", "hwarss")
    assert other.hyph_char_edits == other.char_edits == 2
    corpus = aggregate([final, internal, other])
    assert corpus["cer"] > corpus["cer_line_end_hyphen_harmonized"]
    assert scoring_record()["primary"] == "raw" and "line_end_hyphen_harmonized" in scoring_record()["sensitivity"]
