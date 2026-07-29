"""Historical-feature evaluator tests (docs/htr-migration-plan.md Stage 9).

Real-data cases use `tests/fixtures/transkribus/sample_page.xml`'s actual PAGE XML content (real
17th/18th-century Swedish court-record text, transcribed by a real Transkribus model per that
fixture's own metadata) and SATRN's real documented output (`providers/satrn/README.md`). A small
number of synthetic strings are added only to isolate specific evaluator behaviors (labeled
synthetic, not claims of real data).
"""

from __future__ import annotations

from archivetrust.htr.evaluation.historical_features import (
    AbbreviationEvaluator,
    HISTORICAL_FEATURE_REGISTRY,
    HistoricalDateEvaluator,
    UnusualCharacterEvaluator,
    evaluate_feature,
    historical_feature_metric_results,
)

# Real lines from tests/fixtures/transkribus/sample_page.xml (see that file directly).
REAL_PAGE_LINE_1 = "Anno 1712 den 3 Januarii holltes ting"
REAL_PAGE_LINE_2 = "medh allmogen aff Sochnen"
REAL_MARGINALIA_LINE = "NB dombook"

SATRN_REAL_OUTPUT = "till den 23 Januarii"  # providers/satrn/README.md


# --- HistoricalDateEvaluator -------------------------------------------------------------------


def test_date_evaluator_extracts_real_anno_and_month_from_page_xml_fixture():
    evaluator = HistoricalDateEvaluator()
    spans = evaluator.extract(REAL_PAGE_LINE_1)
    assert any("Anno 1712" in span for span in spans)
    assert any("Januarii" in span for span in spans)


def test_date_evaluator_finds_no_date_in_the_second_real_page_xml_line():
    evaluator = HistoricalDateEvaluator()
    assert evaluator.extract(REAL_PAGE_LINE_2) == ()


def test_date_evaluator_matches_satrns_real_documented_output_as_a_false_positive_against_empty_ground_truth():
    """SATRN's real output ('till den 23 Januarii') contains a date-shaped phrase the ground
    truth (a name/testimony line with no date at all) does not -- a genuine false positive this
    evaluator should surface, not silently ignore."""
    evaluator = HistoricalDateEvaluator()
    ground_truth = "bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt"
    result = evaluate_feature(evaluator, reference=ground_truth, hypothesis=SATRN_REAL_OUTPUT)
    assert result.reference_spans == ()
    assert len(result.hypothesis_spans) >= 1
    assert result.false_positives >= 1
    assert result.recall is None  # no ground-truth dates to recall
    assert result.precision == 0.0  # every hypothesis date span was unsupported


def test_date_evaluator_perfect_match_on_real_page_xml_line():
    evaluator = HistoricalDateEvaluator()
    result = evaluate_feature(evaluator, reference=REAL_PAGE_LINE_1, hypothesis=REAL_PAGE_LINE_1)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


# --- AbbreviationEvaluator ----------------------------------------------------------------------


def test_abbreviation_evaluator_finds_nb_in_real_marginalia_line():
    evaluator = AbbreviationEvaluator()
    spans = evaluator.extract(REAL_MARGINALIA_LINE)
    assert spans == ("NB",)


def test_abbreviation_evaluator_finds_nothing_in_a_line_with_no_known_abbreviation():
    evaluator = AbbreviationEvaluator()
    assert evaluator.extract(REAL_PAGE_LINE_1) == ()


def test_abbreviation_evaluator_synthetic_kongl_majt_case():
    """Synthetic sentence (not from a fixture) exercising a second known abbreviation, "K.M:t"
    (Kongl. Maj:t / Royal Majesty), common in Swedish 17th-century legal documents."""
    evaluator = AbbreviationEvaluator()
    sentence = "Effter K.M:t befalning skall NB dombook föras."
    spans = evaluator.extract(sentence)
    assert "K.M:t" in spans
    assert "NB" in spans


# --- UnusualCharacterEvaluator -------------------------------------------------------------------


def test_unusual_character_evaluator_flags_long_s_archaic_character():
    """Archaic-spelling/historical-character test case (task brief requirement): the long s,
    'ſ' (U+017F), a genuine 17th-century Swedish/German-influenced handwriting convention --
    synthetic sentence, not from a fixture, engineered to isolate this one character."""
    evaluator = UnusualCharacterEvaluator()
    synthetic_sentence = "Han ſade att han war fri man."
    spans = evaluator.extract(synthetic_sentence)
    assert "ſ" in spans


def test_unusual_character_evaluator_finds_nothing_unusual_in_standard_modern_swedish_text():
    evaluator = UnusualCharacterEvaluator()
    assert evaluator.extract(REAL_PAGE_LINE_2) == ()


def test_unusual_character_evaluator_real_page_xml_lines_are_all_standard_characters():
    """Sanity check against real data: none of the real PAGE XML fixture's lines happen to use
    archaic characters -- confirms the evaluator does not over-fire on ordinary real Swedish text.
    """
    evaluator = UnusualCharacterEvaluator()
    for line in (REAL_PAGE_LINE_1, REAL_PAGE_LINE_2, REAL_MARGINALIA_LINE):
        assert evaluator.extract(line) == ()


def test_unusual_character_evaluator_precision_recall_on_synthetic_archaic_text():
    evaluator = UnusualCharacterEvaluator()
    reference = "Han ſade att han war fri man."
    hypothesis = "Han sade att han war fri man."  # OCR normalized away the long s
    result = evaluate_feature(evaluator, reference=reference, hypothesis=hypothesis)
    assert result.reference_spans == ("ſ",)
    assert result.hypothesis_spans == ()
    assert result.recall == 0.0  # the archaic character was present in truth, missed entirely
    # Nothing was predicted at all (hypothesis_spans == ()) while ground truth had a real
    # instance -- precision is defined as 0.0 here (there was an evaluable population), distinct
    # from the both-empty case (test_date_evaluator_matches_..._false_positive... 's sibling case)
    # where precision is undefined (None) because nothing existed on either side to score.
    assert result.precision == 0.0
    assert result.f1 is None  # precision + recall == 0 -- harmonic mean is undefined, not 0.0


# --- historical_feature_metric_results / registry -----------------------------------------------


def test_historical_feature_metric_results_never_fabricates_a_value_for_undefined_precision():
    evaluator = HistoricalDateEvaluator()
    results = historical_feature_metric_results(
        evaluator,
        reference="bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt",
        hypothesis=SATRN_REAL_OUTPUT,
        method_run_id="method_run_1",
    )
    # recall (and therefore f1) is None here (no reference dates) -- must not appear as a
    # fabricated 0.0/1.0 result; only precision (a real, defined 0.0) is produced.
    assert len(results) == 1
    assert results[0].value == 0.0


def test_registry_contains_all_three_concrete_evaluators():
    assert set(HISTORICAL_FEATURE_REGISTRY) == {"historical_dates", "abbreviations", "unusual_characters"}
    for evaluator in HISTORICAL_FEATURE_REGISTRY.values():
        assert callable(evaluator.extract)
        assert isinstance(evaluator.feature_name, str) and evaluator.feature_name
