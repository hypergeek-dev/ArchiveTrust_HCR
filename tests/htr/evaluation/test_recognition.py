"""Recognition metric tests (docs/htr-migration-plan.md Stage 9).

Real-data cases use the actual fixture ground truth
(`tests/fixtures/htr/trolldomskommissionen_sample_line.txt`) against the real, documented model
outputs recorded in `providers/satrn/README.md` / `providers/florence2_htr/README.md`'s "Measured
real-inference behavior" tables -- these are genuine (if inaccurate) recognition results, not
strings invented for this test file. Synthetic cases are labeled as such and exist only to pin
down edit-operation classification edge cases real fixture data cannot reliably isolate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.evaluation.metrics import levenshtein
from archivetrust.htr.evaluation import definitions
from archivetrust.htr.evaluation.recognition import (
    classify_char_edits,
    classify_word_edits,
    compute_recognition_metrics,
    exact_line_accuracy,
    recognition_metric_results,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
GROUND_TRUTH_TEXT = (
    FIXTURES / "htr" / "trolldomskommissionen_sample_line.txt"
).read_text(encoding="utf-8").strip()

# Real, documented model outputs (see module docstring) for the same real fixture line.
SATRN_REAL_OUTPUT = "till den 23 Januarii"
FLORENCE2_REAL_PARSED_OUTPUT = "Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff"
FLORENCE2_REAL_RAW_OUTPUT = "</s><s>Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff</s>"


# --- Real fixture / real documented adapter output ------------------------------------------


def test_ground_truth_fixture_loads_and_is_real_swedish_text():
    assert "bekiendt" in GROUND_TRUTH_TEXT
    assert "å" in GROUND_TRUTH_TEXT or "ä" in GROUND_TRUTH_TEXT  # å / ä present


def test_satrn_real_output_scores_a_genuine_high_cer_against_ground_truth():
    """SATRN's own README documents this as a real, non-trivial recognition error (a
    preprocessing-domain mismatch, not an adapter bug) -- this test computes the actual honest
    number from real data, it does not assert a suspiciously good score."""
    metrics = compute_recognition_metrics(GROUND_TRUTH_TEXT, SATRN_REAL_OUTPUT)
    assert metrics.character_error_rate_normalized == pytest.approx(0.7931034482758621)
    assert metrics.exact_match_normalized is False
    assert metrics.word_error_rate_normalized == 1.0  # not one reference word survived intact


def test_florence2_real_output_scores_better_than_satrn_but_still_far_from_exact():
    """Florence-2's raw decoder output happens to preserve more of the reference's word count/
    shape than SATRN's on this real fixture (both are real, both are wrong) -- this test records
    that real, measured relative difference rather than assuming one adapter must beat the other.
    """
    florence_metrics = compute_recognition_metrics(GROUND_TRUTH_TEXT, FLORENCE2_REAL_PARSED_OUTPUT)
    satrn_metrics = compute_recognition_metrics(GROUND_TRUTH_TEXT, SATRN_REAL_OUTPUT)
    assert florence_metrics.character_error_rate_normalized < satrn_metrics.character_error_rate_normalized
    assert florence_metrics.exact_match_normalized is False


def test_florence2_raw_decoder_output_has_much_worse_cer_than_its_own_parsed_text():
    """Demonstrates *why* raw and normalized/parsed CER must never overwrite each other: scoring
    Florence-2's genuinely raw output (special tokens intact) against ground truth produces a
    materially different, worse number than scoring its parsed text -- collapsing the two into one
    stored value would misrepresent which stage was actually measured."""
    raw_metrics = compute_recognition_metrics(GROUND_TRUTH_TEXT, FLORENCE2_REAL_RAW_OUTPUT)
    parsed_metrics = compute_recognition_metrics(GROUND_TRUTH_TEXT, FLORENCE2_REAL_PARSED_OUTPUT)
    assert raw_metrics.character_error_rate_normalized > parsed_metrics.character_error_rate_normalized


# --- Raw vs. normalized are genuinely separate, never one overwriting the other -------------


def test_swedish_nfc_decomposed_vs_precomposed_a_ring_differs_raw_but_matches_normalized():
    """å/ä/ö-specific Unicode NFC test (task brief requirement): a decomposed 'a' + combining ring
    above (U+0061 U+030A) is visually identical to precomposed 'å' (U+00E5) but is a different code
    point sequence -- raw CER must see them as different, normalized CER must not."""
    precomposed = "gård om natten"
    decomposed = "ga" + "̊" + "rd om natten"
    assert precomposed != decomposed  # sanity: genuinely different code point sequences
    metrics = compute_recognition_metrics(precomposed, decomposed)
    assert metrics.exact_match_raw is False
    assert metrics.character_error_rate_raw > 0.0
    assert metrics.exact_match_normalized is True
    assert metrics.character_error_rate_normalized == 0.0


def test_raw_and_normalized_cer_are_stored_as_genuinely_separate_metric_results():
    reference = "Hej  \n  DÄR"  # extra whitespace + decomposable letters, deliberately messy
    hypothesis = "hej där"
    metrics = compute_recognition_metrics(reference, hypothesis)
    # Raw: case differs ("Hej"≠"hej", "DÄR"≠"där") and whitespace differs -- high raw CER.
    assert metrics.character_error_rate_raw > metrics.character_error_rate_normalized
    results = recognition_metric_results(reference, hypothesis, method_run_id="method_run_1")
    by_definition_id = {r.metric_definition_id: r.value for r in results}
    assert len(by_definition_id) == len(results)  # every MetricResult keyed by a distinct definition
    # Two distinct, genuinely separate stored values -- neither overwrote the other.
    raw_value = by_definition_id[definitions.CHARACTER_ERROR_RATE_RAW.metric_definition_id]
    normalized_value = by_definition_id[definitions.CHARACTER_ERROR_RATE_NORMALIZED.metric_definition_id]
    assert raw_value == pytest.approx(metrics.character_error_rate_raw)
    assert normalized_value == pytest.approx(metrics.character_error_rate_normalized)
    assert raw_value != normalized_value


# --- Edit-operation classification: cross-check against the retained Levenshtein primitive ---


def test_char_edit_classification_cross_checks_against_retained_levenshtein_primitive():
    """`evaluation.metrics.levenshtein` only returns a distance -- this asserts the classifying
    variant's total is always exactly that distance, proving it is an extension, not a divergent
    reimplementation."""
    cases = [
        ("", ""),
        ("a", ""),
        ("", "a"),
        ("kitten", "sitting"),
        (GROUND_TRUTH_TEXT, SATRN_REAL_OUTPUT),
        (GROUND_TRUTH_TEXT, FLORENCE2_REAL_PARSED_OUTPUT),
    ]
    for reference, hypothesis in cases:
        counts = classify_char_edits(reference, hypothesis)
        assert counts.total_edits == levenshtein(reference, hypothesis)
        assert counts.matches + counts.substitutions + counts.deletions == len(reference)
        assert counts.matches + counts.substitutions + counts.insertions == len(hypothesis)


# --- Synthetic edit-operation edge cases (labeled synthetic, not real data) ------------------


def test_synthetic_pure_insertion_case():
    counts = classify_char_edits("ab", "abc")
    assert counts == classify_char_edits("ab", "abc")
    assert counts.insertions == 1
    assert counts.deletions == 0
    assert counts.substitutions == 0
    assert counts.matches == 2


def test_synthetic_pure_deletion_case():
    counts = classify_char_edits("abc", "ab")
    assert counts.deletions == 1
    assert counts.insertions == 0
    assert counts.substitutions == 0
    assert counts.matches == 2


def test_synthetic_pure_substitution_case():
    counts = classify_char_edits("abc", "abd")
    assert counts.substitutions == 1
    assert counts.insertions == 0
    assert counts.deletions == 0
    assert counts.matches == 2


def test_synthetic_engineered_one_of_each_char_edit_operation():
    """Synthetic string engineered (with long unique-context padding around each edit so the
    minimum-cost alignment is not ambiguous) to contain exactly one deletion ('X'), one
    substitution ('Y'->'Z'), and one insertion ('W' appended)."""
    reference = "korpX flygYr ravn"
    hypothesis = "korp flygZr ravnW"
    counts = classify_char_edits(reference, hypothesis)
    assert counts.deletions == 1
    assert counts.substitutions == 1
    assert counts.insertions == 1
    assert counts.total_edits == levenshtein(reference, hypothesis) == 3


def test_synthetic_engineered_one_of_each_word_edit_operation():
    reference_words = ["korp", "EXTRA", "flyger", "gammal", "ravn"]
    hypothesis_words = ["korp", "flyger", "substituted", "ravn", "new"]
    counts = classify_word_edits(reference_words, hypothesis_words)
    assert counts.deletions == 1  # "EXTRA"
    assert counts.substitutions == 1  # "gammal" -> "substituted"
    assert counts.insertions == 1  # "new"
    assert counts.matches == 3  # "korp", "flyger", "ravn"


# --- exact_line_accuracy ----------------------------------------------------------------------


def test_exact_line_accuracy_aligned_lines():
    reference_lines = ["Anno 1712 den 3 Januarii", "medh allmogen aff Sochnen", "NB dombook"]
    hypothesis_lines = ["Anno 1712 den 3 Januarii", "medh allmogen av Sochnen", "NB dombook"]
    # Real PAGE XML fixture lines (tests/fixtures/transkribus/sample_page.xml) -- line 2 has a
    # deliberate one-word difference ("aff" -> "av") to exercise a genuine partial mismatch.
    assert exact_line_accuracy(reference_lines, hypothesis_lines) == pytest.approx(2 / 3)


def test_exact_line_accuracy_missing_hypothesis_line_counts_as_a_miss_not_a_truncated_comparison():
    reference_lines = ["one", "two", "three"]
    hypothesis_lines = ["one", "two"]
    assert exact_line_accuracy(reference_lines, hypothesis_lines) == pytest.approx(2 / 3)


def test_exact_line_accuracy_both_empty_is_trivially_perfect():
    assert exact_line_accuracy([], []) == 1.0


# --- exact_word_accuracy differs from 1 - WER whenever insertions are present ----------------


def test_exact_word_accuracy_diverges_from_one_minus_wer_when_hypothesis_has_insertions():
    reference = "en katt sitter"
    hypothesis = "en katt sitter dar helt still"  # three inserted words
    metrics = compute_recognition_metrics(reference, hypothesis)
    assert metrics.exact_word_accuracy == 1.0  # every reference word was matched
    assert metrics.word_error_rate_normalized > 0.0  # WER penalizes the insertions
    assert metrics.exact_word_accuracy != pytest.approx(1.0 - metrics.word_error_rate_normalized)
