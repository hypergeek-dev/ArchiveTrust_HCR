from __future__ import annotations

import difflib

import pytest

from archivetrust.htr.training.full_run.evaluation_metrics import (
    character_error_rate,
    confidence_buckets,
    corpus_metrics,
    levenshtein,
    rejection_coverage,
    word_error_rate,
)


@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0), ("abc", "abc", 0),
    ("abc", "abd", 1),        # substitution
    ("abc", "ab", 1),         # deletion
    ("ab", "abc", 1),         # insertion
    ("kitten", "sitting", 3), # the canonical example
    ("", "abc", 3), ("abc", "", 3),
])
def test_levenshtein_matches_known_values(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric():
    assert levenshtein("Koor a 5 mk", "koor 3 if.") == levenshtein("koor 3 if.", "Koor a 5 mk")


def test_difflib_is_not_a_substitute_for_levenshtein():
    """The real error from the first shard review: difflib finds matching blocks, never a minimum
    edit script, so it cannot account for substitutions and overstates distance. This test pins the
    discrepancy so nobody reintroduces the shortcut."""
    ref, hyp = "Suin a a mk", "3uin 1/ rp"
    true_distance = levenshtein(ref, hyp)
    sm = difflib.SequenceMatcher(None, ref, hyp)
    difflib_distance = len(ref) + len(hyp) - 2 * sum(b.size for b in sm.get_matching_blocks())
    assert difflib_distance > true_distance, "difflib should overestimate here"
    assert character_error_rate(ref, hyp) == true_distance / len(ref)


def test_cer_on_empty_reference_is_defined():
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "spurious") == 1.0


def test_wer_counts_word_level_edits():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0
    assert word_error_rate("the quick brown fox", "the slow brown fox") == pytest.approx(0.25)


def test_wer_handles_repeated_words_without_collapsing_them():
    """A naive set/dict mapping could merge duplicates and undercount."""
    assert word_error_rate("a a a", "a a a") == 0.0
    assert word_error_rate("a a a", "a b a") == pytest.approx(1 / 3)


def test_corpus_cer_is_edit_weighted_not_a_mean_of_per_line_rates():
    """One long accurate line plus one short bad line: the corpus figure must be dominated by the
    long line's characters, while the per-line mean is not."""
    pairs = [("a" * 100, "a" * 100), ("xy", "qq")]
    m = corpus_metrics(pairs)
    assert m.total_reference_chars == 102
    assert m.total_edits == 2
    assert m.corpus_cer == pytest.approx(2 / 102)
    assert m.mean_per_line_cer == pytest.approx((0.0 + 1.0) / 2)
    assert m.corpus_cer < m.mean_per_line_cer


def test_corpus_metrics_on_empty_input_is_safe():
    m = corpus_metrics([])
    assert m.sample_count == 0 and m.corpus_cer == 0.0


def test_confidence_buckets_report_counts_means_and_medians():
    records = [(0.95, 0.0), (0.85, 0.1), (0.5, 0.3), (0.1, 0.9)]
    buckets = confidence_buckets(records)
    high = [b for b in buckets if b.lower == 0.8][0]
    assert high.sample_count == 2
    assert high.mean_cer == pytest.approx(0.05)
    assert high.median_cer == pytest.approx(0.05)


def test_empty_confidence_buckets_are_reported_not_dropped():
    """A coverage gap is calibration evidence in its own right."""
    buckets = confidence_buckets([(0.95, 0.0)])
    assert len(buckets) == 5
    empty = [b for b in buckets if b.sample_count == 0]
    assert empty and all(b.mean_cer is None and b.median_cer is None for b in empty)


def test_rejection_coverage_reports_accepted_fraction_and_its_error_rate():
    records = [(0.9, 0.05), (0.8, 0.10), (0.4, 0.50), (0.2, 0.80)]
    rows = rejection_coverage(records, thresholds=(0.0, 0.5, 0.85))
    by_t = {r["threshold"]: r for r in rows}
    assert by_t[0.0]["coverage"] == 1.0
    assert by_t[0.5]["accepted_count"] == 2
    assert by_t[0.5]["mean_cer_of_accepted"] == pytest.approx(0.075)
    assert by_t[0.85]["accepted_count"] == 1
    # Accepting only high-confidence lines should lower the error rate of what is accepted.
    assert by_t[0.85]["mean_cer_of_accepted"] < by_t[0.0]["mean_cer_of_accepted"]
