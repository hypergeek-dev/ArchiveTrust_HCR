"""Tests for the reference-free reliability heuristics.

Two properties are load-bearing and are asserted directly rather than assumed:

* **no accuracy signal exists in the module** -- checked by inspecting the public surface, so an
  added CER/WER helper would fail the suite rather than pass unnoticed;
* **failure and empty output are never conflated** -- `text=None` (the adapter failed) and
  `text=""` (the adapter succeeded and returned nothing) are different facts and different flags.
"""

from __future__ import annotations

import pytest

from archivetrust.htr.screening import (
    HEURISTIC_CATALOG,
    RELIABILITY_HEURISTICS_VERSION,
    ReliabilityThresholds,
    evaluate_output,
    length_outlier_bounds,
    percentile,
    repeated_output_summary,
    timing_summary,
)
from archivetrust.htr.screening import reliability_signals


def test_no_accuracy_signal_is_exposed() -> None:
    """The corpus has no ground truth; an accuracy helper here would be a category error."""
    forbidden = ("cer", "wer", "accuracy", "ground_truth", "edit_distance", "levenshtein")
    surface = [name.lower() for name in dir(reliability_signals) if not name.startswith("_")]
    for name in surface:
        assert not any(token in name for token in forbidden), (
            f"{name!r} looks like a reference-comparing metric; this module is reference-free "
            "by construction"
        )


def test_heuristic_catalog_documents_every_threshold_it_names() -> None:
    thresholds = ReliabilityThresholds()
    assert HEURISTIC_CATALOG, "the catalog must not be empty"
    for definition in HEURISTIC_CATALOG:
        assert definition.does_not_mean, definition.heuristic_id
        assert definition.scope in {"output", "crop", "run"}
        for field in definition.threshold_fields:
            assert hasattr(thresholds, field), (
                f"{definition.heuristic_id} names threshold {field!r} which does not exist"
            )


def test_thresholds_are_frozen_and_hashable() -> None:
    thresholds = ReliabilityThresholds()
    with pytest.raises(Exception):
        thresholds.very_short_output_max_length = 99  # type: ignore[misc]
    assert thresholds.configuration_hash.startswith("reliability_thresholds_")
    assert (
        ReliabilityThresholds(very_short_output_max_length=5).configuration_hash
        != thresholds.configuration_hash
    )


def test_version_is_declared() -> None:
    assert RELIABILITY_HEURISTICS_VERSION


def test_failure_is_not_conflated_with_empty_output() -> None:
    failed = evaluate_output(
        None, method="satrn", raw_response={"category": "timeout", "message": "worker timed out"}
    )
    assert failed.produced_output is False
    assert failed.failure_category == "timeout"
    assert failed.empty_or_whitespace_only is False

    empty = evaluate_output("   ", method="satrn")
    assert empty.produced_output is True
    assert empty.empty_or_whitespace_only is True
    assert empty.failure_category is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("aaaaaaaaaaaaaaaa", True),
        ("abababababababab", True),
        ("abcabcabcabcabcabc", True),
        ("ababab", False),  # only 3 repeats, below degenerate_min_ngram_repeats
        ("aa", False),  # below degenerate_min_length
        ("Till förnämnde Aspring", False),
    ],
)
def test_degenerate_repetition(text: str, expected: bool) -> None:
    assert evaluate_output(text, method="florence2").degenerate_repetition is expected


def test_truncation_uses_the_per_method_ceiling() -> None:
    long_text = "x" * 95
    assert evaluate_output(long_text, method="satrn").truncation_suspected is True
    # The same string is nowhere near Florence-2's much higher ceiling.
    assert evaluate_output(long_text, method="florence2").truncation_suspected is False
    # An adapter that reports its own truncation is believed regardless of length.
    assert (
        evaluate_output("short", method="florence2", raw_response={"output_truncated": True})
        .truncation_suspected
        is True
    )


def test_very_short_output_flag() -> None:
    assert evaluate_output("63", method="satrn").very_short_output is True
    assert evaluate_output("63.", method="satrn").very_short_output is False


def test_repeated_output_requires_different_inputs() -> None:
    """The same output twice for the *same* crop hash is not a repeat across inputs."""
    same_input = repeated_output_summary([("crop_a", "hello"), ("crop_a", "hello")])
    assert same_input.repeat_group_count == 0
    assert same_input.distinct_inputs == 1

    different_inputs = repeated_output_summary([("crop_a", "hello"), ("crop_b", "hello")])
    assert different_inputs.repeat_group_count == 1
    assert different_inputs.largest_repeat_group_size == 2
    assert different_inputs.most_repeated_output == "hello"


def test_repeated_output_ratio_and_flag() -> None:
    observations = [(f"crop_{i}", "staden den 27 dennes") for i in range(6)]
    observations += [(f"crop_x{i}", f"distinct line {i}") for i in range(9)]
    summary = repeated_output_summary(observations)
    assert summary.distinct_inputs == 15
    assert summary.distinct_outputs == 10
    assert summary.distinct_ratio == pytest.approx(10 / 15, abs=1e-4)
    assert summary.outputs_in_repeat_groups == 6
    assert summary.flagged is True

    clean = repeated_output_summary([(f"crop_{i}", f"line {i}") for i in range(15)])
    assert clean.distinct_ratio == 1.0
    assert clean.flagged is False


def test_repeated_output_collapses_whitespace() -> None:
    summary = repeated_output_summary([("a", "the  line"), ("b", "the line ")])
    assert summary.repeat_group_count == 1


def test_length_outlier_bounds_refuses_small_samples() -> None:
    result = length_outlier_bounds([10, 12, 14])
    assert result["assessable"] is False
    assert "at least" in str(result["reason"])


def test_length_outlier_bounds_refuses_zero_mad() -> None:
    result = length_outlier_bounds([20] * 30)
    assert result["assessable"] is False
    assert "zero" in str(result["reason"])


def test_length_outlier_bounds_identifies_an_outlier() -> None:
    lengths = [30 + (i % 5) for i in range(40)] + [500]
    bounds = length_outlier_bounds(lengths)
    assert bounds["assessable"] is True
    assert 500 > float(bounds["upper_bound"])  # type: ignore[arg-type]
    assert 32 <= float(bounds["upper_bound"])  # type: ignore[arg-type]


def test_percentile_and_timing_summary() -> None:
    assert percentile([], 0.5) is None
    assert percentile([1.0], 0.9) == 1.0
    assert percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)

    summary = timing_summary([1.0, 2.0, 3.0, 100.0])
    assert summary["count"] == 4
    assert summary["min"] == 1.0
    assert summary["max"] == 100.0
    assert summary["median"] == pytest.approx(2.5)
    assert float(summary["p90"]) > 3.0  # type: ignore[arg-type]

    empty = timing_summary([])
    assert empty["count"] == 0
    assert empty["median"] is None
