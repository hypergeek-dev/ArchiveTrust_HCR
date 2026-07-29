from __future__ import annotations

from archivetrust.domain.calibration.statistics import required_sample_size, wilson_ci


def test_required_sample_size_at_p_half_matches_known_benchmark_figure():
    # Benchmark #1's own contested/paragraph pattern: N=4866, margin +/-10% -> 95.
    assert required_sample_size(0.10, 4866, p=0.5) == 95


def test_required_sample_size_shrinks_as_p_moves_away_from_worst_case():
    population = 10_000
    worst_case = required_sample_size(0.05, population, p=0.5)
    for p in (0.4, 0.3, 0.1, 0.9, 0.99):
        assert required_sample_size(0.05, population, p=p) <= worst_case


def test_required_sample_size_is_symmetric_in_p():
    population = 5000
    assert required_sample_size(0.05, population, p=0.2) == required_sample_size(
        0.05, population, p=0.8
    )


def test_required_sample_size_zero_population_is_zero():
    assert required_sample_size(0.1, 0, p=0.5) == 0
    assert required_sample_size(0.1, -5, p=0.5) == 0


def test_wilson_ci_undefined_at_zero_reviews():
    assert wilson_ci(0, 0) is None


def test_wilson_ci_narrows_as_n_grows():
    narrow = wilson_ci(950, 1000)
    wide = wilson_ci(9, 10)
    assert narrow is not None and wide is not None
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_ci_brackets_observed_proportion():
    ci = wilson_ci(80, 100)
    assert ci is not None
    lo, hi = ci
    assert lo < 0.8 < hi
