"""Sample-size and interval statistics for confidence calibration (Milestone 11, Phase 5).

Promoted out of `scripts/confidence_calibration_sufficiency.py` (2026-07-14) into real, tested,
importable domain code -- the script now imports from here instead of carrying its own copy of
these formulas, so there is exactly one implementation to trust.

`required_sample_size` takes an explicit `p`, deliberately -- callers decide whether the
worst-case `p=0.5` assumption (no evidence yet) or an empirical proportion (evidence exists)
applies; this module has no opinion and performs no I/O. Milestone 11 Phase 5's rule -- "never
continue using worst-case assumptions after sufficient evidence exists" -- lives in
`domain/calibration/progress.py`, which is the one place that decides which `p` to pass here.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054
"""Exact two-sided 95% critical value -- not rounded to 1.96 casually."""


def required_sample_size(margin_of_error: float, population: int, *, p: float = 0.5) -> int:
    """Standard sample-size formula for estimating a population proportion within
    `margin_of_error` at 95% confidence (`n0 = z^2 * p * (1-p) / E^2`), finite-population corrected
    (`n = n0 / (1 + (n0-1)/N)`) against a known, bounded `population`.

    `p=0.5` is the maximum-variance assumption -- the standard, defensible default when no prior
    estimate of the true proportion exists, since it guarantees the requested margin of error
    regardless of what the true proportion turns out to be. Once a pattern has real evidence, pass
    its empirical proportion instead -- required-n shrinks accordingly (Phase 5).
    """
    if population <= 0:
        return 0
    n0 = (Z_95**2) * p * (1 - p) / (margin_of_error**2)
    if n0 <= 0:
        return 0
    n_fpc = n0 / (1 + (n0 - 1) / population)
    return math.ceil(n_fpc)


def wilson_ci(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval -- more reliable than the naive Wald interval at small n or extreme
    proportions (both are common in this dataset). Returns `None` for `n=0` (undefined, not a
    fabricated `(0, 0)` or `(0, 1)`)."""
    if n == 0:
        return None
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    spread = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    return ((center - spread) / denom, (center + spread) / denom)
