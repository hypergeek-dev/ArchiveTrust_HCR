"""Pattern discovery over a live `TelemetryIndex` (Milestone 11, Phase 4).

Split out of `domain/calibration/pattern.py` because discovering patterns needs
`review.triage.review_reason_for` (to know which slots are queued under the current
`TriagePolicy`) and `learning.analytics.index.TelemetryIndex` -- both upper-layer concerns the
domain layer must never import (`tests/domain/test_dependency_direction.py`,
`tests/review/test_separation.py`). The pure `CalibrationPattern` shape itself still lives in
`domain/calibration/pattern.py`; only the "how do we find them" logic lives here, generalized
from `scripts/confidence_calibration_sufficiency.py`'s original pattern-discovery loop.
"""

from __future__ import annotations

from collections.abc import Callable

from archivetrust.domain.calibration.pattern import CalibrationPattern, classification_type_key
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.review.triage import TriagePolicy, review_reason_for


def by_classification_and_type(index: TelemetryIndex) -> tuple[CalibrationPattern, ...]:
    """The primary pattern grain: `(classification, observation_type)`, exactly what
    `review_reason_for` routes on."""
    return discover_patterns(index, key=classification_type_key)


def by_provider_participation_count(index: TelemetryIndex) -> tuple[CalibrationPattern, ...]:
    """Corpus-wide cut by how many independent providers contributed to a slot."""

    def key(c) -> str:
        n = len(index.contributing_providers(c))
        return {1: "single-provider", 2: "two-provider", 3: "three-provider"}.get(
            n, f"{n}-provider"
        )

    return discover_patterns(index, key=key)


def by_single_contributing_provider(index: TelemetryIndex) -> tuple[CalibrationPattern, ...]:
    """Single-source slots broken out by which one provider contributed them. Slots with more
    than one contributing provider are excluded (their pattern is provider-participation-count,
    not provider identity)."""

    def key(c):
        providers = index.contributing_providers(c)
        if len(providers) != 1:
            return None
        return f"single-source/{next(iter(providers)).provider_id}"

    return discover_patterns(index, key=key)


def discover_patterns(
    index: TelemetryIndex,
    *,
    key: Callable[..., str | None],
    triage_policy: TriagePolicy | None = None,
) -> tuple[CalibrationPattern, ...]:
    """Group the corpus's latest canonical slots by `key`, discovering patterns directly from
    measured data rather than a hardcoded list. `key` returning `None` excludes a slot from every
    pattern (e.g. multi-provider slots under the single-provider-identity grouping).
    """
    policy = triage_policy or TriagePolicy()
    counts: dict[str, dict[str, int]] = {}
    for canonical in index.latest_canonicals():
        name = key(canonical)
        if name is None:
            continue
        row = counts.setdefault(name, {"n": 0, "queued": 0})
        row["n"] += 1
        if review_reason_for(canonical, policy) is not None:
            row["queued"] += 1
    return tuple(
        CalibrationPattern(name=name, population=row["n"], queued_for_review=row["queued"])
        for name, row in sorted(counts.items(), key=lambda kv: -kv[1]["n"])
    )
