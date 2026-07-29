"""Replaceable sampling strategies (Milestone 11, Phase 2).

Every strategy is a pure function over already-persisted telemetry (a `TelemetryIndex`) -- like
`review/triage.py`, it never invokes a provider (LP-2, read-only) and is deterministic given its
inputs and (where randomness is involved) an explicit seed, so a given corpus + policy always
produces the same sample (reproducibility is a hard requirement of this milestone, not a nicety).

Strategies are registered by name in `STRATEGY_REGISTRY` so callers can select one by
configuration rather than the code hardcoding a single approach (spec: "Do not hardcode one
strategy. Sampling strategies should be replaceable.").
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.calibration.information_gain import expected_information_gain
from archivetrust.domain.calibration.pattern import classification_type_key
from archivetrust.domain.calibration.progress import CalibrationProgress
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.review.sampling.intent import ReviewIntent


class SampledItem(BaseModel):
    """One slot a sampling strategy selected, with an explanation a reviewer can be shown
    verbatim (Phase 6: "Reviewers should always know why an item appears in the queue.")."""

    model_config = ConfigDict(frozen=True)

    semantic_slot_id: str
    canonical_observation_id: str
    document_ref: str
    observation_type: ObservationType
    pattern: str
    intent: ReviewIntent
    strategy_name: str
    explanation: str
    value_score: float | None = None


class SamplingStrategy(Protocol):
    name: str

    def select(
        self,
        candidates: Sequence[CanonicalObservation],
        *,
        document_ref_by_slot: Mapping[str, str],
        progress_by_pattern: Mapping[str, CalibrationProgress],
        corpus_size: int,
        k: int,
        intent: ReviewIntent,
        seed: int | None = None,
    ) -> tuple[SampledItem, ...]:
        """Select up to `k` items from `candidates` (typically a pattern's or the corpus's latest
        canonical slots). `progress_by_pattern` and `corpus_size` are supplied so a strategy can
        weight by calibration state or corpus share without recomputing either itself."""
        ...


def _item(
    canonical: CanonicalObservation,
    *,
    document_ref_by_slot: Mapping[str, str],
    strategy_name: str,
    intent: ReviewIntent,
    explanation: str,
    value_score: float | None = None,
) -> SampledItem | None:
    document_ref = document_ref_by_slot.get(canonical.semantic_slot_id)
    if document_ref is None:
        return None
    return SampledItem(
        semantic_slot_id=canonical.semantic_slot_id,
        canonical_observation_id=canonical.canonical_observation_id,
        document_ref=document_ref,
        observation_type=canonical.observation_type,
        pattern=classification_type_key(canonical),
        intent=intent,
        strategy_name=strategy_name,
        explanation=explanation,
        value_score=value_score,
    )


class RandomSamplingStrategy:
    """Uniform random sampling, seeded for reproducibility. The simplest unbiased estimator of a
    pattern's true correctness -- the right default when no calibration state should influence
    selection."""

    name = "random_sampling"

    def select(
        self,
        candidates: Sequence[CanonicalObservation],
        *,
        document_ref_by_slot: Mapping[str, str],
        progress_by_pattern: Mapping[str, CalibrationProgress],
        corpus_size: int,
        k: int,
        intent: ReviewIntent,
        seed: int | None = None,
    ) -> tuple[SampledItem, ...]:
        rng = random.Random(seed)
        chosen = rng.sample(list(candidates), k=min(k, len(candidates)))
        items = (
            _item(
                c,
                document_ref_by_slot=document_ref_by_slot,
                strategy_name=self.name,
                intent=intent,
                explanation=f"Randomly sampled (seed={seed}) for {intent.value} review -- an "
                "unbiased estimate of this pattern's true correctness rate.",
            )
            for c in chosen
        )
        return tuple(item for item in items if item is not None)


class StratifiedSamplingStrategy:
    """Allocates `k` proportionally to each pattern's population share, then samples randomly
    within each stratum -- guarantees every pattern gets sampled in proportion to its size,
    including patterns the operational queue never reaches."""

    name = "stratified_sampling"

    def select(
        self,
        candidates: Sequence[CanonicalObservation],
        *,
        document_ref_by_slot: Mapping[str, str],
        progress_by_pattern: Mapping[str, CalibrationProgress],
        corpus_size: int,
        k: int,
        intent: ReviewIntent,
        seed: int | None = None,
    ) -> tuple[SampledItem, ...]:
        rng = random.Random(seed)
        by_pattern: dict[str, list[CanonicalObservation]] = {}
        for c in candidates:
            by_pattern.setdefault(classification_type_key(c), []).append(c)

        total = len(candidates)
        items: list[SampledItem] = []
        for pattern_name, members in by_pattern.items():
            share = len(members) / total if total else 0.0
            stratum_k = round(share * k)
            chosen = rng.sample(members, k=min(stratum_k, len(members)))
            for c in chosen:
                item = _item(
                    c,
                    document_ref_by_slot=document_ref_by_slot,
                    strategy_name=self.name,
                    intent=intent,
                    explanation=f"Stratified sample from pattern '{pattern_name}' "
                    f"({share:.1%} of corpus) for {intent.value} review -- ensures this pattern "
                    "is represented in proportion to its size, whether or not the operational "
                    "queue ever reaches it.",
                )
                if item is not None:
                    items.append(item)
        return tuple(items)


class WeightedSamplingStrategy:
    """Weights each pattern's items by expected information gain (`domain/calibration/
    information_gain.py`) -- spends review budget where it moves calibration the most, not evenly
    or randomly."""

    name = "weighted_sampling"

    def select(
        self,
        candidates: Sequence[CanonicalObservation],
        *,
        document_ref_by_slot: Mapping[str, str],
        progress_by_pattern: Mapping[str, CalibrationProgress],
        corpus_size: int,
        k: int,
        intent: ReviewIntent,
        seed: int | None = None,
    ) -> tuple[SampledItem, ...]:
        rng = random.Random(seed)
        weights: list[float] = []
        pool: list[CanonicalObservation] = []
        gains: dict[str, float] = {}
        for c in candidates:
            pattern_name = classification_type_key(c)
            progress = progress_by_pattern.get(pattern_name)
            if progress is None:
                continue
            if pattern_name not in gains:
                share = progress.population / corpus_size if corpus_size else 0.0
                gains[pattern_name] = expected_information_gain(progress, corpus_share=share)
            weight = gains[pattern_name]
            if weight <= 0:
                continue
            pool.append(c)
            weights.append(weight)

        if not pool:
            return ()
        n = min(k, len(pool))
        chosen_idx = _weighted_sample_without_replacement(rng, weights, n)
        items: list[SampledItem] = []
        for i in chosen_idx:
            c = pool[i]
            pattern_name = classification_type_key(c)
            item = _item(
                c,
                document_ref_by_slot=document_ref_by_slot,
                strategy_name=self.name,
                intent=intent,
                explanation=f"Selected from pattern '{pattern_name}' by expected information "
                f"gain ({gains[pattern_name]:.3g}) for {intent.value} review -- the pattern where "
                "one more review reduces calibration uncertainty the most per unit of sampling "
                "cost.",
                value_score=gains[pattern_name],
            )
            if item is not None:
                items.append(item)
        return tuple(items)


class UncertaintySamplingStrategy:
    """Prioritizes patterns furthest from their statistically required sample size (lowest
    `progress_pct_of_loosest_requirement`) -- closes the biggest calibration gaps first."""

    name = "uncertainty_sampling"

    def select(
        self,
        candidates: Sequence[CanonicalObservation],
        *,
        document_ref_by_slot: Mapping[str, str],
        progress_by_pattern: Mapping[str, CalibrationProgress],
        corpus_size: int,
        k: int,
        intent: ReviewIntent,
        seed: int | None = None,
    ) -> tuple[SampledItem, ...]:
        rng = random.Random(seed)

        def _urgency(c: CanonicalObservation) -> float:
            progress = progress_by_pattern.get(classification_type_key(c))
            if progress is None:
                return 0.0
            pct = progress.progress_pct_of_loosest_requirement
            return 100.0 - pct if pct is not None else 100.0

        ranked = sorted(candidates, key=_urgency, reverse=True)
        # Break exact ties randomly rather than by corpus iteration order.
        top_urgency = _urgency(ranked[0]) if ranked else 0.0
        tied = [c for c in ranked if _urgency(c) == top_urgency]
        rng.shuffle(tied)
        ordered = tied + [c for c in ranked if _urgency(c) != top_urgency]

        items: list[SampledItem] = []
        for c in ordered[:k]:
            pattern_name = classification_type_key(c)
            progress = progress_by_pattern.get(pattern_name)
            pct = progress.progress_pct_of_loosest_requirement if progress else None
            item = _item(
                c,
                document_ref_by_slot=document_ref_by_slot,
                strategy_name=self.name,
                intent=intent,
                explanation=f"Pattern '{pattern_name}' is furthest from its required calibration "
                f"sample size ({pct if pct is not None else 0}% of the loosest (+/-10%) "
                f"requirement met) -- selected for {intent.value} review to close the largest "
                "remaining calibration gap.",
                value_score=(100.0 - pct) if pct is not None else None,
            )
            if item is not None:
                items.append(item)
        return tuple(items)


class RarePatternSamplingStrategy:
    """Weights inversely by pattern population -- deliberately over-samples rare patterns
    (rare providers, rare observation types) that a proportional strategy would rarely touch,
    directly targeting coverage blind spots (Phase 7)."""

    name = "rare_pattern_sampling"

    def select(
        self,
        candidates: Sequence[CanonicalObservation],
        *,
        document_ref_by_slot: Mapping[str, str],
        progress_by_pattern: Mapping[str, CalibrationProgress],
        corpus_size: int,
        k: int,
        intent: ReviewIntent,
        seed: int | None = None,
    ) -> tuple[SampledItem, ...]:
        rng = random.Random(seed)
        pattern_population: dict[str, int] = {}
        for c in candidates:
            name = classification_type_key(c)
            pattern_population[name] = pattern_population.get(name, 0) + 1

        weights = [1.0 / pattern_population[classification_type_key(c)] for c in candidates]
        pool = list(candidates)
        if not pool:
            return ()
        n = min(k, len(pool))
        chosen_idx = _weighted_sample_without_replacement(rng, weights, n)
        items: list[SampledItem] = []
        for i in chosen_idx:
            c = pool[i]
            pattern_name = classification_type_key(c)
            item = _item(
                c,
                document_ref_by_slot=document_ref_by_slot,
                strategy_name=self.name,
                intent=intent,
                explanation=f"Pattern '{pattern_name}' has only {pattern_population[pattern_name]} "
                f"slot(s) in the corpus -- weighted inversely by population for {intent.value} "
                "review so rare patterns are not permanently starved of evidence.",
            )
            if item is not None:
                items.append(item)
        return tuple(items)


def _weighted_sample_without_replacement(
    rng: random.Random, weights: Sequence[float], k: int
) -> list[int]:
    """Efraimidis-Spirakis weighted sampling without replacement: each item gets key
    `u ** (1/weight)` for `u ~ Uniform(0,1)`, and the top-`k` keys are selected. Deterministic
    given `rng`'s seed."""
    keyed = [(rng.random() ** (1.0 / w), i) for i, w in enumerate(weights)]
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    return [i for _, i in keyed[:k]]


STRATEGY_REGISTRY: dict[str, SamplingStrategy] = {
    s.name: s
    for s in (
        RandomSamplingStrategy(),
        StratifiedSamplingStrategy(),
        WeightedSamplingStrategy(),
        UncertaintySamplingStrategy(),
        RarePatternSamplingStrategy(),
    )
}


def strategy_by_name(name: str) -> SamplingStrategy:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown sampling strategy '{name}'; registered: {sorted(STRATEGY_REGISTRY)}"
        ) from None
