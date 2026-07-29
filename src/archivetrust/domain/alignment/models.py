"""Alignment Observability domain model (ROADMAP.md S5.12; Constitution Article 24).

These types record what the grouping mechanism decided; they never make a grouping decision of
their own. Deliberately carry **no** confidence, score, or magnitude — Alignment Confidence is an
explicit non-goal (S5.12). An `AlignmentAttempt` states *what* was grouped and *why* (the rationale
string the algorithm already produced), never *how strongly*.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.comparison.clustering import Cluster, ClusteringBasisCode
from archivetrust.domain.ontology.types import ObservationType

ALIGNMENT_ALGORITHM_VERSION = 1
"""Versioned independently of the Reconciliation Policy and ontology (S5.12): a future alignment
strategy bumps this so replay can identify which strategy produced a grouping, and so strategies can
be compared on the same corpus. This is the *only* alignment version; multiple algorithms are a
future milestone, not built now."""


class AlignmentOutcome(str, Enum):
    """Every Observation ends in exactly one of these — no Observation silently disappears from the
    record (Constitution Article 18, Article 24).
    """

    ALIGNED = "aligned"
    """Grouped with at least one other Observation into a comparison group (the corroboration
    hypothesis: these plausibly refer to the same semantic object)."""

    UNALIGNED = "unaligned"
    """The sole member of its comparison group — no other Observation aligned to it. This is the
    explicit recorded form of "no suitable comparison group exists for this Observation," never a
    silent absence."""


class AlignmentAttempt(BaseModel):
    """One grouping decision, made observable (S5.12). Records what the alignment mechanism
    considered and selected, and why — the durable answer to "why were these Observations grouped
    together?", available from stored telemetry without re-running any provider.

    `comparison_group_id` is identical to the resulting Canonical Observation's `semantic_slot_id`,
    which is what lets a future Human Review signal ("this comparison group is incorrect") target a
    grouping unambiguously.
    """

    model_config = ConfigDict(frozen=True)

    alignment_attempt_id: str
    algorithm_name: str
    algorithm_version: int
    comparison_group_id: str
    observation_type: ObservationType
    page: int | None
    candidate_observation_ids: tuple[str, ...]
    """Every Observation the mechanism *considered* for this group (the pool it was drawn from) —
    distinct from those it *selected*. Recording candidates, not just members, is what makes the
    grouping decision auditable rather than merely stated."""
    selected_observation_ids: tuple[str, ...]
    """The Observations actually grouped together (the group's members)."""
    alignment_rationale: str
    """Why the mechanism grouped these — the rationale it already produced (`clustering_basis`),
    surfaced here, not a new judgment."""
    metadata: dict[str, str] = {}


class ExcludedCandidatePair(BaseModel):
    """One pair of Observations that were pooled as candidates for the same comparison group but
    were structurally excluded from merging by the alignment mechanism itself, independent of this
    document's content (Constitution Article 27, `ARCHITECTURE_TELEMETRY_STANDARD.md` S1.1/Article
    27). Symmetric with `AlignmentAttempt`'s candidate/selected distinction: this is the durable
    record of *why* two considered candidates never became one selection, for the specific
    structural mechanisms Article 27 covers -- not every pair with merely low content-based
    affinity, which is an ordinary scored outcome, not an exclusion.
    """

    model_config = ConfigDict(frozen=True)

    candidate_observation_id: str
    compared_against_observation_id: str
    excluding_mechanism: str
    """Which mechanism excluded this pair, e.g. `"clustering.ambiguous_multi_per_provider_guard"` --
    named after the code path in `domain/comparison/clustering.py`, not a generic label."""
    basis_code: ClusteringBasisCode
    """A stable, enumerable reason code (Constitution Article 26) -- always
    `ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER` today, the only structural exclusion this
    module's vocabulary currently defines."""
    structural: bool
    """Whether this exclusion is a structural property of the candidate pair/mechanism (true for
    every case this model currently represents) rather than incidental to this one document."""


class AlignmentObservationState(BaseModel):
    """One Observation's terminal alignment state — the per-Observation guarantee that nothing is
    dropped (Article 18, Article 24).
    """

    model_config = ConfigDict(frozen=True)

    observation_id: str
    outcome: AlignmentOutcome
    comparison_group_id: str
    """The group it landed in (its own singleton group when UNALIGNED)."""
    alignment_attempt_id: str


class AlignmentResult(BaseModel):
    """The complete, observable outcome of one alignment run over a set of Observations.

    Carries both the authoritative grouping the Comparison Engine consumes (`clusters`, unchanged
    from what clustering produced) and the observability record built from it (`attempts`,
    `observation_states`). The engine reads `clusters`; the telemetry bridge reads the rest.
    """

    model_config = ConfigDict(frozen=True)

    algorithm_name: str
    algorithm_version: int
    clusters: tuple[Cluster, ...]
    attempts: tuple[AlignmentAttempt, ...]
    observation_states: tuple[AlignmentObservationState, ...]
    excluded_pairs: tuple[ExcludedCandidatePair, ...] = ()
    """Structural exclusions Article 27 requires be recorded (S1.1/Article 27) -- additive, empty
    by default so every pre-existing `AlignmentResult` construction remains valid."""

    def state_of(self, observation_id: str) -> AlignmentObservationState:
        for state in self.observation_states:
            if state.observation_id == observation_id:
                return state
        raise KeyError(f"no recorded alignment state for observation {observation_id!r}")

    def attempt_for_group(self, comparison_group_id: str) -> AlignmentAttempt:
        for attempt in self.attempts:
            if attempt.comparison_group_id == comparison_group_id:
                return attempt
        raise KeyError(f"no alignment attempt for comparison group {comparison_group_id!r}")

    @property
    def unaligned_observation_ids(self) -> tuple[str, ...]:
        return tuple(
            s.observation_id
            for s in self.observation_states
            if s.outcome == AlignmentOutcome.UNALIGNED
        )
