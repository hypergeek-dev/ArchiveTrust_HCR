"""The Reconciliation Policy (MILESTONE4_COMPARISON_ENGINE.md S1, C10).

A single versioned object holding every threshold, weight, and tie-break rule the engine uses.
C10 ("no hidden heuristics") is enforced by convention here: any constant a comparison function
needs must be a named field on this type, sourced from the Policy instance it is given, never a
literal buried in code -- a literal threshold in comparison code is, by definition, a hidden
decision and violates C10.

**Default values below are placeholders, not empirically validated.** Per S19, table-lattice
tolerance and coarse-geometry clustering behavior are explicitly gated behind corpus validation
before being frozen -- these defaults make the engine runnable and testable now, not authoritative
tuning.

**Determinism note (S9, item 6 of S17's adversarial review):** the spec calls for fixed-precision
decimal arithmetic to guarantee byte-stable replay across platforms. This implementation instead
uses Python `float` with scores rounded via `round(value, rounding_ndigits)` at every score
boundary -- adequate for Python-only replay (the only replay path this milestone builds; the
journal is a Python object store, not a cross-language wire format) but not a full guarantee
against exotic cross-platform floating-point drift. Recorded here as a known simplification, not
silently claimed as the spec's full guarantee.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from archivetrust.domain.comparison.capability_matrix import Capability


class ReconciliationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_version: int

    # Phase B -- clustering (S2).
    cluster_join_threshold: float = 0.3
    """Minimum geometric affinity (IoU for pixel-accurate pairs; containment/ordinal score for
    coarse pairs) for two Observations to join the same cluster."""

    enforce_scope_compatible_alignment: bool = True
    """F5 scope-aware alignment gate. When enabled, Observations must have compatible
    `ObservationScopeClassification` values before geometry/ordinal affinity can merge them.
    This prevents page-scale transcripts from being compared as if they were paragraph or line
    fragments. Disable only for counterfactual replay against the pre-F5 behavior."""

    # Phase E -- structural reconciliation (S4).
    edge_accept_threshold: float = 0.5
    """Minimum capability-weighted support for a candidate parent/child edge to be elected into
    the ReconciledObservationGraph rather than recorded as proposed/uncorroborated."""

    max_rank_aggregation_children: int = 12
    """Policy cap (S4, S7, S14): above this many siblings, reading-order aggregation falls back
    to a deterministic positional-median heuristic instead of the (bounded, but still costlier)
    exact aggregation, and the fallback is recorded when triggered."""

    # Phase D -- text reconciliation (S5).
    aggressive_text_normalization: bool = False
    """Case-folding / diacritic stripping. Off by default -- "for archival documents a diacritic
    *is* content" (S5.1)."""

    text_corroboration_edit_distance_threshold: float = 0.15
    """Normalized edit distance (S5.3) at or below which two text candidates are judged to agree
    strongly enough to count as CORROBORATED rather than CONTESTED."""

    # Phase D -- table reconciliation (S6).
    table_separator_tolerance: float = 4.0
    """1-D clustering tolerance (in bounding-box coordinate units) for merging near-but-distinct
    projected row/column separator positions into one canonical lattice line. The single most
    outcome-determining table Policy value (S6, S18 R4) -- explicitly gated behind corpus
    validation, per S19."""

    # Phase F -- scoring.
    capability_weight: dict[Capability, float] = Field(
        default_factory=lambda: {
            Capability.NATIVE: 1.0,
            Capability.DERIVED: 0.7,
            Capability.PARTIAL: 0.4,
            Capability.NO: 0.0,
        }
    )
    """Weights structural/coverage claims by capability tag, never by provider name (C6)."""

    rounding_ndigits: int = 6
    """Policy-pinned rounding applied to every computed score (S9's "fixed-precision" intent --
    see this module's docstring for the Decimal-vs-float simplification note)."""

    def round_score(self, value: float) -> float:
        return round(value, self.rounding_ndigits)
