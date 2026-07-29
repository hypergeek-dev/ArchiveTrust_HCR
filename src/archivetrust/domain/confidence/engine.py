"""The Confidence Engine (ROADMAP.md Milestone 5, S5.3, S5.3.1; Constitution Article 13).

Produces `CanonicalConfidence` per Canonical Observation, derived from Comparison Confidence and
*referenced, unblended* Provider Confidence -- completing what Milestone 4 deliberately leaves
`None` (`MILESTONE1_DOMAIN_MODEL.md` S1.3).

**Why no code path here can blend Provider Confidence across providers (C8, S5.3.1), structurally,
not just by convention:** `compute_canonical_confidence` only ever reads Provider Confidence in
the `UNCORROBORATED_SINGLE_SOURCE` branch, where by definition exactly one provider contributed --
there is only ever one Provider Confidence value in scope at the point it is used, never two to
combine. In the `CORROBORATED`/`CONTESTED` branches, the *only* input is
`comparison_confidence.magnitude` (already a legitimate cross-provider signal, per S5.3.1's own
instruction to rely on Comparison Confidence for exactly this). No branch sums, averages, or
otherwise compares two different providers' Provider Confidence values.
"""

from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.models import CanonicalConfidence, ComparisonClassification
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.graph.reconciled_graph import ReconciledObservationGraph
from archivetrust.domain.ontology.base import Observation


def compute_canonical_confidence(
    canonical_observation: CanonicalObservation,
    contributing_observations: tuple[Observation, ...],
    policy: ConfidencePolicy,
) -> CanonicalConfidence:
    comparison_confidence = canonical_observation.comparison_confidence
    classification = comparison_confidence.classification

    if classification == ComparisonClassification.CORROBORATED:
        magnitude = comparison_confidence.magnitude or 0.0
        value = policy.round_score(magnitude)
        derivation = (
            f"CORROBORATED: Canonical Confidence taken directly from Comparison Confidence "
            f"magnitude {magnitude} (Confidence Policy v{policy.confidence_policy_version}). No "
            "Provider Confidence blending -- agreement strength alone is the cross-provider "
            "trust signal (ROADMAP.md S5.3.1)."
        )
        return CanonicalConfidence(value=value, derivation=derivation)

    if classification == ComparisonClassification.CONTESTED:
        magnitude = comparison_confidence.magnitude or 0.0
        value = policy.round_score(magnitude * policy.contested_penalty_factor)
        derivation = (
            f"CONTESTED: Comparison Confidence magnitude {magnitude} penalized by "
            f"contested_penalty_factor={policy.contested_penalty_factor} (Confidence Policy "
            f"v{policy.confidence_policy_version}) -- a high magnitude under CONTESTED means "
            "'strong but conflicting signal,' never 'trustworthy' "
            "(MILESTONE4_COMPARISON_ENGINE.md S9)."
        )
        return CanonicalConfidence(value=value, derivation=derivation)

    # UNCORROBORATED_SINGLE_SOURCE: exactly one provider contributed, so reading its own
    # Provider Confidence here is never a cross-provider blend (S5.3.1: "Provider Confidence may
    # be used unconditionally only within one provider's own values").
    base = policy.single_source_base_confidence
    sole_contributor = contributing_observations[0] if contributing_observations else None
    provider_confidence = sole_contributor.provider_confidence if sole_contributor else None
    if provider_confidence is not None:
        value = policy.round_score(base * provider_confidence.value)
        derivation = (
            f"UNCORROBORATED_SINGLE_SOURCE: base confidence {base} scaled by "
            f"{sole_contributor.provider_id}'s own Provider Confidence "
            f"{provider_confidence.value} (Confidence Policy v{policy.confidence_policy_version}) "
            "-- within-provider only, never blended with another provider's value."
        )
    else:
        value = policy.round_score(base)
        derivation = (
            f"UNCORROBORATED_SINGLE_SOURCE: base confidence {base} (Confidence Policy "
            f"v{policy.confidence_policy_version}); no Provider Confidence was available from "
            "the sole contributing provider."
        )
    return CanonicalConfidence(value=value, derivation=derivation)


def apply_confidence_engine(
    reconciled_graph: ReconciledObservationGraph,
    observations_by_id: dict[str, Observation],
    policy: ConfidencePolicy,
) -> ReconciledObservationGraph:
    """Returns a new `ReconciledObservationGraph` with every member Canonical Observation's
    `canonical_confidence` populated. Never mutates the input (C9) -- `CanonicalObservation` is
    frozen; this produces new instances at the same `reconciliation_sequence`, completing rather
    than superseding them (MILESTONE1_DOMAIN_MODEL.md S1.3: Milestone 5 *completes* the object
    Milestone 4 left unfinished, this is not a new decision epoch).
    """
    completed = []
    for canonical_observation in reconciled_graph.canonical_observations:
        contributing = tuple(
            observations_by_id[ref.observation_id]
            for ref in canonical_observation.contributing_observations
            if ref.observation_id in observations_by_id
        )
        canonical_confidence = compute_canonical_confidence(canonical_observation, contributing, policy)
        completed.append(canonical_observation.model_copy(update={"canonical_confidence": canonical_confidence}))

    return ReconciledObservationGraph(
        reconciliation_sequence=reconciled_graph.reconciliation_sequence,
        canonical_observations=tuple(completed),
    )
