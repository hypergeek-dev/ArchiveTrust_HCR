"""Evolution Center (ROADMAP_V2.md S11).

Where the Quality Center answers "what has been true," the Evolution Center answers "what should
change" -- but it **detects pressure for change; it never makes changes** (S11, S7). Every output
is an `EvolutionCandidate`: a structured artifact naming a signal, its magnitude, the telemetry it
was drawn from, and the specific area a human might change. A candidate is advisory input to an
explicit architectural decision, never a ticket that gets auto-applied and never a write back into
the Trust Engine (Guiding Principle 2, S7's "research never changes production automatically").

Thresholds live on a **versioned `EvolutionPolicy`**, never as hidden constants -- the same
discipline the Trust Engine's Reconciliation Policy follows (ROADMAP.md S10), so a candidate can
always be reproduced against the exact policy that produced it (Guiding Principle 7). Two detectors
are implemented at this stage (correction hotspots, single-source concepts); the remaining S11
signals (fallback-mapping frequency, unknown provider concepts, disagreement hotspots, semantic
clustering of rationale, provider drift) are documented there and deferred, per the
sequential-milestones principle (Guiding Principle 10).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.feedback.models import CorrectionAction
from archivetrust.domain.ontology.types import ObservationType
from archivetrust.learning.analytics.index import TelemetryIndex
from archivetrust.learning.source import TelemetrySource

EVOLUTION_POLICY_VERSION = 1


class EvolutionPolicy(BaseModel):
    """Versioned thresholds for signal detection (ROADMAP_V2.md S11; ROADMAP.md S10 precedent).

    Values here are deliberate placeholders pending corpus validation, exactly as the Trust
    Engine's own tolerance parameters are (ROADMAP.md S13) -- they are recorded on every candidate
    so a later, better-calibrated policy version supersedes rather than silently overwrites them.
    """

    model_config = ConfigDict(frozen=True)

    version: int = EVOLUTION_POLICY_VERSION
    correction_rate_threshold: float = 0.25
    """A slot-correction rate at or above this, for a type with enough slots to matter, raises a
    hotspot candidate."""
    minimum_slots_for_hotspot: int = 4
    """Below this many slots of a type, a correction rate is too small a sample to act on -- avoids
    raising a candidate off one or two corrected slots."""


class EvolutionSignalKind(str, Enum):
    CORRECTION_HOTSPOT = "correction_hotspot"
    """An ObservationType is corrected by humans unusually often (S11: correction frequency)."""
    SINGLE_SOURCE_CONCEPT = "single_source_concept"
    """An ObservationType is only ever produced by one provider, so agreement can never
    corroborate it (S11 unknown-concept / coverage pressure; ROADMAP.md S9 Capability Matrix:
    single-source concepts cannot benefit from agreement-based trust)."""


class ChangeArea(str, Enum):
    """Which architectural surface a candidate suggests examining. Advisory only -- naming the
    area does not authorize a change to it (S7)."""

    ONTOLOGY = "ontology"
    PROMPT_CONTRACT = "prompt_contract"
    PROVIDER = "provider"
    WORKFLOW = "workflow"
    CAPABILITY_MATRIX = "capability_matrix"


class EvolutionCandidate(BaseModel):
    """One evidence-backed suggestion that some architectural surface may warrant a human decision.

    `supporting_refs` are the telemetry-derived ids (Canonical Observation ids, correction ids)
    the signal was computed from, so a maintainer can trace the candidate straight back to the
    evidence (ROADMAP.md Guiding Principle 5, applied to the Learning Platform's own outputs). The
    candidate asserts nothing about the archive and changes nothing -- it is a reproducible reading
    of the telemetry, tagged with the policy version that produced it.
    """

    model_config = ConfigDict(frozen=True)

    signal_kind: EvolutionSignalKind
    change_area: ChangeArea
    subject: str
    """What the candidate is about -- an ObservationType value, a provider id, etc."""
    magnitude: float
    """A 0..1 strength for ranking candidates; its meaning is signal-specific and spelled out in
    `rationale`, never a bare number standing on its own (S10: no unexplainable insights)."""
    rationale: str
    supporting_refs: tuple[str, ...]
    policy_version: int


def detect_correction_hotspots(
    source: TelemetrySource, policy: EvolutionPolicy | None = None
) -> tuple[EvolutionCandidate, ...]:
    """ObservationTypes whose human-correction rate crosses the policy threshold (S11).

    A high correction rate for one type is pressure on whichever layer that type's corrections
    keep landing in -- but the *kind* of change is a human's call, so the candidate points at the
    ontology surface generically and carries the evidence, not a prescription.
    """
    policy = policy or EvolutionPolicy()
    index = TelemetryIndex.build(source)

    slots_by_type: dict[ObservationType, set[str]] = {}
    for canonical in index.latest_canonicals():
        slots_by_type.setdefault(canonical.observation_type, set()).add(canonical.semantic_slot_id)

    corrected_slots_by_type: dict[ObservationType, set[str]] = {}
    correction_refs_by_type: dict[ObservationType, list[str]] = {}
    for correction in index.corrections_submitted:
        if correction.action == CorrectionAction.ACCEPT.value:
            continue  # A confirmation is not correction pressure (see quality.py's same rule).
        target = index.canonical_by_id.get(correction.target_canonical_observation_id)
        if target is None:
            continue
        corrected_slots_by_type.setdefault(target.observation_type, set()).add(target.semantic_slot_id)
        correction_refs_by_type.setdefault(target.observation_type, []).append(
            correction.correction_id
        )

    candidates = []
    for obs_type, slots in slots_by_type.items():
        if len(slots) < policy.minimum_slots_for_hotspot:
            continue
        corrected = corrected_slots_by_type.get(obs_type, set())
        rate = len(corrected) / len(slots)
        if rate >= policy.correction_rate_threshold:
            candidates.append(
                EvolutionCandidate(
                    signal_kind=EvolutionSignalKind.CORRECTION_HOTSPOT,
                    change_area=ChangeArea.ONTOLOGY,
                    subject=obs_type.value,
                    magnitude=rate,
                    rationale=(
                        f"{len(corrected)} of {len(slots)} '{obs_type.value}' slots "
                        f"({rate:.0%}) were corrected by a human, at or above the policy "
                        f"threshold of {policy.correction_rate_threshold:.0%}. This type warrants "
                        "review of its ontology definition, the providers that produce it, or the "
                        "reconciliation policy that reconciles it -- a human decides which."
                    ),
                    supporting_refs=tuple(sorted(correction_refs_by_type.get(obs_type, []))),
                    policy_version=policy.version,
                )
            )
    candidates.sort(key=lambda c: (-c.magnitude, c.subject))
    return tuple(candidates)


def detect_single_source_concepts(
    source: TelemetrySource, policy: EvolutionPolicy | None = None
) -> tuple[EvolutionCandidate, ...]:
    """ObservationTypes that never achieve cross-provider corroboration (S11; ROADMAP.md S9).

    A type whose every current Canonical Observation is `UNCORROBORATED_SINGLE_SOURCE` cannot earn
    agreement-based trust no matter how the Comparison Engine improves -- the only remedies are
    architectural (add a provider that covers the concept, or accept it as inherently single-
    source in the Capability Matrix). That is exactly the kind of pressure a human should see, not
    a value the system should quietly patch.
    """
    policy = policy or EvolutionPolicy()
    index = TelemetryIndex.build(source)

    slots_by_type: dict[ObservationType, set[str]] = {}
    corroborated_types: set[ObservationType] = set()
    refs_by_type: dict[ObservationType, list[str]] = {}
    for canonical in index.latest_canonicals():
        obs_type = canonical.observation_type
        slots_by_type.setdefault(obs_type, set()).add(canonical.semantic_slot_id)
        refs_by_type.setdefault(obs_type, []).append(canonical.canonical_observation_id)
        if (
            canonical.comparison_confidence.classification
            != ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE
        ):
            corroborated_types.add(obs_type)

    candidates = []
    for obs_type, slots in slots_by_type.items():
        if obs_type in corroborated_types:
            continue  # At least one slot was corroborated -- not a single-source concept.
        candidates.append(
            EvolutionCandidate(
                signal_kind=EvolutionSignalKind.SINGLE_SOURCE_CONCEPT,
                change_area=ChangeArea.CAPABILITY_MATRIX,
                subject=obs_type.value,
                magnitude=1.0,
                rationale=(
                    f"All {len(slots)} current '{obs_type.value}' slots are single-source "
                    "(UNCORROBORATED_SINGLE_SOURCE): no independent provider ever corroborated "
                    "this concept, so agreement-based trust is structurally unavailable for it "
                    "(ROADMAP.md S9). Consider whether a provider covering this concept should be "
                    "added, or whether it is inherently single-source -- a human decides."
                ),
                supporting_refs=tuple(sorted(refs_by_type.get(obs_type, []))),
                policy_version=policy.version,
            )
        )
    candidates.sort(key=lambda c: c.subject)
    return tuple(candidates)
