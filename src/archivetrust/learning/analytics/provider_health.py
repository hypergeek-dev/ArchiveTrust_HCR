"""Provider Health analyzers (ROADMAP_V2.md S5.2, S9, S10).

Per-provider operational intelligence: how often each provider was invoked, how often its output
was rejected at the contract boundary, how often its Observations contributed to an accepted
Canonical Observation, and how often the slots it contributed to were subsequently corrected by a
human.

**Cross-provider comparability invariant (ROADMAP.md S5.3.1, restated for the Learning Platform).**
Every figure here is computed *within* one (provider, provider_version) at a time. This module
never averages or ranks providers by raw Provider Confidence, because those scales are not
comparable across providers without explicit, versioned calibration -- which does not exist yet.
Provider *health* is measured by behavior that is comparable (invocation, rejection, contribution,
downstream correction), never by blended self-reported confidence.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.telemetry.events import ProviderInvocationOutcome
from archivetrust.learning.analytics.index import ProviderRef, TelemetryIndex
from archivetrust.learning.source import TelemetrySource


class ProviderHealth(BaseModel):
    """Behavioral health of one (provider, provider_version). No confidence magnitudes appear
    here, by design (see module docstring)."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str

    invocation_count: int
    """ProviderObservationAttempted events -- what was tried, whether or not it produced anything
    (ROADMAP.md S12's Article-18 discipline, read back out as a health denominator)."""
    rejection_count: int
    """EvidenceRejected events -- responses that failed schema/contract validation. Rising
    rejections are a Prompt Contract Health signal for probabilistic providers (S11.1)."""
    contributing_slot_count: int
    """Distinct semantic slots whose current Canonical Observation this provider contributed to."""
    corrected_slot_count: int
    """Of those slots, how many received a human correction -- a proxy for how often trusting this
    provider's contribution turned out to need fixing."""
    no_observation_invocations: int
    """Invocations where the provider ran to completion but produced nothing (Operational
    Hardening milestone, Priority 8/13) -- distinct from a failure below; a fact about the archive
    (empty page, no detectable text), not about the provider."""
    failed_invocations: int
    """Invocations where the provider could not complete (Priority 8/13). Historical events
    recorded before `ProviderInvocationOutcome` existed count toward neither -- they have no
    outcome to classify, not zero of each."""

    @property
    def correction_rate(self) -> float | None:
        """Fraction of this provider's contributing slots that were corrected. None when the
        provider contributed to no slot, so no-data is never reported as flawless."""
        if self.contributing_slot_count == 0:
            return None
        return self.corrected_slot_count / self.contributing_slot_count


def provider_health(source: TelemetrySource) -> tuple[ProviderHealth, ...]:
    """Per-provider health, sorted by correction rate descending (most-corrected first), ties
    broken by provider id/version for determinism (GP 7)."""
    index = TelemetryIndex.build(source)

    invocations: dict[ProviderRef, int] = {}
    no_observations: dict[ProviderRef, int] = {}
    failed: dict[ProviderRef, int] = {}
    for attempt in index.provider_attempts:
        ref = ProviderRef(attempt.provider_id, attempt.provider_version)
        invocations[ref] = invocations.get(ref, 0) + 1
        if attempt.outcome == ProviderInvocationOutcome.NO_OBSERVATIONS:
            no_observations[ref] = no_observations.get(ref, 0) + 1
        elif attempt.outcome == ProviderInvocationOutcome.FAILED:
            failed[ref] = failed.get(ref, 0) + 1

    rejections: dict[ProviderRef, int] = {}
    for rejection in index.evidence_rejections:
        ref = ProviderRef(rejection.provider_id, rejection.provider_version)
        rejections[ref] = rejections.get(ref, 0) + 1

    # Which slots each provider contributed to (via the current Canonical Observation per slot),
    # and which of those slots were corrected.
    contributing_slots: dict[ProviderRef, set[str]] = {}
    for canonical in index.latest_canonicals():
        for ref in index.contributing_providers(canonical):
            contributing_slots.setdefault(ref, set()).add(canonical.semantic_slot_id)

    corrected_slots: set[str] = set()
    for correction in index.corrections_submitted:
        target = index.canonical_by_id.get(correction.target_canonical_observation_id)
        if target is not None:
            corrected_slots.add(target.semantic_slot_id)

    all_refs = set(invocations) | set(rejections) | set(contributing_slots)
    results = []
    for ref in all_refs:
        slots = contributing_slots.get(ref, set())
        results.append(
            ProviderHealth(
                provider_id=ref.provider_id,
                provider_version=ref.provider_version,
                invocation_count=invocations.get(ref, 0),
                rejection_count=rejections.get(ref, 0),
                contributing_slot_count=len(slots),
                corrected_slot_count=len(slots & corrected_slots),
                no_observation_invocations=no_observations.get(ref, 0),
                failed_invocations=failed.get(ref, 0),
            )
        )
    results.sort(
        key=lambda h: (
            -(h.correction_rate if h.correction_rate is not None else -1.0),
            h.provider_id,
            h.provider_version,
        )
    )
    return tuple(results)
