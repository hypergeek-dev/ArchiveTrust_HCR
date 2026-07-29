"""Bridges Confidence Engine output into `ConfidenceChanged` telemetry (ROADMAP.md S5.3, S12;
Constitution Article 13: "Telemetry (`ConfidenceChanged`) must indicate which of the three levels
changed and why.")

**Sequencing note:** this only emits `ConfidenceChanged`. It does not re-emit
`CanonicalDecisionCreated` -- the confidence-completed `CanonicalObservation` should be the one a
caller passes to `domain.comparison.telemetry.comparison_result_to_events` in the first place
(i.e., run `apply_confidence_engine` before emitting `CanonicalDecisionCreated`, not after). Two
separate `CanonicalDecisionCreated` events for the same `canonical_observation_id` would be
recorded as two distinct chain entries by `Journal._record_canonical_observation` -- a real
double-emission bug this ordering avoids, not a limitation of `ConfidenceChanged` itself.
"""

from __future__ import annotations

from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import ConfidenceChanged, ConfidenceLevel


def confidence_changed_events(
    canonical_observations: tuple[CanonicalObservation, ...],
    *,
    document_ref: str,
    policy: ConfidencePolicy,
) -> tuple[ConfidenceChanged, ...]:
    events = []
    for canonical in canonical_observations:
        if canonical.canonical_confidence is None:
            continue
        events.append(
            ConfidenceChanged(
                event_id=new_id("event"),
                document_ref=document_ref,
                subject_id=canonical.canonical_observation_id,
                level=ConfidenceLevel.CANONICAL,
                previous_value=None,
                new_value=canonical.canonical_confidence.value,
                reason=canonical.canonical_confidence.derivation,
                confidence_policy_version=policy.confidence_policy_version,
            )
        )
    return tuple(events)
