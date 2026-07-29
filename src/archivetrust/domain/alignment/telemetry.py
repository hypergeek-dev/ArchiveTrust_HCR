"""Bridges an `AlignmentResult` into telemetry (ROADMAP.md S5.12, S12; Constitution Articles 24, 27).

Emits, per alignment run: one `AlignmentAttempted` per comparison group (the grouping hypothesis,
its candidates, selections, and rationale), one `ObservationAligned` / `ObservationLeftUnaligned`
per Observation, so that every Observation's terminal state is on the replayable record and none
silently disappears (Article 18), and one `CandidateExcluded` per structurally-excluded candidate
pair (Article 27) -- the exclusion-side symmetry `AlignmentAttempted` alone does not provide. Every
event carries `alignment_algorithm_version`, so replay can identify which alignment strategy
produced a grouping.

Follows the same shape as `domain.comparison.telemetry` and `domain.confidence.telemetry`: a pure
function from a domain result to a tuple of immutable events, wired into the stream by the
application layer (`application.pipeline`).
"""

from __future__ import annotations

from archivetrust.domain.alignment.models import AlignmentOutcome, AlignmentResult
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    AlignmentAttempted,
    CandidateExcludedBatch,
    CandidateExclusionRecord,
    ObservationAligned,
    ObservationLeftUnaligned,
    TelemetryEvent,
)

MAX_EXCLUSIONS_PER_EVENT = 1000


def alignment_events(
    result: AlignmentResult, *, document_ref: str
) -> tuple[TelemetryEvent, ...]:
    events: list[TelemetryEvent] = []

    for attempt in result.attempts:
        events.append(
            AlignmentAttempted(
                event_id=new_id("event"),
                document_ref=document_ref,
                alignment_algorithm_version=attempt.algorithm_version,
                alignment_attempt_id=attempt.alignment_attempt_id,
                comparison_group_id=attempt.comparison_group_id,
                algorithm_name=attempt.algorithm_name,
                candidate_observation_ids=attempt.candidate_observation_ids,
                selected_observation_ids=attempt.selected_observation_ids,
                alignment_rationale=attempt.alignment_rationale,
            )
        )

    for state in result.observation_states:
        if state.outcome == AlignmentOutcome.ALIGNED:
            events.append(
                ObservationAligned(
                    event_id=new_id("event"),
                    document_ref=document_ref,
                    alignment_algorithm_version=result.algorithm_version,
                    observation_id=state.observation_id,
                    comparison_group_id=state.comparison_group_id,
                    alignment_attempt_id=state.alignment_attempt_id,
                )
            )
        else:
            events.append(
                ObservationLeftUnaligned(
                    event_id=new_id("event"),
                    document_ref=document_ref,
                    alignment_algorithm_version=result.algorithm_version,
                    observation_id=state.observation_id,
                    comparison_group_id=state.comparison_group_id,
                    alignment_attempt_id=state.alignment_attempt_id,
                    reason=(
                        "sole member of its comparison group; no other Observation aligned to it "
                        "(single-source under the current alignment strategy)"
                    ),
                )
            )

    for offset in range(0, len(result.excluded_pairs), MAX_EXCLUSIONS_PER_EVENT):
        batch = result.excluded_pairs[offset : offset + MAX_EXCLUSIONS_PER_EVENT]
        events.append(
            CandidateExcludedBatch(
                event_id=new_id("event"),
                document_ref=document_ref,
                alignment_algorithm_version=result.algorithm_version,
                exclusions=tuple(
                    CandidateExclusionRecord(
                        candidate_observation_id=pair.candidate_observation_id,
                        compared_against_observation_id=pair.compared_against_observation_id,
                        excluding_mechanism=pair.excluding_mechanism,
                        basis_code=pair.basis_code,
                        structural=pair.structural,
                    )
                    for pair in batch
                ),
            )
        )

    return tuple(events)
