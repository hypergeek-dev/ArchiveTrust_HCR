"""Registers the real baseline knowledge into a durable store, with real correlation and causation.

The orchestration half of `baseline_knowledge.py`: that module holds the entities and the real ids,
this one appends them to a `DurableHtrResearchStore` in the right causal order. The split mirrors
`htr/experiment/`'s own (`baseline_template.py` holds the definition, `baseline_execution.py` runs it)
and keeps `baseline_knowledge.py` free of any persistence import, so it stays readable as a statement
of what was observed rather than of how it was stored.

**Correlation** for every event here is the `ExperimentRun` the knowledge was extracted from --
`experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` for the four controlled-run records and
`experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73` for the Transkribus one. This is the same rule the
run's own events follow (`docs/architecture/htr-telemetry.md` §8: "`correlation_id` = the
`ExperimentRun.id`"), which is what makes a query for "everything about the controlled run" return its
execution telemetry *and* the knowledge extracted from it, in one correlation, across the two files.

**Causation** crosses from this stream into the run's log: each `ResearchObservationCreated`'s
`causation_id` is the `event_id` of the run event that supplied its decisive evidence (see
`baseline_knowledge.OBSERVATION_CAUSING_EVENT`), each `CandidateFindingCreated` is caused by the
`ResearchObservationCreated` of the observation it rests on, and each `FindingStatusChanged` is caused
by the `FindingReviewed` that preceded it. `causation_chain` walks the whole thing when the two
streams are concatenated -- the reason `tests/htr/knowledge/test_baseline_knowledge.py` replays them
together.
"""

from __future__ import annotations

from typing import Any

from archivetrust.htr.knowledge.baseline_knowledge import (
    CONTROLLED_RUN_ID,
    END_TO_END_RUN_ID,
    EXTRACTED_AT,
    OBSERVATION_CAUSING_EVENT,
    baseline_candidate_findings,
    florence2_relative_result_observation,
    gpu_memory_contradiction,
    gpu_memory_variability_observation,
    satrn_confidence_disagreement_observation,
    satrn_omitted_text_observation,
    transkribus_comparability_observation,
)
from archivetrust.htr.knowledge.models import (
    FindingStatus,
    ResearchFinding,
    ResearchObservation,
)

OBSERVATION_BUILDERS_BY_NAME = {
    "satrn_omitted_text": satrn_omitted_text_observation,
    "satrn_confidence_disagreement": satrn_confidence_disagreement_observation,
    "florence2_relative_result": florence2_relative_result_observation,
    "gpu_memory_variability": gpu_memory_variability_observation,
    "transkribus_comparability": transkribus_comparability_observation,
}
"""Insertion-ordered, and the order matters: the two SATRN observations are registered before the
comparison observation whose finding cites them."""

OBSERVATION_CORRELATION = {
    "satrn_omitted_text": CONTROLLED_RUN_ID,
    "satrn_confidence_disagreement": CONTROLLED_RUN_ID,
    "florence2_relative_result": CONTROLLED_RUN_ID,
    "gpu_memory_variability": CONTROLLED_RUN_ID,
    "transkribus_comparability": END_TO_END_RUN_ID,
}
"""Which run's unit of work each observation belongs to. The Transkribus comparability observation
correlates to the *end-to-end* run, not the controlled one, because that is the run whose method run
and fixture it is about -- even though the boundary it describes is the controlled comparison's."""

FINDING_SOURCE_OBSERVATION = {
    "florence2_lower_error_rates": "florence2_relative_result",
    "satrn_omitted_reference_text": "satrn_omitted_text",
    "satrn_confidence_not_aligned": "satrn_confidence_disagreement",
    "transkribus_not_comparable": "transkribus_comparability",
    "florence2_environment_reproducible": "gpu_memory_variability",
}
"""The observation whose `ResearchObservationCreated` causes each finding's
`CandidateFindingCreated` -- the layer-10-to-layer-11 causal edge. A finding resting on two
observations (`florence2_lower_error_rates` does) still has one causal parent, since `causation_id` is
single-valued; the full list is in `ResearchFinding.supporting_observations`."""


class BaselineKnowledge:
    """What `register_baseline_knowledge` produced: the registered entities and their event ids.

    A plain result holder, not a store -- the durable store already holds the entities; this is what
    the caller needs to keep walking the graph (event ids for further causation, entities for further
    transitions).
    """

    def __init__(
        self,
        *,
        observations: dict[str, ResearchObservation],
        observation_event_ids: dict[str, str],
        findings: dict[str, ResearchFinding],
        finding_event_ids: dict[str, str],
    ) -> None:
        self.observations = observations
        self.observation_event_ids = observation_event_ids
        self.findings = findings
        self.finding_event_ids = finding_event_ids


def register_baseline_knowledge(store: Any, *, at: str = EXTRACTED_AT) -> BaselineKnowledge:
    """Registers the five real observations and five candidate findings into `store`.

    `store` is a `DurableHtrResearchStore` (typed `Any` to keep this module's imports free of the
    persistence layer, the same reason `durable_store.py` types its own review-record parameters
    `Any`). Every registration is scoped to its run's correlation and caused by the real run event
    that supplied its evidence.

    Returns a `BaselineKnowledge` so a caller can continue the workflow -- which
    `scripts/register_baseline_knowledge.py` does, transitioning two of the findings.
    """
    observations: dict[str, ResearchObservation] = {}
    observation_event_ids: dict[str, str] = {}

    for name, builder in OBSERVATION_BUILDERS_BY_NAME.items():
        observation = builder(at=at)
        with store.correlated_to(OBSERVATION_CORRELATION[name]):
            event_id = store.register_research_observation(
                observation, caused_by=OBSERVATION_CAUSING_EVENT[name]
            )
        observations[name] = observation
        observation_event_ids[name] = event_id

    findings = baseline_candidate_findings(observations, at=at)
    finding_event_ids: dict[str, str] = {}
    for name, finding in findings.items():
        source = FINDING_SOURCE_OBSERVATION[name]
        with store.correlated_to(OBSERVATION_CORRELATION[source]):
            finding_event_ids[name] = store.register_candidate_finding(
                finding, caused_by=observation_event_ids[source]
            )

    return BaselineKnowledge(
        observations=observations,
        observation_event_ids=observation_event_ids,
        findings=findings,
        finding_event_ids=finding_event_ids,
    )


TRANSKRIBUS_REVIEW_REASONING = (
    "Reviewed against the committed log. The claim is verifiable by inspection rather than by "
    "measurement: the fixture's own content is an unrelated passage, the method run carries "
    "input_crop_id = null, and no MetricCalculated for CER or WER exists for it anywhere in the log "
    "(asserted by tests/htr/persistence/test_real_baseline_reconstruction.py::"
    "test_no_invalid_transkribus_cer_or_wer_exists_anywhere_in_the_log against every recorded metric "
    "*and* every MetricCalculated event). The exclusion was also declared in advance by the "
    "experiment's own exclusion_criteria, so it is not a post-hoc rationalisation of an inconvenient "
    "result. Advancing to Under review to record that a human has examined it."
)

TRANSKRIBUS_PROVISIONAL_REASONING = (
    "Provisionally supported, and deliberately not Supported. The evidence in scope is as strong as "
    "this kind of claim gets -- the fixture's non-correspondence is a fact about a file, not an "
    "estimate, and it is independently asserted by a test -- but 'Supported' in this lifecycle "
    "requires reproduction in an experiment run outside the finding's own scope, and only one "
    "end-to-end run exists. The honest ceiling is therefore Provisionally supported. It would reach "
    "Supported the moment a second run over the same fixture recorded the same absence, which costs "
    "no GPU time; nobody has run it."
)

ENVIRONMENT_REVIEW_REASONING = (
    "Reviewed against the two committed records this claim rests on: the Florence-2 adapter README's "
    "documented peak GPU memory from an earlier session, and this run's measured figure. They do not "
    "agree. Advancing to Under review so the disagreement is examined rather than left implicit in a "
    "README."
)

ENVIRONMENT_DISPUTE_REASONING = (
    "Disputed. The two measurements differ by a factor of ~3.3 (3983 MB documented vs. 1210.64 MiB "
    "measured), so the reproducibility this finding claims is contradicted by the repository's own "
    "records. Disputed rather than Rejected because which figure is representative is genuinely "
    "unknown: the earlier session left no durable record of its torch version or device state, so "
    "there is no basis for declaring either measurement wrong. The SATRN control reproduced to within "
    "0.3 MB across the same pair of sessions, which rules out 'this repository cannot measure GPU "
    "memory' as the explanation and is why the dispute is specific to Florence-2. The candidate "
    "finding, its supporting observation, and the contradiction all remain readable -- nothing was "
    "deleted to resolve this."
)


def demonstrate_review_workflow(
    store: Any, knowledge: BaselineKnowledge, *, reviewer: str, at: str = EXTRACTED_AT
) -> dict[str, ResearchFinding]:
    """Runs two of the candidate findings through the real status workflow.

    Produces, from real data only:

    * `transkribus_not_comparable`: `Candidate -> Under review -> Provisionally supported`. **Not**
      `Supported` -- `transition_finding_status` requires reproduction evidence naming a run outside
      the finding's scope, and this repository has exactly one end-to-end run. That refusal is the
      mechanism working, not a limitation worked around.
    * `florence2_environment_reproducible`: `Candidate -> Under review -> Disputed`, with a
      `ContradictoryEvidence` entry citing two real committed records.

    No `Superseded` example is produced, because none exists: superseding requires a second finding
    derived from a second run, and there is one baseline run. The *mechanism* is proven by
    `tests/htr/knowledge/test_lifecycle.py::test_superseding_preserves_both_findings_and_their_history`
    rather than by fabricating a second run to force an example.

    `reviewer` is the attributable human actor. It is a required parameter with no default: this
    function cannot invent one, and a finding does not advance past `Candidate` without one.
    """
    results: dict[str, ResearchFinding] = {}

    transkribus = knowledge.findings["transkribus_not_comparable"]
    with store.correlated_to(END_TO_END_RUN_ID):
        transkribus, _ = store.record_finding_transition(
            transkribus,
            FindingStatus.UNDER_REVIEW,
            reviewer=reviewer,
            reasoning=TRANSKRIBUS_REVIEW_REASONING,
            at=at,
            caused_by=knowledge.finding_event_ids["transkribus_not_comparable"],
        )
        transkribus, _ = store.record_finding_transition(
            transkribus,
            FindingStatus.PROVISIONALLY_SUPPORTED,
            reviewer=reviewer,
            reasoning=TRANSKRIBUS_PROVISIONAL_REASONING,
            at=at,
        )
    results["transkribus_not_comparable"] = transkribus

    environment = knowledge.findings["florence2_environment_reproducible"]
    with store.correlated_to(CONTROLLED_RUN_ID):
        environment, _ = store.record_finding_transition(
            environment,
            FindingStatus.UNDER_REVIEW,
            reviewer=reviewer,
            reasoning=ENVIRONMENT_REVIEW_REASONING,
            at=at,
            caused_by=knowledge.finding_event_ids["florence2_environment_reproducible"],
        )
        environment, _ = store.record_finding_transition(
            environment,
            FindingStatus.DISPUTED,
            reviewer=reviewer,
            reasoning=ENVIRONMENT_DISPUTE_REASONING,
            at=at,
            contradiction=gpu_memory_contradiction(
                observation_id=knowledge.observations["gpu_memory_variability"].observation_id,
                recorded_by=reviewer,
                at=at,
            ),
        )
    results["florence2_environment_reproducible"] = environment

    return results
