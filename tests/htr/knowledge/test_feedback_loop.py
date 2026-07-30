"""The research-question feedback loop, and the destroy-and-reconstruct proof for its whole chain.

`docs/knowledge-lifecycle.md`'s Phase 11. The load-bearing test here is
`test_the_whole_cycle_survives_destruction_of_the_store`, which follows
`tests/htr/persistence/test_read_model_reconstruction.py::test_read_model_survives_destruction_of_the_store`
exactly: register through a durable store over a real file, confirm the events are on disk, `del` every
in-memory object and force `gc.collect()`, construct a **brand-new** `FileTelemetrySink` over the same
path, replay, and only then assert. Nothing is carried across in memory and nothing is serialized as a
whole-store blob -- the follow-up brief is explicit that a blob round trip does not count.

What that test proves is not "a question can be stored" but that the four-link chain

    observation -> candidate finding -> research question -> drafted experiment version

is reconstructable *by following explicit ids in the replayed projection*, with no step inferred from
a timestamp, an append position, or a shared name.
"""

from __future__ import annotations

import gc
import json

import pytest

from archivetrust.application.htr_journal import HtrJournal, causation_chain
from archivetrust.htr.knowledge.baseline_knowledge import EXTRACTED_AT
from archivetrust.htr.knowledge.models import (
    ContradictionSourceKind,
    ContradictoryEvidence,
    Hypothesis,
    ResearchQuestion,
    ResearchQuestionStatus,
)
from archivetrust.htr.knowledge.questions import (
    DRAFT_NOT_EXECUTED_CAVEAT,
    NO_HYPOTHESIS_ATTACHED,
    DraftedExperimentDefinition,
    draft_experiment_from_question,
    question_from_contradiction,
    question_from_finding,
    satrn_confidence_calibration_question,
)
from archivetrust.htr.knowledge.registration import (
    FEEDBACK_LOOP_SAMPLE_SIZE_TARGET,
    register_baseline_knowledge,
    register_feedback_loop,
)
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.research_store import HtrResearchStore, UnknownEntityError
from archivetrust.infrastructure.storage.telemetry_sink import (
    FileTelemetrySink,
    InMemoryTelemetrySink,
)

ASKER = "hypergeek-dev"


def _registered(store) -> tuple:
    """The baseline knowledge plus the feedback loop, registered into `store`."""
    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    loop = register_feedback_loop(
        store,
        observation=knowledge.observations["satrn_confidence_disagreement"],
        finding=knowledge.findings["satrn_confidence_not_aligned"],
        created_by=ASKER,
        at=EXTRACTED_AT,
        caused_by=knowledge.finding_event_ids["satrn_confidence_not_aligned"],
    )
    return knowledge, loop


# -- The model ------------------------------------------------------------------------------------


def test_a_question_must_name_what_provoked_it() -> None:
    """A question with no originating record cannot be constructed. Without this the whole entity is
    a to-do list item rather than a link in the knowledge cycle."""
    with pytest.raises(ValueError, match="at least one of originating_observation_id"):
        ResearchQuestion(
            question_id="research_question_x",
            statement="Is this a question?",
            motivation="It has no origin.",
            created_by=ASKER,
            created_at=EXTRACTED_AT,
        )


def test_a_question_cannot_claim_to_be_answered_without_naming_the_finding() -> None:
    with pytest.raises(ValueError, match="must name the ResearchFinding that answers it"):
        ResearchQuestion(
            question_id="research_question_x",
            statement="Q",
            motivation="M",
            originating_observation_id="research_observation_x",
            status=ResearchQuestionStatus.ANSWERED,
            created_by=ASKER,
            created_at=EXTRACTED_AT,
        )


def test_a_drafted_experiment_pointer_cannot_coexist_with_open_status() -> None:
    """Drafting an experiment is an act on the question, so the pointer and the status cannot
    disagree -- and the model refuses rather than trusting a caller to keep them in step."""
    with pytest.raises(ValueError, match="but status is 'Open'"):
        ResearchQuestion(
            question_id="research_question_x",
            statement="Q",
            motivation="M",
            originating_finding_id="research_finding_x",
            created_experiment_id="experiment_x",
            created_by=ASKER,
            created_at=EXTRACTED_AT,
        )


def test_a_second_draft_is_refused_rather_than_overwriting_the_first_link() -> None:
    question = ResearchQuestion.create(
        statement="Q",
        motivation="M",
        created_by=ASKER,
        created_at=EXTRACTED_AT,
        originating_observation_id="research_observation_x",
    ).with_drafted_experiment(
        experiment_id="experiment_a", experiment_version_id="experiment_version_a"
    )
    with pytest.raises(ValueError, match="already names drafted experiment"):
        question.with_drafted_experiment(
            experiment_id="experiment_b", experiment_version_id="experiment_version_b"
        )
    assert question.created_experiment_id == "experiment_a"


def test_a_hypothesis_is_rebound_to_the_question_it_is_attached_to() -> None:
    """`with_hypothesis` rebinds `research_question_id` rather than accepting a mismatched one, and the
    model validator refuses a mismatch that bypasses it."""
    question = ResearchQuestion.create(
        statement="Q",
        motivation="M",
        created_by=ASKER,
        created_at=EXTRACTED_AT,
        originating_observation_id="research_observation_x",
    )
    stray = Hypothesis.create(
        research_question_id="research_question_somewhere_else",
        statement="H",
        falsification_criterion="F",
        author=ASKER,
        created_at=EXTRACTED_AT,
    )
    attached = question.with_hypothesis(stray)
    assert attached.hypotheses[0].research_question_id == question.question_id

    with pytest.raises(ValueError, match="but is attached to"):
        question.model_copy(update={"hypotheses": (stray,)}).model_validate(
            question.model_copy(update={"hypotheses": (stray,)}).model_dump()
        )


def test_history_of_hypotheses_only_grows() -> None:
    question = ResearchQuestion.create(
        statement="Q",
        motivation="M",
        created_by=ASKER,
        created_at=EXTRACTED_AT,
        originating_observation_id="research_observation_x",
    )
    first = Hypothesis.create(
        research_question_id=question.question_id,
        statement="H1",
        falsification_criterion="F1",
        author=ASKER,
        created_at=EXTRACTED_AT,
    )
    second = Hypothesis.create(
        research_question_id=question.question_id,
        statement="H2",
        falsification_criterion="F2",
        author=ASKER,
        created_at=EXTRACTED_AT,
    )
    both = question.with_hypothesis(first).with_hypothesis(second)
    assert [h.statement for h in both.hypotheses] == ["H1", "H2"]
    # The input is frozen and untouched, like every other entity in this package.
    assert question.hypotheses == ()


def test_a_contradiction_can_raise_a_question_without_touching_the_disputed_finding() -> None:
    """The follow-up requires a contradiction to be able to raise a question. Both the contradiction
    and the finding it was appended to are recorded, and the finding is unchanged."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    disputed = knowledge.findings["florence2_environment_reproducible"]
    contradiction = ContradictoryEvidence.create(
        source_kind=ContradictionSourceKind.RESEARCH_OBSERVATION,
        source_id=knowledge.observations["gpu_memory_variability"].observation_id,
        description="Two committed figures differ by a factor of ~3.3.",
        recorded_by=ASKER,
        recorded_at=EXTRACTED_AT,
    )
    question = question_from_contradiction(
        contradiction,
        disputed_finding=disputed,
        statement="Which of the two peak-GPU-memory figures is representative?",
        motivation="Neither session recorded enough environment to tell.",
        created_by=ASKER,
        at=EXTRACTED_AT,
    )
    assert question.originating_contradiction_id == contradiction.contradiction_id
    assert question.originating_finding_id == disputed.finding_id
    # Raising a question changes nothing about the finding it was raised from.
    assert store.finding(disputed.finding_id) == disputed


# -- Drafting -------------------------------------------------------------------------------------


def test_the_draft_carries_the_originating_scopes_exact_checkpoints() -> None:
    """The one part of a follow-up that must not drift: re-testing a calibration claim at a different
    model revision answers a different question. The revisions come from the originating scope's
    positionally-paired `method_ids`/`model_version_ids`, not from an adapter probe."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    knowledge, loop = _registered(store)
    observation = knowledge.observations["satrn_confidence_disagreement"]

    assert loop.draft.experiment_version.method_ids == observation.scope.method_ids
    assert loop.draft.definition.method_versions == {
        "satrn": observation.scope.model_version_ids[0]
    }


def test_a_draft_is_never_executed() -> None:
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    _knowledge, loop = _registered(store)
    version_id = loop.draft.experiment_version.experiment_version_id

    assert store.experiment_runs(experiment_version_id=version_id) == ()
    assert DRAFT_NOT_EXECUTED_CAVEAT in loop.draft.definition.scope_caveats
    assert "never executed" in loop.draft.experiment.name


def test_a_draft_with_no_hypothesis_says_so_rather_than_inventing_one() -> None:
    """`hypothesis` is a required non-empty field on `BaselineExperimentDefinition`, so a draft from a
    question with no hypothesis has to put *something* there. It puts the honest absence."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    observation = knowledge.observations["satrn_omitted_text"]
    finding = knowledge.findings["satrn_omitted_reference_text"]
    bare = question_from_finding(
        finding,
        statement="Does the omission recur on other lines?",
        motivation="One occurrence is recorded.",
        created_by=ASKER,
        at=EXTRACTED_AT,
    )
    assert bare.hypotheses == ()
    draft = draft_experiment_from_question(
        bare,
        originating_scope=observation.scope,
        research_project_id="research_project_x",
        dataset_version_id="dataset_version_x",
        created_at=EXTRACTED_AT,
    )
    assert draft.definition.hypothesis == NO_HYPOTHESIS_ATTACHED
    assert draft.definition.hypothesis_id is None


def test_the_drafted_definition_still_parses_as_a_baseline_definition() -> None:
    """`DraftedExperimentDefinition` subclasses the real one and adds only ids, so
    `baseline_execution.py::build_research_report_from_store`'s
    `BaselineExperimentDefinition.model_validate_json(pipeline_configuration_ref)` keeps working -- no
    committed artifact changes meaning."""
    from archivetrust.htr.experiment.baseline_template import BaselineExperimentDefinition

    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    _knowledge, loop = _registered(store)
    ref = loop.draft.experiment_version.pipeline_configuration_ref

    reparsed = BaselineExperimentDefinition.model_validate_json(ref)
    assert reparsed.research_question == loop.question.statement
    # And the subclass's added ids really are inside that same deterministic, sorted JSON.
    payload = json.loads(ref)
    assert payload["research_question_id"] == loop.question.question_id
    assert payload["originating_observation_id"] == loop.question.originating_observation_id
    assert payload["originating_finding_id"] == loop.question.originating_finding_id


def test_a_drafted_definition_must_reach_back_to_evidence_not_only_to_a_question() -> None:
    with pytest.raises(ValueError, match="originating_observation_id/originating_finding_id"):
        DraftedExperimentDefinition(
            research_question="Q",
            hypothesis="H",
            dataset_description="D",
            dataset_version_note="N",
            inclusion_criteria="I",
            exclusion_criteria="E",
            sampling_strategy="S",
            transcription_convention="T",
            ground_truth_requirements="G",
            method_ids=("satrn",),
            method_versions={"satrn": "rev"},
            segmentation_strategy="SS",
            controlled_variables=(),
            independent_variables=(),
            dependent_variables=(),
            metric_names=(),
            failure_policy="preserve_and_continue",
            acceptance_criteria="A",
            hardware_environment_requirements={},
            software_environment_requirements={},
            random_seed=0,
            reporting_strategy="R",
            research_question_id="research_question_x",
        )


def test_the_sampling_strategy_states_the_target_and_that_no_such_sample_exists() -> None:
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    _knowledge, loop = _registered(store)
    strategy = loop.draft.definition.sampling_strategy

    assert str(FEEDBACK_LOOP_SAMPLE_SIZE_TARGET) in strategy
    assert "No such sample" in strategy


def test_registering_the_loop_refuses_an_unlinked_observation_finding_pair() -> None:
    """A question may claim to originate from both an observation and a finding only when the finding
    really rests on that observation, which makes the chain a traversal rather than an assertion."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    with pytest.raises(ValueError, match="not among finding"):
        register_feedback_loop(
            store,
            observation=knowledge.observations["gpu_memory_variability"],
            finding=knowledge.findings["satrn_confidence_not_aligned"],
            created_by=ASKER,
            at=EXTRACTED_AT,
        )


def test_replay_refuses_to_invent_a_question_a_draft_event_advances() -> None:
    """An `ExperimentDraftedFromQuestion` with no preceding `ResearchQuestionRaised` means the log is
    incomplete. That must surface, not silently produce a question that appears already under
    investigation with nothing recording that it was ever asked."""
    from archivetrust.domain.telemetry.events import (
        HTR_RESEARCH_SCOPE,
        ExperimentDraftedFromQuestion,
    )

    orphan = ResearchQuestion.create(
        statement="Q",
        motivation="M",
        created_by=ASKER,
        created_at=EXTRACTED_AT,
        originating_observation_id="research_observation_missing",
    ).with_drafted_experiment(
        experiment_id="experiment_x", experiment_version_id="experiment_version_x"
    )
    event = ExperimentDraftedFromQuestion(
        event_id="event_orphan",
        document_ref=HTR_RESEARCH_SCOPE,
        question_ref=orphan.question_id,
        experiment_ref="experiment_x",
        experiment_version_ref="experiment_version_x",
        question=orphan,
    )
    with pytest.raises(UnknownEntityError):
        HtrJournal().replay([event])


# -- The destroy-and-reconstruct proof for the whole cycle -----------------------------------------


def test_the_whole_cycle_survives_destruction_of_the_store(tmp_path) -> None:
    """observation -> candidate finding -> research question -> drafted experiment version, replayed
    off disk and traversed by explicit id at every hop.

    The four ids are captured as plain strings *before* the store is destroyed, and every assertion
    afterwards reaches the entity by following a pointer stored on the previous one -- never by
    searching for it, and never by assuming append order.
    """
    path = tmp_path / "telemetry" / "htr_knowledge_events.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    knowledge, loop = _registered(store)

    # 1. The four ids, as plain values.
    observation_id = knowledge.observations["satrn_confidence_disagreement"].observation_id
    finding_id = knowledge.findings["satrn_confidence_not_aligned"].finding_id
    question_id = loop.question.question_id
    experiment_id = loop.draft.experiment.experiment_id
    experiment_version_id = loop.draft.experiment_version.experiment_version_id
    hypothesis_id = loop.question.hypotheses[0].hypothesis_id

    # 2. The events really are on disk, as self-describing records.
    assert path.exists()
    kinds = [
        json.loads(line)["kind"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for expected in (
        "ResearchObservationCreated",
        "CandidateFindingCreated",
        "ResearchQuestionRaised",
        "ExperimentCreated",
        "ExperimentVersionCreated",
        "ExperimentDraftedFromQuestion",
    ):
        assert expected in kinds, f"{expected} was never written to the durable stream"
    assert (path.parent / "htr_knowledge_events.jsonl.chain.jsonl").exists()

    # 3. Destroy every in-memory object. Nothing below may reach them.
    del store, knowledge, loop
    gc.collect()

    # 4. A brand-new sink over the same file, and a fresh projection replayed from it alone.
    reopened_sink = FileTelemetrySink(path)
    reconstructed = HtrJournal().replay(reopened_sink.all_events())
    # Deliberately *not* an `id()` comparison against the destroyed store: once it is freed, CPython
    # may hand the same address to the replacement, so that check passes or fails by allocator luck.
    # `HtrJournal.replay` builds a plain `HtrResearchStore`, never the `DurableHtrResearchStore`
    # subclass that was registered through -- which proves this is a different object *by type*, and
    # cannot flake.
    assert isinstance(reconstructed, HtrResearchStore)
    assert not isinstance(reconstructed, DurableHtrResearchStore)

    # 5. Link 1: the observation exists, with its typed evidence intact.
    observation = reconstructed.research_observation(observation_id)
    assert observation is not None
    assert observation.supporting_evidence, "an observation must name what it was extracted from"

    # 6. Link 2: finding -> observation, by the id stored on the finding.
    finding = reconstructed.finding(finding_id)
    assert finding is not None
    assert observation_id in finding.supporting_observations

    # 7. Link 3: question -> finding *and* -> observation, by the ids stored on the question.
    question = reconstructed.research_question(question_id)
    assert question is not None
    assert question.originating_finding_id == finding_id
    assert question.originating_observation_id == observation_id
    assert question.status is ResearchQuestionStatus.BEING_INVESTIGATED
    assert [h.hypothesis_id for h in question.hypotheses] == [hypothesis_id]

    # 8. Link 4: question -> drafted experiment version, forwards by the question's own pointer.
    assert question.created_experiment_id == experiment_id
    assert question.created_experiment_version_id == experiment_version_id
    experiment = reconstructed.experiment(experiment_id)
    version = reconstructed.experiment_version(experiment_version_id)
    assert experiment is not None and version is not None
    assert version.experiment_id == experiment_id

    # 9. And link 4 *backwards*, from the stored configuration itself -- not derived from step 8.
    payload = json.loads(version.pipeline_configuration_ref)
    assert payload["research_question_id"] == question_id
    assert payload["originating_observation_id"] == observation_id
    assert payload["originating_finding_id"] == finding_id
    assert reconstructed.research_questions_for_experiment(experiment_id) == (question,)

    # 10. The drafted experiment is still a draft after a full round trip.
    assert reconstructed.experiment_runs(experiment_version_id=experiment_version_id) == ()


def test_the_causal_chain_walks_from_the_finding_into_the_drafted_experiment(tmp_path) -> None:
    """The same four links again, this time by `causation_id` pointer alone -- no timestamp, no append
    position, no shared domain id. `causation_chain` consults nothing else."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    knowledge, loop = _registered(store)
    observation_event = knowledge.observation_event_ids["satrn_confidence_disagreement"]
    del store

    events = tuple(FileTelemetrySink(path).all_events())
    chain = causation_chain(events, from_event_id=observation_event)
    kinds = [event.kind.value for event in chain]

    assert kinds[:5] == [
        "ResearchObservationCreated",
        "CandidateFindingCreated",
        "ResearchQuestionRaised",
        "ExperimentCreated",
        "ExperimentVersionCreated",
    ]
    assert kinds[5] == "ExperimentDraftedFromQuestion"
    for parent, child in zip(chain, chain[1:], strict=False):
        assert child.causation_id == parent.event_id
    assert chain[2].question_ref == loop.question.question_id


def test_the_committed_feedback_artifact_links_the_documented_ids() -> None:
    """The committed `htr_knowledge_feedback_events.jsonl` links the observation and finding the
    documentation quotes, not a freshly-minted pair.

    This is what makes `scripts/register_research_question.py`'s "read the committed logs, do not
    regenerate them" discipline falsifiable: if a future pass re-registered the baseline knowledge and
    re-pointed the question, this test fails rather than the three knowledge documents quietly going
    stale.
    """
    from itertools import chain as chain_iter
    from pathlib import Path

    base = Path(__file__).resolve().parents[3] / "docs" / "experiments" / "baseline-comparison"
    feedback_path = base / "htr_knowledge_feedback_events.jsonl"
    if not feedback_path.exists():
        pytest.skip(f"{feedback_path} has not been generated in this checkout")

    events = [
        FileTelemetrySink(base / name).all_events()
        for name in (
            "htr_research_events.jsonl",
            "htr_knowledge_events.jsonl",
            "htr_knowledge_feedback_events.jsonl",
        )
    ]
    projection = HtrJournal().replay(chain_iter(*events))
    questions = projection.research_questions()

    assert len(questions) == 1
    question = questions[0]
    assert question.originating_observation_id == (
        "research_observation_067cd6a4ae80439289aa8256efffa835"
    )
    assert question.originating_finding_id == "research_finding_0f77a4f5443e4e79b429d8076596c4eb"
    assert question.status is ResearchQuestionStatus.BEING_INVESTIGATED
    assert len(question.hypotheses) == 1

    # The originating pair really is linked in the committed knowledge, not merely named together.
    finding = projection.finding(question.originating_finding_id)
    assert finding is not None
    assert question.originating_observation_id in finding.supporting_observations

    # And the drafted experiment is committed, reachable, and unexecuted.
    version = projection.experiment_version(question.created_experiment_version_id)
    assert version is not None
    assert json.loads(version.pipeline_configuration_ref)["research_question_id"] == (
        question.question_id
    )
    assert (
        projection.experiment_runs(experiment_version_id=version.experiment_version_id) == ()
    )


def test_the_real_question_is_read_off_a_gap_the_observation_itself_names() -> None:
    """The SATRN calibration question is not invented: the observation it comes from says in its own
    tags that it is not a calibration claim, which is the gap the question attacks."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    observation = knowledge.observations["satrn_confidence_disagreement"]

    assert "not_a_calibration_claim" in observation.tags
    assert "this_run_and_this_sample_only" in observation.tags

    question = satrn_confidence_calibration_question(
        observation=observation, created_by=ASKER, at=EXTRACTED_AT
    )
    assert question.originating_observation_id == observation.observation_id
    assert "larger than one line crop" in question.statement
    # The hypothesis states what would refute it -- non-empty by construction, and really populated.
    assert question.hypotheses[0].falsification_criterion.startswith("Refuted if")
