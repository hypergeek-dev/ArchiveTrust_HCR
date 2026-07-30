"""The research-question feedback loop: from a knowledge record back to a drafted experiment.

`docs/knowledge-lifecycle.md`'s "Known gaps for the next phase" names this as Phase 11 -- "the
observation -> question -> experiment-draft feedback loop". This module is it. It closes the cycle the
three knowledge layers open and then stop at:

    Observation -> Candidate finding -> Research question -> Hypothesis -> Experiment
        -> Results -> Reviewed finding -> New research question

**Why a sibling module rather than more of `models.py`.** `htr/knowledge/models.py` imports nothing
but `pydantic` and `domain.shared.ids`, which is `docs/architecture/htr-telemetry.md` §5's condition
for `domain/telemetry/events.py` to carry its entities as **typed embedded objects** instead of
`record` dicts. `ResearchQuestion` and `Hypothesis` therefore live there and travel on the wire
typed. Drafting an experiment, by contrast, needs `htr/experiment/baseline_template.py`, which imports
the three method adapters for their real `METHOD_ID` constants -- so putting this function in
`models.py` would drag `providers/*` into the module `events.py` imports and silently demote both new
entities to dicts. The split is the same one `baseline_knowledge.py` (entities, no persistence import)
and `registration.py` (orchestration) already make in this package.

**Nothing here is duplicated from `baseline_template.py`.** `draft_experiment_from_question` builds a
`DraftedExperimentDefinition` -- a *subclass* of the real `BaselineExperimentDefinition`, adding only
the three id fields that link a draft back to what provoked it -- and then hands it to
`build_baseline_experiment`, which is the one place in this codebase that constructs an `Experiment`
plus its `ExperimentVersion` from a definition. Subclassing rather than copying means
`ExperimentVersion.pipeline_configuration_ref` gains the question id *inside the same deterministic,
sorted JSON* the baseline already round-trips, and
`baseline_execution.py::build_research_report_from_store`'s
`BaselineExperimentDefinition.model_validate_json(ref)` keeps parsing it unchanged (Pydantic ignores
the extra keys by default, and the base model's own shape is untouched -- so no committed artifact
changes meaning).

**A draft is not a run.** `draft_experiment_from_question` constructs entities and returns them. It
starts no `ExperimentRun`, touches no GPU, imports no adapter runtime, and registers nothing -- the
same "build() returns constructed entities; registering/running them is the caller's decision" rule
`presentation/htr_experiment_builder_viewmodel.py` and `baseline_template.py` already follow. Every
drafted definition says so in its own `scope_caveats`, so a reader of the stored configuration cannot
mistake a draft for a result.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.htr.experiment.baseline_template import (
    BaselineExperimentDefinition,
    build_baseline_experiment,
    default_baseline_definition,
)
from archivetrust.htr.experiment.models import Experiment, ExperimentVersion
from archivetrust.htr.knowledge.models import (
    ContradictoryEvidence,
    Hypothesis,
    ResearchFinding,
    ResearchObservation,
    ResearchQuestion,
    ResearchScope,
)

REVISION_NOT_IN_ORIGINATING_SCOPE = (
    "not named by the originating scope -- no model revision to carry forward into the draft"
)
"""Placeholder for `default_baseline_definition`'s two required revision arguments when the
originating scope does not name that method at all.

It never reaches the drafted definition's own `method_versions`, which is rebuilt from the scope's
positionally-paired `method_ids`/`model_version_ids` and therefore contains exactly the methods the
originating record actually covered. Kept as a named constant rather than an empty string so that if
it ever *does* surface somewhere it reads as the absence it is.
"""

DRAFT_NOT_EXECUTED_CAVEAT = (
    "THIS IS AN UNEXECUTED DRAFT. No ExperimentRun exists for this ExperimentVersion, no inference "
    "has been performed, and no metric has been computed. It records what a researcher proposed to "
    "run in order to answer one research question -- nothing about what any method does."
)

DRAFT_SAMPLE_UNAVAILABLE_CAVEAT = (
    "The corpus this draft would need does not exist in this repository. The originating record is "
    "scoped to a single line crop, and answering its question requires many more ground-truthed "
    "lines than the one real shared HTR asset this repository contains. Executing this draft as-is "
    "would reproduce the same N=1 limitation it was raised to escape."
)

NO_HYPOTHESIS_ATTACHED = (
    "No hypothesis has been attached to the originating research question yet, so this draft asserts "
    "none. A hypothesis is a separate, attributable act (Hypothesis.create -> "
    "ResearchQuestion.with_hypothesis) and inventing one here to fill the field would put a "
    "prediction nobody made into a stored experiment configuration."
)


class DraftedExperimentDefinition(BaselineExperimentDefinition):
    """A `BaselineExperimentDefinition` that names the research question it was drafted from.

    Three added fields, all ids, all of which end up inside
    `ExperimentVersion.pipeline_configuration_ref` because `to_ref()` is inherited and dumps the whole
    model. That is what makes the backward edge -- experiment version -> question -> finding or
    observation -> evidence -- walkable from the durable record alone, rather than only forwards from
    `ResearchQuestion.created_experiment_version_id`. Both directions are stored; neither is inferred
    from the other.

    A subclass rather than a fourth free-form string on the base model: the base's field set is the
    brief's "Experiment definition" list and these are not part of it -- they are provenance about how
    *this* definition came to be written.
    """

    model_config = ConfigDict(frozen=True)

    research_question_id: str
    hypothesis_id: str | None = None
    originating_observation_id: str | None = None
    originating_finding_id: str | None = None

    @model_validator(mode="after")
    def _draft_names_what_provoked_it(self) -> "DraftedExperimentDefinition":
        if not self.research_question_id.strip():
            raise ValueError(
                "DraftedExperimentDefinition.research_question_id must name the ResearchQuestion "
                "this configuration was drafted from"
            )
        if not (self.originating_observation_id or self.originating_finding_id):
            raise ValueError(
                "DraftedExperimentDefinition must carry at least one of "
                "originating_observation_id/originating_finding_id, copied from the question, so the "
                "stored configuration itself reaches back to the evidence -- not only through the "
                "question record"
            )
        return self


class DraftedExperiment(BaseModel):
    """What `draft_experiment_from_question` returns.

    The `BuiltBaselineExperiment` shape plus the updated question, because drafting changes the
    question too (`created_experiment_id`/`created_experiment_version_id` are now set and its status
    has moved off `Open`). Returning the updated question rather than mutating the caller's is forced:
    `ResearchQuestion` is frozen.
    """

    model_config = ConfigDict(frozen=True)

    experiment: Experiment
    experiment_version: ExperimentVersion
    definition: DraftedExperimentDefinition
    question: ResearchQuestion
    """The question *after* `with_drafted_experiment`. The caller persists this state; the pre-draft
    state stays in whatever `ResearchQuestionRaised` event already recorded it."""

    @model_validator(mode="after")
    def _every_link_is_an_explicit_id(self) -> "DraftedExperiment":
        """The four-way link this whole module exists to produce, asserted on the way out.

        Not a test-only check, for the same reason `lifecycle.py::_assert_history_only_grew` is not:
        "the chain is linked by explicit ids, never inferred" is the property, and a future edit that
        returned a mismatched triple would otherwise be caught only by whichever test happened to
        look.
        """
        if self.question.created_experiment_id != self.experiment.experiment_id:
            raise AssertionError(
                f"drafted question names experiment {self.question.created_experiment_id!r} but the "
                f"drafted experiment is {self.experiment.experiment_id!r}"
            )
        if (
            self.question.created_experiment_version_id
            != self.experiment_version.experiment_version_id
        ):
            raise AssertionError(
                "drafted question and drafted experiment version disagree: "
                f"{self.question.created_experiment_version_id!r} != "
                f"{self.experiment_version.experiment_version_id!r}"
            )
        if self.definition.research_question_id != self.question.question_id:
            raise AssertionError(
                f"definition names question {self.definition.research_question_id!r} but the drafted "
                f"question is {self.question.question_id!r}"
            )
        if self.experiment_version.experiment_id != self.experiment.experiment_id:
            raise AssertionError(
                "drafted ExperimentVersion belongs to a different Experiment than the drafted one"
            )
        return self


def question_from_observation(
    observation: ResearchObservation,
    *,
    statement: str,
    motivation: str,
    created_by: str,
    at: str,
) -> ResearchQuestion:
    """Raises a `ResearchQuestion` provoked by one observation, linked by its real id."""
    return ResearchQuestion.create(
        statement=statement,
        motivation=motivation,
        created_by=created_by,
        created_at=at,
        originating_observation_id=observation.observation_id,
    )


def question_from_finding(
    finding: ResearchFinding,
    *,
    statement: str,
    motivation: str,
    created_by: str,
    at: str,
    include_supporting_observation: bool = True,
) -> ResearchQuestion:
    """Raises a `ResearchQuestion` provoked by one finding.

    `include_supporting_observation` also records the finding's *first* supporting observation, so the
    question reaches the evidence in one hop rather than two. Defaults to `True` because a finding
    always has at least one (`supporting_observations` is non-empty by construction) and a question
    that named only the finding would need the finding record resolved before it could be traced to
    anything measured.
    """
    return ResearchQuestion.create(
        statement=statement,
        motivation=motivation,
        created_by=created_by,
        created_at=at,
        originating_finding_id=finding.finding_id,
        originating_observation_id=(
            finding.supporting_observations[0] if include_supporting_observation else None
        ),
    )


def question_from_contradiction(
    contradiction: ContradictoryEvidence,
    *,
    disputed_finding: ResearchFinding,
    statement: str,
    motivation: str,
    created_by: str,
    at: str,
) -> ResearchQuestion:
    """Raises a `ResearchQuestion` provoked by a recorded contradiction.

    The follow-up requires a contradiction to be able to raise a question, and a contradiction is not
    a finding -- it is one append-only entry *on* a disputed one. Both ids are recorded: the
    contradiction (what specifically disagrees) and the finding it was appended to (what it disagrees
    with). Nothing about the disputed finding is changed; a question is raised alongside it.
    """
    return ResearchQuestion.create(
        statement=statement,
        motivation=motivation,
        created_by=created_by,
        created_at=at,
        originating_contradiction_id=contradiction.contradiction_id,
        originating_finding_id=disputed_finding.finding_id,
    )


def definition_for_question(
    question: ResearchQuestion,
    *,
    originating_scope: ResearchScope,
    title: str,
    sample_size_target: int | None = None,
    base_definition: BaselineExperimentDefinition | None = None,
) -> DraftedExperimentDefinition:
    """Builds the drafted experiment's definition from the question and its originating scope.

    Every value that can be sourced from a durable record is sourced from one rather than restated:

    * `method_ids`/`method_versions` come from `originating_scope`'s positionally-paired
      `method_ids`/`model_version_ids`, so the draft carries the *exact checkpoints* the originating
      record was scoped to. This is the one part of the definition a follow-up must not drift on --
      re-testing a calibration claim against a different revision answers a different question.
    * `research_question` is the question's own `statement`, verbatim.
    * `hypothesis` is the attached hypotheses' statements and falsification criteria, or
      `NO_HYPOTHESIS_ATTACHED` when there are none. Never invented.
    * everything else -- dataset description, inclusion/exclusion criteria, metric names, environment
      requirements -- is inherited unchanged from `default_baseline_definition`, because those facts
      are about this repository and have not changed.
    """
    base = base_definition or _base_definition_for(originating_scope)
    revisions = dict(
        zip(originating_scope.method_ids, originating_scope.model_version_ids, strict=True)
    )
    hypothesis_id = question.hypotheses[-1].hypothesis_id if question.hypotheses else None
    return DraftedExperimentDefinition(
        **{
            **base.model_dump(),
            "title": title,
            "research_question": question.statement,
            "hypothesis": _hypothesis_text(question),
            "method_ids": originating_scope.method_ids,
            "method_versions": revisions,
            "sampling_strategy": _sampling_strategy(originating_scope, sample_size_target),
            "scope_caveats": (
                *base.scope_caveats,
                DRAFT_NOT_EXECUTED_CAVEAT,
                DRAFT_SAMPLE_UNAVAILABLE_CAVEAT,
            ),
        },
        research_question_id=question.question_id,
        hypothesis_id=hypothesis_id,
        originating_observation_id=question.originating_observation_id,
        originating_finding_id=question.originating_finding_id,
    )


def draft_experiment_from_question(
    question: ResearchQuestion,
    *,
    originating_scope: ResearchScope,
    research_project_id: str,
    dataset_version_id: str,
    created_at: str,
    title: str | None = None,
    sample_size_target: int | None = None,
    version: int = 1,
    base_definition: BaselineExperimentDefinition | None = None,
) -> DraftedExperiment:
    """Drafts a real, valid, **unexecuted** `Experiment` + `ExperimentVersion` from one question.

    Construction is delegated entirely to `baseline_template.build_baseline_experiment`; this function
    decides *what* the definition says, never *how* an experiment is built. Returns the two entities,
    the definition that produced `pipeline_configuration_ref`, and the question advanced to
    `Being investigated` with both ids filled in.

    `dataset_version_id` is the caller's decision and is not defaulted from the originating scope,
    even though the scope usually names one. A follow-up that means to widen a sample will need a
    *different* `DatasetVersion` than the one the original result was scoped to, and silently reusing
    the old one would draft an experiment that cannot answer the question it was raised for. Passing
    the same id back is legitimate and explicit; having it appear by default would not be.

    Raises if `question` already names a drafted experiment -- see
    `ResearchQuestion.with_drafted_experiment`.
    """
    definition = definition_for_question(
        question,
        originating_scope=originating_scope,
        title=title or _default_title(question),
        sample_size_target=sample_size_target,
        base_definition=base_definition,
    )
    built = build_baseline_experiment(
        definition,
        research_project_id=research_project_id,
        dataset_version_id=dataset_version_id,
        created_at=created_at,
        version=version,
    )
    return DraftedExperiment(
        experiment=built.experiment,
        experiment_version=built.experiment_version,
        definition=definition,
        question=question.with_drafted_experiment(
            experiment_id=built.experiment.experiment_id,
            experiment_version_id=built.experiment_version.experiment_version_id,
        ),
    )


# -- Internals -------------------------------------------------------------------------------------


def _base_definition_for(scope: ResearchScope) -> BaselineExperimentDefinition:
    """`default_baseline_definition` with the originating scope's real revisions.

    Its two required revision arguments are filled from the scope where it names that method and with
    `REVISION_NOT_IN_ORIGINATING_SCOPE` where it does not -- which never survives into the returned
    definition, because `definition_for_question` overwrites `method_versions` with the scope's own
    pairs.
    """
    from archivetrust.providers.florence2_htr.adapter import METHOD_ID as FLORENCE2_METHOD_ID
    from archivetrust.providers.satrn.adapter import METHOD_ID as SATRN_METHOD_ID

    revisions = dict(zip(scope.method_ids, scope.model_version_ids, strict=True))
    return default_baseline_definition(
        satrn_model_revision=revisions.get(SATRN_METHOD_ID, REVISION_NOT_IN_ORIGINATING_SCOPE),
        florence2_model_revision=revisions.get(
            FLORENCE2_METHOD_ID, REVISION_NOT_IN_ORIGINATING_SCOPE
        ),
    )


def _hypothesis_text(question: ResearchQuestion) -> str:
    if not question.hypotheses:
        return NO_HYPOTHESIS_ATTACHED
    return " ".join(
        f"{hypothesis.statement} Refuted if: {hypothesis.falsification_criterion}"
        for hypothesis in question.hypotheses
    )


def _sampling_strategy(scope: ResearchScope, sample_size_target: int | None) -> str:
    target = (
        f"a target of at least {sample_size_target} {scope.unit_of_analysis.value}s"
        if sample_size_target is not None
        else "a sample larger than the originating scope's"
    )
    return (
        f"The originating record covers {scope.sample_size} "
        f"{scope.unit_of_analysis.value}{'' if scope.sample_size == 1 else 's'}. This draft proposes "
        f"{target}, drawn from ground-truthed lines of the same dataset version. No such sample "
        "exists in this repository yet, so no seed, stratification or draw is recorded here -- "
        "stating one would describe a selection nobody has made."
    )


def _default_title(question: ResearchQuestion) -> str:
    return f"Follow-up experiment drafted from research question {question.question_id}"


# -- The one real question this repository's own data raises ----------------------------------------

SATRN_CALIBRATION_QUESTION = (
    "Does SATRN's reported confidence track its measured transcription accuracy on a sample larger "
    "than one line crop -- i.e. is the confidence/accuracy disagreement recorded on the single "
    "controlled baseline line a property of this checkpoint's calibration, or a property of that one "
    "crop?"
)
"""Raised by the real `confidence_anomaly` observation on the committed 2026-07-30 baseline run.

Chosen because it is the question that observation's own scope *forces*: the observation records a
confidence/accuracy disagreement and then says, in its description and in two of its tags
(`this_run_and_this_sample_only`, `not_a_calibration_claim`), that one sample cannot measure
calibration. That is a stated gap with an obvious next experiment, which is exactly what a research
question is for -- and it is not a question this phase can answer, only ask.
"""

SATRN_CALIBRATION_MOTIVATION = (
    "The observation records reported confidence 0.6666051723062992 against measured CER "
    "0.7931034482758621 and WER 1.0 on one crop in one run, and states explicitly that this is not a "
    "claim about SATRN's confidence calibration because one sample cannot measure a distributional "
    "property. htr/evaluation/failures.py::classify_reliability flagged the run only because the "
    "disagreement (0.460) exceeded its own 0.35 threshold, and the candidate finding derived from "
    "the observation carries that threshold sensitivity as a stated limitation. The gap is therefore "
    "named in the record rather than inferred: what is unknown is whether the disagreement recurs."
)

SATRN_CALIBRATION_HYPOTHESIS = (
    "SATRN's reported confidence is not monotonically related to its measured character error rate "
    "on 17th-century Swedish court-record lines at this checkpoint."
)

SATRN_CALIBRATION_FALSIFICATION = (
    "Refuted if, over a sample of ground-truthed lines large enough to estimate a rank correlation, "
    "reported confidence and measured CER are monotonically related (higher confidence accompanying "
    "lower CER) with the relationship holding across the sample rather than only in aggregate. "
    "Refuted equally by the disagreement failing to recur at all: a single non-recurring "
    "disagreement is evidence about one crop, which is what the originating observation already says."
)


def satrn_confidence_calibration_question(
    *,
    observation: ResearchObservation,
    created_by: str,
    at: str,
) -> ResearchQuestion:
    """The real research question the SATRN confidence-anomaly observation raises, with a hypothesis
    already attached.

    Takes the registered observation rather than an id string so the link cannot be to an observation
    that was never registered, and so `originating_observation_id` is always the id actually in the
    log rather than a freshly minted one.
    """
    question = question_from_observation(
        observation,
        statement=SATRN_CALIBRATION_QUESTION,
        motivation=SATRN_CALIBRATION_MOTIVATION,
        created_by=created_by,
        at=at,
    )
    return question.with_hypothesis(
        Hypothesis.create(
            research_question_id=question.question_id,
            statement=SATRN_CALIBRATION_HYPOTHESIS,
            falsification_criterion=SATRN_CALIBRATION_FALSIFICATION,
            author=created_by,
            created_at=at,
        )
    )


SATRN_CALIBRATION_DRAFT_TITLE = (
    "SATRN confidence-calibration follow-up (drafted, never executed)"
)
"""The drafted experiment's `Experiment.name`. Says "drafted, never executed" in the name itself,
because `Experiment.name` is what every listing surface shows and a follow-up experiment with no run
behind it must not be readable as one that produced something."""
