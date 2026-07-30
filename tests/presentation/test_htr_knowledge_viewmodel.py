"""`ResearchKnowledgeViewModel` -- asserted against the real committed baseline knowledge.

No mocks and no stub models: every test registers the actual five observations and five candidate
findings from `htr/knowledge/baseline_knowledge.py` (whose own ids and numbers are re-verified against
the committed run log by `tests/htr/knowledge/test_baseline_knowledge.py`), runs the real review
workflow over two of them, and raises the real research question. So an assertion here about a metric
id, a scope sentence or a breadcrumb is an assertion about data this repository actually recorded.

The tests that matter most are the ones that would catch a *comfortable* answer: that an empty
collection reports why it is empty rather than looking like a load failure, that
`recurring_failure_patterns` refuses to call five single occurrences recurrences, that
`reproduced_findings` is derived from revision history rather than from a status label, and that an
evidence reference this read model cannot open comes back unresolved with the reason instead of a
plausible label over nothing.
"""

from __future__ import annotations

import pytest

from archivetrust.htr.knowledge.baseline_knowledge import (
    CONTROLLED_RUN_ID,
    DATASET_ID,
    DATASET_VERSION_ID,
    EXPERIMENT_ID,
    EXTRACTED_AT,
    FLORENCE2_CER_NORMALIZED_METRIC_ID,
    PROJECT_ID,
    SATRN_CER_NORMALIZED_METRIC_ID,
    SATRN_METHOD_ID,
    SATRN_METHOD_RUN_ID,
    SATRN_MODEL_REVISION,
)
from archivetrust.htr.knowledge.models import (
    EvidenceReference,
    EvidenceReferenceKind,
    FindingConfidence,
    FindingStatus,
    ObservationType,
    ResearchScope,
    ScopeUnit,
)
from archivetrust.htr.knowledge.registration import (
    demonstrate_review_workflow,
    register_baseline_knowledge,
    register_feedback_loop,
)
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.presentation.desktop_v2_pages import DesktopV2Page
from archivetrust.presentation.htr_knowledge_viewmodel import (
    EMPTY_FIELD_NOTES,
    _TARGET_PAGE_BY_EVIDENCE_KIND,
    KnowledgeFilter,
    ResearchKnowledgeViewModel,
)
from archivetrust.htr.research_store import HtrResearchStore

REVIEWER = "hypergeek-dev"


@pytest.fixture()
def knowledge_store():
    """A durable store holding the real baseline knowledge, its two real transitions, and the real
    research question drafted from the confidence-anomaly observation."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    knowledge = register_baseline_knowledge(store, at=EXTRACTED_AT)
    demonstrate_review_workflow(store, knowledge, reviewer=REVIEWER, at=EXTRACTED_AT)
    loop = register_feedback_loop(
        store,
        observation=knowledge.observations["satrn_confidence_disagreement"],
        finding=knowledge.findings["satrn_confidence_not_aligned"],
        created_by=REVIEWER,
        at=EXTRACTED_AT,
        caused_by=knowledge.finding_event_ids["satrn_confidence_not_aligned"],
    )
    return store, knowledge, loop


@pytest.fixture()
def vm(knowledge_store):
    store, _knowledge, _loop = knowledge_store
    return ResearchKnowledgeViewModel(store)


# -- The snapshot's collections --------------------------------------------------------------------


def test_the_snapshot_sorts_the_real_findings_into_their_real_statuses(vm) -> None:
    """The committed statuses are exactly one Provisionally supported, one Disputed and three
    Candidate -- the same set `docs/knowledge-lifecycle.md` quotes."""
    snapshot = vm.snapshot()

    assert snapshot.observation_count == 5
    assert snapshot.finding_count == 5
    assert len(snapshot.candidate_findings) == 3
    assert len(snapshot.provisionally_supported_findings) == 1
    assert len(snapshot.disputed_findings) == 1
    assert snapshot.supported_findings == ()
    assert snapshot.superseded_findings == ()


def test_every_empty_collection_reports_why_rather_than_looking_like_a_load_failure(vm) -> None:
    notes = dict(vm.snapshot().empty_field_notes)

    assert "supported_findings" in notes
    assert "reproduction evidence" in notes["supported_findings"]
    assert "reproduced_findings" in notes
    assert "revision history" in notes["reproduced_findings"]
    assert "superseded_findings" in notes
    # A collection that is *not* empty must not carry a note.
    assert "candidate_findings" not in notes
    assert "disputed_findings" not in notes


def test_every_note_key_is_a_real_snapshot_field(vm) -> None:
    """A note keyed to a field that no longer exists would never be shown. Asserted rather than
    reviewed, so renaming a snapshot field without its note fails here."""
    fields = set(vm.snapshot().model_dump())
    assert set(EMPTY_FIELD_NOTES) <= fields


def test_the_comparison_boundary_observation_is_its_own_collection(vm) -> None:
    """`experiment_validity_boundary` exists so a statement about what an experiment can measure is
    never filed as one about the method that ran inside it, and the UI keeps that separation."""
    boundary = vm.snapshot().comparison_boundary_observations

    assert len(boundary) == 1
    row = boundary[0]
    assert row.observation_type == ObservationType.EXPERIMENT_VALIDITY_BOUNDARY.value
    assert row.observation_type_label == "What this comparison cannot measure"
    assert "not_a_method_performance_observation" in row.tags


# -- Scope reaches the UI -------------------------------------------------------------------------


def test_every_row_carries_the_domains_own_scope_sentence_and_derived_sample_size(vm) -> None:
    """`scope_description` is `ResearchScope.describe()` verbatim and `sample_size` is the derived
    property -- so `N=1` reaches the screen whether or not a page thinks to show it."""
    snapshot = vm.snapshot()

    for row in snapshot.recent_observations:
        assert row.sample_size == 1
        assert row.scope_description.startswith("1 ")
        assert f"experiment {EXPERIMENT_ID}" in row.scope_description
    for row in snapshot.candidate_findings + snapshot.disputed_findings:
        assert row.sample_size == 1
        assert row.limitations, "a scoped claim with no stated limitation is not honest"


def test_the_unverified_hypothesis_stays_in_its_own_field_all_the_way_to_the_ui(vm) -> None:
    """One observation holds a candidate explanation this project has not measured. It must stay
    separable from the measured facts without a page parsing prose."""
    rows = [
        row
        for row in vm.observations()
        if row.observation_type == ObservationType.REPRODUCIBILITY_ANOMALY.value
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row.unverified_hypothesis is not None
    assert "UNVERIFIED" in row.unverified_hypothesis
    assert "allocator" in row.unverified_hypothesis
    # And nowhere in the factual field.
    assert "allocator" not in row.description


def test_a_project_and_a_collection_are_traversals_not_invented_fields(knowledge_store) -> None:
    """An observation carries a dataset version, not a project or a collection. Both are reached
    through the corpus, and are `None`/empty when the corpus is not in the projection rather than
    guessed."""
    store, _knowledge, _loop = knowledge_store
    with_corpus = ResearchKnowledgeViewModel(store).observations()

    # The committed knowledge log alone carries no Dataset/DatasetVersion -- those live in the run log.
    assert all(row.project_id is None for row in with_corpus)
    assert all(row.collection_ids == () for row in with_corpus)
    # The dataset ids the observation itself names are still present, because they are its own fields.
    assert all(row.dataset_id == DATASET_ID for row in with_corpus)
    assert all(row.dataset_version_id == DATASET_VERSION_ID for row in with_corpus)


# -- Filters are real ------------------------------------------------------------------------------


def test_filters_narrow_by_real_stored_relationships(vm) -> None:
    assert len(vm.observations(filter=KnowledgeFilter(method_id=SATRN_METHOD_ID))) == 3
    assert len(vm.observations(filter=KnowledgeFilter(model_version_id=SATRN_MODEL_REVISION))) == 3
    assert len(vm.observations(filter=KnowledgeFilter(experiment_run_id=CONTROLLED_RUN_ID))) == 4
    assert len(vm.observations(filter=KnowledgeFilter(tag="n=1"))) == 3
    assert (
        len(
            vm.observations(
                filter=KnowledgeFilter(observation_type=ObservationType.CONFIDENCE_ANOMALY.value)
            )
        )
        == 1
    )
    assert len(vm.findings(filter=KnowledgeFilter(status=FindingStatus.DISPUTED.value))) == 1
    assert (
        len(vm.findings(filter=KnowledgeFilter(confidence=FindingConfidence.HIGH.value))) == 1
    )
    assert len(vm.findings(filter=KnowledgeFilter(dataset_version_id=DATASET_VERSION_ID))) == 5
    assert len(vm.findings(filter=KnowledgeFilter(experiment_id=EXPERIMENT_ID))) == 5


def test_a_tag_filter_reaches_findings_through_their_supporting_observations(vm) -> None:
    """`ResearchFinding` has no `tags` field. Rather than a tag filter silently matching no finding --
    which would read as "no finding is tagged that" -- it traverses `supporting_observations`."""
    tagged = vm.findings(filter=KnowledgeFilter(tag="not_a_calibration_claim"))

    assert len(tagged) == 1
    assert "confidence_calibration_disagreement" in tagged[0].tags
    assert tagged[0].observation_types == (ObservationType.CONFIDENCE_ANOMALY.value,)


def test_an_observation_type_filter_reaches_findings_through_what_they_rest_on(vm) -> None:
    rows = vm.findings(
        filter=KnowledgeFilter(
            observation_type=ObservationType.EXPERIMENT_VALIDITY_BOUNDARY.value
        )
    )

    assert len(rows) == 1
    assert rows[0].review_status == FindingStatus.PROVISIONALLY_SUPPORTED.value


def test_a_filter_that_matches_nothing_says_the_snapshot_is_filtered(vm) -> None:
    """An empty table under an active filter and an empty table over an empty store are different
    facts, and a page must be able to tell a reader which it is looking at."""
    snapshot = vm.snapshot(filter=KnowledgeFilter(method_id="no_such_method"))

    assert snapshot.filtered is True
    assert snapshot.observation_count == 0
    assert vm.snapshot().filtered is False


def test_a_status_value_from_the_other_enum_matches_nothing_rather_than_raising(vm) -> None:
    """One status control, two enums. An `ObservationReviewStatus` value against findings is honestly
    empty -- findings have no such status -- rather than an exception a UI has to catch."""
    assert vm.findings(filter=KnowledgeFilter(status="Unreviewed")) == ()
    assert len(vm.observations(filter=KnowledgeFilter(status="Unreviewed"))) == 5


# -- Recurring failure patterns are measured, not asserted -----------------------------------------


def test_failure_patterns_refuse_to_call_a_single_occurrence_a_recurrence(vm) -> None:
    """Five one-off observations in one run are five one-off observations. A "recurring failures" list
    that said otherwise would manufacture exactly the recurrence
    `ObservationType.MODEL_LIMITATION` was chosen over `RECURRING_RECOGNITION_FAILURE` to avoid."""
    patterns = vm.snapshot().recurring_failure_patterns

    assert {row.observation_type for row in patterns} == {
        ObservationType.MODEL_LIMITATION.value,
        ObservationType.CONFIDENCE_ANOMALY.value,
        ObservationType.REPRODUCIBILITY_ANOMALY.value,
    }
    for row in patterns:
        assert row.occurrence_count == 1
        assert row.distinct_experiment_run_count == 1
        assert row.recurrence_established is False
        assert row.total_sample_size == 1


def test_failure_patterns_exclude_the_types_that_are_not_failures(vm) -> None:
    """The relative-result observation is a measurement of two methods differing and the Transkribus
    one is a statement about experiment validity. Grouping either as a failure would misfile it."""
    grouped = {row.observation_type for row in vm.snapshot().recurring_failure_patterns}

    assert ObservationType.UNEXPECTED_METHOD_DISAGREEMENT.value not in grouped
    assert ObservationType.EXPERIMENT_VALIDITY_BOUNDARY.value not in grouped


def test_a_pattern_that_really_spans_two_runs_reports_recurrence(knowledge_store) -> None:
    """The derivation is real, not permanently `False`: a second observation of the same type from a
    second run flips `recurrence_established`. Built from a synthetic second run, because this
    repository has one -- the *mechanism* is what is under test, and fabricating a run into the
    committed artifact to demonstrate it is what must not happen."""
    from archivetrust.htr.knowledge.models import (
        ObservationConfidence,
        ResearchObservation,
    )

    store, _knowledge, _loop = knowledge_store
    second_run = "experiment_run_synthetic_second"
    scope = ResearchScope(
        experiment_id=EXPERIMENT_ID,
        experiment_version_id="experiment_version_synthetic",
        experiment_run_ids=(second_run,),
        unit_of_analysis=ScopeUnit.LINE_CROP,
        covered_unit_ids=("input_crop_synthetic",),
        method_ids=(SATRN_METHOD_ID,),
        model_version_ids=(SATRN_MODEL_REVISION,),
    )
    store.register_research_observation(
        ResearchObservation.create(
            observation_type=ObservationType.CONFIDENCE_ANOMALY,
            title="A second recorded confidence/accuracy disagreement",
            description="Synthetic: exercises the grouping, not a claim about SATRN.",
            scope=scope,
            supporting_evidence=(
                EvidenceReference(
                    kind=EvidenceReferenceKind.EXPERIMENT_RUN, reference_id=second_run
                ),
            ),
            source_experiment_id=EXPERIMENT_ID,
            source_experiment_run_id=second_run,
            affected_method=SATRN_METHOD_ID,
            author_or_source_component="tests",
            creation_timestamp=EXTRACTED_AT,
            observation_confidence=ObservationConfidence.LOW,
            tags=("n=1",),
        )
    )

    pattern = next(
        row
        for row in ResearchKnowledgeViewModel(store).snapshot().recurring_failure_patterns
        if row.observation_type == ObservationType.CONFIDENCE_ANOMALY.value
    )
    assert pattern.occurrence_count == 2
    assert pattern.distinct_experiment_run_count == 2
    assert pattern.recurrence_established is True
    assert pattern.shared_tags == ("n=1",)


# -- Reproduction is derived from history, not from a label ----------------------------------------


def test_no_committed_finding_is_reported_as_reproduced(vm) -> None:
    assert vm.snapshot().reproduced_findings == ()
    assert all(row.reproduced is False for row in vm.findings())
    assert all(row.reproducing_run_ids == () for row in vm.findings())


def test_reproduction_is_read_off_the_revision_that_earned_it(knowledge_store) -> None:
    """A finding taken legitimately to `Supported` reports the run that reproduced it -- and reports it
    from the revision's own evidence, not from the status word."""
    store, knowledge, _loop = knowledge_store
    finding = knowledge.findings["transkribus_not_comparable"]
    current = store.finding(finding.finding_id)
    assert current.review_status is FindingStatus.PROVISIONALLY_SUPPORTED

    second_run = "experiment_run_synthetic_second"
    promoted, _event = store.record_finding_transition(
        current,
        FindingStatus.SUPPORTED,
        reviewer=REVIEWER,
        reasoning="Synthetic reproduction, to exercise the derivation.",
        at=EXTRACTED_AT,
        reproduction_evidence=(
            EvidenceReference(
                kind=EvidenceReferenceKind.EXPERIMENT_RUN, reference_id=second_run
            ),
        ),
    )
    row = next(
        candidate
        for candidate in ResearchKnowledgeViewModel(store).findings()
        if candidate.finding_id == promoted.finding_id
    )

    assert row.reproduced is True
    assert row.reproducing_run_ids == (second_run,)
    assert second_run not in row.experiment_run_ids, (
        "reproduction must be a run the finding's own scope does not cover"
    )


def test_a_status_label_alone_does_not_make_a_finding_reproduced() -> None:
    """The guard the derivation exists for: a `Supported` finding whose revision names only a run
    already in its own scope is reported as *not* reproduced. `lifecycle.py` refuses to produce one, so
    this constructs it directly -- and the ViewModel still declines to believe the label."""
    from archivetrust.htr.knowledge.models import FindingRevision, ResearchFinding

    scope = ResearchScope(
        experiment_id=EXPERIMENT_ID,
        experiment_version_id="experiment_version_x",
        experiment_run_ids=(CONTROLLED_RUN_ID,),
        unit_of_analysis=ScopeUnit.LINE_CROP,
        covered_unit_ids=("input_crop_x",),
    )
    forged = ResearchFinding(
        finding_id="research_finding_forged",
        statement="A claim wearing a Supported label.",
        scope=scope,
        supporting_observations=("research_observation_x",),
        limitations=("Constructed directly, bypassing the transition function.",),
        confidence_level=FindingConfidence.LOW,
        author="tests",
        reviewer=REVIEWER,
        review_status=FindingStatus.SUPPORTED,
        creation_date=EXTRACTED_AT,
        revision_history=(
            FindingRevision.create(
                from_status=FindingStatus.PROVISIONALLY_SUPPORTED,
                to_status=FindingStatus.SUPPORTED,
                actor=REVIEWER,
                reasoning="Evidence recycled from the finding's own run.",
                revised_at=EXTRACTED_AT,
                evidence_refs=(
                    EvidenceReference(
                        kind=EvidenceReferenceKind.EXPERIMENT_RUN,
                        reference_id=CONTROLLED_RUN_ID,
                    ),
                ),
            ),
        ),
    )
    store = HtrResearchStore()
    store.register_finding(forged)

    row = ResearchKnowledgeViewModel(store).findings()[0]
    assert row.review_status == FindingStatus.SUPPORTED.value
    assert row.reproduced is False
    assert ResearchKnowledgeViewModel(store).snapshot().reproduced_findings == ()


# -- Contradiction preservation, both directions ---------------------------------------------------


def test_a_disputed_finding_keeps_its_full_history_and_both_sides(vm) -> None:
    row = vm.snapshot().disputed_findings[0]

    assert [(r.from_status, r.to_status) for r in row.revisions] == [
        ("Candidate", "Under review"),
        ("Under review", "Disputed"),
    ]
    assert all(revision.reasoning for revision in row.revisions)
    assert len(row.contradictions) == 1
    contradiction = row.contradictions[0]
    assert contradiction.source_kind == "research_observation"
    assert contradiction.recorded_by == REVIEWER
    # The disputed claim itself is untouched -- nothing was deleted to resolve it.
    assert "reproducible across sessions" in row.statement
    assert row.limitations


# -- Evidence navigation ---------------------------------------------------------------------------


def test_every_evidence_reference_kind_has_a_navigation_answer() -> None:
    """Exhaustive over `EvidenceReferenceKind`, so adding a kind without deciding where it leads fails
    here rather than silently producing an unnavigable link."""
    assert set(_TARGET_PAGE_BY_EVIDENCE_KIND) == set(EvidenceReferenceKind)
    for page in _TARGET_PAGE_BY_EVIDENCE_KIND.values():
        assert page is None or isinstance(page, DesktopV2Page)


def test_a_metric_result_reference_resolves_to_the_run_that_produced_it(knowledge_store) -> None:
    """The evidence chain is walkable from a method run; a metric result on its own has nowhere upward
    to go, so the navigable target is the run and the label is the measured value."""
    store, _knowledge, _loop = knowledge_store
    # The corpus and metric records live in the run log, which this store does not hold -- so this
    # reference is honestly unresolved here. Registering the metric makes it resolvable, which is what
    # this asserts.
    from archivetrust.htr.experiment.models import MethodRun, MetricResult

    store.register_method_run(
        MethodRun.create(
            experiment_run_id=CONTROLLED_RUN_ID,
            method_id=SATRN_METHOD_ID,
            evidence_id="evidence_x",
            outcome="succeeded",
            started_at=EXTRACTED_AT,
            model_version_id=SATRN_MODEL_REVISION,
        ).model_copy(update={"method_run_id": SATRN_METHOD_RUN_ID})
    )
    store.register_metric_result(
        MetricResult(
            metric_result_id=SATRN_CER_NORMALIZED_METRIC_ID,
            metric_definition_id="metric_definition_cer",
            method_run_id=SATRN_METHOD_RUN_ID,
            value=0.7931034482758621,
        )
    )

    link = ResearchKnowledgeViewModel(store).resolve_evidence_reference(
        EvidenceReference(
            kind=EvidenceReferenceKind.METRIC_RESULT,
            reference_id=SATRN_CER_NORMALIZED_METRIC_ID,
        )
    )
    assert link.resolved is True
    assert link.target_kind == "method_run"
    assert link.target_id == SATRN_METHOD_RUN_ID
    assert link.target_page == DesktopV2Page.RESEARCH_EVIDENCE.value
    assert "0.7931034482758621" in link.label


def test_a_telemetry_event_reference_is_unresolved_with_the_reason_not_a_plausible_label(
    vm,
) -> None:
    """`HtrResearchStore` has no event bucket. That is a property of the read model, not of the
    evidence, and the difference between "this evidence is missing" and "this surface cannot open it"
    has to survive to the UI."""
    link = vm.resolve_evidence_reference(
        EvidenceReference(
            kind=EvidenceReferenceKind.TELEMETRY_EVENT,
            reference_id="event_b53e520f7a8745c4ab7c32772e07a888",
            stream="docs/experiments/baseline-comparison/htr_research_events.jsonl",
        )
    )

    assert link.resolved is False
    assert link.target_page is None
    assert "does not index" in link.detail
    assert "htr_research_events.jsonl" in link.detail


def test_a_dangling_reference_is_shown_as_broken_rather_than_dropped(vm) -> None:
    link = vm.resolve_evidence_reference(
        EvidenceReference(
            kind=EvidenceReferenceKind.METHOD_RUN, reference_id="method_run_never_registered"
        )
    )

    assert link.resolved is False
    assert "not registered in this read model" in link.detail


def test_an_observation_carries_structured_evidence_ids_not_only_free_text(vm) -> None:
    """The follow-up's requirement: enough structured data for a UI to link to the evidence. Event ids,
    method-run ids and metric-result ids are separate fields, not sentences to parse."""
    row = next(
        candidate
        for candidate in vm.observations()
        if candidate.observation_type == ObservationType.UNEXPECTED_METHOD_DISAGREEMENT.value
    )

    assert FLORENCE2_CER_NORMALIZED_METRIC_ID in row.metric_result_ids
    assert SATRN_CER_NORMALIZED_METRIC_ID in row.metric_result_ids
    assert SATRN_METHOD_RUN_ID in row.method_run_ids
    assert row.telemetry_event_ids
    assert len(row.evidence) == 13
    assert {link.kind for link in row.evidence} >= {
        EvidenceReferenceKind.METRIC_RESULT.value,
        EvidenceReferenceKind.METHOD_RUN.value,
        EvidenceReferenceKind.EXPERIMENT_RUN.value,
    }


def test_a_breadcrumb_comes_from_the_existing_evidence_chain_viewmodel(knowledge_store) -> None:
    """Not reimplemented here: the injected `HtrEvidenceChainViewModel` produces the hops, and a
    reference to a run it cannot trace yields an empty breadcrumb rather than a fabricated one."""
    from archivetrust.presentation.htr_evidence_viewmodel import HtrEvidenceChainViewModel
    from tests.presentation._htr_fixtures import build_fixture_corpus

    corpus = build_fixture_corpus()
    chain = HtrEvidenceChainViewModel(corpus.store)
    vm = ResearchKnowledgeViewModel(corpus.store, evidence_chain=chain)

    link = vm.resolve_evidence_reference(
        EvidenceReference(
            kind=EvidenceReferenceKind.METHOD_RUN, reference_id=corpus.satrn_run_line_0
        )
    )
    expected = chain.chain_for_method_run(corpus.satrn_run_line_0)

    assert link.resolved is True
    assert len(link.breadcrumb) == len(expected.links)
    assert link.breadcrumb[0].startswith("result_stage:")
    assert link.breadcrumb[-1].startswith("project:")


# -- The research question -------------------------------------------------------------------------


def test_the_open_research_question_reaches_its_provocation_and_its_draft(vm, knowledge_store) -> None:
    _store, knowledge, loop = knowledge_store
    rows = vm.snapshot().unresolved_research_questions

    assert len(rows) == 1
    row = rows[0]
    assert row.question_id == loop.question.question_id
    assert row.status_label == "Being investigated"
    assert (
        row.originating_observation_id
        == knowledge.observations["satrn_confidence_disagreement"].observation_id
    )
    assert (
        row.originating_finding_id
        == knowledge.findings["satrn_confidence_not_aligned"].finding_id
    )
    assert row.originating_summary == (
        knowledge.observations["satrn_confidence_disagreement"].title
    )
    assert len(row.hypotheses) == 1
    assert row.hypotheses[0].falsification_criterion.startswith("Refuted if")
    assert row.created_experiment_id == loop.draft.experiment.experiment_id
    assert row.created_experiment_version_id == (
        loop.draft.experiment_version.experiment_version_id
    )
    assert row.drafted_experiment_has_runs is False
    assert {link.kind for link in row.evidence} == {"research_observation", "research_finding"}
    assert all(link.resolved for link in row.evidence)


def test_the_observation_and_finding_report_the_question_they_raised(vm, knowledge_store) -> None:
    """The feedback edge, forwards, from both ends -- so a reader looking at a record sees what it
    made somebody ask without querying separately."""
    _store, knowledge, loop = knowledge_store
    observation_id = knowledge.observations["satrn_confidence_disagreement"].observation_id
    finding_id = knowledge.findings["satrn_confidence_not_aligned"].finding_id

    observation_row = next(
        row for row in vm.observations() if row.observation_id == observation_id
    )
    finding_row = next(row for row in vm.findings() if row.finding_id == finding_id)

    assert observation_row.raised_question_ids == (loop.question.question_id,)
    assert finding_row.raised_question_ids == (loop.question.question_id,)


def test_research_questions_are_legitimately_empty_when_none_was_raised() -> None:
    """The field was designed to be empty, not stubbed. A store with knowledge but no question shows
    none, with the reason."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    register_baseline_knowledge(store, at=EXTRACTED_AT)
    snapshot = ResearchKnowledgeViewModel(store).snapshot()

    assert snapshot.unresolved_research_questions == ()
    assert snapshot.research_question_count == 0
    assert "unresolved_research_questions" in dict(snapshot.empty_field_notes)


def test_an_empty_store_produces_an_empty_snapshot_with_every_reason_stated() -> None:
    snapshot = ResearchKnowledgeViewModel(HtrResearchStore()).snapshot()

    assert snapshot.observation_count == 0
    assert snapshot.finding_count == 0
    assert snapshot.filtered is False
    notes = dict(snapshot.empty_field_notes)
    assert set(notes) == set(EMPTY_FIELD_NOTES)


# -- Read-only ------------------------------------------------------------------------------------


def test_the_viewmodel_has_no_way_to_change_anything(vm) -> None:
    """A ViewModel that could promote a finding would be a second, un-audited path past
    `lifecycle.py::transition_finding_status`. Asserted mechanically rather than by review."""
    mutators = [
        name
        for name in dir(vm)
        if not name.startswith("_")
        and any(
            verb in name
            for verb in ("register", "record", "transition", "advance", "promote", "set", "update")
        )
    ]
    assert mutators == []
