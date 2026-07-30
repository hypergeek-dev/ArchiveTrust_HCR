"""The status state machine, and the two requirements it exists to make true:

1. **A finding never loses its history.** Every transition appends a `FindingRevision` and preserves
   every prior one, in order, unmodified.
2. **A contradiction preserves both sides.** When finding B disputes finding A, neither is deleted nor
   overwritten; both remain independently readable with full history intact, and the relationship is
   queryable in both directions.

`test_a_contradiction_leaves_both_findings_independently_readable` is the follow-up's explicitly
required contradiction-preservation proof, and
`test_superseding_preserves_both_findings_and_their_history` is the mechanism proof that stands in for
a real `Superseded` example, since this repository has exactly one baseline experiment run and
superseding honestly needs two.
"""

from __future__ import annotations

import pytest

from archivetrust.htr.knowledge.lifecycle import (
    ALLOWED_TRANSITIONS,
    InvalidFindingTransitionError,
    can_transition,
    transition_finding_status,
)
from archivetrust.htr.knowledge.models import (
    ContradictionSourceKind,
    ContradictoryEvidence,
    EvidenceReference,
    EvidenceReferenceKind,
    FindingConfidence,
    FindingStatus,
    ResearchFinding,
    ResearchScope,
    ScopeUnit,
)
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink

RUN = "experiment_run_test0000000000000000000000000001"
SECOND_RUN = "experiment_run_test0000000000000000000000000002"
EXPERIMENT = "experiment_test000000000000000000000000000001"
VERSION = "experiment_version_test00000000000000000000001"
CROP = "input_crop_test0000000000000000000000000000001"
REVIEWER = "test-reviewer"
AT = "2026-07-30T04:00:00+00:00"


def a_scope(
    run_ids: tuple[str, ...] = (RUN,), covered: tuple[str, ...] = (CROP,)
) -> ResearchScope:
    return ResearchScope(
        experiment_id=EXPERIMENT,
        experiment_version_id=VERSION,
        experiment_run_ids=run_ids,
        unit_of_analysis=ScopeUnit.LINE_CROP,
        covered_unit_ids=covered,
        method_ids=("satrn",),
        model_version_ids=("abc123",),
    )


def a_finding(statement: str = "a scoped claim") -> ResearchFinding:
    return ResearchFinding.create(
        statement=statement,
        scope=a_scope(),
        supporting_observations=("research_observation_test0001",),
        limitations=("N=1",),
        confidence_level=FindingConfidence.LOW,
        author="tests",
        creation_date=AT,
    )


def a_contradiction(source_id: str, description: str = "the numbers disagree") -> ContradictoryEvidence:
    return ContradictoryEvidence.create(
        source_kind=ContradictionSourceKind.RESEARCH_FINDING,
        source_id=source_id,
        description=description,
        recorded_by=REVIEWER,
        recorded_at=AT,
    )


def reproduction_in(run_id: str) -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference(kind=EvidenceReferenceKind.EXPERIMENT_RUN, reference_id=run_id),
    )


def under_review(finding: ResearchFinding) -> ResearchFinding:
    return transition_finding_status(
        finding, FindingStatus.UNDER_REVIEW, reviewer=REVIEWER, reasoning="examined", at=AT
    )


# -- The state machine itself ----------------------------------------------------------------------


def test_every_status_has_an_explicit_edge_set():
    """Exhaustive over `FindingStatus`, so adding a status without deciding its transitions fails
    here rather than silently producing a dead end."""
    assert set(ALLOWED_TRANSITIONS) == set(FindingStatus)
    assert ALLOWED_TRANSITIONS[FindingStatus.SUPERSEDED] == frozenset(), (
        "Superseded is terminal: a finding that has been replaced does not change status again"
    )


def test_candidate_cannot_jump_straight_to_provisionally_supported():
    """`Under review` is the step that records a human actually looking, so it cannot be skipped."""
    assert not can_transition(FindingStatus.CANDIDATE, FindingStatus.PROVISIONALLY_SUPPORTED)
    with pytest.raises(InvalidFindingTransitionError, match="not a permitted transition"):
        transition_finding_status(
            a_finding(),
            FindingStatus.PROVISIONALLY_SUPPORTED,
            reviewer=REVIEWER,
            reasoning="skipping ahead",
            at=AT,
        )


def test_candidate_cannot_jump_straight_to_supported():
    with pytest.raises(InvalidFindingTransitionError, match="not a permitted transition"):
        transition_finding_status(
            a_finding(), FindingStatus.SUPPORTED, reviewer=REVIEWER, reasoning="no", at=AT
        )


def test_a_transition_requires_non_empty_reasoning():
    with pytest.raises(InvalidFindingTransitionError, match="non-empty `reasoning`"):
        transition_finding_status(
            a_finding(), FindingStatus.UNDER_REVIEW, reviewer=REVIEWER, reasoning="   ", at=AT
        )


def test_a_transition_requires_an_attributable_reviewer():
    with pytest.raises(InvalidFindingTransitionError, match="non-empty `reviewer`"):
        transition_finding_status(
            a_finding(), FindingStatus.UNDER_REVIEW, reviewer="", reasoning="examined", at=AT
        )


def test_a_no_op_transition_is_refused():
    with pytest.raises(InvalidFindingTransitionError, match="already at status"):
        transition_finding_status(
            a_finding(), FindingStatus.CANDIDATE, reviewer=REVIEWER, reasoning="again", at=AT
        )


# -- Supported requires reproduction, not just optimism --------------------------------------------


def test_supported_requires_reproduction_evidence():
    provisional = transition_finding_status(
        under_review(a_finding()),
        FindingStatus.PROVISIONALLY_SUPPORTED,
        reviewer=REVIEWER,
        reasoning="evidence in scope supports it",
        at=AT,
    )
    with pytest.raises(InvalidFindingTransitionError, match="requires `reproduction_evidence`"):
        transition_finding_status(
            provisional, FindingStatus.SUPPORTED, reviewer=REVIEWER, reasoning="looks solid", at=AT
        )


def test_supported_refuses_evidence_from_the_findings_own_run():
    """The rule that makes `Supported` genuinely unreachable for a single-run experiment.

    Recycling the finding's own run as "reproduction" would make `Supported` and `Provisionally
    supported` synonyms. This is exactly why the real Transkribus finding stops at `Provisionally
    supported`: only one end-to-end run exists to draw evidence from.
    """
    provisional = transition_finding_status(
        under_review(a_finding()),
        FindingStatus.PROVISIONALLY_SUPPORTED,
        reviewer=REVIEWER,
        reasoning="evidence in scope supports it",
        at=AT,
    )
    with pytest.raises(InvalidFindingTransitionError, match="already inside this finding's scope"):
        transition_finding_status(
            provisional,
            FindingStatus.SUPPORTED,
            reviewer=REVIEWER,
            reasoning="it reproduced",
            at=AT,
            reproduction_evidence=reproduction_in(RUN),
        )


def test_supported_accepts_reproduction_from_a_different_run():
    provisional = transition_finding_status(
        under_review(a_finding()),
        FindingStatus.PROVISIONALLY_SUPPORTED,
        reviewer=REVIEWER,
        reasoning="evidence in scope supports it",
        at=AT,
    )
    supported = transition_finding_status(
        provisional,
        FindingStatus.SUPPORTED,
        reviewer=REVIEWER,
        reasoning="held again in a second run",
        at=AT,
        reproduction_evidence=reproduction_in(SECOND_RUN),
    )
    assert supported.review_status is FindingStatus.SUPPORTED
    assert supported.revision_history[-1].evidence_refs[0].reference_id == SECOND_RUN


# -- History is append-only ------------------------------------------------------------------------


def test_every_transition_appends_a_revision_and_preserves_all_prior_ones():
    finding = a_finding()
    assert finding.revision_history == ()

    reviewing = under_review(finding)
    provisional = transition_finding_status(
        reviewing,
        FindingStatus.PROVISIONALLY_SUPPORTED,
        reviewer=REVIEWER,
        reasoning="in-scope evidence supports it",
        at=AT,
    )
    supported = transition_finding_status(
        provisional,
        FindingStatus.SUPPORTED,
        reviewer=REVIEWER,
        reasoning="reproduced elsewhere",
        at=AT,
        reproduction_evidence=reproduction_in(SECOND_RUN),
    )

    assert [(r.from_status.value, r.to_status.value) for r in supported.revision_history] == [
        ("Candidate", "Under review"),
        ("Under review", "Provisionally supported"),
        ("Provisionally supported", "Supported"),
    ]
    # Each earlier state's history is a genuine prefix of the later one's -- nothing rewritten.
    assert supported.revision_history[:1] == reviewing.revision_history
    assert supported.revision_history[:2] == provisional.revision_history
    assert all(r.reasoning.strip() for r in supported.revision_history)


def test_the_original_finding_object_is_never_mutated():
    finding = a_finding()
    under_review(finding)
    assert finding.review_status is FindingStatus.CANDIDATE
    assert finding.revision_history == ()
    assert finding.reviewer is None


# -- Contradiction preservation (the follow-up's required proof) -----------------------------------


def test_disputing_requires_a_named_contradiction():
    with pytest.raises(InvalidFindingTransitionError, match="requires a `contradiction`"):
        transition_finding_status(
            under_review(a_finding()),
            FindingStatus.DISPUTED,
            reviewer=REVIEWER,
            reasoning="I disagree",
            at=AT,
        )


def test_a_contradiction_leaves_both_findings_independently_readable(tmp_path):
    """**The required contradiction-preservation test.**

    Create finding A and take it all the way to `Supported`. Create finding B whose evidence
    contradicts A. Transition A to `Disputed` referencing B. Then assert, from a store replayed off
    disk, that:

    * both A and B exist independently and neither was deleted or overwritten;
    * A's full history -- including the revisions that made it `Supported` -- is intact;
    * A's own supporting evidence and statement are unchanged by the dispute;
    * B is untouched, still at its own status, with its own history;
    * the relationship is queryable in both directions.
    """
    path = tmp_path / "knowledge.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))

    finding_a = a_finding("A: the measured value is X")
    store.register_candidate_finding(finding_a)
    finding_a, _ = store.record_finding_transition(
        finding_a, FindingStatus.UNDER_REVIEW, reviewer=REVIEWER, reasoning="examined A", at=AT
    )
    finding_a, _ = store.record_finding_transition(
        finding_a,
        FindingStatus.PROVISIONALLY_SUPPORTED,
        reviewer=REVIEWER,
        reasoning="in-scope evidence supports A",
        at=AT,
    )
    finding_a, _ = store.record_finding_transition(
        finding_a,
        FindingStatus.SUPPORTED,
        reviewer=REVIEWER,
        reasoning="A reproduced in a second run",
        at=AT,
        reproduction_evidence=reproduction_in(SECOND_RUN),
    )
    assert finding_a.review_status is FindingStatus.SUPPORTED
    supported_history_length = len(finding_a.revision_history)
    a_statement = finding_a.statement
    a_evidence = finding_a.supporting_observations

    finding_b = a_finding("B: the measured value is not X")
    store.register_candidate_finding(finding_b)
    finding_b, _ = store.record_finding_transition(
        finding_b, FindingStatus.UNDER_REVIEW, reviewer=REVIEWER, reasoning="examined B", at=AT
    )

    finding_a, _ = store.record_finding_transition(
        finding_a,
        FindingStatus.DISPUTED,
        reviewer=REVIEWER,
        reasoning="B measured a different value for the same quantity",
        at=AT,
        contradiction=a_contradiction(finding_b.finding_id, "B contradicts A"),
    )

    # Replayed from disk, not read from the live store -- the preservation must be durable.
    del store
    from archivetrust.application.htr_journal import HtrJournal

    replayed = HtrJournal().replay(FileTelemetrySink(path).all_events())

    a = replayed.finding(finding_a.finding_id)
    b = replayed.finding(finding_b.finding_id)
    assert a is not None, "the disputed finding must still exist -- disputing is not deleting"
    assert b is not None, "the contradicting finding must exist independently"

    assert a.review_status is FindingStatus.DISPUTED
    assert a.statement == a_statement, "a dispute does not rewrite the claim"
    assert a.supporting_observations == a_evidence, "a dispute does not remove A's own support"
    assert len(a.revision_history) == supported_history_length + 1
    assert [r.to_status.value for r in a.revision_history] == [
        "Under review",
        "Provisionally supported",
        "Supported",
        "Disputed",
    ], "every state A passed through, including Supported, is still readable"
    assert a.revision_history[2].evidence_refs[0].reference_id == SECOND_RUN, (
        "the reproduction evidence that earned Supported survives the dispute"
    )

    assert b.review_status is FindingStatus.UNDER_REVIEW, "B is untouched by disputing A"
    assert len(b.revision_history) == 1
    assert b.contradictory_evidence == (), "B is not retroactively marked as contradicted"

    # Queryable in both directions.
    assert [c.source_id for c in a.contradictory_evidence] == [b.finding_id]
    assert [f.finding_id for f in replayed.findings_contradicting(b.finding_id)] == [a.finding_id]

    statuses = {f.finding_id: f.review_status for f in replayed.findings()}
    assert statuses == {
        finding_a.finding_id: FindingStatus.DISPUTED,
        finding_b.finding_id: FindingStatus.UNDER_REVIEW,
    }, "both findings are listed independently; neither was folded into the other"


def test_contradictory_evidence_accumulates_and_is_never_dropped():
    """A second dispute adds to the list; the first is not replaced."""
    finding = under_review(a_finding())
    disputed = transition_finding_status(
        finding,
        FindingStatus.DISPUTED,
        reviewer=REVIEWER,
        reasoning="first contradiction",
        at=AT,
        contradiction=a_contradiction("research_finding_first", "first"),
    )
    reopened = transition_finding_status(
        disputed,
        FindingStatus.UNDER_REVIEW,
        reviewer=REVIEWER,
        reasoning="re-examining after the dispute",
        at=AT,
    )
    assert [c.description for c in reopened.contradictory_evidence] == ["first"], (
        "re-opening a disputed finding keeps the record of what disputed it"
    )
    disputed_again = transition_finding_status(
        reopened,
        FindingStatus.DISPUTED,
        reviewer=REVIEWER,
        reasoning="second contradiction",
        at=AT,
        contradiction=a_contradiction("research_finding_second", "second"),
    )
    assert [c.description for c in disputed_again.contradictory_evidence] == ["first", "second"]


# -- Superseding: the mechanism, in the absence of a real two-run example ---------------------------


def test_superseding_requires_a_pointer_to_the_superseding_finding():
    with pytest.raises(InvalidFindingTransitionError, match="requires `superseded_by`"):
        transition_finding_status(
            a_finding(), FindingStatus.SUPERSEDED, reviewer=REVIEWER, reasoning="replaced", at=AT
        )


def test_superseding_preserves_both_findings_and_their_history(tmp_path):
    """**The mechanism proof that stands in for a real `Superseded` example.**

    No real superseded finding exists in this repository: superseding a scoped empirical claim needs a
    second experiment run producing a better-scoped successor, and there is exactly one baseline run.
    Fabricating a second run to manufacture an example is precisely what the brief forbids, so this
    test proves the *mechanism* on synthetic findings instead, and
    `docs/knowledge-lifecycle.md` says plainly that no real example exists yet.
    """
    path = tmp_path / "knowledge.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))

    old = a_finding("the claim, scoped to one run")
    store.register_candidate_finding(old)
    old, _ = store.record_finding_transition(
        old, FindingStatus.UNDER_REVIEW, reviewer=REVIEWER, reasoning="examined", at=AT
    )

    successor = ResearchFinding.create(
        statement="the same claim, scoped to two runs",
        scope=a_scope((RUN, SECOND_RUN), (CROP, "input_crop_test0000000000000000000000000000002")),
        supporting_observations=("research_observation_test0001", "research_observation_test0002"),
        limitations=("N=2",),
        confidence_level=FindingConfidence.MODERATE,
        author="tests",
        creation_date=AT,
    )
    store.register_candidate_finding(successor)

    old, _ = store.record_finding_transition(
        old,
        FindingStatus.SUPERSEDED,
        reviewer=REVIEWER,
        reasoning="a wider-scoped finding over two runs replaces this single-run one",
        at=AT,
        superseded_by=successor.finding_id,
    )

    del store
    from archivetrust.application.htr_journal import HtrJournal

    replayed = HtrJournal().replay(FileTelemetrySink(path).all_events())

    superseded = replayed.finding(old.finding_id)
    replacement = replayed.finding(successor.finding_id)
    assert superseded is not None, "a superseded finding is kept, never deleted"
    assert replacement is not None
    assert superseded.review_status is FindingStatus.SUPERSEDED
    assert superseded.superseded_by == successor.finding_id, "the chain stays walkable forwards"
    assert superseded.revision_history[-1].superseding_finding_id == successor.finding_id
    assert [r.to_status.value for r in superseded.revision_history] == [
        "Under review",
        "Superseded",
    ]
    assert replacement.review_status is FindingStatus.CANDIDATE, (
        "superseding an old finding does not promote the new one -- the successor still has to be "
        "reviewed on its own merits"
    )
    assert superseded.scope.sample_size == 1 and replacement.scope.sample_size == 2


def test_a_superseded_finding_cannot_change_status_again():
    superseded = transition_finding_status(
        a_finding(),
        FindingStatus.SUPERSEDED,
        reviewer=REVIEWER,
        reasoning="replaced",
        at=AT,
        superseded_by="research_finding_successor",
    )
    with pytest.raises(InvalidFindingTransitionError, match="terminal"):
        transition_finding_status(
            superseded, FindingStatus.UNDER_REVIEW, reviewer=REVIEWER, reasoning="reopening", at=AT
        )
