"""The finding-status state machine and `transition_finding_status`.

A pure module: it computes the next `ResearchFinding` and raises when a transition is not permitted.
It emits no telemetry -- `htr/persistence/durable_store.py::DurableHtrResearchStore
.record_finding_transition` is the producer that appends `FindingReviewed`/`FindingStatusChanged`
around a call to this function. That split is the same one `htr/evaluation/*` already has and that
`docs/htr-telemetry-knowledge-gap-analysis.md` §7 endorses: computation stays pure, persistence is
the caller's job.

**The state machine, and why each edge exists.**

```
Draft ──────────────► Candidate ─────────► Under review ──┬──► Provisionally supported ──► Supported
  │                       │                     │         ├──► Disputed                      │
  │                       │                     └─────────┴──► Rejected                      │
  │                       │                                                                  │
  └── Rejected            └── Rejected                    Disputed ◄──────────────────────────┘
                                                             │
                                                             ├──► Under review  (re-examination)
                                                             └──► Rejected

any status ──► Superseded   (requires a pointer to the superseding finding)
Superseded ──► (terminal)
```

Four rules are enforced beyond the edge list itself:

1. **Non-empty `reasoning` for every transition.** The follow-up requires it for any transition away
   from `Candidate`; this module requires it universally, because a `Draft -> Candidate` promotion
   with no stated reason is no more auditable than a silent `-> Supported`.
2. **`Provisionally supported -> Supported` requires reproduction evidence from a different
   experiment run.** Not merely a non-empty evidence list -- the evidence must name at least one
   `experiment_run` the finding's own scope does not already cover. Reproduction means "it happened
   again", and evidence drawn from the same single run cannot show that. This is the rule that makes
   `Supported` genuinely unreachable for the 2026-07-30 baseline, which has one controlled run: the
   honest ceiling there is `Provisionally supported`, and the type system says so rather than a
   comment.
3. **`-> Superseded` requires `superseded_by`.** A superseded finding always points at its
   successor.
4. **`-> Disputed` requires a `ContradictoryEvidence` entry**, appended to whatever the finding
   already carried. Nothing is ever removed from `contradictory_evidence` or `revision_history`; the
   function asserts the resulting history extends the prior one prefix-for-prefix before returning.
"""

from __future__ import annotations

from archivetrust.htr.knowledge.models import (
    ContradictoryEvidence,
    EvidenceReference,
    EvidenceReferenceKind,
    FindingRevision,
    FindingStatus,
    ResearchFinding,
)


class InvalidFindingTransitionError(ValueError):
    """Raised when a status transition is not permitted, or is permitted but under-evidenced.

    A `ValueError` subclass rather than a new exception hierarchy, matching
    `htr/experiment/models.py::ExperimentImmutableError`'s precedent for "a caller asked for a
    mutation the domain rules forbid".
    """


ALLOWED_TRANSITIONS: dict[FindingStatus, frozenset[FindingStatus]] = {
    FindingStatus.DRAFT: frozenset(
        {FindingStatus.CANDIDATE, FindingStatus.REJECTED, FindingStatus.SUPERSEDED}
    ),
    FindingStatus.CANDIDATE: frozenset(
        {FindingStatus.UNDER_REVIEW, FindingStatus.REJECTED, FindingStatus.SUPERSEDED}
    ),
    FindingStatus.UNDER_REVIEW: frozenset(
        {
            FindingStatus.PROVISIONALLY_SUPPORTED,
            FindingStatus.DISPUTED,
            FindingStatus.REJECTED,
            FindingStatus.SUPERSEDED,
        }
    ),
    FindingStatus.PROVISIONALLY_SUPPORTED: frozenset(
        {
            FindingStatus.SUPPORTED,
            FindingStatus.DISPUTED,
            FindingStatus.REJECTED,
            FindingStatus.SUPERSEDED,
        }
    ),
    FindingStatus.SUPPORTED: frozenset({FindingStatus.DISPUTED, FindingStatus.SUPERSEDED}),
    FindingStatus.DISPUTED: frozenset(
        {FindingStatus.UNDER_REVIEW, FindingStatus.REJECTED, FindingStatus.SUPERSEDED}
    ),
    FindingStatus.REJECTED: frozenset({FindingStatus.SUPERSEDED}),
    FindingStatus.SUPERSEDED: frozenset(),
}
"""The complete edge list. Exhaustive over `FindingStatus` -- `test_every_status_has_an_explicit_edge_set`
asserts that, so adding a status without deciding its transitions fails a test rather than silently
producing a dead end.

`Candidate -> Provisionally supported` is deliberately absent: a candidate must pass through
`Under review`, which is the step that records a human actually looking. `Supported -> Provisionally
supported` is absent too -- walking a claim back is a dispute, recorded as one, not a quiet
downgrade."""


def can_transition(current: FindingStatus, new_status: FindingStatus) -> bool:
    """Whether the edge `current -> new_status` exists. Pure predicate, no evidence rules applied."""
    return new_status in ALLOWED_TRANSITIONS[current]


def transition_finding_status(
    finding: ResearchFinding,
    new_status: FindingStatus,
    *,
    reviewer: str,
    reasoning: str,
    at: str,
    reproduction_evidence: tuple[EvidenceReference, ...] = (),
    superseded_by: str | None = None,
    contradiction: ContradictoryEvidence | None = None,
) -> ResearchFinding:
    """Returns a **new** `ResearchFinding` at `new_status`, with one `FindingRevision` appended.

    Never mutates `finding` (it is frozen) and never discards any part of its history. Raises
    `InvalidFindingTransitionError` when the edge does not exist or its evidence requirement is
    unmet -- see the module docstring's four rules.

    `at` is the transition timestamp, passed in rather than read from a clock, matching every
    `create` classmethod in `htr/corpus/models.py` and `htr/experiment/models.py` (the domain layer
    does not own a clock, so a test can produce a deterministic history).
    """
    current = finding.review_status

    if new_status is current:
        raise InvalidFindingTransitionError(
            f"finding {finding.finding_id} is already at status {current.value!r}; a no-op "
            "transition would append a revision recording no change"
        )
    if not can_transition(current, new_status):
        permitted = sorted(s.value for s in ALLOWED_TRANSITIONS[current])
        raise InvalidFindingTransitionError(
            f"{current.value!r} -> {new_status.value!r} is not a permitted transition; from "
            f"{current.value!r} the permitted targets are {permitted or ['(terminal)']}"
        )
    if not reasoning.strip():
        raise InvalidFindingTransitionError(
            f"a non-empty `reasoning` is required for every transition (here "
            f"{current.value!r} -> {new_status.value!r}); an unexplained status change is not "
            "reviewable"
        )
    if not reviewer.strip():
        raise InvalidFindingTransitionError(
            "a non-empty `reviewer` is required: no finding changes status without an attributable "
            "actor (docs/architecture/htr-event-model.md §1)"
        )

    if new_status is FindingStatus.SUPPORTED:
        _require_reproduction_from_another_run(finding, reproduction_evidence)
    if new_status is FindingStatus.SUPERSEDED and not (superseded_by or "").strip():
        raise InvalidFindingTransitionError(
            "transitioning to 'Superseded' requires `superseded_by`, the id of the finding that "
            "supersedes this one; superseding without a successor pointer breaks the chain"
        )
    if new_status is not FindingStatus.SUPERSEDED and superseded_by is not None:
        raise InvalidFindingTransitionError(
            f"`superseded_by` was supplied for a {new_status.value!r} transition; only 'Superseded' "
            "carries a successor pointer"
        )

    contradictions = finding.contradictory_evidence
    if new_status is FindingStatus.DISPUTED:
        if contradiction is None and not contradictions:
            raise InvalidFindingTransitionError(
                "transitioning to 'Disputed' requires a `contradiction` (or a finding that already "
                "carries one): a dispute must name the record that contradicts the claim, so both "
                "sides stay independently readable"
            )
        if contradiction is not None:
            contradictions = (*contradictions, contradiction)
    elif contradiction is not None:
        # Contradictory evidence is preserved on any transition it is offered with -- it is never
        # dropped just because the resulting status is not `Disputed` (a re-examination that
        # ultimately rejects a contradiction still keeps the record of it).
        contradictions = (*contradictions, contradiction)

    revision = FindingRevision.create(
        from_status=current,
        to_status=new_status,
        actor=reviewer,
        reasoning=reasoning,
        revised_at=at,
        evidence_refs=reproduction_evidence,
        superseding_finding_id=superseded_by,
        contradicting_finding_id=(
            contradiction.source_id
            if contradiction is not None and new_status is FindingStatus.DISPUTED
            else None
        ),
    )
    history = (*finding.revision_history, revision)

    updated = finding.model_copy(
        update={
            "review_status": new_status,
            "reviewer": reviewer,
            "revision_history": history,
            "contradictory_evidence": contradictions,
            "superseded_by": superseded_by if new_status is FindingStatus.SUPERSEDED else None,
        }
    )
    # `model_copy` bypasses validators, so re-validate: the invariants on `ResearchFinding` (a
    # Superseded finding points at a successor, a Disputed one carries a contradiction, a reviewed
    # one names a reviewer) must hold for the object this function hands back, not only for objects
    # built through `__init__`.
    updated = ResearchFinding.model_validate(updated.model_dump())

    _assert_history_only_grew(before=finding, after=updated)
    return updated


def _require_reproduction_from_another_run(
    finding: ResearchFinding, reproduction_evidence: tuple[EvidenceReference, ...]
) -> None:
    """Rule 2 of the module docstring: `Supported` needs evidence from a run this finding's scope
    does not already cover.

    A non-empty list is not enough. `Provisionally supported` already means "the evidence in scope
    supports this"; `Supported` has to mean something more, and the only thing it can honestly mean
    for a scoped empirical claim is that it held up somewhere else. Evidence recycled from the
    finding's own single run would make the two statuses synonyms.
    """
    if not reproduction_evidence:
        raise InvalidFindingTransitionError(
            "transitioning to 'Supported' requires `reproduction_evidence`: at least one "
            "EvidenceReference of kind 'experiment_run' naming a run outside this finding's scope"
        )
    run_refs = tuple(
        ref.reference_id
        for ref in reproduction_evidence
        if ref.kind is EvidenceReferenceKind.EXPERIMENT_RUN
    )
    if not run_refs:
        raise InvalidFindingTransitionError(
            "`reproduction_evidence` contains no EvidenceReference of kind 'experiment_run'; "
            f"got kinds {sorted({ref.kind.value for ref in reproduction_evidence})}. Reproduction "
            "is a claim about another run, so it must name one."
        )
    already_in_scope = set(finding.scope.experiment_run_ids)
    novel = [run_id for run_id in run_refs if run_id not in already_in_scope]
    if not novel:
        raise InvalidFindingTransitionError(
            f"`reproduction_evidence` names only runs already inside this finding's scope "
            f"({sorted(already_in_scope)}). 'Supported' requires the claim to have held in a run the "
            "finding was not derived from -- evidence from the same run cannot show reproduction."
        )


def _assert_history_only_grew(*, before: ResearchFinding, after: ResearchFinding) -> None:
    """Proves the transition appended rather than rewrote.

    An internal assertion rather than a test-only check, because "this entity must never lose its
    history" is the requirement the whole revision mechanism exists to satisfy, and a future edit to
    `transition_finding_status` that replaced instead of extended would otherwise be caught only by
    whichever test happened to look.
    """
    old, new = before.revision_history, after.revision_history
    if len(new) != len(old) + 1 or new[: len(old)] != old:
        raise AssertionError(
            "transition_finding_status must append exactly one FindingRevision and preserve every "
            f"prior one; history went from {len(old)} to {len(new)} entries with a changed prefix"
        )
    old_contradictions = before.contradictory_evidence
    if after.contradictory_evidence[: len(old_contradictions)] != old_contradictions:
        raise AssertionError(
            "transition_finding_status must preserve every prior ContradictoryEvidence entry"
        )
