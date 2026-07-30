"""Research Knowledge ViewModel -- the UI surface over `htr/knowledge/`.

Phase 10 of `docs/knowledge-lifecycle.md`'s "Known gaps for the next phase": "**No frontend.** The
Research Knowledge ViewModel is Phase 10 ... `HtrResearchStore` now has the query surface both will
need (`research_observations`, `findings`, `findings_contradicting`), but nothing consumes it yet."
This module consumes it.

Follows the established `presentation/htr_*_viewmodel.py` pattern exactly: the constructor takes the
store, `snapshot()` returns one frozen model a Qt page renders, and no page computes a status, groups
a pattern, or formats a domain value itself.

**Every field is derived from a real record or is honestly empty.** Four of the ten collections the
follow-up asks for are empty against this repository's data, and each is empty for a *different*
reason that is reported rather than left to look like a load failure (`EMPTY_FIELD_NOTES`, surfaced on
the snapshot as `empty_field_notes`):

* `supported_findings` -- reaching `Supported` requires reproduction evidence from an experiment run
  outside the finding's scope (`docs/knowledge-lifecycle.md` rule 3) and this repository has one run
  per scope. The honest ceiling is `Provisionally supported`, which is why
  `provisionally_supported_findings` exists as its own field rather than being folded in.
* `reproduced_findings` -- derived, not asserted: a finding qualifies only if its
  `revision_history` actually contains a `Provisionally supported -> Supported` transition whose
  `evidence_refs` name an `experiment_run` its own scope does not cover. Empty follows from
  `supported_findings` being empty; it is computed independently anyway, so it starts reporting the
  moment a second run exists.
* `superseded_findings` -- superseding needs a better-scoped successor, which needs a second run.
* `unresolved_research_questions` -- non-empty as of Phase 11
  (`htr/knowledge/questions.py`), but designed to be legitimately empty: a deployment that has raised
  no question shows none, and this field was not stubbed with placeholder rows while the entity did
  not exist.

`recurring_failure_patterns` is the one field where "derive it, do not hardcode it" needed a real
decision: it groups the store's observations by `observation_type` and reports, per group, how many
distinct experiment runs and how many observations it spans -- so `recurrence_established` is `False`
for every current group, because each is a single occurrence in a single run. A pattern list that
reported the baseline's five one-off observations as "recurring failures" would be the exact
overclaim `ObservationType.MODEL_LIMITATION` was chosen over `RECURRING_RECOGNITION_FAILURE` to avoid.

**Scope is never dropped on the way to the UI.** Every row carries `scope_description` (the domain's
own `ResearchScope.describe()`, not re-derived prose) and `sample_size` (the derived property, not a
number this layer computes), so `N=1` reaches the screen whether or not a page thinks to show it.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.knowledge.models import (
    ContradictoryEvidence,
    EvidenceReference,
    EvidenceReferenceKind,
    FindingRevision,
    FindingStatus,
    ObservationType,
    ResearchFinding,
    ResearchObservation,
    ResearchQuestion,
    ResearchQuestionStatus,
)
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.presentation.desktop_v2_pages import DesktopV2Page
from archivetrust.presentation.display_names import (
    evidence_reference_kind_label,
    finding_status_label,
    hypothesis_relationship_label,
    knowledge_confidence_label,
    method_label,
    observation_review_status_label,
    observation_type_label,
    research_question_status_label,
    short_ref,
)
from archivetrust.presentation.htr_evidence_viewmodel import HtrEvidenceChainViewModel

DEFAULT_RECENT_LIMIT = 20
"""How many observations `snapshot().recent_observations` carries. A cap rather than everything,
because "recent" is a triage list; `observations()` returns the unbounded, filtered set."""

FAILURE_PATTERN_OBSERVATION_TYPES = frozenset(
    {
        ObservationType.RECURRING_RECOGNITION_FAILURE,
        ObservationType.SEGMENTATION_PROBLEM,
        ObservationType.MODEL_LIMITATION,
        ObservationType.CONFIDENCE_ANOMALY,
        ObservationType.PERFORMANCE_BOTTLENECK,
        ObservationType.ENVIRONMENT_ISSUE,
        ObservationType.REPRODUCIBILITY_ANOMALY,
    }
)
"""Which of the fourteen observation types can constitute a *failure* pattern.

Named explicitly rather than "everything that is not a success", because three of the excluded members
are deliberately not failures and grouping them here would misfile them:
`UNEXPECTED_METHOD_DISAGREEMENT` is a measurement of two methods differing (the baseline's
Florence-2/SATRN comparison), `EXPERIMENT_VALIDITY_BOUNDARY` is a statement about what an experiment
can measure and never about the method that ran on it, and `HANDWRITING_FEATURE`/
`DOCUMENT_LAYOUT_FEATURE`/`REVIEWER_OBSERVATION`/`POSSIBLE_HYPOTHESIS` are not outcomes at all.
"""

EMPTY_FIELD_NOTES: dict[str, str] = {
    "supported_findings": (
        "No finding is Supported. Reaching that status requires reproduction evidence naming an "
        "experiment run outside the finding's own scope, and this repository has one run per scope. "
        "Provisionally supported is the honest ceiling here, not an omission."
    ),
    "reproduced_findings": (
        "No finding has been reproduced. This list is derived from revision history -- a "
        "'Provisionally supported -> Supported' transition whose evidence names a run the finding's "
        "scope does not cover -- so it will populate itself the moment a second run records one."
    ),
    "superseded_findings": (
        "No finding has been superseded. Superseding means replacing a scoped claim with a "
        "better-scoped successor, which needs a second experiment run."
    ),
    "unresolved_research_questions": (
        "No open research question. A question is raised from an observation, finding or "
        "contradiction; none is outstanding in this deployment."
    ),
    "recurring_failure_patterns": (
        "No failure-type observation is recorded, so there is nothing to group."
    ),
    "comparison_boundary_observations": (
        "No observation records what a comparison cannot measure. Absence here is not reassurance -- "
        "it can equally mean nobody has written the boundary down."
    ),
    "recent_observations": (
        "No research observation is recorded. Observations are extracted from durable records by an "
        "explicit step; none has been run against this deployment's telemetry."
    ),
    "candidate_findings": (
        "No candidate finding. A finding rests on at least one observation, so none can exist until "
        "one does."
    ),
    "findings_under_review": ("No finding is under review."),
    "disputed_findings": ("No finding is disputed."),
    "provisionally_supported_findings": ("No finding is provisionally supported."),
}
"""Why each collection is empty, when it is. Keyed by snapshot field name.

Stated as data rather than as prose in a page, so one place changes when a reason stops holding, and so
a page can render the reason next to the empty table instead of an unexplained blank -- the same
discipline `htr_review_center_viewmodel.CORRECTION_TIME_GAP` already applies to a missing field.
"""

_NOT_IN_THIS_PROJECTION = (
    "Recorded durably in the telemetry stream, which this read model does not index by id"
)

_TARGET_PAGE_BY_EVIDENCE_KIND: dict[EvidenceReferenceKind, DesktopV2Page | None] = {
    EvidenceReferenceKind.METHOD_RUN: DesktopV2Page.RESEARCH_EVIDENCE,
    EvidenceReferenceKind.METRIC_RESULT: DesktopV2Page.RESEARCH_EVIDENCE,
    EvidenceReferenceKind.RELIABILITY_CLASSIFICATION: DesktopV2Page.RESEARCH_EVIDENCE,
    EvidenceReferenceKind.EVIDENCE_RECORD: DesktopV2Page.RESEARCH_EVIDENCE,
    EvidenceReferenceKind.INPUT_CROP: DesktopV2Page.DATASETS,
    EvidenceReferenceKind.TEXT_LINE: DesktopV2Page.DATASETS,
    EvidenceReferenceKind.GROUND_TRUTH_TEXT: DesktopV2Page.DATASETS,
    EvidenceReferenceKind.EXPERIMENT_RUN: DesktopV2Page.EXPERIMENTS,
    EvidenceReferenceKind.REPRODUCIBILITY_MANIFEST: DesktopV2Page.EXPERIMENTS,
    EvidenceReferenceKind.RESEARCH_OBSERVATION: DesktopV2Page.RESEARCH_KNOWLEDGE,
    EvidenceReferenceKind.RESEARCH_FINDING: DesktopV2Page.RESEARCH_KNOWLEDGE,
    EvidenceReferenceKind.TELEMETRY_EVENT: None,
    EvidenceReferenceKind.EXTERNAL_DOCUMENT: None,
}
"""Which navigation page can show each kind of evidence reference, or `None` where none can.

Exhaustive over `EvidenceReferenceKind` -- `test_every_evidence_reference_kind_has_a_navigation_answer`
asserts that, so adding a kind without deciding where it leads fails a test rather than silently
producing an unnavigable link. The two `None`s are honest: no page in this application replays the HTR
telemetry stream by event id (the operational Document Evidence page replays per Archive Object, and
HTR research events carry the `htr:research` sentinel instead), and an `external_document` reference
names a committed repository file rather than any record this application stores.
"""


class EvidenceLink(BaseModel):
    """One resolved `EvidenceReference`: what it points at, whether it is here, and where to go.

    The return type of `ResearchKnowledgeViewModel.resolve_evidence_reference`, and also what every
    observation and finding row carries -- one type, so a page renders a reference the same way
    wherever it appears.

    Deliberately shaped like `htr_evidence_viewmodel.py::ChainLink` (kind / entity id / label / detail
    / `resolved`), because that ViewModel already solved "walk from an id to a human-readable
    breadcrumb, and show a broken link rather than hiding it". `breadcrumb` is literally its output:
    for a reference this projection can trace to a method run, `HtrEvidenceChainViewModel
    .chain_for_method_run` supplies the hops, and they are not re-derived here.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    """The `EvidenceReferenceKind` value, verbatim."""
    kind_label: str
    reference_id: str
    label: str
    detail: str = ""
    note: str | None = None
    """The reference's own optional annotation, carried through unchanged. Never load-bearing --
    `kind`/`reference_id` are (`EvidenceReference.note`'s own docstring)."""
    stream: str | None = None
    """Which durable stream the referenced record lives in, when the reference says."""
    resolved: bool = True
    """`False` when this read model cannot produce the referenced entity. Two different causes, both
    surfaced rather than hidden: the id is genuinely dangling, or the record is durable but is not
    indexed by this projection (a telemetry event id, an `Evidence` record). `detail` says which."""
    target_kind: str | None = None
    """The entity kind a UI would open -- `"method_run"`, `"input_crop"`, `"experiment_run"`, ... --
    which is not always this reference's own kind: a `metric_result` reference resolves to the method
    run that produced it, because that is what the evidence chain is walkable from."""
    target_id: str | None = None
    target_page: str | None = None
    """The `DesktopV2Page` value a UI would navigate to, or `None` where no page can show this kind.
    A value rather than a page object so the Qt layer stays the only thing that knows about widgets."""
    breadcrumb: tuple[str, ...] = ()
    """The evidence chain from the target upward, one label per hop, from
    `HtrEvidenceChainViewModel`. Empty when the target is not something that chain starts from."""


class RevisionRow(BaseModel):
    """One recorded status transition on a finding. Append-only history, rendered in order."""

    model_config = ConfigDict(frozen=True)

    revision_id: str
    revised_at: str
    from_status: str
    from_status_label: str
    to_status: str
    to_status_label: str
    actor: str
    reasoning: str
    evidence: tuple[EvidenceLink, ...] = ()
    superseding_finding_id: str | None = None
    contradicting_finding_id: str | None = None


class ContradictionRow(BaseModel):
    """One `ContradictoryEvidence` entry on a finding.

    Rendered *alongside* the claim it disputes, never in place of it: nothing is removed from a
    disputed finding, so a UI that showed only the contradiction would misrepresent the record.
    """

    model_config = ConfigDict(frozen=True)

    contradiction_id: str
    source_kind: str
    source_id: str
    description: str
    recorded_by: str
    recorded_at: str
    evidence: tuple[EvidenceLink, ...] = ()


class ObservationRow(BaseModel):
    """One `ResearchObservation`, with its scope and its evidence links intact."""

    model_config = ConfigDict(frozen=True)

    observation_id: str
    observation_type: str
    observation_type_label: str
    title: str
    description: str
    unverified_hypothesis: str | None = None
    """Held in its own field all the way to the UI, exactly as on the entity. Everything in
    `description` is quoted from a durable record; this is what somebody thinks might explain it, and
    a page must be able to render them differently without parsing prose."""
    scope_description: str
    """`ResearchScope.describe()`, verbatim. Not re-derived: the domain owns this sentence and it
    always states the sample size."""
    sample_size: int
    unit_of_analysis: str
    project_id: str | None = None
    """Resolved through `dataset_id -> Dataset.project_id`. An observation carries no project id of
    its own -- there is no such field -- so this is a traversal, and `None` when the dataset is not in
    this projection."""
    dataset_id: str | None = None
    dataset_version_id: str | None = None
    collection_ids: tuple[str, ...] = ()
    """Resolved through `dataset_version_id -> DatasetVersion.collection_ids`."""
    experiment_id: str
    experiment_version_id: str
    experiment_run_ids: tuple[str, ...] = ()
    method_ids: tuple[str, ...] = ()
    method_labels: tuple[str, ...] = ()
    model_version_ids: tuple[str, ...] = ()
    method_run_ids: tuple[str, ...] = ()
    metric_result_ids: tuple[str, ...] = ()
    telemetry_event_ids: tuple[str, ...] = ()
    covered_unit_ids: tuple[str, ...] = ()
    normalization_profile: str | None = None
    affected_method: str | None = None
    affected_method_label: str | None = None
    affected_model_version: str | None = None
    affected_document_or_segment_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    observation_confidence: str
    observation_confidence_label: str
    review_status: str
    review_status_label: str
    author_or_source_component: str
    creation_timestamp: str
    evidence: tuple[EvidenceLink, ...] = ()
    raised_question_ids: tuple[str, ...] = ()
    """Questions this observation provoked. The feedback edge, forwards."""


class FindingRow(BaseModel):
    """One `ResearchFinding`, with its full history, contradictions and evidence links."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    statement: str
    review_status: str
    review_status_label: str
    confidence_level: str
    confidence_label: str
    scope_description: str
    sample_size: int
    unit_of_analysis: str
    research_question: str | None = None
    hypothesis_relationship: str | None = None
    hypothesis_relationship_label: str | None = None
    supporting_observation_ids: tuple[str, ...] = ()
    observation_types: tuple[str, ...] = ()
    """The `ObservationType`s of this finding's supporting observations, resolved from the store. A
    finding has no type of its own -- it is a claim, not a kind of fact -- so filtering findings by
    observation type has to traverse `supporting_observations`, and this field is that traversal made
    visible rather than recomputed inside the filter."""
    supporting_metric_ids: tuple[str, ...] = ()
    supporting_experiment_ids: tuple[str, ...] = ()
    experiment_id: str
    experiment_version_id: str
    experiment_run_ids: tuple[str, ...] = ()
    project_id: str | None = None
    affected_datasets: tuple[str, ...] = ()
    affected_dataset_versions: tuple[str, ...] = ()
    collection_ids: tuple[str, ...] = ()
    affected_methods: tuple[str, ...] = ()
    affected_method_labels: tuple[str, ...] = ()
    affected_model_versions: tuple[str, ...] = ()
    affected_document_types: tuple[str, ...] = ()
    affected_handwriting_periods: tuple[str, ...] = ()
    transcription_convention: str | None = None
    limitations: tuple[str, ...] = ()
    """Non-empty on the entity by construction, and carried through in full. On an N=1 run the
    limitations are most of the content, so a row that truncated them would misrepresent the claim."""
    author: str
    reviewer: str | None = None
    creation_date: str
    superseded_by: str | None = None
    revisions: tuple[RevisionRow, ...] = ()
    contradictions: tuple[ContradictionRow, ...] = ()
    contradicted_finding_ids: tuple[str, ...] = ()
    """Findings whose own `contradictory_evidence` names *this* one -- the other direction of the
    relationship, from `HtrResearchStore.findings_contradicting`. Both are stored independently; this
    row shows both because neither is derivable from the other by deletion."""
    evidence: tuple[EvidenceLink, ...] = ()
    """Evidence offered for this finding's *transitions* and *contradictions*. The measurement
    evidence lives on the supporting observations, which the finding names rather than duplicating --
    so this list is empty for a `Candidate` finding, correctly."""
    reproduced: bool = False
    """`True` only when `revision_history` records a `Provisionally supported -> Supported` transition
    whose evidence names an experiment run this finding's scope does not cover. Derived, never
    asserted."""
    reproducing_run_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    """Union of the tags on this finding's supporting observations. A `ResearchFinding` has no `tags`
    field; this is a traversal, so a tag filter can reach findings at all rather than silently
    matching none."""
    raised_question_ids: tuple[str, ...] = ()


class HypothesisRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    statement: str
    falsification_criterion: str
    author: str
    created_at: str


class ResearchQuestionRow(BaseModel):
    """One `ResearchQuestion` and the draft, if any, that came out of it."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    statement: str
    motivation: str
    status: str
    status_label: str
    originating_observation_id: str | None = None
    originating_finding_id: str | None = None
    originating_contradiction_id: str | None = None
    originating_summary: str | None = None
    """The originating observation's title or finding's statement, resolved from this projection so a
    reader sees what provoked the question without navigating away. `None` when the originating record
    is not in this projection."""
    hypotheses: tuple[HypothesisRow, ...] = ()
    created_by: str
    created_at: str
    created_experiment_id: str | None = None
    created_experiment_version_id: str | None = None
    created_experiment_name: str | None = None
    drafted_experiment_has_runs: bool = False
    """Whether the drafted experiment has actually been executed. `False` means drafted-only, which is
    the state of the one real drafted experiment in this repository -- and a UI must be able to say so
    rather than listing it beside experiments that produced results."""
    answered_by_finding_id: str | None = None
    evidence: tuple[EvidenceLink, ...] = ()
    """Links back to the originating records, as typed references, so the question is navigable to its
    provocation the same way an observation is navigable to its measurements."""


class FailurePatternRow(BaseModel):
    """One observation type grouped across the observations that share it.

    **Not a claim that anything recurs.** `occurrence_count` and `distinct_experiment_run_count` are
    counts of what is recorded, and `recurrence_established` is `False` unless the group genuinely
    spans more than one occurrence. On this repository's data every group is a single occurrence in a
    single run, which is exactly why the baseline's SATRN observation is typed `model_limitation`
    rather than `recurring_recognition_failure`.
    """

    model_config = ConfigDict(frozen=True)

    observation_type: str
    observation_type_label: str
    occurrence_count: int
    distinct_experiment_run_count: int
    distinct_method_count: int
    recurrence_established: bool
    observation_ids: tuple[str, ...] = ()
    affected_methods: tuple[str, ...] = ()
    affected_method_labels: tuple[str, ...] = ()
    shared_tags: tuple[str, ...] = ()
    """Tags present on *every* observation in the group -- an intersection, not a union, because a tag
    only characterises a pattern if it holds across it."""
    total_sample_size: int
    """Sum of the groups' scopes' derived sample sizes. On this data it equals the occurrence count,
    and a group of five one-crop observations must not read as a 5-crop study."""


class KnowledgeFilter(BaseModel):
    """The real filter parameters, as one frozen value.

    Every field narrows an actual stored relationship -- none is cosmetic. `project_id` and
    `collection_id` are traversals rather than fields on the knowledge entities (an observation names a
    dataset version; a project and a collection are reached through it), and are resolved against the
    store rather than matched textually.

    `tag` reaches findings through their supporting observations, since `ResearchFinding` has no `tags`
    field; a tag filter that silently matched no finding would look like "no finding is tagged that"
    rather than "findings are not tagged".
    """

    model_config = ConfigDict(frozen=True)

    project_id: str | None = None
    dataset_id: str | None = None
    dataset_version_id: str | None = None
    collection_id: str | None = None
    experiment_id: str | None = None
    experiment_version_id: str | None = None
    experiment_run_id: str | None = None
    method_id: str | None = None
    model_version_id: str | None = None
    observation_type: str | None = None
    status: str | None = None
    """A `FindingStatus` value for findings, or an `ObservationReviewStatus` value for observations.
    One field because a UI has one status control; each collection matches it against its own enum, and
    a value belonging to the other enum simply matches nothing rather than raising -- an empty result
    for "no finding is Accepted" is the honest answer, since findings have no such status."""
    confidence: str | None = None
    """`low`/`moderate`/`high`, matched against `observation_confidence` or `confidence_level`."""
    tag: str | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            value is not None for value in self.model_dump().values()
        )


class ResearchKnowledgeSnapshot(BaseModel):
    """Everything the Research Knowledge page renders, in one frozen value."""

    model_config = ConfigDict(frozen=True)

    recent_observations: tuple[ObservationRow, ...] = ()
    comparison_boundary_observations: tuple[ObservationRow, ...] = ()
    candidate_findings: tuple[FindingRow, ...] = ()
    findings_under_review: tuple[FindingRow, ...] = ()
    provisionally_supported_findings: tuple[FindingRow, ...] = ()
    """Beyond the follow-up's list, and the reason it is here rather than folded into
    `supported_findings`: the one finding this repository has taken through review reached
    `Provisionally supported` and nothing further, because `Supported` requires reproduction in another
    run. Merging the two would either hide the real reviewed finding or misreport it as reproduced."""
    supported_findings: tuple[FindingRow, ...] = ()
    disputed_findings: tuple[FindingRow, ...] = ()
    superseded_findings: tuple[FindingRow, ...] = ()
    rejected_findings: tuple[FindingRow, ...] = ()
    reproduced_findings: tuple[FindingRow, ...] = ()
    unresolved_research_questions: tuple[ResearchQuestionRow, ...] = ()
    answered_research_questions: tuple[ResearchQuestionRow, ...] = ()
    recurring_failure_patterns: tuple[FailurePatternRow, ...] = ()
    observation_count: int = 0
    finding_count: int = 0
    research_question_count: int = 0
    filtered: bool = False
    """Whether a filter narrowed these collections. A page must say so: an empty table under an active
    filter and an empty table over an empty store are different facts."""
    empty_field_notes: tuple[tuple[str, str], ...] = ()
    """`(field_name, reason)` for every collection above that is empty, from `EMPTY_FIELD_NOTES`. A
    tuple of pairs rather than a dict so the model stays frozen-comparable and the order is stable."""


class ResearchKnowledgeViewModel:
    """Read surface over one `HtrResearchStore`'s research-knowledge layers.

    Read-only by construction: it has no method that registers, transitions, or advances anything.
    Raising a question and drafting an experiment are `htr/knowledge/questions.py`'s and the durable
    store's business, and a ViewModel that could promote a finding would be a second, un-audited path
    past `lifecycle.py::transition_finding_status`.
    """

    def __init__(
        self,
        store: HtrResearchStore,
        *,
        evidence_chain: HtrEvidenceChainViewModel | None = None,
    ) -> None:
        self._store = store
        self._chain = evidence_chain or HtrEvidenceChainViewModel(store)
        """The existing evidence-chain ViewModel, reused rather than reimplemented -- it already walks
        an id to a human-readable breadcrumb and already surfaces a dangling hop instead of hiding it.
        Injectable so a caller can share one instance rather than building a second over the same
        store."""

    # -- The snapshot -----------------------------------------------------------------------------

    def snapshot(
        self,
        *,
        filter: KnowledgeFilter | None = None,
        recent_limit: int = DEFAULT_RECENT_LIMIT,
    ) -> ResearchKnowledgeSnapshot:
        """Everything the page needs, under one optional filter."""
        active = filter or KnowledgeFilter()
        observations = self.observations(filter=active)
        findings = self.findings(filter=active)
        questions = self.research_questions(filter=active)

        by_status: dict[FindingStatus, tuple[FindingRow, ...]] = {
            status: tuple(row for row in findings if row.review_status == status.value)
            for status in FindingStatus
        }
        fields: dict[str, tuple] = {
            "recent_observations": observations[:recent_limit],
            "comparison_boundary_observations": tuple(
                row
                for row in observations
                if row.observation_type
                == ObservationType.EXPERIMENT_VALIDITY_BOUNDARY.value
            ),
            "candidate_findings": by_status[FindingStatus.CANDIDATE]
            + by_status[FindingStatus.DRAFT],
            "findings_under_review": by_status[FindingStatus.UNDER_REVIEW],
            "provisionally_supported_findings": by_status[
                FindingStatus.PROVISIONALLY_SUPPORTED
            ],
            "supported_findings": by_status[FindingStatus.SUPPORTED],
            "disputed_findings": by_status[FindingStatus.DISPUTED],
            "superseded_findings": by_status[FindingStatus.SUPERSEDED],
            "rejected_findings": by_status[FindingStatus.REJECTED],
            "reproduced_findings": tuple(row for row in findings if row.reproduced),
            "unresolved_research_questions": tuple(
                row
                for row in questions
                if row.status != ResearchQuestionStatus.ANSWERED.value
            ),
            "answered_research_questions": tuple(
                row
                for row in questions
                if row.status == ResearchQuestionStatus.ANSWERED.value
            ),
            "recurring_failure_patterns": self.failure_patterns(filter=active),
        }
        return ResearchKnowledgeSnapshot(
            **fields,
            observation_count=len(observations),
            finding_count=len(findings),
            research_question_count=len(questions),
            filtered=not active.is_empty,
            empty_field_notes=tuple(
                (name, EMPTY_FIELD_NOTES[name])
                for name, rows in fields.items()
                if not rows and name in EMPTY_FIELD_NOTES
            ),
        )

    # -- Filtered queries -------------------------------------------------------------------------

    def observations(self, *, filter: KnowledgeFilter | None = None) -> tuple[ObservationRow, ...]:
        """Every recorded observation matching `filter`, newest first."""
        active = filter or KnowledgeFilter()
        rows = [
            self.observation_row(observation)
            for observation in self._store.research_observations()
        ]
        rows = [row for row in rows if self._observation_matches(row, active)]
        return tuple(
            sorted(rows, key=lambda row: (row.creation_timestamp, row.observation_id), reverse=True)
        )

    def findings(self, *, filter: KnowledgeFilter | None = None) -> tuple[FindingRow, ...]:
        """Every recorded finding matching `filter`, newest first."""
        active = filter or KnowledgeFilter()
        rows = [self.finding_row(finding) for finding in self._store.findings()]
        rows = [row for row in rows if self._finding_matches(row, active)]
        return tuple(
            sorted(rows, key=lambda row: (row.creation_date, row.finding_id), reverse=True)
        )

    def research_questions(
        self, *, filter: KnowledgeFilter | None = None
    ) -> tuple[ResearchQuestionRow, ...]:
        """Every raised research question matching `filter`, newest first.

        Legitimately empty in a deployment where nobody has raised one. The only filter fields that
        apply are `status` (against `ResearchQuestionStatus`) and, through the originating record,
        `experiment_id`; the rest describe scopes a question does not have -- a question is precisely
        the thing that has *no* scope yet, which is why an experiment gets drafted from it.
        """
        active = filter or KnowledgeFilter()
        rows = [
            self.research_question_row(question)
            for question in self._store.research_questions()
        ]
        if active.status is not None:
            rows = [row for row in rows if row.status == active.status]
        if active.experiment_id is not None:
            rows = [row for row in rows if row.created_experiment_id == active.experiment_id]
        return tuple(
            sorted(rows, key=lambda row: (row.created_at, row.question_id), reverse=True)
        )

    def failure_patterns(
        self, *, filter: KnowledgeFilter | None = None
    ) -> tuple[FailurePatternRow, ...]:
        """Failure-type observations grouped by `observation_type`, with recurrence *measured*.

        The derivation, in full: take the filtered observations, keep those whose type is in
        `FAILURE_PATTERN_OBSERVATION_TYPES`, group by that type, and for each group count the
        observations, the distinct `source_experiment_run_id`s and the distinct affected methods.
        `recurrence_established` is `True` only when more than one observation is in the group -- a
        single observation is a single occurrence however it is typed, and a "recurring failure" list
        that said otherwise would manufacture the recurrence the enum is careful not to assert.

        Groups are ordered by occurrence count descending, so the most-recorded pattern is first.
        """
        rows = self.observations(filter=filter)
        grouped: dict[str, list[ObservationRow]] = defaultdict(list)
        failure_values = {member.value for member in FAILURE_PATTERN_OBSERVATION_TYPES}
        for row in rows:
            if row.observation_type in failure_values:
                grouped[row.observation_type].append(row)

        patterns: list[FailurePatternRow] = []
        for observation_type, group in grouped.items():
            methods = tuple(
                sorted({row.affected_method for row in group if row.affected_method})
            )
            tag_sets = [set(row.tags) for row in group]
            shared = set.intersection(*tag_sets) if tag_sets else set()
            patterns.append(
                FailurePatternRow(
                    observation_type=observation_type,
                    observation_type_label=observation_type_label(observation_type),
                    occurrence_count=len(group),
                    distinct_experiment_run_count=len(
                        {run_id for row in group for run_id in row.experiment_run_ids}
                    ),
                    distinct_method_count=len(methods),
                    recurrence_established=len(group) > 1,
                    observation_ids=tuple(row.observation_id for row in group),
                    affected_methods=methods,
                    affected_method_labels=tuple(method_label(m) for m in methods),
                    shared_tags=tuple(sorted(shared)),
                    total_sample_size=sum(row.sample_size for row in group),
                )
            )
        return tuple(
            sorted(patterns, key=lambda row: (-row.occurrence_count, row.observation_type))
        )

    # -- Single-entity rows -----------------------------------------------------------------------

    def observation_row(self, observation: ResearchObservation) -> ObservationRow:
        """One observation as view state, with every structured id a UI needs to link its evidence."""
        scope = observation.scope
        return ObservationRow(
            observation_id=observation.observation_id,
            observation_type=observation.observation_type.value,
            observation_type_label=observation_type_label(observation.observation_type),
            title=observation.title,
            description=observation.description,
            unverified_hypothesis=observation.unverified_hypothesis,
            scope_description=scope.describe(),
            sample_size=scope.sample_size,
            unit_of_analysis=scope.unit_of_analysis.value,
            project_id=self._project_for_dataset(scope.dataset_id),
            dataset_id=scope.dataset_id,
            dataset_version_id=scope.dataset_version_id,
            collection_ids=self._collections_for_dataset_version(scope.dataset_version_id),
            experiment_id=scope.experiment_id,
            experiment_version_id=scope.experiment_version_id,
            experiment_run_ids=scope.experiment_run_ids,
            method_ids=scope.method_ids,
            method_labels=tuple(method_label(m) for m in scope.method_ids),
            model_version_ids=scope.model_version_ids,
            method_run_ids=scope.method_run_ids,
            metric_result_ids=observation.evidence_ids(EvidenceReferenceKind.METRIC_RESULT),
            telemetry_event_ids=observation.evidence_ids(
                EvidenceReferenceKind.TELEMETRY_EVENT
            ),
            covered_unit_ids=scope.covered_unit_ids,
            normalization_profile=scope.normalization_profile,
            affected_method=observation.affected_method,
            affected_method_label=(
                method_label(observation.affected_method)
                if observation.affected_method is not None
                else None
            ),
            affected_model_version=observation.affected_model_version,
            affected_document_or_segment_ids=observation.affected_document_or_segment_ids,
            tags=observation.tags,
            observation_confidence=observation.observation_confidence.value,
            observation_confidence_label=knowledge_confidence_label(
                observation.observation_confidence
            ),
            review_status=observation.review_status.value,
            review_status_label=observation_review_status_label(observation.review_status),
            author_or_source_component=observation.author_or_source_component,
            creation_timestamp=observation.creation_timestamp,
            evidence=tuple(
                self.resolve_evidence_reference(ref)
                for ref in observation.supporting_evidence
            ),
            raised_question_ids=tuple(
                question.question_id
                for question in self._store.research_questions(
                    originating_observation_id=observation.observation_id
                )
            ),
        )

    def finding_row(self, finding: ResearchFinding) -> FindingRow:
        """One finding as view state, with its complete history and both sides of any contradiction."""
        scope = finding.scope
        reproducing = self._reproducing_run_ids(finding)
        supporting = tuple(
            self._store.research_observation(observation_id)
            for observation_id in finding.supporting_observations
        )
        return FindingRow(
            finding_id=finding.finding_id,
            statement=finding.statement,
            review_status=finding.review_status.value,
            review_status_label=finding_status_label(finding.review_status),
            confidence_level=finding.confidence_level.value,
            confidence_label=knowledge_confidence_label(finding.confidence_level),
            scope_description=scope.describe(),
            sample_size=scope.sample_size,
            unit_of_analysis=scope.unit_of_analysis.value,
            research_question=finding.research_question,
            hypothesis_relationship=(
                finding.hypothesis_relationship.value
                if finding.hypothesis_relationship is not None
                else None
            ),
            hypothesis_relationship_label=(
                hypothesis_relationship_label(finding.hypothesis_relationship)
                if finding.hypothesis_relationship is not None
                else None
            ),
            supporting_observation_ids=finding.supporting_observations,
            observation_types=tuple(
                sorted(
                    {
                        observation.observation_type.value
                        for observation in supporting
                        if observation is not None
                    }
                )
            ),
            supporting_metric_ids=finding.supporting_metrics,
            supporting_experiment_ids=finding.supporting_experiments,
            experiment_id=scope.experiment_id,
            experiment_version_id=scope.experiment_version_id,
            experiment_run_ids=scope.experiment_run_ids,
            project_id=self._project_for_dataset(scope.dataset_id),
            affected_datasets=finding.affected_datasets,
            affected_dataset_versions=finding.affected_dataset_versions,
            collection_ids=self._collections_for_dataset_version(scope.dataset_version_id),
            affected_methods=finding.affected_methods,
            affected_method_labels=tuple(method_label(m) for m in finding.affected_methods),
            affected_model_versions=finding.affected_model_versions,
            affected_document_types=finding.affected_document_types,
            affected_handwriting_periods=finding.affected_handwriting_periods,
            transcription_convention=finding.transcription_convention,
            limitations=finding.limitations,
            author=finding.author,
            reviewer=finding.reviewer,
            creation_date=finding.creation_date,
            superseded_by=finding.superseded_by,
            revisions=tuple(self._revision_row(revision) for revision in finding.revision_history),
            contradictions=tuple(
                self._contradiction_row(contradiction)
                for contradiction in finding.contradictory_evidence
            ),
            contradicted_finding_ids=tuple(
                other.finding_id
                for other in self._store.findings_contradicting(finding.finding_id)
            ),
            evidence=tuple(
                link
                for revision in finding.revision_history
                for link in (
                    self.resolve_evidence_reference(ref) for ref in revision.evidence_refs
                )
            )
            + tuple(
                link
                for contradiction in finding.contradictory_evidence
                for link in (
                    self.resolve_evidence_reference(ref) for ref in contradiction.evidence_refs
                )
            ),
            reproduced=bool(reproducing),
            reproducing_run_ids=reproducing,
            tags=tuple(
                sorted(
                    {
                        tag
                        for observation in supporting
                        if observation is not None
                        for tag in observation.tags
                    }
                )
            ),
            raised_question_ids=tuple(
                question.question_id
                for question in self._store.research_questions(
                    originating_finding_id=finding.finding_id
                )
            ),
        )

    def research_question_row(self, question: ResearchQuestion) -> ResearchQuestionRow:
        """One research question as view state, including whether its drafted experiment ever ran."""
        drafted = (
            self._store.experiment(question.created_experiment_id)
            if question.created_experiment_id is not None
            else None
        )
        runs = (
            self._store.experiment_runs(
                experiment_version_id=question.created_experiment_version_id
            )
            if question.created_experiment_version_id is not None
            else ()
        )
        return ResearchQuestionRow(
            question_id=question.question_id,
            statement=question.statement,
            motivation=question.motivation,
            status=question.status.value,
            status_label=research_question_status_label(question.status),
            originating_observation_id=question.originating_observation_id,
            originating_finding_id=question.originating_finding_id,
            originating_contradiction_id=question.originating_contradiction_id,
            originating_summary=self._originating_summary(question),
            hypotheses=tuple(
                HypothesisRow(
                    hypothesis_id=hypothesis.hypothesis_id,
                    statement=hypothesis.statement,
                    falsification_criterion=hypothesis.falsification_criterion,
                    author=hypothesis.author,
                    created_at=hypothesis.created_at,
                )
                for hypothesis in question.hypotheses
            ),
            created_by=question.created_by,
            created_at=question.created_at,
            created_experiment_id=question.created_experiment_id,
            created_experiment_version_id=question.created_experiment_version_id,
            created_experiment_name=drafted.name if drafted is not None else None,
            drafted_experiment_has_runs=bool(runs),
            answered_by_finding_id=question.answered_by_finding_id,
            evidence=tuple(
                self.resolve_evidence_reference(ref)
                for ref in self._question_evidence_refs(question)
            ),
        )

    # -- Evidence navigation ----------------------------------------------------------------------

    def resolve_evidence_reference(self, ref: EvidenceReference) -> EvidenceLink:
        """One `EvidenceReference` resolved to a human-readable label and a navigable target.

        The support helper the follow-up's evidence-navigation requirement asks for. Reuses
        `htr_evidence_viewmodel.py::HtrEvidenceChainViewModel`, which already turns an id into a
        breadcrumb, rather than walking the corpus a second time here.

        **A reference this projection cannot produce comes back `resolved=False` with the reason**,
        never as a plausible-looking label over nothing. There are two honest reasons, and `detail`
        distinguishes them: the id is dangling (nothing with that id was ever registered), or the
        record is durable but this read model does not index it by id -- which is true of every
        `telemetry_event` reference, because `HtrResearchStore` has no event bucket, and of every
        `evidence_record` reference, because `Evidence` deliberately belongs to the retained substrate
        rather than to this store (`docs/htr-domain-design.md` §2).
        """
        target_page = _TARGET_PAGE_BY_EVIDENCE_KIND[ref.kind]
        base: dict[str, object] = {
            "kind": ref.kind.value,
            "kind_label": evidence_reference_kind_label(ref.kind),
            "reference_id": ref.reference_id,
            "note": ref.note,
            "stream": ref.stream,
            "target_page": target_page.value if target_page is not None else None,
        }
        resolver = {
            EvidenceReferenceKind.METHOD_RUN: self._resolve_method_run,
            EvidenceReferenceKind.METRIC_RESULT: self._resolve_metric_result,
            EvidenceReferenceKind.RELIABILITY_CLASSIFICATION: self._resolve_failure_record,
            EvidenceReferenceKind.EXPERIMENT_RUN: self._resolve_experiment_run,
            EvidenceReferenceKind.INPUT_CROP: self._resolve_input_crop,
            EvidenceReferenceKind.TEXT_LINE: self._resolve_text_line,
            EvidenceReferenceKind.REPRODUCIBILITY_MANIFEST: self._resolve_manifest,
            EvidenceReferenceKind.RESEARCH_OBSERVATION: self._resolve_observation,
            EvidenceReferenceKind.RESEARCH_FINDING: self._resolve_finding,
        }.get(ref.kind)
        if resolver is None:
            return EvidenceLink(**base, **self._unindexed(ref))
        return EvidenceLink(**base, **resolver(ref.reference_id))

    def evidence_for_observation(self, observation_id: str) -> tuple[EvidenceLink, ...]:
        """Every resolved evidence link for one observation, or `()` when it is not registered."""
        observation = self._store.research_observation(observation_id)
        if observation is None:
            return ()
        return tuple(
            self.resolve_evidence_reference(ref) for ref in observation.supporting_evidence
        )

    # -- Internals: resolvers ---------------------------------------------------------------------

    def _resolve_method_run(self, method_run_id: str) -> dict[str, object]:
        run = self._store.method_run(method_run_id)
        if run is None:
            return self._dangling("method_run", method_run_id)
        return {
            "label": f"{method_label(run.method_id)} run",
            "detail": (
                f"{run.outcome}; model revision {run.model_version_id or 'not recorded'}; "
                f"started {run.started_at}"
            ),
            "target_kind": "method_run",
            "target_id": method_run_id,
            "breadcrumb": self._breadcrumb(method_run_id),
        }

    def _resolve_metric_result(self, metric_result_id: str) -> dict[str, object]:
        result = next(
            (
                candidate
                for candidate in self._store.metric_results()
                if candidate.metric_result_id == metric_result_id
            ),
            None,
        )
        if result is None:
            return self._dangling("metric_result", metric_result_id)
        definition = self._store.metric_definition(result.metric_definition_id)
        name = definition.name if definition is not None else result.metric_definition_id
        version = f" v{definition.version}" if definition is not None else ""
        # The target is the *method run*, not the metric: the evidence chain is walkable from a run,
        # and a metric result on its own has nowhere upward to go.
        return {
            "label": f"{name}{version} = {result.value}",
            "detail": f"computed for method run {short_ref(result.method_run_id, prefix=20)}",
            "target_kind": "method_run",
            "target_id": result.method_run_id,
            "breadcrumb": self._breadcrumb(result.method_run_id),
        }

    def _resolve_failure_record(self, failure_record_id: str) -> dict[str, object]:
        record = next(
            (
                candidate
                for candidate in self._store.failures()
                if candidate.failure_record_id == failure_record_id
            ),
            None,
        )
        if record is None:
            return self._dangling("reliability_classification", failure_record_id)
        return {
            "label": record.category or "Reliability issue",
            "detail": record.reason,
            "target_kind": "method_run",
            "target_id": record.method_run_id,
            "breadcrumb": self._breadcrumb(record.method_run_id),
        }

    def _resolve_experiment_run(self, experiment_run_id: str) -> dict[str, object]:
        run = self._store.experiment_run(experiment_run_id)
        if run is None:
            return self._dangling("experiment_run", experiment_run_id)
        mode = (
            "end-to-end (each method used its own segmentation)"
            if run.is_end_to_end
            else "controlled (every method read the same input crop)"
        )
        return {
            "label": f"Experiment run, {mode}",
            "detail": (
                f"started {run.started_at}"
                + (f", completed {run.completed_at}" if run.completed_at else ", never completed")
            ),
            "target_kind": "experiment_run",
            "target_id": experiment_run_id,
        }

    def _resolve_input_crop(self, crop_id: str) -> dict[str, object]:
        crop = self._store.input_crop(crop_id)
        if crop is None:
            return self._dangling("input_crop", crop_id)
        return {
            "label": f"Input crop {short_ref(crop.hash, prefix=18)}",
            "detail": f"{crop.byte_size} bytes at {crop.storage_path}",
            "target_kind": "input_crop",
            "target_id": crop_id,
        }

    def _resolve_text_line(self, text_line_id: str) -> dict[str, object]:
        line = self._store.text_line(text_line_id)
        if line is None:
            return self._dangling("text_line", text_line_id)
        ground_truth = self._store.ground_truth_for_line(text_line_id)
        return {
            "label": f"Line {line.reading_order_index}",
            "detail": (
                f"ground truth: {ground_truth}"
                if ground_truth is not None
                else "no ground truth recorded for this line"
            ),
            "target_kind": "text_line",
            "target_id": text_line_id,
        }

    def _resolve_manifest(self, manifest_id: str) -> dict[str, object]:
        manifest = next(
            (
                candidate
                for candidate in self._store.manifests()
                if candidate.manifest_id == manifest_id
            ),
            None,
        )
        if manifest is None:
            return self._dangling("reproducibility_manifest", manifest_id)
        return {
            "label": "Reproducibility manifest",
            "detail": (
                f"experiment run {short_ref(manifest.experiment_run_id, prefix=20)}; "
                f"commit {manifest.git_commit or 'not recorded'}"
            ),
            "target_kind": "experiment_run",
            "target_id": manifest.experiment_run_id,
        }

    def _resolve_observation(self, observation_id: str) -> dict[str, object]:
        observation = self._store.research_observation(observation_id)
        if observation is None:
            return self._dangling("research_observation", observation_id)
        return {
            "label": observation.title,
            "detail": (
                f"{observation_type_label(observation.observation_type)}; "
                f"{observation.scope.sample_size} "
                f"{observation.scope.unit_of_analysis.value}(s) in scope"
            ),
            "target_kind": "research_observation",
            "target_id": observation_id,
        }

    def _resolve_finding(self, finding_id: str) -> dict[str, object]:
        finding = self._store.finding(finding_id)
        if finding is None:
            return self._dangling("research_finding", finding_id)
        return {
            "label": finding.statement,
            "detail": finding_status_label(finding.review_status),
            "target_kind": "research_finding",
            "target_id": finding_id,
        }

    @staticmethod
    def _dangling(kind: str, reference_id: str) -> dict[str, object]:
        return {
            "label": short_ref(reference_id, prefix=24),
            "detail": f"Referenced as a {kind} but not registered in this read model",
            "resolved": False,
        }

    @staticmethod
    def _unindexed(ref: EvidenceReference) -> dict[str, object]:
        """A reference that is durable but that this projection does not index by id.

        Not a broken link and not silently omitted: the record exists, in the stream `ref.stream`
        names, and the reason it cannot be resolved here is a property of the read model rather than of
        the evidence. Saying so is the difference between "this evidence is missing" and "this surface
        cannot open it".
        """
        return {
            "label": short_ref(ref.reference_id, prefix=24),
            "detail": (
                ref.note
                or (
                    f"{_NOT_IN_THIS_PROJECTION}"
                    + (f" (stream: {ref.stream})" if ref.stream else "")
                )
            ),
            "resolved": False,
        }

    def _breadcrumb(self, method_run_id: str) -> tuple[str, ...]:
        chain = self._chain.chain_for_method_run(method_run_id)
        return tuple(f"{link.kind}: {link.label}" for link in chain.links)

    # -- Internals: rows -------------------------------------------------------------------------

    def _revision_row(self, revision: FindingRevision) -> RevisionRow:
        return RevisionRow(
            revision_id=revision.revision_id,
            revised_at=revision.revised_at,
            from_status=revision.from_status.value,
            from_status_label=finding_status_label(revision.from_status),
            to_status=revision.to_status.value,
            to_status_label=finding_status_label(revision.to_status),
            actor=revision.actor,
            reasoning=revision.reasoning,
            evidence=tuple(
                self.resolve_evidence_reference(ref) for ref in revision.evidence_refs
            ),
            superseding_finding_id=revision.superseding_finding_id,
            contradicting_finding_id=revision.contradicting_finding_id,
        )

    def _contradiction_row(self, contradiction: ContradictoryEvidence) -> ContradictionRow:
        return ContradictionRow(
            contradiction_id=contradiction.contradiction_id,
            source_kind=contradiction.source_kind.value,
            source_id=contradiction.source_id,
            description=contradiction.description,
            recorded_by=contradiction.recorded_by,
            recorded_at=contradiction.recorded_at,
            evidence=tuple(
                self.resolve_evidence_reference(ref) for ref in contradiction.evidence_refs
            ),
        )

    def _question_evidence_refs(
        self, question: ResearchQuestion
    ) -> tuple[EvidenceReference, ...]:
        refs: list[EvidenceReference] = []
        if question.originating_observation_id is not None:
            refs.append(
                EvidenceReference(
                    kind=EvidenceReferenceKind.RESEARCH_OBSERVATION,
                    reference_id=question.originating_observation_id,
                    note="the observation that raised this question",
                )
            )
        if question.originating_finding_id is not None:
            refs.append(
                EvidenceReference(
                    kind=EvidenceReferenceKind.RESEARCH_FINDING,
                    reference_id=question.originating_finding_id,
                    note="the finding whose stated limitation this question attacks",
                )
            )
        return tuple(refs)

    def _originating_summary(self, question: ResearchQuestion) -> str | None:
        if question.originating_observation_id is not None:
            observation = self._store.research_observation(
                question.originating_observation_id
            )
            if observation is not None:
                return observation.title
        if question.originating_finding_id is not None:
            finding = self._store.finding(question.originating_finding_id)
            if finding is not None:
                return finding.statement
        return None

    @staticmethod
    def _reproducing_run_ids(finding: ResearchFinding) -> tuple[str, ...]:
        """The experiment runs that actually reproduced this finding, from its revision history.

        A run counts only if a `Provisionally supported -> Supported` revision names it as an
        `experiment_run` evidence reference *and* the finding's own scope does not already cover it --
        the same rule `lifecycle.py::_require_reproduction_from_another_run` enforces on the way in.
        Re-deriving it here rather than trusting `review_status == 'Supported'` means a finding that
        somehow reached that status without qualifying evidence is reported as unreproduced, not as
        reproduced on the strength of its label.
        """
        in_scope = set(finding.scope.experiment_run_ids)
        return tuple(
            sorted(
                {
                    ref.reference_id
                    for revision in finding.revision_history
                    if revision.from_status is FindingStatus.PROVISIONALLY_SUPPORTED
                    and revision.to_status is FindingStatus.SUPPORTED
                    for ref in revision.evidence_refs
                    if ref.kind is EvidenceReferenceKind.EXPERIMENT_RUN
                    and ref.reference_id not in in_scope
                }
            )
        )

    # -- Internals: filtering and traversal -------------------------------------------------------

    def _project_for_dataset(self, dataset_id: str | None) -> str | None:
        if dataset_id is None:
            return None
        dataset = self._store.dataset(dataset_id)
        return dataset.project_id if dataset is not None else None

    def _collections_for_dataset_version(
        self, dataset_version_id: str | None
    ) -> tuple[str, ...]:
        if dataset_version_id is None:
            return ()
        version = self._store.dataset_version(dataset_version_id)
        return version.collection_ids if version is not None else ()

    @staticmethod
    def _observation_matches(row: ObservationRow, active: KnowledgeFilter) -> bool:
        checks = (
            (active.project_id, lambda value: row.project_id == value),
            (active.dataset_id, lambda value: row.dataset_id == value),
            (active.dataset_version_id, lambda value: row.dataset_version_id == value),
            (active.collection_id, lambda value: value in row.collection_ids),
            (active.experiment_id, lambda value: row.experiment_id == value),
            (
                active.experiment_version_id,
                lambda value: row.experiment_version_id == value,
            ),
            (active.experiment_run_id, lambda value: value in row.experiment_run_ids),
            (
                active.method_id,
                lambda value: value in row.method_ids or row.affected_method == value,
            ),
            (
                active.model_version_id,
                lambda value: value in row.model_version_ids
                or row.affected_model_version == value,
            ),
            (active.observation_type, lambda value: row.observation_type == value),
            (active.status, lambda value: row.review_status == value),
            (active.confidence, lambda value: row.observation_confidence == value),
            (active.tag, lambda value: value in row.tags),
        )
        return all(value is None or predicate(value) for value, predicate in checks)

    @staticmethod
    def _finding_matches(row: FindingRow, active: KnowledgeFilter) -> bool:
        checks = (
            (active.project_id, lambda value: row.project_id == value),
            (active.dataset_id, lambda value: value in row.affected_datasets),
            (
                active.dataset_version_id,
                lambda value: value in row.affected_dataset_versions,
            ),
            (active.collection_id, lambda value: value in row.collection_ids),
            (active.experiment_id, lambda value: row.experiment_id == value),
            (
                active.experiment_version_id,
                lambda value: row.experiment_version_id == value,
            ),
            (active.experiment_run_id, lambda value: value in row.experiment_run_ids),
            (active.method_id, lambda value: value in row.affected_methods),
            (
                active.model_version_id,
                lambda value: value in row.affected_model_versions,
            ),
            # `observation_type` describes an observation, not a claim. A finding matches when any
            # observation it rests on is of that type -- which is a real relationship
            # (`supporting_observations`), not a field this row invents.
            (active.observation_type, lambda value: value in row.observation_types),
            (active.status, lambda value: row.review_status == value),
            (active.confidence, lambda value: row.confidence_level == value),
            (active.tag, lambda value: value in row.tags),
        )
        return all(value is None or predicate(value) for value, predicate in checks)


__all__ = [
    "DEFAULT_RECENT_LIMIT",
    "EMPTY_FIELD_NOTES",
    "FAILURE_PATTERN_OBSERVATION_TYPES",
    "ContradictionRow",
    "EvidenceLink",
    "FailurePatternRow",
    "FindingRow",
    "HypothesisRow",
    "KnowledgeFilter",
    "ObservationRow",
    "ResearchKnowledgeSnapshot",
    "ResearchKnowledgeViewModel",
    "ResearchQuestionRow",
    "RevisionRow",
]
