"""Research-knowledge entities (event-model doc §1 layers 10-12).

Frozen Pydantic types only. This module's imports are deliberately limited to `pydantic` and
`archivetrust.domain.shared.ids`, exactly as `htr/corpus/models.py` and `htr/experiment/models.py`
are -- that is the condition `docs/architecture/htr-telemetry.md` §5 sets for
`domain/telemetry/events.py` to carry an entity as a **typed embedded object** rather than a
`record: dict`. Adding an import of `review/`, `evaluation/` or `providers/` here would silently
demote these entities to dicts on the wire and invert the stated dependency direction.

**The layering this module exists to enforce** (event-model doc §1):

    a telemetry event is not automatically a research observation;
    a research observation is not automatically a finding;
    a finding is not automatically accepted knowledge.

Each of the three arrows is blocked structurally, not by convention:

1. *event -> observation*: a `ResearchObservation` cannot be constructed without a non-empty
   `supporting_evidence` list of **typed** `EvidenceReference`s, so producing one is always an
   explicit extraction step naming what it read. No constructor takes a `TelemetryEvent`.
2. *observation -> finding*: a `ResearchFinding` cannot be constructed without at least one
   `supporting_observations` entry, and `ResearchFinding.create` refuses any status other than
   `Draft`/`Candidate` -- reinforced by a model validator so bypassing `create` does not help.
3. *finding -> accepted knowledge*: every status past `Candidate` requires an attributable
   `reviewer` and a `FindingRevision` in `revision_history`, both validated on the model itself. The
   only sanctioned way to produce one is `htr/knowledge/lifecycle.py::transition_finding_status`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivetrust.domain.shared.ids import new_id

# -- Enumerations ---------------------------------------------------------------------------------


class ObservationType(str, Enum):
    """The fourteen observation kinds the knowledge follow-up specifies.

    An enum rather than a free string so "what kinds of things have we observed?" is a closed,
    countable question -- and so an observation about *experiment validity* can never be silently
    filed as one about *method performance*, which is precisely the confusion the Transkribus
    comparability observation exists to prevent.
    """

    SUCCESSFUL_RECOGNITION_BEHAVIOR = "successful_recognition_behavior"
    RECURRING_RECOGNITION_FAILURE = "recurring_recognition_failure"
    SEGMENTATION_PROBLEM = "segmentation_problem"
    HANDWRITING_FEATURE = "handwriting_feature"
    DOCUMENT_LAYOUT_FEATURE = "document_layout_feature"
    MODEL_LIMITATION = "model_limitation"
    CONFIDENCE_ANOMALY = "confidence_anomaly"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    ENVIRONMENT_ISSUE = "environment_issue"
    REVIEWER_OBSERVATION = "reviewer_observation"
    POSSIBLE_HYPOTHESIS = "possible_hypothesis"
    UNEXPECTED_METHOD_DISAGREEMENT = "unexpected_method_disagreement"
    EXPERIMENT_VALIDITY_BOUNDARY = "experiment_validity_boundary"
    """An observation about what a comparison *can and cannot* measure. Deliberately not a
    method-performance type: recording that a fixture is outside a controlled set is a statement
    about the experiment, never about the method that ran on it."""
    REPRODUCIBILITY_ANOMALY = "reproducibility_anomaly"


class EvidenceReferenceKind(str, Enum):
    """What kind of durable record an `EvidenceReference` points at.

    Typed rather than a bare id string because "this observation rests on
    `metric_result_40f23f77...`" and "this observation rests on `event_69f2a7b3...`" are different
    claims about different layers, and a reader must not have to pattern-match an id prefix to tell
    which. Every member names a record that exists durably in a telemetry stream.
    """

    TELEMETRY_EVENT = "telemetry_event"
    EXPERIMENT_RUN = "experiment_run"
    METHOD_RUN = "method_run"
    METRIC_RESULT = "metric_result"
    RELIABILITY_CLASSIFICATION = "reliability_classification"
    """A `FailureRecord` produced by `htr/evaluation/failures.py::classify_reliability`."""
    EVIDENCE_RECORD = "evidence_record"
    """An `Evidence.evidence_id` in the retained content-addressed substrate."""
    INPUT_CROP = "input_crop"
    TEXT_LINE = "text_line"
    GROUND_TRUTH_TEXT = "ground_truth_text"
    RESEARCH_OBSERVATION = "research_observation"
    RESEARCH_FINDING = "research_finding"
    REPRODUCIBILITY_MANIFEST = "reproducibility_manifest"
    EXTERNAL_DOCUMENT = "external_document"
    """A committed repository artifact that is not a telemetry record -- an adapter README, a
    fixture. The one member whose target is not in an event stream, which is why it is named
    separately instead of being passed off as a `telemetry_event`."""


class ScopeUnit(str, Enum):
    """The unit of analysis a `ResearchScope` enumerates instances of."""

    LINE_CROP = "line_crop"
    TEXT_LINE = "text_line"
    PAGE = "page"
    DOCUMENT = "document"
    METHOD_RUN = "method_run"
    EXPERIMENT_RUN = "experiment_run"


class ObservationConfidence(str, Enum):
    """**The observer's epistemic confidence in the observation itself.**

    Explicitly *not* a recognition confidence. `Evidence.provider_confidence`,
    `RecognitionResult.confidence` and the `RecognitionMetrics` family all describe how sure a
    *model* was about a transcription; this describes how sure a *researcher or extraction
    component* is that the observation it wrote down is a real property of the recorded run. The
    field that carries it is named `observation_confidence` for the same reason -- a bare
    `confidence` field on an entity that sits one layer above recognition output would be read as the
    model's number by every reader who has been looking at adapter code.

    Deliberately ordinal-with-three-values rather than a float: an observation's epistemic standing
    is not measured, and a `0.72` would imply a calibration that does not exist.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ObservationReviewStatus(str, Enum):
    """Whether a human has looked at an observation. An observation is *not* a finding, so it has no
    `Supported`/`Disputed` states -- accepting an observation means "yes, this is really what the
    records show", never "yes, this generalizes"."""

    UNREVIEWED = "Unreviewed"
    UNDER_REVIEW = "Under review"
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"


class FindingStatus(str, Enum):
    """The eight statuses of the knowledge lifecycle, exactly as specified.

    Transitions between them are not free: see `htr/knowledge/lifecycle.py`, which owns the state
    machine. The values carry their display spelling (`"Under review"`, `"Provisionally supported"`)
    because these are read by humans in a research context and abbreviating them to
    `under_review` would create a second vocabulary to translate between.
    """

    DRAFT = "Draft"
    CANDIDATE = "Candidate"
    UNDER_REVIEW = "Under review"
    PROVISIONALLY_SUPPORTED = "Provisionally supported"
    SUPPORTED = "Supported"
    DISPUTED = "Disputed"
    SUPERSEDED = "Superseded"
    REJECTED = "Rejected"


INITIAL_FINDING_STATUSES = frozenset({FindingStatus.DRAFT, FindingStatus.CANDIDATE})
"""The only two statuses a finding may be *constructed* in. Every other status is reachable only
through a recorded transition, so "this finding is Supported" always has a revision explaining who
said so and why."""

REVIEWED_FINDING_STATUSES = frozenset(FindingStatus) - INITIAL_FINDING_STATUSES
"""Statuses that require an attributable `reviewer` and at least one `FindingRevision`."""


class FindingConfidence(str, Enum):
    """How much weight the finding's author places on the claim, given its scope and limitations.
    Distinct from `ObservationConfidence` (which is about a single recorded fact) and from any
    model's recognition confidence."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class HypothesisRelationship(str, Enum):
    """How a finding relates to the experiment's stated hypothesis. `NO_HYPOTHESIS_ASSERTED` exists
    because the real baseline experiment's `hypothesis` field says exactly that -- "No directional
    hypothesis is asserted about which method performs better" -- and forcing it into
    `supports`/`contradicts` would invent a hypothesis the experiment declined to make."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    UNTESTED = "untested"
    NO_HYPOTHESIS_ASSERTED = "no_hypothesis_asserted"


class ContradictionSourceKind(str, Enum):
    """What kind of record contradicts a finding."""

    RESEARCH_FINDING = "research_finding"
    RESEARCH_OBSERVATION = "research_observation"
    METRIC_RESULT = "metric_result"
    TELEMETRY_EVENT = "telemetry_event"
    EXTERNAL_DOCUMENT = "external_document"


# -- Value objects --------------------------------------------------------------------------------

_WILDCARD_PLACEHOLDERS = frozenset({"", "*", "all", "any", "n/a", "various", "-"})
"""Strings that would turn an enumerated scope back into an unbounded one. Refused by
`ResearchScope`, because a scope whose covered units are `("*",)` is a general claim wearing a
scope's clothes."""


class EvidenceReference(BaseModel):
    """One typed pointer from a knowledge record into a durable record that supports it.

    The follow-up requires supporting evidence be "a list of typed evidence references -- event ids,
    method_run ids, metric_result ids -- not free-text notes". `kind` + `reference_id` are that
    pointer and are both required. `note` is optional and is *annotation on* the reference, never a
    substitute for it: there is no way to record a reference that has no resolvable id.
    """

    model_config = ConfigDict(frozen=True)

    kind: EvidenceReferenceKind
    reference_id: str = Field(min_length=1)
    stream: str | None = None
    """Which durable stream the record lives in, when it is not this record's own. The baseline
    knowledge records point at events in `htr_research_events.jsonl` while themselves living in
    `htr_knowledge_events.jsonl`, and a reference that did not say so would be unresolvable."""
    note: str | None = None
    """Human-readable annotation. Never load-bearing -- `kind`/`reference_id` are."""

    @model_validator(mode="after")
    def _reference_id_is_not_a_placeholder(self) -> "EvidenceReference":
        if self.reference_id.strip().casefold() in _WILDCARD_PLACEHOLDERS:
            raise ValueError(
                f"EvidenceReference.reference_id {self.reference_id!r} is a placeholder, not a "
                "resolvable durable-record id"
            )
        return self


class ResearchScope(BaseModel):
    """**The structured scope every observation and finding is bound to.** Not free text.

    The follow-up insists repeatedly that scope must never be implicit, and that the model must make
    a scoped result structurally unreadable as a general claim. Four mechanisms do that here, and
    none of them is a naming convention:

    1. **`experiment_id`, `experiment_version_id` and at least one `experiment_run_id` are
       required.** There is no way to describe a result without naming the exact configuration that
       produced it.
    2. **`covered_unit_ids` must be non-empty and must enumerate real ids.** Placeholders (`"*"`,
       `"all"`, `""`) are refused. A scope can only ever cover units it can name.
    3. **`sample_size` is a derived property, not a field.** It is `len(covered_unit_ids)`, so it
       cannot be inflated, rounded, or left out. `N=1` shows up in every rendering of a one-crop
       scope whether the author remembered to mention it or not.
    4. **Every named method must carry its exact model version.** `method_ids` and
       `model_version_ids` are validated to the same length and are paired positionally, so
       "Florence-2 beat SATRN" cannot be scoped without stating *which checkpoints*.

    `describe()` renders the scope as one sentence, which is what documentation and UI should quote
    rather than re-deriving prose from the fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    """`extra="forbid"` is load-bearing here, not tidiness. Pydantic's default is to *ignore* unknown
    keys, so `ResearchScope(..., sample_size=500)` would otherwise construct successfully and silently
    drop the inflated figure -- a caller could believe they had set it. Forbidding extras makes
    mechanism 3 ("`sample_size` is derived, not a field") fail loudly instead of quietly, which is
    what `test_sample_size_is_derived_from_covered_units_and_cannot_be_asserted` asserts."""

    experiment_id: str = Field(min_length=1)
    experiment_version_id: str = Field(min_length=1)
    experiment_run_ids: tuple[str, ...] = Field(min_length=1)
    unit_of_analysis: ScopeUnit
    covered_unit_ids: tuple[str, ...] = Field(min_length=1)
    """The concrete instances this scope covers -- crop ids, page ids, method-run ids. The single
    most important field on this type: it is what makes the scope finite."""
    dataset_id: str | None = None
    dataset_version_id: str | None = None
    method_ids: tuple[str, ...] = ()
    model_version_ids: tuple[str, ...] = ()
    """Positionally paired with `method_ids` -- `model_version_ids[i]` is the exact revision
    `method_ids[i]` ran at."""
    method_run_ids: tuple[str, ...] = ()
    normalization_profile: str | None = None
    """Which normalization rules the metrics in scope were computed under. A CER is not comparable
    across normalization profiles, so a scope that omits this is under-specified."""
    ground_truth_ref: str | None = None
    configuration_ref: str | None = None

    @property
    def sample_size(self) -> int:
        """`len(covered_unit_ids)`. Derived, never stored -- see the class docstring, mechanism 3."""
        return len(self.covered_unit_ids)

    @model_validator(mode="after")
    def _scope_is_genuinely_specific(self) -> "ResearchScope":
        for field_name in ("experiment_run_ids", "covered_unit_ids", "method_ids", "method_run_ids"):
            for value in getattr(self, field_name):
                if value.strip().casefold() in _WILDCARD_PLACEHOLDERS:
                    raise ValueError(
                        f"ResearchScope.{field_name} contains the placeholder {value!r}; a scope "
                        "must enumerate real ids, never stand in for 'everything'"
                    )
        if len(self.method_ids) != len(self.model_version_ids):
            raise ValueError(
                "ResearchScope.method_ids and .model_version_ids must be the same length and are "
                f"paired positionally: got {len(self.method_ids)} method(s) and "
                f"{len(self.model_version_ids)} model version(s). A method named in a scope without "
                "its exact model revision makes the scope unreproducible."
            )
        if self.dataset_id is not None and self.dataset_version_id is None:
            raise ValueError(
                "ResearchScope names a dataset_id but no dataset_version_id; a Dataset is a moving "
                "target and only its immutable DatasetVersion snapshot scopes a result "
                "(docs/htr-domain-design.md §3)"
            )
        return self

    def describe(self) -> str:
        """One sentence naming everything this scope is bound to. Always states the sample size."""
        methods = (
            ", ".join(
                f"{method}@{revision}"
                for method, revision in zip(self.method_ids, self.model_version_ids, strict=True)
            )
            or "no method named"
        )
        parts = [
            f"{self.sample_size} {self.unit_of_analysis.value}"
            f"{'' if self.sample_size == 1 else 's'} "
            f"({', '.join(self.covered_unit_ids)})",
            f"experiment {self.experiment_id}",
            f"version {self.experiment_version_id}",
            f"run(s) {', '.join(self.experiment_run_ids)}",
            f"method(s) {methods}",
        ]
        if self.dataset_version_id is not None:
            parts.append(f"dataset version {self.dataset_version_id}")
        if self.normalization_profile is not None:
            parts.append(f"normalization {self.normalization_profile}")
        if self.ground_truth_ref is not None:
            parts.append(f"ground truth {self.ground_truth_ref}")
        return "; ".join(parts)


class ContradictoryEvidence(BaseModel):
    """A record that contradicts the finding carrying it.

    **Never deletes or edits the finding it contradicts.** The follow-up's contradiction-preservation
    requirement is that both sides stay independently readable, so a contradiction is stored *on* the
    disputed finding as one more append-only entry, and the contradicting record keeps its own
    independent identity and history. `htr/knowledge/lifecycle.py` only ever appends to
    `ResearchFinding.contradictory_evidence`; nothing removes from it.
    """

    model_config = ConfigDict(frozen=True)

    contradiction_id: str
    source_kind: ContradictionSourceKind
    source_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    recorded_by: str = Field(min_length=1)
    recorded_at: str = Field(min_length=1)
    evidence_refs: tuple[EvidenceReference, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        source_kind: ContradictionSourceKind,
        source_id: str,
        description: str,
        recorded_by: str,
        recorded_at: str,
        evidence_refs: tuple[EvidenceReference, ...] = (),
    ) -> "ContradictoryEvidence":
        return cls(
            contradiction_id=new_id("contradiction"),
            source_kind=source_kind,
            source_id=source_id,
            description=description,
            recorded_by=recorded_by,
            recorded_at=recorded_at,
            evidence_refs=evidence_refs,
        )


class FindingRevision(BaseModel):
    """One entry in a `ResearchFinding`'s `revision_history` -- the record of a status transition.

    Every field except the two optional pointers is required, and `reasoning` has `min_length=1`, so
    a revision can never record *that* something changed without recording *why* and *who*. This is
    the entity that makes "a finding never loses its history" true rather than aspirational: a
    transition appends one of these and rewrites nothing.
    """

    model_config = ConfigDict(frozen=True)

    revision_id: str
    revised_at: str = Field(min_length=1)
    from_status: FindingStatus
    to_status: FindingStatus
    actor: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    evidence_refs: tuple[EvidenceReference, ...] = ()
    """Evidence offered *for this transition specifically* -- e.g. the reproduction evidence a
    `Provisionally supported -> Supported` transition requires."""
    superseding_finding_id: str | None = None
    contradicting_finding_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        from_status: FindingStatus,
        to_status: FindingStatus,
        actor: str,
        reasoning: str,
        revised_at: str,
        evidence_refs: tuple[EvidenceReference, ...] = (),
        superseding_finding_id: str | None = None,
        contradicting_finding_id: str | None = None,
    ) -> "FindingRevision":
        return cls(
            revision_id=new_id("finding_revision"),
            revised_at=revised_at,
            from_status=from_status,
            to_status=to_status,
            actor=actor,
            reasoning=reasoning,
            evidence_refs=evidence_refs,
            superseding_finding_id=superseding_finding_id,
            contradicting_finding_id=contradicting_finding_id,
        )


# -- Entities -------------------------------------------------------------------------------------


class ResearchObservation(BaseModel):
    """Layer 10: one recorded, scoped, evidence-backed fact about what durable records show.

    An observation is **not** a finding. It claims only "this is what the records for this scope
    say", never "this is how these methods behave". The distinction is load-bearing: four of the five
    real baseline observations would be indefensible as general claims and are perfectly defensible
    as observations, precisely because `scope` pins them to one run and `sample_size` is derived
    rather than asserted.

    `unverified_hypothesis` is the one field on this type that is explicitly *not* a fact -- see its
    own docstring.
    """

    model_config = ConfigDict(frozen=True)

    observation_id: str
    observation_type: ObservationType
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scope: ResearchScope
    supporting_evidence: tuple[EvidenceReference, ...] = Field(min_length=1)
    """Non-empty by construction. This is mechanism 1 of the module docstring's three: an observation
    that names no durable record it was extracted from cannot exist, so a telemetry event can never
    "become" an observation without an explicit extraction step saying what it read."""
    source_experiment_id: str = Field(min_length=1)
    source_experiment_run_id: str = Field(min_length=1)
    affected_method: str | None = None
    affected_model_version: str | None = None
    affected_dataset_id: str | None = None
    affected_dataset_version_id: str | None = None
    affected_document_or_segment_ids: tuple[str, ...] = ()
    author_or_source_component: str = Field(min_length=1)
    creation_timestamp: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    observation_confidence: ObservationConfidence
    """The *observer's* epistemic confidence -- see `ObservationConfidence`. Never a model's
    recognition confidence."""
    review_status: ObservationReviewStatus = ObservationReviewStatus.UNREVIEWED
    unverified_hypothesis: str | None = None
    """A candidate *explanation* for the observed fact, held separately from every factual field on
    this entity and never asserted as established.

    This field exists because the GPU-memory-variability observation has an obvious plausible cause
    (CUDA allocator/caching state differing between sessions) that this project has **not measured**.
    Writing it into `description` would make a guess indistinguishable from the two measured numbers
    in the same paragraph. Keeping it here makes the epistemic status structural: everything in
    `description` is what the records show, everything here is what someone thinks might explain it,
    and a reader or exporter can tell the two apart without parsing prose.
    """

    @model_validator(mode="after")
    def _scope_covers_the_source_run(self) -> "ResearchObservation":
        if self.source_experiment_run_id not in self.scope.experiment_run_ids:
            raise ValueError(
                f"ResearchObservation.source_experiment_run_id {self.source_experiment_run_id!r} is "
                f"not among scope.experiment_run_ids {self.scope.experiment_run_ids!r}: an "
                "observation cannot be sourced from a run its own scope does not cover"
            )
        if self.source_experiment_id != self.scope.experiment_id:
            raise ValueError(
                "ResearchObservation.source_experiment_id must equal scope.experiment_id "
                f"({self.source_experiment_id!r} != {self.scope.experiment_id!r})"
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        observation_type: ObservationType,
        title: str,
        description: str,
        scope: ResearchScope,
        supporting_evidence: tuple[EvidenceReference, ...],
        source_experiment_id: str,
        source_experiment_run_id: str,
        author_or_source_component: str,
        creation_timestamp: str,
        observation_confidence: ObservationConfidence,
        affected_method: str | None = None,
        affected_model_version: str | None = None,
        affected_dataset_id: str | None = None,
        affected_dataset_version_id: str | None = None,
        affected_document_or_segment_ids: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        review_status: ObservationReviewStatus = ObservationReviewStatus.UNREVIEWED,
        unverified_hypothesis: str | None = None,
    ) -> "ResearchObservation":
        return cls(
            observation_id=new_id("research_observation"),
            observation_type=observation_type,
            title=title,
            description=description,
            scope=scope,
            supporting_evidence=supporting_evidence,
            source_experiment_id=source_experiment_id,
            source_experiment_run_id=source_experiment_run_id,
            affected_method=affected_method,
            affected_model_version=affected_model_version,
            affected_dataset_id=affected_dataset_id,
            affected_dataset_version_id=affected_dataset_version_id,
            affected_document_or_segment_ids=affected_document_or_segment_ids,
            author_or_source_component=author_or_source_component,
            creation_timestamp=creation_timestamp,
            tags=tags,
            observation_confidence=observation_confidence,
            review_status=review_status,
            unverified_hypothesis=unverified_hypothesis,
        )

    def evidence_ids(self, kind: EvidenceReferenceKind | None = None) -> tuple[str, ...]:
        """The referenced ids, optionally filtered to one kind. Convenience for tests and exporters
        that need to resolve an observation's evidence against a replayed store."""
        return tuple(
            ref.reference_id
            for ref in self.supporting_evidence
            if kind is None or ref.kind is kind
        )


class ResearchFinding(BaseModel):
    """Layers 11-12: a scoped claim a human is (or is not yet) willing to stand behind.

    **Constructible only at `Draft` or `Candidate`.** `create` refuses anything else, and
    `_status_beyond_candidate_requires_a_recorded_review` refuses it on the model too, so
    constructing `ResearchFinding(review_status=SUPPORTED, ...)` directly fails as well. Reaching any
    other status requires `htr/knowledge/lifecycle.py::transition_finding_status`, which appends a
    `FindingRevision` naming a reviewer and a reason. That is what makes "a finding is not
    automatically accepted knowledge" a property of the type rather than a policy.

    **History is append-only.** `revision_history` grows; the transition function asserts the new
    history extends the old one prefix-for-prefix and never rewrites an entry.
    `contradictory_evidence` behaves the same way -- a dispute adds a record, it does not replace the
    finding or remove its support.
    """

    model_config = ConfigDict(frozen=True)

    finding_id: str
    statement: str = Field(min_length=1)
    scope: ResearchScope
    research_question: str | None = None
    hypothesis_relationship: HypothesisRelationship | None = None
    supporting_experiments: tuple[str, ...] = ()
    supporting_observations: tuple[str, ...] = Field(min_length=1)
    """Non-empty by construction -- mechanism 2 of the module docstring. A finding always says which
    observations it rests on, and never re-derives its evidence independently of them."""
    supporting_metrics: tuple[str, ...] = ()
    affected_datasets: tuple[str, ...] = ()
    affected_dataset_versions: tuple[str, ...] = ()
    affected_document_types: tuple[str, ...] = ()
    affected_handwriting_periods: tuple[str, ...] = ()
    affected_methods: tuple[str, ...] = ()
    affected_model_versions: tuple[str, ...] = ()
    transcription_convention: str | None = None
    """Nullable, and `None` for every finding from the 2026-07-30 baseline: no versioned
    `TranscriptionConvention` record was ever created for that run's ground truth
    (`docs/experiments/baseline-comparison/README.md`, "Honest gaps in this run"). A non-nullable
    field here would have forced a fabricated convention id."""
    confidence_level: FindingConfidence
    limitations: tuple[str, ...] = Field(min_length=1)
    """Non-empty by construction. A scoped finding with no stated limitation is either not scoped or
    not honest; on an N=1 run the limitation *is* most of the content."""
    contradictory_evidence: tuple[ContradictoryEvidence, ...] = ()
    author: str = Field(min_length=1)
    reviewer: str | None = None
    review_status: FindingStatus
    creation_date: str = Field(min_length=1)
    revision_history: tuple[FindingRevision, ...] = ()
    superseded_by: str | None = None
    """The finding that replaced this one. Required whenever `review_status` is `Superseded`, and
    forbidden otherwise -- so a superseded finding always points at its successor and nothing else
    silently claims to be superseded."""

    @model_validator(mode="after")
    def _status_beyond_candidate_requires_a_recorded_review(self) -> "ResearchFinding":
        status = self.review_status
        if status in INITIAL_FINDING_STATUSES:
            if self.revision_history and status is FindingStatus.CANDIDATE:
                # A finding may legitimately return to Candidate only via a recorded transition,
                # which is exactly what a non-empty history proves happened.
                pass
        else:
            if not self.revision_history:
                raise ValueError(
                    f"a ResearchFinding cannot be constructed at status {status.value!r}: statuses "
                    f"other than {sorted(s.value for s in INITIAL_FINDING_STATUSES)} are reachable "
                    "only through htr/knowledge/lifecycle.py::transition_finding_status, which "
                    "records a FindingRevision naming a reviewer and a reason"
                )
            if self.reviewer is None:
                raise ValueError(
                    f"a ResearchFinding at status {status.value!r} must name an attributable "
                    "reviewer -- no finding advances past Candidate without one "
                    "(docs/architecture/htr-event-model.md §1)"
                )
        if status is FindingStatus.SUPERSEDED and self.superseded_by is None:
            raise ValueError(
                "a Superseded ResearchFinding must name the finding that supersedes it in "
                "`superseded_by`; superseding without a successor pointer loses the chain"
            )
        if status is not FindingStatus.SUPERSEDED and self.superseded_by is not None:
            raise ValueError(
                f"ResearchFinding.superseded_by is set but review_status is {status.value!r}; only "
                "a Superseded finding carries a successor pointer"
            )
        if status is FindingStatus.DISPUTED and not self.contradictory_evidence:
            raise ValueError(
                "a Disputed ResearchFinding must carry at least one ContradictoryEvidence entry -- "
                "a dispute with no recorded contradiction is an unexplained status"
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        statement: str,
        scope: ResearchScope,
        supporting_observations: tuple[str, ...],
        limitations: tuple[str, ...],
        confidence_level: FindingConfidence,
        author: str,
        creation_date: str,
        status: FindingStatus = FindingStatus.CANDIDATE,
        research_question: str | None = None,
        hypothesis_relationship: HypothesisRelationship | None = None,
        supporting_experiments: tuple[str, ...] = (),
        supporting_metrics: tuple[str, ...] = (),
        affected_datasets: tuple[str, ...] = (),
        affected_dataset_versions: tuple[str, ...] = (),
        affected_document_types: tuple[str, ...] = (),
        affected_handwriting_periods: tuple[str, ...] = (),
        affected_methods: tuple[str, ...] = (),
        affected_model_versions: tuple[str, ...] = (),
        transcription_convention: str | None = None,
        contradictory_evidence: tuple[ContradictoryEvidence, ...] = (),
    ) -> "ResearchFinding":
        """Constructs a fresh finding. `status` may only be `Draft` or `Candidate`.

        A fresh `ResearchFinding.create(..., status=FindingStatus.SUPPORTED)` raises rather than
        producing an unreviewed "Supported" claim -- the follow-up's explicit requirement, and the
        reason this classmethod validates `status` itself instead of trusting the caller.
        """
        if status not in INITIAL_FINDING_STATUSES:
            raise ValueError(
                f"ResearchFinding.create refuses status {status.value!r}: a finding is created at "
                f"{sorted(s.value for s in INITIAL_FINDING_STATUSES)} and reaches any other status "
                "only through an explicit, recorded transition "
                "(htr/knowledge/lifecycle.py::transition_finding_status)"
            )
        return cls(
            finding_id=new_id("research_finding"),
            statement=statement,
            scope=scope,
            research_question=research_question,
            hypothesis_relationship=hypothesis_relationship,
            supporting_experiments=supporting_experiments,
            supporting_observations=supporting_observations,
            supporting_metrics=supporting_metrics,
            affected_datasets=affected_datasets,
            affected_dataset_versions=affected_dataset_versions,
            affected_document_types=affected_document_types,
            affected_handwriting_periods=affected_handwriting_periods,
            affected_methods=affected_methods,
            affected_model_versions=affected_model_versions,
            transcription_convention=transcription_convention,
            confidence_level=confidence_level,
            limitations=limitations,
            contradictory_evidence=contradictory_evidence,
            author=author,
            reviewer=None,
            review_status=status,
            creation_date=creation_date,
            revision_history=(),
            superseded_by=None,
        )
