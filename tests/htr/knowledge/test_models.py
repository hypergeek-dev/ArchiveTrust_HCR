"""The three arrows of `docs/architecture/htr-event-model.md` §1, blocked structurally.

    a telemetry event is not automatically a research observation;
    a research observation is not automatically a finding;
    a finding is not automatically accepted knowledge.

Each test below attacks one of those from the direction a careless caller would: constructing an
observation with no evidence, a finding with no observation, a finding already `Supported`, or a scope
that quietly means "everything". If any of them passes, the layering is a convention rather than a
property of the types.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.htr.knowledge.models import (
    EvidenceReference,
    EvidenceReferenceKind,
    FindingConfidence,
    FindingStatus,
    ObservationConfidence,
    ObservationType,
    ResearchFinding,
    ResearchObservation,
    ResearchScope,
    ScopeUnit,
)

RUN = "experiment_run_test0000000000000000000000000001"
OTHER_RUN = "experiment_run_test0000000000000000000000000002"
EXPERIMENT = "experiment_test000000000000000000000000000001"
VERSION = "experiment_version_test00000000000000000000001"
CROP = "input_crop_test0000000000000000000000000000001"
AT = "2026-07-30T04:00:00+00:00"


def a_scope(**overrides) -> ResearchScope:
    fields = {
        "experiment_id": EXPERIMENT,
        "experiment_version_id": VERSION,
        "experiment_run_ids": (RUN,),
        "unit_of_analysis": ScopeUnit.LINE_CROP,
        "covered_unit_ids": (CROP,),
        "method_ids": ("satrn",),
        "model_version_ids": ("abc123",),
    }
    fields.update(overrides)
    return ResearchScope(**fields)


def an_evidence_ref() -> EvidenceReference:
    return EvidenceReference(
        kind=EvidenceReferenceKind.METRIC_RESULT, reference_id="metric_result_test0001"
    )


def an_observation(**overrides) -> ResearchObservation:
    fields = {
        "observation_type": ObservationType.MODEL_LIMITATION,
        "title": "a title",
        "description": "a description",
        "scope": a_scope(),
        "supporting_evidence": (an_evidence_ref(),),
        "source_experiment_id": EXPERIMENT,
        "source_experiment_run_id": RUN,
        "author_or_source_component": "tests",
        "creation_timestamp": AT,
        "observation_confidence": ObservationConfidence.MODERATE,
    }
    fields.update(overrides)
    return ResearchObservation.create(**fields)


def a_finding(**overrides) -> ResearchFinding:
    fields = {
        "statement": "a statement",
        "scope": a_scope(),
        "supporting_observations": ("research_observation_test0001",),
        "limitations": ("N=1",),
        "confidence_level": FindingConfidence.LOW,
        "author": "tests",
        "creation_date": AT,
    }
    fields.update(overrides)
    return ResearchFinding.create(**fields)


# -- Arrow 1: an event does not become an observation ----------------------------------------------


def test_an_observation_cannot_exist_without_typed_supporting_evidence():
    """`supporting_evidence` is `min_length=1`, so extraction always names what it read."""
    with pytest.raises(ValidationError):
        an_observation(supporting_evidence=())


def test_evidence_references_are_typed_not_bare_strings():
    """A caller cannot pass a note where a reference belongs -- `kind` and `reference_id` are both
    required, so every entry resolves to a durable record."""
    with pytest.raises(ValidationError):
        EvidenceReference(kind=EvidenceReferenceKind.METRIC_RESULT, reference_id="")
    with pytest.raises(ValidationError):
        EvidenceReference(reference_id="metric_result_x")  # type: ignore[call-arg]


def test_a_placeholder_reference_id_is_refused():
    with pytest.raises(ValidationError, match="placeholder"):
        EvidenceReference(kind=EvidenceReferenceKind.METRIC_RESULT, reference_id="*")


def test_an_observation_cannot_be_sourced_from_a_run_outside_its_own_scope():
    with pytest.raises(ValidationError, match="not among scope.experiment_run_ids"):
        an_observation(source_experiment_run_id=OTHER_RUN)


# -- Scope: structurally incapable of expressing a general claim -----------------------------------


def test_sample_size_is_derived_from_covered_units_and_cannot_be_asserted():
    """The single most important property of `ResearchScope`: N is computed, not claimed.

    There is no `sample_size` field to set, so an author cannot inflate, round, or omit it -- an N=1
    scope renders as `1 line_crop` in `describe()` whether or not anyone remembered to say so.
    """
    scope = a_scope()
    assert scope.sample_size == 1
    assert "1 line_crop" in scope.describe()
    with pytest.raises(ValidationError):
        ResearchScope(**{**scope.model_dump(), "sample_size": 500})

    wider = a_scope(covered_unit_ids=(CROP, "input_crop_test0002"))
    assert wider.sample_size == 2, "sample size must follow the enumerated units, nothing else"


def test_a_scope_cannot_cover_everything():
    """A wildcard is a general claim wearing a scope's clothes, so it is refused outright."""
    for placeholder in ("*", "all", "", "any"):
        with pytest.raises(ValidationError, match="placeholder"):
            a_scope(covered_unit_ids=(placeholder,))


def test_a_scope_must_have_at_least_one_covered_unit_and_one_run():
    with pytest.raises(ValidationError):
        a_scope(covered_unit_ids=())
    with pytest.raises(ValidationError):
        a_scope(experiment_run_ids=())


def test_naming_a_method_without_its_exact_model_revision_is_refused():
    """"Florence-2 beat SATRN" cannot be scoped without stating which checkpoints."""
    with pytest.raises(ValidationError, match="paired positionally"):
        a_scope(method_ids=("satrn", "florence2_htr"), model_version_ids=("abc123",))


def test_a_dataset_must_be_scoped_by_its_immutable_version():
    with pytest.raises(ValidationError, match="dataset_version_id"):
        a_scope(dataset_id="dataset_test0001", dataset_version_id=None)


# -- Arrow 2: an observation does not become a finding ---------------------------------------------


def test_a_finding_cannot_exist_without_a_supporting_observation():
    with pytest.raises(ValidationError):
        a_finding(supporting_observations=())


def test_a_finding_cannot_exist_without_a_stated_limitation():
    """On a scoped empirical claim the limitations are most of the content; zero of them means the
    finding is either not scoped or not honest."""
    with pytest.raises(ValidationError):
        a_finding(limitations=())


# -- Arrow 3: a finding does not become accepted knowledge -----------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        FindingStatus.UNDER_REVIEW,
        FindingStatus.PROVISIONALLY_SUPPORTED,
        FindingStatus.SUPPORTED,
        FindingStatus.DISPUTED,
        FindingStatus.SUPERSEDED,
        FindingStatus.REJECTED,
    ],
)
def test_create_refuses_every_status_beyond_draft_and_candidate(status):
    """The follow-up's explicit requirement: `create(..., status=Supported)` must raise."""
    with pytest.raises(ValueError, match="refuses status"):
        a_finding(status=status)


def test_create_permits_draft_and_candidate():
    assert a_finding(status=FindingStatus.DRAFT).review_status is FindingStatus.DRAFT
    assert a_finding().review_status is FindingStatus.CANDIDATE, "Candidate is the default"


def test_bypassing_create_does_not_help():
    """The same rule on the model itself, so `ResearchFinding(review_status=SUPPORTED, ...)` with an
    empty `revision_history` fails too. Without this, `create` would be a suggestion."""
    fields = a_finding().model_dump()
    fields["review_status"] = FindingStatus.SUPPORTED.value
    fields["reviewer"] = "someone"
    with pytest.raises(ValidationError, match="reachable\n?\\s*only through"):
        ResearchFinding.model_validate(fields)


def test_a_reviewed_finding_must_name_a_reviewer():
    fields = a_finding().model_dump()
    fields["review_status"] = FindingStatus.UNDER_REVIEW.value
    fields["revision_history"] = [
        {
            "revision_id": "finding_revision_test0001",
            "revised_at": AT,
            "from_status": FindingStatus.CANDIDATE.value,
            "to_status": FindingStatus.UNDER_REVIEW.value,
            "actor": "someone",
            "reasoning": "because",
            "evidence_refs": [],
            "superseding_finding_id": None,
            "contradicting_finding_id": None,
        }
    ]
    fields["reviewer"] = None
    with pytest.raises(ValidationError, match="attributable reviewer"):
        ResearchFinding.model_validate(fields)


def test_a_superseded_finding_must_point_at_its_successor():
    fields = a_finding().model_dump()
    fields["review_status"] = FindingStatus.SUPERSEDED.value
    fields["reviewer"] = "someone"
    fields["revision_history"] = [
        {
            "revision_id": "finding_revision_test0002",
            "revised_at": AT,
            "from_status": FindingStatus.CANDIDATE.value,
            "to_status": FindingStatus.SUPERSEDED.value,
            "actor": "someone",
            "reasoning": "because",
            "evidence_refs": [],
            "superseding_finding_id": None,
            "contradicting_finding_id": None,
        }
    ]
    with pytest.raises(ValidationError, match="must name the finding that supersedes it"):
        ResearchFinding.model_validate(fields)


# -- The confidence-naming requirement -------------------------------------------------------------


def test_observation_confidence_is_named_unambiguously_and_is_not_a_recognition_confidence():
    """The field is `observation_confidence`, never `confidence`.

    The follow-up asks specifically that this not be confusable with `RecognitionMetrics`/adapter
    confidence. A bare `confidence` attribute would be read as the model's number by anyone arriving
    from adapter code, so its absence is asserted rather than left to review.
    """
    observation = an_observation()
    assert observation.observation_confidence is ObservationConfidence.MODERATE
    assert "observation_confidence" in ResearchObservation.model_fields
    assert "confidence" not in ResearchObservation.model_fields
    assert not isinstance(observation.observation_confidence, float), (
        "an observation's epistemic standing is ordinal, not a measured float -- a 0.72 would imply a "
        "calibration that does not exist"
    )


def test_the_unverified_hypothesis_field_is_separate_from_every_factual_field():
    """An explanation must be structurally distinguishable from a measurement, not merely worded
    cautiously inside `description`."""
    observation = an_observation(unverified_hypothesis="might be the allocator")
    assert observation.unverified_hypothesis == "might be the allocator"
    assert "might be the allocator" not in observation.description
    assert an_observation().unverified_hypothesis is None, (
        "the field defaults to None: most observations have no hypothesis attached, and an empty "
        "string would be indistinguishable from a hypothesis someone forgot to write"
    )
