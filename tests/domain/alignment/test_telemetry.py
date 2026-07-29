from __future__ import annotations

from archivetrust.domain.alignment.models import ALIGNMENT_ALGORITHM_VERSION
from archivetrust.domain.alignment.service import ClusteringAlignmentService
from archivetrust.domain.alignment.telemetry import alignment_events
from archivetrust.domain.comparison.clustering import ClusteringBasisCode
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.telemetry.events import (
    AlignmentAttempted,
    CandidateExcludedBatch,
    ObservationAligned,
    ObservationLeftUnaligned,
)

from tests.domain.alignment._helpers import box, evidence, heading, observation, paragraph


def _result():
    ev_a = evidence("docling", "Chapter 1", box(0, 0, 100, 20))
    ev_b = evidence("tesseract", "Chapter I", box(2, 1, 98, 19))
    ev_solo = evidence("qwen", "Aside", box(0, 400, 100, 420))
    obs_a = observation("docling", heading("Chapter 1"), ev_a)
    obs_b = observation("tesseract", heading("Chapter I"), ev_b)
    obs_solo = observation("qwen", paragraph("Aside"), ev_solo)
    evidence_by_id = {
        ev_a.evidence_id: ev_a,
        ev_b.evidence_id: ev_b,
        ev_solo.evidence_id: ev_solo,
    }
    result = ClusteringAlignmentService().align((obs_a, obs_b, obs_solo), evidence_by_id, ReconciliationPolicy(policy_version=1))
    return result, obs_a, obs_b, obs_solo


def test_emits_one_attempt_per_group_and_one_state_per_observation():
    result, obs_a, obs_b, obs_solo = _result()
    events = alignment_events(result, document_ref="doc-1")

    attempts = [e for e in events if isinstance(e, AlignmentAttempted)]
    aligned = [e for e in events if isinstance(e, ObservationAligned)]
    unaligned = [e for e in events if isinstance(e, ObservationLeftUnaligned)]

    assert len(attempts) == len(result.attempts)
    # Every observation is accounted for exactly once across aligned + unaligned.
    accounted = {e.observation_id for e in aligned} | {e.observation_id for e in unaligned}
    assert accounted == {obs_a.observation_id, obs_b.observation_id, obs_solo.observation_id}
    assert obs_solo.observation_id in {e.observation_id for e in unaligned}


def test_events_carry_alignment_algorithm_version():
    result, *_ = _result()
    events = alignment_events(result, document_ref="doc-1")
    assert all(e.alignment_algorithm_version == ALIGNMENT_ALGORITHM_VERSION for e in events)


def test_attempt_event_links_group_to_candidates_and_selection():
    result, *_ = _result()
    events = alignment_events(result, document_ref="doc-1")
    attempt_event = next(e for e in events if isinstance(e, AlignmentAttempted))
    # The event is a faithful projection of the domain attempt.
    domain_attempt = result.attempt_for_group(attempt_event.comparison_group_id)
    assert attempt_event.candidate_observation_ids == domain_attempt.candidate_observation_ids
    assert attempt_event.selected_observation_ids == domain_attempt.selected_observation_ids
    assert attempt_event.alignment_rationale == domain_attempt.alignment_rationale


def test_unaligned_event_has_a_reason():
    result, _a, _b, obs_solo = _result()
    events = alignment_events(result, document_ref="doc-1")
    unaligned = next(
        e for e in events if isinstance(e, ObservationLeftUnaligned) and e.observation_id == obs_solo.observation_id
    )
    assert unaligned.reason  # never a silent absence


def test_emits_one_candidate_excluded_batch_per_alignment_result():
    # Constitution Article 27: a provider contributing two same-type, no-geometry Observations to
    # one page makes every pair drawing on them a structural exclusion, not an ordinary low score.
    ev_a1 = evidence("docling", "Fragment A", None)
    ev_a2 = evidence("docling", "Fragment B", None)
    obs_a1 = observation("docling", paragraph("Fragment A"), ev_a1)
    obs_a2 = observation("docling", paragraph("Fragment B"), ev_a2)
    evidence_by_id = {ev_a1.evidence_id: ev_a1, ev_a2.evidence_id: ev_a2}
    result = ClusteringAlignmentService().align(
        (obs_a1, obs_a2), evidence_by_id, ReconciliationPolicy(policy_version=1)
    )

    events = alignment_events(result, document_ref="doc-1")
    excluded = [e for e in events if isinstance(e, CandidateExcludedBatch)]

    assert len(excluded) == 1
    event = excluded[0].exclusions[0]
    assert {event.candidate_observation_id, event.compared_against_observation_id} == {
        obs_a1.observation_id,
        obs_a2.observation_id,
    }
    assert event.basis_code == ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER
    assert event.structural is True
    assert excluded[0].alignment_algorithm_version == ALIGNMENT_ALGORITHM_VERSION


def test_scope_mismatch_exclusion_is_emitted_in_candidate_batch():
    ev_unit = evidence("docling", "Short paragraph.", box(0, 0, 100, 80))
    ev_extended = evidence(
        "paddleocr_vl",
        " ".join(["whole page transcript"] * 80),
        box(0, 0, 100, 80),
    )
    obs_unit = observation("docling", paragraph(ev_unit.raw_output), ev_unit)
    obs_extended = observation("paddleocr_vl", paragraph(ev_extended.raw_output), ev_extended)
    result = ClusteringAlignmentService().align(
        (obs_unit, obs_extended),
        {
            ev_unit.evidence_id: ev_unit,
            ev_extended.evidence_id: ev_extended,
        },
        ReconciliationPolicy(policy_version=1),
    )

    events = alignment_events(result, document_ref="doc-1")
    excluded = next(e for e in events if isinstance(e, CandidateExcludedBatch))
    event = excluded.exclusions[0]

    assert event.basis_code == ClusteringBasisCode.SCOPE_MISMATCH
    assert event.excluding_mechanism == "alignment.scope_compatibility_gate"
    assert event.structural is False
