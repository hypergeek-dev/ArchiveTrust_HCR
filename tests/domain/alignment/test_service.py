from __future__ import annotations

import itertools

from archivetrust.domain.alignment.models import ALIGNMENT_ALGORITHM_VERSION, AlignmentOutcome
from archivetrust.domain.alignment.service import ClusteringAlignmentService
from archivetrust.domain.comparison.clustering import ClusteringBasisCode, cluster_observations
from archivetrust.domain.comparison.policy import ReconciliationPolicy

from tests.domain.alignment._helpers import box, evidence, heading, observation, paragraph


def _policy() -> ReconciliationPolicy:
    return ReconciliationPolicy(policy_version=1)


def _ambiguous_multi_per_provider_pool():
    """One provider ('docling') contributes two same-type Observations to the same page, with no
    bounding box (geometry absent, not merely coarse) -- the real, currently-shipped structural
    exclusion case (Constitution Article 27): every pair drawing on docling's two observations is
    ambiguous, since ordinal position within the arbitrary id-sorted pool carries no spatial
    meaning once a provider contributed more than one same-type Observation to the page.
    """
    ev_a1 = evidence("docling", "Fragment A", None)
    ev_a2 = evidence("docling", "Fragment B", None)
    ev_b = evidence("paddleocr_vl", "Whole page transcript", None)
    obs_a1 = observation("docling", paragraph("Fragment A"), ev_a1)
    obs_a2 = observation("docling", paragraph("Fragment B"), ev_a2)
    obs_b = observation("paddleocr_vl", paragraph("Whole page transcript"), ev_b)
    evidence_by_id = {
        ev_a1.evidence_id: ev_a1,
        ev_a2.evidence_id: ev_a2,
        ev_b.evidence_id: ev_b,
    }
    return (obs_a1, obs_a2, obs_b), evidence_by_id


def test_ambiguous_multi_per_provider_case_is_recorded_as_excluded_not_silently_unaligned():
    observations, evidence_by_id = _ambiguous_multi_per_provider_pool()

    result = ClusteringAlignmentService().align(observations, evidence_by_id, _policy())

    # Behavior preservation (Phase 3's own regression requirement): three singleton clusters,
    # exactly as cluster_observations() alone already produces -- unchanged.
    direct = cluster_observations(observations, evidence_by_id, _policy())
    assert [c.member_observation_ids for c in result.clusters] == [
        c.member_observation_ids for c in direct
    ]
    assert all(len(c.member_observation_ids) == 1 for c in result.clusters)

    # The new fact Article 27 requires: every one of the 3 candidate pairs in this pool is recorded
    # as a structural exclusion, not merely inferred from three separate Unaligned states.
    assert len(result.excluded_pairs) == 3
    assert all(pair.structural for pair in result.excluded_pairs)
    assert all(
        pair.basis_code == ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER
        for pair in result.excluded_pairs
    )
    assert all(
        pair.excluding_mechanism == "clustering.ambiguous_multi_per_provider_guard"
        for pair in result.excluded_pairs
    )
    excluded_ids = {
        frozenset((pair.candidate_observation_id, pair.compared_against_observation_id))
        for pair in result.excluded_pairs
    }
    all_ids = {o.observation_id for o in observations}
    expected_pairs = {frozenset(pair) for pair in itertools.combinations(all_ids, 2)}
    assert excluded_ids == expected_pairs


def test_non_ambiguous_pools_produce_no_excluded_pairs():
    observations, evidence_by_id = _two_provider_aligned_slot()

    result = ClusteringAlignmentService().align(observations, evidence_by_id, _policy())

    assert result.excluded_pairs == ()


def test_scope_mismatch_is_recorded_as_policy_exclusion():
    ev_unit = evidence("docling", "Short paragraph.", box(0, 0, 100, 80))
    ev_extended = evidence(
        "paddleocr_vl",
        " ".join(["whole page transcript"] * 80),
        box(0, 0, 100, 80),
    )
    obs_unit = observation("docling", paragraph(ev_unit.raw_output), ev_unit)
    obs_extended = observation("paddleocr_vl", paragraph(ev_extended.raw_output), ev_extended)
    evidence_by_id = {
        ev_unit.evidence_id: ev_unit,
        ev_extended.evidence_id: ev_extended,
    }

    result = ClusteringAlignmentService().align(
        (obs_unit, obs_extended), evidence_by_id, _policy()
    )

    assert len(result.clusters) == 2
    assert len(result.excluded_pairs) == 1
    excluded = result.excluded_pairs[0]
    assert {
        excluded.candidate_observation_id,
        excluded.compared_against_observation_id,
    } == {obs_unit.observation_id, obs_extended.observation_id}
    assert excluded.basis_code == ClusteringBasisCode.SCOPE_MISMATCH
    assert excluded.excluding_mechanism == "alignment.scope_compatibility_gate"
    assert excluded.structural is False


def _two_provider_aligned_slot():
    ev_a = evidence("docling", "Chapter 1", box(0, 0, 100, 20))
    ev_b = evidence("tesseract", "Chapter I", box(2, 1, 98, 19))
    obs_a = observation("docling", heading("Chapter 1"), ev_a)
    obs_b = observation("tesseract", heading("Chapter I"), ev_b)
    evidence_by_id = {ev_a.evidence_id: ev_a, ev_b.evidence_id: ev_b}
    return (obs_a, obs_b), evidence_by_id


def test_service_does_not_change_the_grouping():
    # Behavior preservation: the service's clusters are exactly what the clustering algorithm
    # produces on its own — alignment observability adds nothing to the grouping decision.
    observations, evidence_by_id = _two_provider_aligned_slot()
    policy = _policy()

    direct = cluster_observations(observations, evidence_by_id, policy)
    via_service = ClusteringAlignmentService().align(observations, evidence_by_id, policy).clusters

    assert [c.member_observation_ids for c in via_service] == [
        c.member_observation_ids for c in direct
    ]


def test_every_observation_ends_aligned_or_unaligned():
    # Article 18 / Article 24: no observation silently disappears.
    observations, evidence_by_id = _two_provider_aligned_slot()
    # Add a lone paragraph that no other provider observed -> must be recorded Unaligned.
    ev_solo = evidence("qwen", "Footnote text", box(0, 400, 100, 420))
    obs_solo = observation("qwen", paragraph("Footnote text"), ev_solo)
    evidence_by_id[ev_solo.evidence_id] = ev_solo
    all_obs = observations + (obs_solo,)

    result = ClusteringAlignmentService().align(all_obs, evidence_by_id, _policy())

    recorded_ids = {s.observation_id for s in result.observation_states}
    assert recorded_ids == {o.observation_id for o in all_obs}  # nothing dropped
    assert result.state_of(all_obs[0].observation_id).outcome is AlignmentOutcome.ALIGNED
    assert result.state_of(all_obs[1].observation_id).outcome is AlignmentOutcome.ALIGNED
    assert result.state_of(obs_solo.observation_id).outcome is AlignmentOutcome.UNALIGNED
    assert obs_solo.observation_id in result.unaligned_observation_ids


def test_attempt_records_candidates_selection_and_rationale():
    observations, evidence_by_id = _two_provider_aligned_slot()
    result = ClusteringAlignmentService().align(observations, evidence_by_id, _policy())

    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    ids = {o.observation_id for o in observations}
    assert set(attempt.selected_observation_ids) == ids
    assert set(attempt.candidate_observation_ids) == ids  # both were considered for this group
    assert attempt.alignment_rationale  # the clustering_basis, surfaced
    assert attempt.algorithm_version == ALIGNMENT_ALGORITHM_VERSION
    assert attempt.comparison_group_id == result.clusters[0].cluster_id  # == semantic_slot_id


def test_candidates_include_all_considered_even_when_split_into_groups():
    # Two disjoint headings on the same page: same candidate pool, two separate groups. Each
    # attempt's candidates are the whole pool; its selection is just its own member.
    ev_a = evidence("docling", "Chapter 1", box(0, 0, 50, 20))
    ev_b = evidence("docling", "Chapter 2", box(500, 500, 550, 520))
    obs_a = observation("docling", heading("Chapter 1"), ev_a)
    obs_b = observation("docling", heading("Chapter 2"), ev_b)
    evidence_by_id = {ev_a.evidence_id: ev_a, ev_b.evidence_id: ev_b}

    result = ClusteringAlignmentService().align((obs_a, obs_b), evidence_by_id, _policy())

    assert len(result.attempts) == 2
    both = {obs_a.observation_id, obs_b.observation_id}
    for attempt in result.attempts:
        assert set(attempt.candidate_observation_ids) == both  # whole pool considered
        assert len(attempt.selected_observation_ids) == 1  # each selected alone
    # Both are single-member groups -> both Unaligned.
    assert set(result.unaligned_observation_ids) == both


def test_align_is_deterministic_regardless_of_input_order():
    observations, evidence_by_id = _two_provider_aligned_slot()
    service = ClusteringAlignmentService()
    first = service.align(observations, evidence_by_id, _policy())
    second = service.align(tuple(reversed(observations)), evidence_by_id, _policy())
    assert [c.member_observation_ids for c in first.clusters] == [
        c.member_observation_ids for c in second.clusters
    ]
