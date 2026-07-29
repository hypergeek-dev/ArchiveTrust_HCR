"""Alignment Observability replay (ROADMAP.md S5.12; Constitution Article 24).

Proves the grouping hypothesis is reconstructable from stored telemetry alone — "why were these
grouped?" and "which observations were left unaligned?" are answerable months later without
re-running any provider.
"""

from __future__ import annotations

import json

from archivetrust.application.journal import Journal
from archivetrust.domain.alignment.service import ClusteringAlignmentService
from archivetrust.domain.alignment.telemetry import alignment_events
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.telemetry.events import (
    AlignmentAttempted,
    CandidateExcluded,
    CandidateExcludedBatch,
    CandidateExclusionRecord,
    parse_event,
)
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink

from tests.domain.alignment._helpers import box, evidence, heading, observation, paragraph

DOC = "doc-1"


def _emit_alignment(sink: InMemoryTelemetrySink):
    ev_a = evidence("docling", "Chapter 1", box(0, 0, 100, 20))
    ev_b = evidence("tesseract", "Chapter I", box(2, 1, 98, 19))
    ev_solo = evidence("qwen", "Aside", box(0, 400, 100, 420))
    obs_a = observation("docling", heading("Chapter 1"), ev_a)
    obs_b = observation("tesseract", heading("Chapter I"), ev_b)
    obs_solo = observation("qwen", paragraph("Aside"), ev_solo)
    evidence_by_id = {e.evidence_id: e for e in (ev_a, ev_b, ev_solo)}
    result = ClusteringAlignmentService().align(
        (obs_a, obs_b, obs_solo), evidence_by_id, ReconciliationPolicy(policy_version=1)
    )
    for event in alignment_events(result, document_ref=DOC):
        sink.append(event)
    return result, (obs_a, obs_b), obs_solo


def test_journal_reconstructs_alignment_from_telemetry_alone():
    sink = InMemoryTelemetrySink()
    result, aligned_pair, obs_solo = _emit_alignment(sink)

    state = Journal().replay(sink.events_for_document(DOC))

    # "Why were these grouped?" — the attempt, with candidates and rationale, survives replay.
    aligned_group = result.state_of(aligned_pair[0].observation_id).comparison_group_id
    attempt = state.alignment_attempt(aligned_group)
    assert attempt is not None
    assert set(attempt.selected_observation_ids) == {o.observation_id for o in aligned_pair}
    assert attempt.alignment_rationale

    # "Which observations were left unaligned?" — reconstructable, nothing silently dropped.
    assert obs_solo.observation_id in state.unaligned_observations()
    for obs in aligned_pair:
        assert obs.observation_id not in state.unaligned_observations()
        assert state.alignment_state(obs.observation_id) is not None


def test_never_aligned_observation_is_distinguishable_from_unaligned():
    sink = InMemoryTelemetrySink()
    _emit_alignment(sink)
    state = Journal().replay(sink.events_for_document(DOC))
    # An observation the alignment never saw returns None (never processed), which is a different
    # fact from an explicit Unaligned record (Article 18 discipline, one level up).
    assert state.alignment_state("observation_never_seen") is None


def test_alignment_event_round_trips_through_serialization():
    sink = InMemoryTelemetrySink()
    _emit_alignment(sink)
    attempt = next(e for e in sink.all_events() if isinstance(e, AlignmentAttempted))
    reparsed = parse_event(json.loads(attempt.model_dump_json()))
    assert isinstance(reparsed, AlignmentAttempted)
    assert reparsed == attempt
    assert reparsed.alignment_algorithm_version == attempt.alignment_algorithm_version


def test_batched_candidate_exclusions_replay_like_legacy_single_events():
    ev_a1 = evidence("docling", "Fragment A", None)
    ev_a2 = evidence("docling", "Fragment B", None)
    obs_a1 = observation("docling", paragraph("Fragment A"), ev_a1)
    obs_a2 = observation("docling", paragraph("Fragment B"), ev_a2)
    result = ClusteringAlignmentService().align(
        (obs_a1, obs_a2),
        {ev_a1.evidence_id: ev_a1, ev_a2.evidence_id: ev_a2},
        ReconciliationPolicy(policy_version=1),
    )
    pair = result.excluded_pairs[0]

    legacy = CandidateExcluded(
        event_id="event-legacy",
        document_ref=DOC,
        alignment_algorithm_version=result.algorithm_version,
        candidate_observation_id=pair.candidate_observation_id,
        compared_against_observation_id=pair.compared_against_observation_id,
        excluding_mechanism=pair.excluding_mechanism,
        basis_code=pair.basis_code,
        structural=pair.structural,
    )
    batched = CandidateExcludedBatch(
        event_id="event-batch",
        document_ref=DOC,
        alignment_algorithm_version=result.algorithm_version,
        exclusions=(
            CandidateExclusionRecord(
                candidate_observation_id=pair.candidate_observation_id,
                compared_against_observation_id=pair.compared_against_observation_id,
                excluding_mechanism=pair.excluding_mechanism,
                basis_code=pair.basis_code,
                structural=pair.structural,
            ),
        ),
    )

    legacy_pair = Journal().replay((legacy,)).excluded_candidate_pairs()[0]
    batch_pair = Journal().replay((batched,)).excluded_candidate_pairs()[0]

    assert legacy_pair.candidate_observation_id == batch_pair.candidate_observation_id
    assert legacy_pair.compared_against_observation_id == batch_pair.compared_against_observation_id
    assert legacy_pair.basis_code == batch_pair.basis_code
