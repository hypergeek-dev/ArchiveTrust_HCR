from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.alignment.models import AlignmentResult, ExcludedCandidatePair
from archivetrust.domain.alignment.telemetry import MAX_EXCLUSIONS_PER_EVENT, alignment_events
from archivetrust.domain.comparison.clustering import ClusteringBasisCode
from archivetrust.domain.telemetry.events import (
    CandidateExcludedBatch,
    CandidateExclusionRecord,
    EvidenceRejected,
)
from archivetrust.infrastructure.storage.telemetry_sink import (
    FileTelemetrySink,
    TelemetryEventSizeError,
)


def _record(index: int) -> CandidateExclusionRecord:
    return CandidateExclusionRecord(
        candidate_observation_id=f"candidate_{index}",
        compared_against_observation_id=f"other_{index}",
        excluding_mechanism="scope_aware_alignment",
        basis_code=ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER,
        structural=True,
    )


def test_candidate_exclusion_batch_cannot_exceed_structural_limit():
    with pytest.raises(ValidationError, match="at most 1000"):
        CandidateExcludedBatch(
            event_id="event_1",
            document_ref="doc1",
            exclusions=tuple(_record(index) for index in range(MAX_EXCLUSIONS_PER_EVENT + 1)),
        )


def test_alignment_bridge_chunks_exclusions_before_event_construction():
    result = AlignmentResult(
        algorithm_name="test",
        algorithm_version=1,
        clusters=(),
        attempts=(),
        observation_states=(),
        excluded_pairs=tuple(
            ExcludedCandidatePair(
                candidate_observation_id=f"candidate_{index}",
                compared_against_observation_id=f"other_{index}",
                excluding_mechanism="test",
                basis_code=ClusteringBasisCode.AMBIGUOUS_MULTI_PER_PROVIDER,
                structural=True,
            )
            for index in range(MAX_EXCLUSIONS_PER_EVENT + 1)
        ),
    )

    batches = alignment_events(result, document_ref="doc1")

    assert [len(batch.exclusions) for batch in batches] == [1000, 1]


def test_file_sink_rejects_any_serialized_event_beyond_one_record_envelope(tmp_path):
    sink = FileTelemetrySink(tmp_path / "events.jsonl", max_event_bytes=256)
    event = EvidenceRejected(
        event_id="event_1",
        document_ref="doc1",
        provider_id="provider",
        provider_version="1",
        invocation_id="invocation",
        processing_stage="raw",
        raw_output="small",
        rejection_reason="x" * 500,
    )
    with pytest.raises(TelemetryEventSizeError, match="maximum is 256"):
        sink.append(event)
    assert (tmp_path / "events.jsonl").stat().st_size == 0
