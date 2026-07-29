from __future__ import annotations

from archivetrust.domain.ontology.payloads import (
    NormalizedTranscriptionPayload,
    ParsedTranscriptionPayload,
    RawTranscriptionPayload,
    RegionPayload,
    TextLinePayload,
)
from archivetrust.domain.ontology.types import ObservationType


def test_text_line_payload_matches_its_observation_type():
    payload = TextLinePayload(reading_order_index=0)
    assert payload.observation_type == ObservationType.TEXT_LINE


def test_region_payload_matches_its_observation_type():
    payload = RegionPayload(region_type="text_block")
    assert payload.observation_type == ObservationType.REGION


def test_transcription_payload_chain_matches_observation_types():
    assert RawTranscriptionPayload(text="raw").observation_type == ObservationType.RAW_TRANSCRIPTION
    assert (
        ParsedTranscriptionPayload(text="parsed").observation_type
        == ObservationType.PARSED_TRANSCRIPTION
    )
    assert (
        NormalizedTranscriptionPayload(text="normalized").observation_type
        == ObservationType.NORMALIZED_TRANSCRIPTION
    )
