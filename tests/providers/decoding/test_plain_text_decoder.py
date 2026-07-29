from __future__ import annotations

from archivetrust.domain.ontology.payloads import ParagraphPayload
from archivetrust.providers.decoding.base import DecodeContext
from archivetrust.providers.decoding.plain_text_decoder import PlainTextDecoder


def _context(**kwargs) -> DecodeContext:
    return DecodeContext(provider_id="some-provider", provider_version="v1", **kwargs)


def test_single_block_with_no_blank_lines_becomes_one_paragraph_preserving_internal_newlines():
    decoder = PlainTextDecoder()
    decoded = decoder.decode("Land Registry Entry\nRecorded on 3 March 1921.", context=_context())
    assert len(decoded.observations) == 1
    assert decoded.observations[0].payload.text == "Land Registry Entry\nRecorded on 3 March 1921."


def test_blank_line_separated_text_splits_into_multiple_paragraph_observations():
    decoder = PlainTextDecoder()
    decoded = decoder.decode("First paragraph.\n\nSecond paragraph.\n\nThird.", context=_context())
    texts = [o.payload.text for o in decoded.observations]
    assert texts == ["First paragraph.", "Second paragraph.", "Third."]
    assert all(isinstance(o.payload, ParagraphPayload) for o in decoded.observations)


def test_whitespace_only_artifact_produces_no_observations_and_no_rejection():
    decoder = PlainTextDecoder()
    decoded = decoder.decode("   \n  \n\t", context=_context())
    assert decoded.observations == ()
    assert decoded.evidence == ()
    assert decoded.rejections == ()


def test_provider_confidence_is_never_fabricated():
    decoder = PlainTextDecoder()
    decoded = decoder.decode("Some text.", context=_context())
    assert decoded.evidence[0].provider_confidence is None


def test_sequence_hint_reflects_paragraph_position():
    decoder = PlainTextDecoder()
    decoded = decoder.decode("One.\n\nTwo.", context=_context())
    hints = [e.supporting_metadata["sequence_hint"] for e in decoded.evidence]
    assert hints == [0, 1]


def test_runtime_and_extra_metadata_are_merged_into_supporting_metadata():
    decoder = PlainTextDecoder()
    decoded = decoder.decode(
        "Text.",
        context=_context(runtime_provenance={"runtime_kind": "vllm"}, extra_metadata={"pixel_width": 1000}),
    )
    metadata = decoded.evidence[0].supporting_metadata
    assert metadata["runtime_kind"] == "vllm"
    assert metadata["pixel_width"] == 1000


def test_artifact_type_is_plain_text():
    assert PlainTextDecoder.artifact_type == "plain_text"
