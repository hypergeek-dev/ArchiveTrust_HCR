from __future__ import annotations

import pytest

from archivetrust.providers.decoding.base import (
    DecodeContext,
    DecodedArtifact,
    DecoderTelemetryKind,
    InMemoryDecoderTelemetrySink,
    run_decoder,
)


class _FakeDecoder:
    artifact_type = "fake"
    decoder_version = "3"

    def decode(self, artifact, *, context):
        from archivetrust.domain.evidence.models import Evidence, ProcessingStage

        evidence = Evidence.create(
            provider=context.provider_id,
            provider_version=context.provider_version,
            raw_output="x",
            processing_stage=ProcessingStage.VLM_INFERENCE,
        )
        return DecodedArtifact(evidence=(evidence,), observations=())


class _FailingDecoder:
    artifact_type = "fake"

    def decode(self, artifact, *, context):
        raise ValueError("boom")


def _context() -> DecodeContext:
    return DecodeContext(provider_id="some-provider", provider_version="v1")


def test_run_decoder_returns_the_decoders_result():
    result = run_decoder(_FakeDecoder(), "artifact", context=_context())
    assert len(result.evidence) == 1
    assert result.observations == ()


def test_run_decoder_stamps_decoder_version_onto_every_evidence():
    """Production Observability Completion milestone, "Artifact Decoder Completion... Implement
    version tracking" -- centralized in `run_decoder` so no individual decoder implementation can
    forget it."""
    result = run_decoder(_FakeDecoder(), "artifact", context=_context())
    assert result.evidence[0].supporting_metadata["decoder_version"] == "3"


def test_run_decoder_does_not_stamp_a_version_when_the_decoder_declares_none():
    class _NoVersionDecoder:
        artifact_type = "fake"

        def decode(self, artifact, *, context):
            return _FakeDecoder().decode(artifact, context=context)

    result = run_decoder(_NoVersionDecoder(), "artifact", context=_context())
    assert "decoder_version" not in result.evidence[0].supporting_metadata


def test_run_decoder_records_artifact_decoded_telemetry_on_success():
    telemetry = InMemoryDecoderTelemetrySink()
    run_decoder(_FakeDecoder(), "artifact", context=_context(), telemetry=telemetry)
    events = telemetry.events()
    assert len(events) == 1
    assert events[0].kind == DecoderTelemetryKind.ARTIFACT_DECODED
    assert events[0].artifact_type == "fake"
    assert events[0].decoder_name == "_FakeDecoder"
    assert events[0].provider_id == "some-provider"


def test_run_decoder_records_artifact_decode_failed_telemetry_and_reraises():
    telemetry = InMemoryDecoderTelemetrySink()
    with pytest.raises(ValueError, match="boom"):
        run_decoder(_FailingDecoder(), "artifact", context=_context(), telemetry=telemetry)
    events = telemetry.events()
    assert len(events) == 1
    assert events[0].kind == DecoderTelemetryKind.ARTIFACT_DECODE_FAILED
    assert events[0].detail == "boom"


def test_run_decoder_works_without_a_telemetry_sink():
    run_decoder(_FakeDecoder(), "artifact", context=_context())  # must not raise
