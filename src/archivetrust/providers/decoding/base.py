"""Native Artifact Decoders (Native Artifact Decoder Architecture addendum, Production Runtime
Completion milestone): the deterministic final stage between a provider's raw inference output --
the "native artifact" -- and Canonical Observations.

Pipeline:

    Inference -> Native Artifact -> ArtifactDecoder -> Canonical Observations

A provider adapter's responsibility ends at producing a native artifact (e.g.
`PaddleOCRVLRawResponse.raw_text`, `SuryaRawResponse.raw_html`, a Docling/Tesseract item list) --
constructing `Evidence`/`Observation` is always a decoder's job, never the adapter's ("No provider
shall construct Canonical Observations directly").

Decoders are keyed by *artifact type*, not by provider (`artifact_type` below): `PlainTextDecoder`
and `HtmlDecoder` are generic and reusable by any future provider that happens to produce the same
shape of native artifact, exactly like two different cameras can hand their film to the same
darkroom. Every decoder in this package is a pure, deterministic function of its input artifact
plus `DecodeContext` -- no network calls, no model inference, no randomness ("No AI shall
participate after inference. Translation must always be deterministic.").
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.providers.base import RejectedRawOutput

TArtifact = TypeVar("TArtifact")


class MappedPayload(BaseModel):
    """Constitution Article 28. What a provider's `label_to_payload` callback returns -- the
    payload plus which `MappingTableEntry` (`domain/ontology/mapping.py`) produced it, so a
    decoder can stamp `mapping_table_entry_id`/`mapping_table_version` onto the resulting
    Evidence's `supporting_metadata`, mirroring how `_stamp_decoder_version` already stamps
    `decoder_version` uniformly (below) -- the same centralized-stamping discipline, applied to a
    per-block rather than per-decoder fact. `None` for a provider callback that hasn't been wired
    to a `MappingTable` yet (never guessed, Article 18's discipline extended here).
    """

    model_config = ConfigDict(frozen=True)

    payload: ObservationPayload
    mapping_table_entry_id: str | None = None
    mapping_table_version: int | None = None


class DecodeContext(BaseModel):
    """Everything a decoder needs about *where* an artifact came from, without needing to know
    which provider produced it -- the provider-agnostic half of what a hand-rolled per-provider
    importer used to carry as private state (page number, prompt, rendering/runtime provenance).
    `extra_metadata` is the escape hatch for a provider-specific fact a decoder should merge into
    every block-level Evidence's `supporting_metadata` without needing a dedicated field here (e.g.
    Surya's page-level `mean_token_prob`) -- never used to smuggle in a fabricated confidence value.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    page_number: int | None = None
    prompt: str | None = None
    pixel_width: float | None = None
    pixel_height: float | None = None
    runtime_provenance: dict[str, str] = {}
    extra_metadata: dict[str, object] = {}


class DecodedArtifact(BaseModel):
    """What a decoder produces -- exactly the pieces a `ProviderRunResult` needs (`evidence`,
    `observations`, `rejections`), returned separately from the `ProviderAttempt` (which needs
    `invocation_id`, owned by the adapter/pipeline, never the decoder).
    """

    model_config = ConfigDict(frozen=True)

    evidence: tuple[Evidence, ...] = ()
    observations: tuple[Observation, ...] = ()
    rejections: tuple[RejectedRawOutput, ...] = ()


class ArtifactDecoder(Protocol, Generic[TArtifact]):
    """`decode(artifact, context) -> DecodedArtifact`. Stateless and deterministic -- exactly what
    makes an implementation safe to share across providers and to unit test without any backend,
    model, or network call.
    """

    artifact_type: str
    """A short, stable name ("plain_text", "html", "labeled_block_list") -- recorded in decoder
    telemetry, never inferred from a class name."""

    decoder_version: str
    """A short, stable version string for *this decoder's own mapping logic* (Production
    Observability Completion milestone: "Artifact Decoder Completion... Implement version
    tracking"), independent of `provider_version` (the model/checkpoint) and
    `reconciliation_policy_version`/`confidence_policy_version` (comparison/confidence policy).
    Bumped only when a decoder's own label-to-payload mapping or structural extraction changes in a
    way that would produce different Canonical Observations from the same native artifact -- so a
    later replay can tell "the model changed" (`provider_version`) apart from "the decoding rules
    changed" (`decoder_version`) as two independently-versioned, independently-reproducible facts.
    `run_decoder` stamps this into every resulting Evidence's `supporting_metadata` uniformly, so no
    individual decoder implementation can forget to record it."""

    def decode(self, artifact: TArtifact, *, context: DecodeContext) -> DecodedArtifact: ...


class DecoderTelemetryKind(str, Enum):
    ARTIFACT_DECODED = "ArtifactDecoded"
    ARTIFACT_DECODE_FAILED = "ArtifactDecodeFailed"


class DecoderTelemetryEvent(BaseModel):
    """Records one decode attempt -- deliberately separate from `RuntimeTelemetryEvent`
    (`runtime/runtime_telemetry.py`, which records *inference*-stage facts: containers, GPU,
    warm-up) and from `domain/telemetry/events.py` (canonical knowledge-evolution events, Article
    16). "This keeps inference telemetry separate from deterministic decoding telemetry" -- a
    decode failure is a parsing/mapping bug, never a model or GPU problem, and should never be
    conflated with either in a diagnostics view.
    """

    model_config = ConfigDict(frozen=True)

    kind: DecoderTelemetryKind
    recorded_at: float
    artifact_type: str
    decoder_name: str
    provider_id: str
    seconds: float | None = None
    observation_count: int | None = None
    detail: str | None = None


class DecoderTelemetrySink(Protocol):
    def record(self, event: DecoderTelemetryEvent) -> None: ...


class InMemoryDecoderTelemetrySink:
    def __init__(self) -> None:
        self._events: list[DecoderTelemetryEvent] = []

    def record(self, event: DecoderTelemetryEvent) -> None:
        self._events.append(event)

    def events(self) -> tuple[DecoderTelemetryEvent, ...]:
        return tuple(self._events)


def run_decoder(
    decoder: ArtifactDecoder[TArtifact],
    artifact: TArtifact,
    *,
    context: DecodeContext,
    telemetry: DecoderTelemetrySink | None = None,
) -> DecodedArtifact:
    """The one place decoder telemetry is recorded (Part: "Record the decoder stage") -- every
    provider importer calls this instead of `decoder.decode(...)` directly, so `ArtifactDecoded`/
    `ArtifactDecodeFailed` are emitted uniformly regardless of which decoder or provider is
    involved, and decoders themselves stay pure (no telemetry-sink dependency to inject/mock).
    """
    started = time.monotonic()
    decoder_name = type(decoder).__name__
    try:
        result = decoder.decode(artifact, context=context)
    except Exception as exc:  # noqa: BLE001 -- re-raised after recording; a decode failure is a
        # real error (unlike a provider's own malformed-output rejection, which is a *result*, not
        # an exception) -- deterministic code raising here means a genuine decoder bug or an
        # artifact shape outside its contract, not a rejected inference.
        if telemetry is not None:
            telemetry.record(
                DecoderTelemetryEvent(
                    kind=DecoderTelemetryKind.ARTIFACT_DECODE_FAILED,
                    recorded_at=time.time(),
                    artifact_type=decoder.artifact_type,
                    decoder_name=decoder_name,
                    provider_id=context.provider_id,
                    seconds=time.monotonic() - started,
                    detail=str(exc),
                )
            )
        raise
    if telemetry is not None:
        telemetry.record(
            DecoderTelemetryEvent(
                kind=DecoderTelemetryKind.ARTIFACT_DECODED,
                recorded_at=time.time(),
                artifact_type=decoder.artifact_type,
                decoder_name=decoder_name,
                provider_id=context.provider_id,
                seconds=time.monotonic() - started,
                observation_count=len(result.observations),
            )
        )
    return _stamp_decoder_version(result, decoder)


def _stamp_decoder_version(result: DecodedArtifact, decoder: ArtifactDecoder) -> DecodedArtifact:
    """Stamps `decoder_version` into every Evidence's `supporting_metadata`, uniformly, regardless
    of which decoder produced it -- centralized here (not left to each decoder implementation) so
    it can never be forgotten by a future decoder (Production Observability Completion milestone:
    "Artifact Decoder Completion... Implement version tracking").
    """
    version = getattr(decoder, "decoder_version", None)
    if version is None or not result.evidence:
        return result
    stamped = tuple(
        evidence.model_copy(
            update={"supporting_metadata": {**evidence.supporting_metadata, "decoder_version": version}}
        )
        for evidence in result.evidence
    )
    return result.model_copy(update={"evidence": stamped})
