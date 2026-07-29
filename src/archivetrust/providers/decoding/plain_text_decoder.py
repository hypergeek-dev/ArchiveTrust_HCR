"""`PlainTextDecoder` (Native Artifact Decoder Architecture addendum): normalizes whitespace,
preserves reading order and real paragraph boundaries, and maps each paragraph into a Canonical
Text Observation. Deterministic and provider-agnostic -- used initially by PaddleOCR-VL's `"OCR:"`
task (a bare plain-text transcription, no labels, no geometry, no confidence), reusable by any
future provider whose native artifact is the same shape.

Does not infer layout and does not hallucinate structure: a paragraph boundary is recognized only
where the source text already has one (a blank line) -- there is no heading detection, no
sentence-splitting, nothing invented that the artifact did not already express.
"""

from __future__ import annotations

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import ParagraphPayload
from archivetrust.providers.decoding.base import DecodeContext, DecodedArtifact


def _normalize_whitespace(text: str) -> str:
    """Normalizes line endings and trims trailing whitespace from each line -- never collapses an
    internal newline within a paragraph into a space, since that newline is part of the artifact's
    own reading order (a caller comparing exact transcribed text expects it preserved)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _split_paragraphs(text: str) -> tuple[str, ...]:
    """Splits on blank-line boundaries -- the one paragraph signal plain text can honestly encode.
    Text with no blank lines at all becomes exactly one paragraph (matches this decoder's only
    predecessor, `PaddleOCRVLAdapter`'s pre-decoder behavior: one whole-page transcription, one
    Observation, for output containing no blank-line breaks).
    """
    if not text:
        return ()
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip() == "":
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append("\n".join(current).strip())
    return tuple(p for p in paragraphs if p)


class PlainTextDecoder:
    artifact_type = "plain_text"
    decoder_version = "1"

    def __init__(
        self,
        *,
        mapping_table_entry_id: str | None = None,
        mapping_table_version: int | None = None,
    ) -> None:
        """`mapping_table_entry_id`/`mapping_table_version` (Constitution Article 28) are constant
        for every paragraph this decoder produces -- unlike `LabeledBlockDecoder`/`HtmlDecoder`,
        there is no per-block label to vary the mapping (a plain-text artifact carries none), so
        these are supplied once at construction (mirroring `processing_stage`/`graph_source`'s own
        per-provider-configuration pattern) rather than computed per call.
        """
        self._mapping_table_entry_id = mapping_table_entry_id
        self._mapping_table_version = mapping_table_version

    def decode(self, artifact: str, *, context: DecodeContext) -> DecodedArtifact:
        paragraphs = _split_paragraphs(_normalize_whitespace(artifact))
        if not paragraphs:
            # A genuinely blank/whitespace-only artifact is not a decode failure (Operational
            # Completion milestone's documented distinction) -- zero observations, never a
            # fabricated rejection for a real, empty transcription.
            return DecodedArtifact()

        evidence_list: list[Evidence] = []
        observations: list[Observation] = []
        for sequence_hint, text in enumerate(paragraphs):
            supporting_metadata: dict[str, object] = {
                "sequence_hint": sequence_hint,
                **context.runtime_provenance,
                **context.extra_metadata,
            }
            if self._mapping_table_entry_id is not None:
                supporting_metadata["mapping_table_entry_id"] = self._mapping_table_entry_id
                supporting_metadata["mapping_table_version"] = self._mapping_table_version

            evidence = Evidence.create(
                provider=context.provider_id,
                provider_version=context.provider_version,
                raw_output=text,
                processing_stage=ProcessingStage.VLM_INFERENCE,
                page=context.page_number,
                provider_confidence=None,  # a bare plain-text artifact carries no confidence signal
                prompt=context.prompt,
                supporting_metadata=supporting_metadata,
            )
            evidence_list.append(evidence)
            observations.append(
                Observation.from_evidence(
                    provider_id=context.provider_id,
                    provider_version=context.provider_version,
                    payload=ParagraphPayload(text=text),
                    evidence=(evidence,),
                    graph_source=f"{context.provider_id}-plain-text-reading-order",
                )
            )

        return DecodedArtifact(evidence=tuple(evidence_list), observations=tuple(observations))
