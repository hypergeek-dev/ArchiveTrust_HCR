"""Transcription payloads (docs/htr-domain-design.md §1, §2, §4; migration Stage 2).

Three stages of one line's recognized text, mirroring the retained
`RawResult -> ParsedResult -> NormalizedResult` chain (§2, §4) at the Observation/payload layer:
`RAW_TRANSCRIPTION` is the method's untouched output text; `PARSED_TRANSCRIPTION` is that output
after structural parsing (e.g. splitting method-specific markup); `NORMALIZED_TRANSCRIPTION` is
the final, convention-normalized text a `CanonicalResult` span can be built from.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class RawTranscriptionPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.RAW_TRANSCRIPTION

    text: str


class ParsedTranscriptionPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.PARSED_TRANSCRIPTION

    text: str


class NormalizedTranscriptionPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.NORMALIZED_TRANSCRIPTION

    text: str
    convention_id: str | None = None
    convention_version: int | None = None
