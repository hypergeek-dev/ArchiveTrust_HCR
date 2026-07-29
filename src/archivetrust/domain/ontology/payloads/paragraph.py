"""Paragraph payload (MILESTONE1_DOMAIN_MODEL.md S5.7)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class IllegibleSpan(BaseModel):
    """A range of `text` that the provider could not confidently transcribe."""

    model_config = ConfigDict(frozen=True)

    start: int
    end: int

    @model_validator(mode="after")
    def _validate_range(self) -> "IllegibleSpan":
        if self.end < self.start:
            raise ValueError("IllegibleSpan.end must be >= start")
        return self


class ParagraphPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.PARAGRAPH

    text: str
    illegible_spans: tuple[IllegibleSpan, ...] = ()
