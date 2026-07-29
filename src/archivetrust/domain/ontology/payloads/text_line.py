"""Text Line payload (docs/htr-domain-design.md §1, §7; migration Stage 2). One detected text
line at the segmentation granularity HTR recognition actually operates on -- the Observation-level
counterpart to `htr.corpus.models.TextLine`."""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class TextLinePayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.TEXT_LINE

    reading_order_index: int
    native_label: str | None = None
