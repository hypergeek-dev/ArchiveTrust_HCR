from __future__ import annotations

from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import HeadingPayload, ParagraphPayload


def evidence(provider: str, text: str, bbox: BoundingBox | None) -> Evidence:
    return Evidence.create(
        provider=provider,
        provider_version="1.0",
        raw_output=text,
        processing_stage=ProcessingStage.OCR,
        page=1,
        bounding_box=bbox,
    )


def observation(provider: str, payload, ev: Evidence) -> Observation:
    return Observation.from_evidence(
        provider_id=provider, provider_version="1.0", payload=payload, evidence=(ev,)
    )


def heading(text: str):
    return HeadingPayload(text=text, level=1)


def paragraph(text: str):
    return ParagraphPayload(text=text)


def box(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, precision=Precision.PIXEL_ACCURATE)
