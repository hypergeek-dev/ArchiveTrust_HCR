"""Geometry helpers shared by clustering (S2) and table reconciliation (S6).

Bounding Box is a derived, read-through property of an Observation's contributing Evidence
(`MILESTONE1_DOMAIN_MODEL.md` S4.2) -- never independently asserted here.
"""

from __future__ import annotations

from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision
from archivetrust.domain.ontology.base import Observation


def bounding_box_of(observation: Observation, evidence_by_id: dict[str, Evidence]) -> BoundingBox | None:
    """The first contributing Evidence's bounding box, if any. Simplification, documented: an
    Observation with multiple contributing Evidence records (rare pre-reconciliation -- most
    provider adapters emit one Evidence per Observation) uses only the first; full multi-Evidence
    geometric agreement is Phase F's job for *Canonical* Observations, not this pre-clustering
    read-through.
    """
    for evidence_id in observation.evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is not None and evidence.bounding_box is not None:
            return evidence.bounding_box
    return None


def page_of(observation: Observation, evidence_by_id: dict[str, Evidence]) -> int | None:
    for evidence_id in observation.evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is not None and evidence.page is not None:
            return evidence.page
    return None


def intersection_over_union(a: BoundingBox, b: BoundingBox) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def both_pixel_accurate(a: BoundingBox | None, b: BoundingBox | None) -> bool:
    return (
        a is not None
        and b is not None
        and a.precision == Precision.PIXEL_ACCURATE
        and b.precision == Precision.PIXEL_ACCURATE
    )
