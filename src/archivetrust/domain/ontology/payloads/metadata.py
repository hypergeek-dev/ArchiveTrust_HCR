"""Metadata payload (MILESTONE1_DOMAIN_MODEL.md S5.8). One Observation per key, preserving
per-key Evidence traceability and confidence, at the cost of Observation-count scale on
metadata-rich documents -- a flagged, not resolved, design tension per S5.8.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class MetadataPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.METADATA

    key: str
    value: str
