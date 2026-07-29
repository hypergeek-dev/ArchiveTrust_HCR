"""Caption payload (MILESTONE1_DOMAIN_MODEL.md S5.12). Text content stays here; which Observation
it captions is a graph fact, expressed as a `related_observations` edge (relation type "captions"),
not a payload field (Graph-Reference Rule, S5.1).
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class CaptionPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.CAPTION

    text: str
