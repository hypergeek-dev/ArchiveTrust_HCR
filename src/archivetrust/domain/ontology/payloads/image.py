"""Image payload (MILESTONE1_DOMAIN_MODEL.md S5.14). Scoped to exclude pixel data -- the Archive
Object owns the pixels (Constitution Article 1); an Image Observation only records that an image
region was detected and what is known about it.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class ImagePayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.IMAGE

    image_type: str | None = None
    alt_text: str | None = None
