"""Page payload (MILESTONE1_DOMAIN_MODEL.md S5.2). The stable top-level container for one page's
worth of claims. No missing metadata or ambiguity found during Milestone 1 review.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class PagePayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.PAGE

    page_number: int
