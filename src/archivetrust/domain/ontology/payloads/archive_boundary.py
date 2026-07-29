"""Archive Boundary payload (MILESTONE1_DOMAIN_MODEL.md S5.11). A boundary claim about a specific
page/region, MVP scope still an open question recorded elsewhere (not resolved by this document).
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class ArchiveBoundaryPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.ARCHIVE_BOUNDARY

    boundary_type: str
    description: str | None = None
