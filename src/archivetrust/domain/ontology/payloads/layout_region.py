"""Layout Region payload (MILESTONE1_DOMAIN_MODEL.md S5.16). The deliberate fallback/catch-all
category for detected regions that don't map to a more specific type. `native_label` is
audit-only, non-branchable metadata -- no downstream logic may branch on its value (that would
violate Provider Independence, Constitution Article 20).
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class LayoutRegionPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.LAYOUT_REGION

    native_label: str
