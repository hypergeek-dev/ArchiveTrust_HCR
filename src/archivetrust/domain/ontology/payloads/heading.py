"""Heading payload (MILESTONE1_DOMAIN_MODEL.md S5.3).

Open question flagged but explicitly not resolved by S5.3: whether `level` needs its own,
finer-than-Observation confidence granularity (a Heading's existence may be certain while its
nesting depth is not). Resolved for Milestone 1, per Constitution Article 8 (no ontology concept
without evidence): no investigated provider's Milestone 0 report establishes a field-level
confidence mechanism, so Confidence remains Observation-granular here, as everywhere else in the
ontology. Revisit if a future provider investigation produces evidence of field-level confidence
output.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class HeadingPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.HEADING

    text: str
    level: int
