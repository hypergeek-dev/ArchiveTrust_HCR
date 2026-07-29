"""Section payload (MILESTONE1_DOMAIN_MODEL.md S5.4).

`label` is deliberately absent: S5.4 recommends it be "a display-only alias resolved from the
Section's Heading child (via the graph), not independently authored text," to avoid a second
source of truth for content the Heading child Observation already owns (the same Graph-Reference
Rule discipline as S5.1, applied to derived-not-authored content rather than a graph reference
field). A Section's display label is therefore computed by callers by traversing
`child_observations` to its Heading, never stored here.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class SectionPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.SECTION

    grouping_confidence_hint: float | None = None
