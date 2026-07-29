"""Region payload (docs/htr-domain-design.md §1, §7; migration Stage 2). One detected region
produced by an independent segmentation stage -- distinct from `LAYOUT_REGION`'s catch-all
provider-native-label role: `REGION` is the segmentation-stage-native unit that owns `TextLine`s
underneath it (§7's `SegmentationAdapter.detect_regions`), never a fallback category."""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class RegionPayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.REGION

    region_type: str | None = None
    order_index: int | None = None
