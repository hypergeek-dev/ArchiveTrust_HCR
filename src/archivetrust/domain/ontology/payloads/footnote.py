"""Footnote payload (MILESTONE1_DOMAIN_MODEL.md S5.15). Same finding as Caption (S5.12): which
Observation it annotates is a graph fact (`related_observations`, relation type "annotates"), not
a payload field.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class FootnotePayload(ObservationPayload):
    observation_type: ClassVar[ObservationType] = ObservationType.FOOTNOTE

    text: str
