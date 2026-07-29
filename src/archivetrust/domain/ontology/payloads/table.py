"""Table payload (MILESTONE1_DOMAIN_MODEL.md S5.13, S6.1).

`TableCellPayload`/`ObservationType.TABLE_CELL` (cell-level child Observations of a Table
Observation, and the cell-level neutral-lattice reconciliation pass they backed,
`domain/comparison/table_reconciliation.py`) were deleted in docs/htr-migration-plan.md Stage 5
(EXECUTED): table semantics don't fit line-level HTR research and no in-scope HTR method produces
structured tables. `TablePayload`/`ObservationType.TABLE` itself is retained as a distinct,
non-cell type.
"""

from __future__ import annotations

from typing import ClassVar

from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.ontology.types import ObservationType


class TablePayload(ObservationPayload):
    """The table as a whole."""

    observation_type: ClassVar[ObservationType] = ObservationType.TABLE

    row_count: int
    column_count: int
    caption_text: str | None = None
