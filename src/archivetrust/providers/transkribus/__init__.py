from __future__ import annotations

from archivetrust.providers.transkribus.adapter import TranskribusAdapter
from archivetrust.providers.transkribus.external_import import (
    ExternalImport,
    ExternalImportFormat,
    ImportAssociation,
    ImportTargetKind,
    associate_external_import,
)

__all__ = [
    "ExternalImport",
    "ExternalImportFormat",
    "ImportAssociation",
    "ImportTargetKind",
    "associate_external_import",
    "TranskribusAdapter",
]
