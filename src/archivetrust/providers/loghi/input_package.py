"""Builds a Loghi container input mount from an already-normalized page (docs/methods/loghi.md
§"Shared page-input policy").

**No new normalization here.** `htr/preprocessing/` already produces a content-addressed, versioned
`NormalizedPageArtifact` per page (`docs/methods/transkribus-swedish-lion-1.md`'s pipeline). This
module's only job is laying that artifact's bytes out at a filesystem path Docker/WSL2 can mount, and
recording the provenance chain (original hash, normalized hash, normalization version, page/dataset
identifiers) on the resulting `LoghiInputPackage` -- exactly the same normalized page hash the Swedish
Lion workflow reads from, so both methods provably consumed byte-identical input where technically
possible (brief: "The same normalized page hash must be referenced by both method workflows").

If Loghi's own containers apply further preprocessing internally, that is recorded as part of the
Loghi pipeline's stage results (`stage_results.py`), never folded back into this module or used to
justify altering what gets normalized here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.preprocessing.models import NormalizedPageArtifact


class LoghiInputPackage(BaseModel):
    """The provenance record for one page handed to a Loghi pipeline run."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    dataset_id: str
    original_image_hash: str
    normalized_image_hash: str
    normalization_version: str
    mounted_input_path: str
    """Filesystem path the container-side `/input` mount will resolve to on the host."""


def build_loghi_input_package(
    *,
    normalized_artifact: NormalizedPageArtifact,
    dataset_id: str,
    mount_root: str,
) -> LoghiInputPackage:
    """Copies `normalized_artifact`'s bytes into `mount_root` (never moves/mutates the original
    normalized-artifact storage) and returns the provenance record `LoghiAdapter`/telemetry attach to
    the resulting pipeline run.
    """
    source_path = Path(normalized_artifact.storage_path)
    destination_dir = Path(mount_root)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name
    shutil.copyfile(source_path, destination_path)

    return LoghiInputPackage(
        page_id=normalized_artifact.page_id,
        dataset_id=dataset_id,
        original_image_hash=normalized_artifact.source_content_hash,
        normalized_image_hash=normalized_artifact.normalized_content_hash,
        normalization_version=normalized_artifact.normalization_version,
        mounted_input_path=str(destination_path),
    )
