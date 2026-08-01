"""Corpus entities (docs/htr-domain-design.md §1, §5).

`ResearchProject → Dataset → DatasetVersion → Collection → Document → Page → Region → TextLine →
InputCrop`. Owned by this package per the domain design's §5 data-ownership table.

**Document is deliberately not a new model here.** Per §5: "`Document`/`Page` reuse the existing
`ArchiveObject` acquisition/blob-store mechanism (`acquisition/`,
`infrastructure/storage/blob_store.py`) rather than reinventing file storage." Every place this
package would otherwise reference "Document" instead stores an `archive_object_ref: str` -- the
id of the existing `archivetrust.acquisition.archive_object.ArchiveObject` -- exactly the
"reference by id, never embed" discipline already enforced on `Observation.evidence_ids`
(Constitution Article 7). No new Document class exists; ArchiveObject *is* the document concept
at this layer.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivetrust.domain.evidence.models import BoundingBox
from archivetrust.domain.shared.ids import new_id


class ResearchProject(BaseModel):
    """The top-level grouping a research effort is organized under (§1)."""

    model_config = ConfigDict(frozen=True)

    project_id: str
    name: str
    description: str | None = None
    created_at: str

    @classmethod
    def create(cls, *, name: str, description: str | None = None, created_at: str) -> "ResearchProject":
        return cls(
            project_id=new_id("research_project"),
            name=name,
            description=description,
            created_at=created_at,
        )


class CorpusProfile(BaseModel):
    """Language/domain/provenance metadata for a `Dataset` (docs/experiments/lion-loghi-comparison/
    dataset-comparability.md). Its own model, not flattened fields on `Dataset` -- it groups every
    field the Swedish-vs-Dutch corpus comparison needs, all optional/`None` when genuinely unknown
    (never a fabricated classification standing in for "not yet assessed").
    """

    model_config = ConfigDict(frozen=True)

    language: str | None = None
    historical_period: str | None = None
    document_type: str | None = None
    source: str | None = None
    rights: str | None = None
    page_count: int | None = None
    writer_diversity: str | None = None
    inclusion_criteria: str | None = None
    exclusion_criteria: str | None = None
    difficulty_classification: str | None = None
    handwriting_style: str | None = None
    image_condition: str | None = None
    layout_class: str | None = None


class Dataset(BaseModel):
    """A named, evolving corpus within a `ResearchProject`. Mutable only in the sense that new
    `DatasetVersion`s are created over time -- the `Dataset` record itself carries no membership
    list (that lives on `DatasetVersion`, §3: "immutable snapshot ... new documents create a new
    DatasetVersion, never mutate an existing one")."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    project_id: str
    name: str
    description: str | None = None
    created_at: str
    corpus_profile: CorpusProfile | None = None
    """Additive field -- `None` for every pre-existing `Dataset` record (the Swedish
    technical-reliability-screening corpus included), which replays unchanged. Populated for the new
    Swedish/Dutch datasets in the Lion-vs-Loghi comparison."""

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        name: str,
        description: str | None = None,
        created_at: str,
        corpus_profile: CorpusProfile | None = None,
    ) -> "Dataset":
        return cls(
            dataset_id=new_id("dataset"),
            project_id=project_id,
            name=name,
            description=description,
            created_at=created_at,
            corpus_profile=corpus_profile,
        )


class DatasetVersion(BaseModel):
    """An immutable snapshot of a `Dataset`'s collection membership at a point in time (§3).
    Never mutated -- a membership change is always a new `DatasetVersion` with an incremented
    `version`, referencing every `Collection` it snapshots by id.
    """

    model_config = ConfigDict(frozen=True)

    dataset_version_id: str
    dataset_id: str
    version: int
    collection_ids: tuple[str, ...]
    created_at: str
    supersedes: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "DatasetVersion":
        if self.version < 1:
            raise ValueError("DatasetVersion.version must be >= 1")
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        version: int,
        collection_ids: tuple[str, ...],
        created_at: str,
        supersedes: str | None = None,
    ) -> "DatasetVersion":
        return cls(
            dataset_version_id=new_id("dataset_version"),
            dataset_id=dataset_id,
            version=version,
            collection_ids=collection_ids,
            created_at=created_at,
            supersedes=supersedes,
        )


class Collection(BaseModel):
    """A named group of Documents (ArchiveObjects) within a Dataset -- e.g. one archival fond or
    one imaging batch (§1)."""

    model_config = ConfigDict(frozen=True)

    collection_id: str
    dataset_id: str
    name: str
    archive_object_refs: tuple[str, ...] = ()
    """`ArchiveObject.id` references -- the Documents this Collection contains. See module
    docstring: Document = ArchiveObject, referenced by id, never embedded."""
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        name: str,
        archive_object_refs: tuple[str, ...] = (),
        created_at: str,
    ) -> "Collection":
        return cls(
            collection_id=new_id("collection"),
            dataset_id=dataset_id,
            name=name,
            archive_object_refs=archive_object_refs,
            created_at=created_at,
        )


class Page(BaseModel):
    """One page of a Document (ArchiveObject), referenced by id -- the unit segmentation operates
    over (§1, §7)."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    archive_object_ref: str
    """`ArchiveObject.id` this page belongs to."""
    page_number: int
    width: int | None = None
    height: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> "Page":
        if self.page_number < 1:
            raise ValueError("Page.page_number must be >= 1")
        return self

    @classmethod
    def create(
        cls,
        *,
        archive_object_ref: str,
        page_number: int,
        width: int | None = None,
        height: int | None = None,
    ) -> "Page":
        return cls(
            page_id=new_id("page"),
            archive_object_ref=archive_object_ref,
            page_number=page_number,
            width=width,
            height=height,
        )


class Region(BaseModel):
    """A detected region on a `Page`, produced by a segmentation stage (§1, §7). Independent of
    any recognition method -- `SegmentationAdapter.detect_regions` (htr/segmentation) is what
    produces these."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    page_id: str
    bounding_box: BoundingBox
    region_type: str | None = None
    """Free-form segmentation-adapter label (e.g. "text_block", "marginalia"); non-branchable
    audit metadata, matching `LayoutRegionPayload.native_label`'s discipline."""
    order_index: int | None = None

    @classmethod
    def create(
        cls,
        *,
        page_id: str,
        bounding_box: BoundingBox,
        region_type: str | None = None,
        order_index: int | None = None,
    ) -> "Region":
        return cls(
            region_id=new_id("region"),
            page_id=page_id,
            bounding_box=bounding_box,
            region_type=region_type,
            order_index=order_index,
        )


class TextLine(BaseModel):
    """A detected text line within a `Region`, produced by `SegmentationAdapter.detect_lines`
    and ordered by `.order_lines` (§7). The unit `InputCrop`s are cropped from."""

    model_config = ConfigDict(frozen=True)

    text_line_id: str
    region_id: str
    bounding_box: BoundingBox
    reading_order_index: int

    @model_validator(mode="after")
    def _validate(self) -> "TextLine":
        if self.reading_order_index < 0:
            raise ValueError("TextLine.reading_order_index must be >= 0")
        return self

    @classmethod
    def create(
        cls, *, region_id: str, bounding_box: BoundingBox, reading_order_index: int
    ) -> "TextLine":
        return cls(
            text_line_id=new_id("text_line"),
            region_id=region_id,
            bounding_box=bounding_box,
            reading_order_index=reading_order_index,
        )


class InputCrop(BaseModel):
    """Hash-addressed image bytes cropped for one `TextLine`, shared byte-identical across every
    method in a controlled comparison (§7: "`InputCrop.hash` must match across all `MethodRun`s in
    a controlled comparison, enforced by an assertion in the experiment runner").

    Bytes are stored out of line (`storage_path`, mirroring `ArchiveObject.storage_path`) --
    `hash` is what identifies the crop's content, not where it is stored.
    """

    model_config = ConfigDict(frozen=True)

    crop_id: str
    hash: str
    """Content address of the raw image bytes (sha256, `prefix="crop"`) -- identical bytes always
    produce the same `hash`, so the same crop is never stored twice."""
    text_line_id: str
    storage_path: str
    byte_size: int
    width: int | None = None
    height: int | None = None

    @model_validator(mode="after")
    def _validate_hash(self) -> "InputCrop":
        if not self.hash.startswith("crop_"):
            raise ValueError(
                "InputCrop.hash must be content-addressed via InputCrop.compute_hash / "
                "InputCrop.create, never hand-set"
            )
        return self

    @staticmethod
    def compute_hash(image_bytes: bytes) -> str:
        digest = hashlib.sha256(image_bytes).hexdigest()
        return f"crop_{digest}"

    @classmethod
    def create(
        cls,
        *,
        image_bytes: bytes,
        text_line_id: str,
        storage_path: str,
        width: int | None = None,
        height: int | None = None,
    ) -> "InputCrop":
        """The only supported construction path -- computes the content-addressed `hash`."""
        return cls(
            crop_id=new_id("input_crop"),
            hash=cls.compute_hash(image_bytes),
            text_line_id=text_line_id,
            storage_path=storage_path,
            byte_size=len(image_bytes),
            width=width,
            height=height,
        )
