from __future__ import annotations

import pytest
from pydantic import ValidationError

from archivetrust.domain.evidence.models import BoundingBox, Precision
from archivetrust.htr.corpus import (
    Collection,
    Dataset,
    DatasetVersion,
    InputCrop,
    Page,
    Region,
    ResearchProject,
    TextLine,
)


def _box() -> BoundingBox:
    return BoundingBox(x0=0, y0=0, x1=10, y1=10, precision=Precision.PIXEL_ACCURATE)


def test_research_project_round_trips():
    project = ResearchProject.create(name="Swedish HTR", created_at="2026-01-01T00:00:00Z")
    restored = ResearchProject.model_validate(project.model_dump())
    assert restored == project


def test_dataset_round_trips():
    project = ResearchProject.create(name="p", created_at="2026-01-01T00:00:00Z")
    dataset = Dataset.create(project_id=project.project_id, name="d", created_at="2026-01-01T00:00:00Z")
    assert dataset.project_id == project.project_id


def test_dataset_version_rejects_version_below_one():
    with pytest.raises(ValidationError):
        DatasetVersion(
            dataset_version_id="dataset_version_x",
            dataset_id="dataset_x",
            version=0,
            collection_ids=(),
            created_at="2026-01-01T00:00:00Z",
        )


def test_dataset_version_valid_construction():
    version = DatasetVersion.create(
        dataset_id="dataset_x", version=1, collection_ids=("collection_a",), created_at="2026-01-01T00:00:00Z"
    )
    assert version.version == 1
    restored = DatasetVersion.model_validate(version.model_dump())
    assert restored == version


def test_collection_references_archive_objects_by_id():
    collection = Collection.create(
        dataset_id="dataset_x",
        name="Batch 1",
        archive_object_refs=("archive_object_1",),
        created_at="2026-01-01T00:00:00Z",
    )
    assert collection.archive_object_refs == ("archive_object_1",)


def test_page_rejects_non_positive_page_number():
    with pytest.raises(ValidationError):
        Page.create(archive_object_ref="archive_object_1", page_number=0)


def test_region_round_trips():
    region = Region.create(page_id="page_1", bounding_box=_box(), region_type="text_block")
    restored = Region.model_validate(region.model_dump())
    assert restored == region


def test_text_line_rejects_negative_reading_order():
    with pytest.raises(ValidationError):
        TextLine.create(region_id="region_1", bounding_box=_box(), reading_order_index=-1)


def test_input_crop_is_content_addressed():
    a = InputCrop.create(
        image_bytes=b"same-bytes", text_line_id="text_line_1", storage_path="crops/a.png"
    )
    b = InputCrop.create(
        image_bytes=b"same-bytes", text_line_id="text_line_2", storage_path="crops/b.png"
    )
    assert a.hash == b.hash

    c = InputCrop.create(
        image_bytes=b"different-bytes", text_line_id="text_line_1", storage_path="crops/c.png"
    )
    assert a.hash != c.hash


def test_input_crop_rejects_hand_set_hash():
    with pytest.raises(ValidationError):
        InputCrop(
            crop_id="input_crop_x",
            hash="not-a-real-hash",
            text_line_id="text_line_1",
            storage_path="crops/x.png",
            byte_size=10,
        )
