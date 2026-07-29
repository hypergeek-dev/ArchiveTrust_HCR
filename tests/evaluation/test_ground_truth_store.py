"""Release WS6: the append-only ground-truth store — identity validation, supersession, export."""

from __future__ import annotations

import json

import pytest

from archivetrust.evaluation.ground_truth import (
    AdjudicationStatus,
    FileGroundTruthStore,
    GroundTruthAnnotation,
    GroundTruthValidationError,
)


def _annotation(annotation_id: str, *, text="Innehåll", supersedes=None, **overrides):
    fields = dict(
        annotation_id=annotation_id,
        archive_object_ref="archive_object_1",
        content_hash="abc123",
        page=1,
        field="page1_heading",
        observation_type="heading",
        text=text,
        annotator="tester",
        method="unit fixture",
        source="test campaign",
        created_at="2026-07-16T00:00:00+00:00",
        supersedes=supersedes,
    )
    fields.update(overrides)
    return GroundTruthAnnotation(**fields)


def test_append_and_read_back(tmp_path):
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    store.append(_annotation("gt_1"), expected_content_hash="abc123")
    (record,) = store.latest()
    assert record.text == "Innehåll"
    assert record.adjudication_status is AdjudicationStatus.UNADJUDICATED


def test_content_hash_mismatch_is_refused(tmp_path):
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    with pytest.raises(GroundTruthValidationError, match="different document"):
        store.append(_annotation("gt_1"), expected_content_hash="OTHER")


def test_supersession_is_append_only_and_resolves_to_latest(tmp_path):
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    store.append(_annotation("gt_1", text="Inehåll"))
    store.append(_annotation("gt_2", text="Innehåll", supersedes="gt_1", revision=2))

    assert len(store.all_records()) == 2  # history is never rewritten
    (latest,) = store.latest()
    assert latest.annotation_id == "gt_2"
    assert latest.text == "Innehåll"


def test_supersedes_unknown_id_is_refused(tmp_path):
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    with pytest.raises(GroundTruthValidationError, match="unknown annotation"):
        store.append(_annotation("gt_2", supersedes="gt_missing"))


def test_illegible_requires_no_text_and_vice_versa(tmp_path):
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    with pytest.raises(GroundTruthValidationError):
        store.append(_annotation("gt_1", text="x", illegible=True))
    with pytest.raises(GroundTruthValidationError):
        store.append(_annotation("gt_2", text=None))
    store.append(_annotation("gt_3", text=None, illegible=True))


def test_export_carries_provenance_and_integrity_sidecar(tmp_path):
    store = FileGroundTruthStore(tmp_path / "annotations.jsonl")
    store.append(_annotation("gt_1", text="Old"))
    store.append(_annotation("gt_2", text="New", supersedes="gt_1", revision=2))

    output = store.export_with_provenance(tmp_path / "dataset.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total_records"] == 2
    assert payload["superseded_records"] == ["gt_1"]
    assert [a["annotation_id"] for a in payload["annotations"]] == ["gt_2"]
    assert (tmp_path / "dataset.json.integrity.json").exists()
