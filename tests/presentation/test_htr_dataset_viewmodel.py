"""Dataset explorer ViewModel: the read-only corpus tree and its per-line detail."""

from __future__ import annotations

import pytest

from archivetrust.htr.corpus.models import InputCrop
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.presentation.htr_dataset_viewmodel import DatasetExplorerViewModel
from tests.presentation._htr_fixtures import (
    ARCHIVE_OBJECT_REF,
    SYNTHETIC_FIXTURE_LINE_0,
    build_fixture_corpus,
)


def _viewmodel(corpus) -> DatasetExplorerViewModel:
    return DatasetExplorerViewModel(corpus.store)


def test_tree_walks_project_to_crop_with_correct_child_counts() -> None:
    corpus = build_fixture_corpus()
    vm = _viewmodel(corpus)

    projects = vm.projects()
    assert [p.label for p in projects] == ["Swedish Historical HTR"]
    assert projects[0].child_count == 1

    datasets = vm.datasets(corpus.project_id)
    assert [d.label for d in datasets] == ["Trolldomskommissionen"]
    assert datasets[0].child_count == 1

    versions = vm.dataset_versions(corpus.dataset_id)
    assert [v.label for v in versions] == ["Version 1"]

    collections = vm.collections(corpus.dataset_version_id)
    assert [c.label for c in collections] == ["Court records 1670s"]

    documents = vm.documents(corpus.collection_id)
    assert [d.node_id for d in documents] == [ARCHIVE_OBJECT_REF]
    assert documents[0].child_count == 1

    pages = vm.pages(ARCHIVE_OBJECT_REF)
    assert [p.label for p in pages] == ["Page 1"]
    assert "2480x3508" in pages[0].detail

    regions = vm.regions(corpus.page_id)
    assert [r.label for r in regions] == ["text_block"]
    assert regions[0].child_count == 2

    lines = vm.text_lines(corpus.region_id)
    assert [line.label for line in lines] == ["Line 0", "Line 1"]

    crops = vm.input_crops(corpus.line_0_id)
    assert len(crops) == 1
    assert crops[0].node_id == corpus.crop_a_id


def test_line_detail_reports_ground_truth_and_every_method_that_ran() -> None:
    corpus = build_fixture_corpus()
    detail = _viewmodel(corpus).line_detail(corpus.line_0_id)

    assert detail is not None
    assert detail.ground_truth == SYNTHETIC_FIXTURE_LINE_0
    assert detail.reading_order_index == 0
    assert detail.page_id == corpus.page_id
    assert set(detail.method_outputs) == {
        "SATRN (Riksarkivet)",
        "Florence-2 (vlm-htr line OCR)",
        "Transkribus Swedish Lion I",
    }


def test_single_crop_hash_marks_the_line_as_a_verified_controlled_comparison() -> None:
    corpus = build_fixture_corpus()
    detail = _viewmodel(corpus).line_detail(corpus.line_0_id)

    assert detail is not None
    assert detail.crop_hashes == (corpus.crop_a_hash,)
    assert detail.shared_crop_verified is True


def test_two_differing_crop_hashes_for_one_line_are_reported_as_not_shared() -> None:
    """Three states, not two: `docs/htr-domain-design.md` §7 requires a controlled comparison to
    hold `InputCrop.hash` identical across every method run, so a line whose crops disagree must be
    visibly NOT controlled -- not quietly treated as fine."""
    corpus = build_fixture_corpus()
    corpus.store.register_input_crop(
        InputCrop.create(
            image_bytes=b"a-differently-cropped-version-of-the-same-line",
            text_line_id=corpus.line_0_id,
            storage_path="crops/a2.png",
        )
    )

    detail = _viewmodel(corpus).line_detail(corpus.line_0_id)
    assert detail is not None
    assert len(detail.crop_hashes) == 2
    assert detail.shared_crop_verified is False


def test_line_with_no_crop_reports_unknown_rather_than_false() -> None:
    from archivetrust.domain.evidence.models import BoundingBox, Precision
    from archivetrust.htr.corpus.models import TextLine

    corpus = build_fixture_corpus()
    bare = TextLine.create(
        region_id=corpus.region_id,
        bounding_box=BoundingBox(x0=0, y0=0, x1=1, y1=1, precision=Precision.PIXEL_ACCURATE),
        reading_order_index=2,
    )
    corpus.store.register_text_line(bare)

    detail = _viewmodel(corpus).line_detail(bare.text_line_id)
    assert detail is not None
    assert detail.crop_hashes == ()
    assert detail.shared_crop_verified is None  # not segmented yet, not "not shared"


def test_collection_referenced_but_not_registered_is_shown_as_a_gap() -> None:
    from archivetrust.htr.corpus.models import DatasetVersion

    corpus = build_fixture_corpus()
    dangling = DatasetVersion.create(
        dataset_id=corpus.dataset_id,
        version=2,
        collection_ids=("collection_never_registered",),
        created_at="2026-07-04T00:00:00Z",
    )
    corpus.store.register_dataset_version(dangling)

    nodes = _viewmodel(corpus).collections(dangling.dataset_version_id)
    assert len(nodes) == 1
    assert "not registered" in nodes[0].detail


def test_unknown_ids_return_empty_rather_than_raising() -> None:
    vm = DatasetExplorerViewModel(HtrResearchStore())
    assert vm.projects() == ()
    assert vm.datasets("nope") == ()
    assert vm.collections("nope") == ()
    assert vm.documents("nope") == ()
    assert vm.line_detail("nope") is None


def test_viewmodel_exposes_no_mutating_method() -> None:
    """The brief calls this a read-only navigation surface; enforce that structurally so a future
    edit cannot quietly add a writer here."""
    public = {name for name in dir(DatasetExplorerViewModel) if not name.startswith("_")}
    forbidden_prefixes = ("set_", "add_", "remove_", "delete_", "update_", "register_", "save")
    assert not [name for name in public if name.startswith(forbidden_prefixes)]


def test_review_status_defaults_to_not_reviewed_never_to_blank() -> None:
    corpus = build_fixture_corpus()
    vm = DatasetExplorerViewModel(
        corpus.store, review_status_by_line={corpus.line_1_id: "Adjudicated"}
    )

    assert vm.line_detail(corpus.line_0_id).review_status == "Not reviewed"
    assert vm.line_detail(corpus.line_1_id).review_status == "Adjudicated"


def test_registering_the_same_entity_twice_is_refused() -> None:
    from archivetrust.htr.research_store import DuplicateRegistrationError

    corpus = build_fixture_corpus()
    page = corpus.store.page(corpus.page_id)
    with pytest.raises(DuplicateRegistrationError):
        corpus.store.register_page(page)
