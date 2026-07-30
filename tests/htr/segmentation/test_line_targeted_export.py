"""The line-targeted Transkribus export: layout-only PAGE XML plus per-line `InputCrop` records.

The point of this export is that all three methods can be given the *same detected lines*. These
tests hold it to that claim and to its stated limits -- notably that the correspondence recorded is
crop *geometry*, not crop bytes, and that nothing here uploads anything.
"""

from __future__ import annotations

import json

import pytest

from archivetrust.htr.preprocessing.line_targeted_export import (
    LINE_TARGETED_MANIFEST_FILENAME,
    build_line_targeted_export,
    read_line_targeted_manifest,
)
from archivetrust.htr.preprocessing.models import PageImageArtifact, RgbNormalizationConfig
from archivetrust.htr.preprocessing.line_targeted_export import LineTargetedSelection
from archivetrust.htr.preprocessing.rgb_normalization import (
    build_normalized_artifact,
    normalize_page_image,
)
from archivetrust.htr.segmentation import CONFOUND_STATEMENT, Florence2LineDetectorAdapter
from archivetrust.providers.transkribus.page_xml import parse_page_xml
from archivetrust.providers.transkribus.page_xml_writer import (
    PAGE_XML_NAMESPACE,
    PageXmlLine,
    PageXmlRegion,
    build_layout_only_page_xml,
)

from tests.htr.segmentation import _fakes

AT = "2026-07-30T12:00:00+00:00"
ARCHIVE_OBJECT_REF = "archive_object_line_targeted_test"


def _segmented(tmp_path, *, spread: bool):
    """Runs the real segmentation adapter (fake detector) and the real normalization transform."""
    image_bytes = (
        _fakes.spread_bytes(1600, 1000, gutter_x=810) if spread else _fakes.single_page_bytes(600, 800)
    )
    boxes = (
        (((60.0, 100.0, 700.0, 160.0), (60.0, 300.0, 700.0, 360.0)), ((10.0, 120.0, 600.0, 180.0),))
        if spread
        else (((60.0, 100.0, 500.0, 160.0), (60.0, 250.0, 500.0, 310.0)),)
    )
    adapter = Florence2LineDetectorAdapter(facade=_fakes.FakeLineDetectorFacade(boxes))
    page = _fakes.page_image("page_lt_1", image_bytes)
    regions = adapter.detect_regions(page)
    lines = tuple(
        line for region in regions for line in adapter.detect_lines(region, page_image=page)
    )
    ordered = adapter.order_lines(lines)
    crops = adapter.crop_lines(ordered, page_image=page, destination=tmp_path / "crops")

    result = normalize_page_image(image_bytes)
    source = PageImageArtifact.create(
        image_bytes=image_bytes, page_id=page.page_id, storage_path=str(tmp_path / "source.png")
    )
    artifact = build_normalized_artifact(
        result, source_artifact=source, storage_path=str(tmp_path / "normalized.png"), executed_at=AT
    )
    selection = LineTargetedSelection(
        page_id=page.page_id,
        archive_object_ref=ARCHIVE_OBJECT_REF,
        page_number=1,
        normalized_artifact=artifact,
        normalized_image_bytes=result.image_bytes,
        regions=regions,
        text_lines=ordered,
        input_crops=crops,
    )
    return adapter, selection, crops


def _build(tmp_path, selection, adapter):
    return build_line_targeted_export(
        (selection,),
        destination=tmp_path / "packages",
        segmentation_adapter_name=adapter.name,
        segmentation_confound_statement=CONFOUND_STATEMENT,
        normalization_version=selection.normalized_artifact.normalization_version,
        configuration_hash=RgbNormalizationConfig().configuration_hash,
        package_id="line_targeted_test_package",
        created_at=AT,
    )


# -- The PAGE XML writer ----------------------------------------------------------------------


def test_layout_only_page_xml_round_trips_through_this_repos_own_parser():
    """Written by `page_xml_writer`, read by `page_xml` -- the parser that reads real Transkribus
    exports. If the writer emitted something that parser could not read, this export would be
    unusable in exactly the workflow it exists for."""
    xml = build_layout_only_page_xml(
        image_filename="page_0001.png",
        image_width=1600,
        image_height=1000,
        regions=(
            PageXmlRegion(
                region_id="region_left",
                x0=0, y0=0, x1=820, y1=1000,
                reading_order_index=0,
                region_type="paragraph",
                lines=(
                    PageXmlLine(
                        line_id="line_a", x0=60, y0=100, x1=700, y1=160,
                        reading_order_index=0, input_crop_id="input_crop_1", input_crop_hash="crop_abc",
                    ),
                ),
            ),
        ),
        created_at=AT,
    )
    parsed = parse_page_xml(xml)

    assert parsed.page_width == 1600
    assert parsed.page_height == 1000
    assert parsed.image_filename == "page_0001.png"
    assert len(parsed.regions) == 1
    (region,) = parsed.regions
    assert region.region_id == "region_left"
    assert len(region.lines) == 1
    (line,) = region.lines
    assert line.line_id == "line_a"
    assert line.polygon == ((60.0, 100.0), (700.0, 100.0), (700.0, 160.0), (60.0, 160.0))
    assert line.baseline == ((60.0, 160.0), (700.0, 160.0))


def test_every_text_equiv_is_empty_because_transkribus_is_meant_to_fill_it():
    """An empty TextEquiv states "text not yet known"; omitting it would be indistinguishable from
    "this file carries no text at all"."""
    xml = build_layout_only_page_xml(
        image_filename="p.png", image_width=100, image_height=100,
        regions=(
            PageXmlRegion(
                region_id="r", x0=0, y0=0, x1=100, y1=100, reading_order_index=0,
                lines=(PageXmlLine(line_id="l", x0=1, y0=1, x1=99, y1=20, reading_order_index=0),),
            ),
        ),
        created_at=AT,
    )
    assert "<TextEquiv>" in xml
    parsed = parse_page_xml(xml)
    assert all(line.text == "" for region in parsed.regions for line in region.lines)


def test_reading_order_is_written_explicitly_not_left_to_document_order():
    xml = build_layout_only_page_xml(
        image_filename="p.png", image_width=100, image_height=100,
        regions=(
            PageXmlRegion(region_id="r_right", x0=50, y0=0, x1=100, y1=100, reading_order_index=1,
                          lines=(PageXmlLine(line_id="l2", x0=51, y0=1, x1=99, y1=20, reading_order_index=0),)),
            PageXmlRegion(region_id="r_left", x0=0, y0=0, x1=50, y1=100, reading_order_index=0,
                          lines=(PageXmlLine(line_id="l1", x0=1, y0=1, x1=49, y1=20, reading_order_index=0),)),
        ),
        created_at=AT,
    )
    assert "<ReadingOrder>" in xml
    parsed = parse_page_xml(xml)
    assert [r.region_id for r in sorted(parsed.regions, key=lambda r: r.reading_order_index)] == [
        "r_left",
        "r_right",
    ]


def test_writer_uses_the_page_namespace_transkribus_emits():
    xml = build_layout_only_page_xml(
        image_filename="p.png", image_width=10, image_height=10,
        regions=(PageXmlRegion(region_id="r", x0=0, y0=0, x1=10, y1=10, reading_order_index=0,
                               lines=(PageXmlLine(line_id="l", x0=1, y0=1, x1=9, y1=5, reading_order_index=0),)),),
        created_at=AT,
    )
    assert PAGE_XML_NAMESPACE in xml


def test_writer_refuses_an_empty_layout():
    with pytest.raises(ValueError, match="at least one region"):
        build_layout_only_page_xml(
            image_filename="p.png", image_width=10, image_height=10, regions=(), created_at=AT
        )


# -- The package ------------------------------------------------------------------------------


def test_package_writes_one_image_and_one_page_xml_per_page(tmp_path):
    adapter, selection, _ = _segmented(tmp_path, spread=True)
    package = _build(tmp_path, selection, adapter)

    assert len(package.image_paths) == 1
    assert len(package.page_xml_paths) == 1
    assert package.image_paths[0].exists()
    assert package.page_xml_paths[0].exists()
    assert (package.directory / LINE_TARGETED_MANIFEST_FILENAME).exists()
    # One upload per page, not one per line -- the reason option (b) was chosen over (a).
    assert len(package.page_xml_paths) < len(selection.text_lines)


def test_manifest_records_the_input_crop_id_and_hash_for_every_line(tmp_path):
    """The whole purpose: a returned Transkribus line can be associated with the exact `InputCrop`
    SATRN and Florence-2 read."""
    adapter, selection, crops = _segmented(tmp_path, spread=True)
    package = _build(tmp_path, selection, adapter)

    (entry,) = package.manifest.entries
    assert entry.line_count == len(selection.text_lines) == 3
    assert len(entry.lines) == 3

    crop_by_line = {crop.text_line_id: crop for crop in crops}
    for line_entry in entry.lines:
        expected = crop_by_line[line_entry.line_id]
        assert line_entry.input_crop_id == expected.crop_id
        assert line_entry.input_crop_hash == expected.hash
        assert line_entry.input_crop_hash.startswith("crop_")


def test_page_xml_line_ids_are_the_text_line_ids_the_manifest_keys_on(tmp_path):
    """A returned file's line ids must map straight back with no lookup table -- so the ids written
    into the XML and the ids recorded in the manifest have to be the same strings."""
    adapter, selection, _ = _segmented(tmp_path, spread=True)
    package = _build(tmp_path, selection, adapter)

    parsed = parse_page_xml(package.page_xml_paths[0].read_text(encoding="utf-8"))
    xml_line_ids = {line.line_id for region in parsed.regions for line in region.lines}
    manifest_line_ids = {line.line_id for entry in package.manifest.entries for line in entry.lines}
    assert xml_line_ids == manifest_line_ids
    assert xml_line_ids == {line.text_line_id for line in selection.text_lines}


def test_crop_hash_lookup_returns_none_for_a_line_the_package_never_exported(tmp_path):
    """An unrecognized line id means Transkribus produced a line this package did not ask for --
    i.e. it re-ran layout analysis. That must be detectable, not papered over with a guess."""
    adapter, selection, _ = _segmented(tmp_path, spread=False)
    package = _build(tmp_path, selection, adapter)

    known = package.manifest.entries[0].lines[0].line_id
    assert package.manifest.crop_hash_for_line(known) is not None
    assert package.manifest.crop_hash_for_line("text_line_transkribus_invented_this") is None


def test_manifest_carries_the_confound_and_the_byte_identity_caveat(tmp_path):
    """Both must travel with a package handed to a colleague, not live only in a docstring."""
    adapter, selection, _ = _segmented(tmp_path, spread=True)
    package = _build(tmp_path, selection, adapter)
    written = json.loads(package.manifest_path.read_text(encoding="utf-8"))

    assert "Florence-2-family" in written["segmentation_confound_statement"]
    assert "not the same bytes" in written["byte_identity_caveat"]
    assert "manually upload" in written["manual_step_required"]
    assert "layout analysis DISABLED" in written["recognition_only_instructions"]
    assert "not cryptographic proof" in written["correspondence_caveat"]
    assert written["segmentation_adapter_name"] == adapter.name


def test_manifest_reads_back_validated(tmp_path):
    adapter, selection, _ = _segmented(tmp_path, spread=True)
    package = _build(tmp_path, selection, adapter)
    assert read_line_targeted_manifest(package.directory) == package.manifest


def test_a_line_without_a_matching_input_crop_is_refused(tmp_path):
    """The crop correspondence is the package's whole purpose; a line that cannot state which crop
    it is would silently weaken exactly the guarantee the manifest exists to record."""
    adapter, selection, crops = _segmented(tmp_path, spread=False)
    crippled = LineTargetedSelection(
        page_id=selection.page_id,
        archive_object_ref=selection.archive_object_ref,
        page_number=selection.page_number,
        normalized_artifact=selection.normalized_artifact,
        normalized_image_bytes=selection.normalized_image_bytes,
        regions=selection.regions,
        text_lines=selection.text_lines,
        input_crops=crops[:1],  # one line now has no crop
    )
    with pytest.raises(ValueError, match="has no InputCrop"):
        _build(tmp_path, crippled, adapter)


def test_spread_produces_two_regions_in_the_page_xml(tmp_path):
    """The spread split survives into what Transkribus is handed: two TextRegions, one per leaf."""
    adapter, selection, _ = _segmented(tmp_path, spread=True)
    package = _build(tmp_path, selection, adapter)
    parsed = parse_page_xml(package.page_xml_paths[0].read_text(encoding="utf-8"))
    assert len(parsed.regions) == 2
    assert package.manifest.entries[0].region_count == 2
