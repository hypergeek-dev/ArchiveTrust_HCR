"""Telemetry emission, failure preservation, and destroy-and-reconstruct replay.

Follows the established pattern in `tests/htr/persistence/test_read_model_reconstruction.py`: the
reconstructed store is built from a **brand-new `FileTelemetrySink` over the same path**, after the
original store and its sink have been deleted. Nothing is carried across in memory and nothing is
reloaded as a whole-store blob.
"""

from __future__ import annotations

import gc
import json

import pytest

from archivetrust.application.htr_journal import HtrJournal, causation_chain
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.preprocessing.models import (
    NormalizationError,
    NormalizationFailureCategory,
    RgbNormalizationConfig,
)
from archivetrust.htr.preprocessing.normalization_service import NormalizationService
from archivetrust.infrastructure.storage.blob_store import (
    BlobIntegrityError,
    ContentAddressedBlobStore,
)
from archivetrust.infrastructure.storage.telemetry_sink import (
    FileTelemetrySink,
    InMemoryTelemetrySink,
)
from tests.htr.preprocessing import _images

PAGE_ID = "page_normalization_test_1"


def _service(tmp_path, *, sink=None, config=None):
    sink = sink if sink is not None else InMemoryTelemetrySink()
    store = DurableHtrResearchStore(sink)
    service = NormalizationService(
        blob_store=ContentAddressedBlobStore(tmp_path / "blobs"), store=store, config=config
    )
    return service, store


def _kinds(sink) -> list[str]:
    return [type(event).__name__ for event in sink.all_events()]


# -- Emission -----------------------------------------------------------------------------------


def test_a_successful_normalization_emits_the_three_success_events(tmp_path):
    sink = InMemoryTelemetrySink()
    service, _ = _service(tmp_path, sink=sink)

    service.normalize_and_record(_images.cmyk(), page_id=PAGE_ID)

    assert _kinds(sink) == [
        "ImageNormalizationStarted",
        "DerivedImageArtifactCreated",
        "ImageNormalizationCompleted",
    ]


def test_the_started_event_is_appended_before_the_transform_runs(tmp_path):
    """Proven via the failure path: an undecodable image still leaves an `ImageNormalizationStarted`
    in the log, which is only possible if the open event precedes the transform."""
    sink = InMemoryTelemetrySink()
    service, _ = _service(tmp_path, sink=sink)

    with pytest.raises(NormalizationError):
        service.normalize_and_record(_images.undecodable(), page_id=PAGE_ID)

    assert _kinds(sink) == ["ImageNormalizationStarted", "ImageNormalizationFailed"]


def test_events_carry_the_full_typed_artifacts(tmp_path):
    sink = InMemoryTelemetrySink()
    service, _ = _service(tmp_path, sink=sink)

    recorded = service.normalize_and_record(_images.palette_with_transparency(), page_id=PAGE_ID)
    started, derived, completed = tuple(sink.all_events())

    assert started.source_artifact == recorded.source_artifact
    assert derived.normalized_artifact == recorded.normalized_artifact
    assert completed.source_content_hash == recorded.source_artifact.content_hash
    assert completed.normalized_content_hash == (
        recorded.normalized_artifact.normalized_content_hash
    )


def test_events_are_correlated_and_causally_chained(tmp_path):
    """`correlation_id` groups the unit of work; `causation_id` makes the chain explicit rather than
    inferred from append order (docs/architecture/htr-event-model.md §4)."""
    sink = InMemoryTelemetrySink()
    service, _ = _service(tmp_path, sink=sink)

    recorded = service.normalize_and_record(
        _images.cmyk(), page_id=PAGE_ID, correlation_id="experiment_run_demo"
    )
    started, derived, completed = tuple(sink.all_events())

    assert {e.correlation_id for e in (started, derived, completed)} == {"experiment_run_demo"}
    assert derived.causation_id == started.event_id
    assert completed.causation_id == derived.event_id

    chain = causation_chain(sink.all_events(), from_event_id=recorded.started_event_id)
    assert [type(e).__name__ for e in chain] == [
        "ImageNormalizationStarted",
        "DerivedImageArtifactCreated",
        "ImageNormalizationCompleted",
    ]


def test_the_provenance_record_links_to_the_original_by_id_and_hash(tmp_path):
    """The full provenance link the specification requires -- an id reference plus a content address,
    never an embedded copy of the original artifact."""
    service, _ = _service(tmp_path)

    recorded = service.normalize_and_record(_images.grayscale_16bit(), page_id=PAGE_ID)
    artifact = recorded.normalized_artifact

    assert artifact.source_artifact_id == recorded.source_artifact.artifact_id
    assert artifact.source_content_hash == recorded.source_artifact.content_hash
    assert artifact.page_id == PAGE_ID
    # Every field the specification enumerates is populated with a real observed value.
    assert artifact.source_color_mode == "I;16"
    assert artifact.source_bit_depth == 16
    assert artifact.source_channel_count == 1
    assert (artifact.source_width, artifact.source_height) == (_images.WIDTH, _images.HEIGHT)
    assert artifact.output_color_mode == "RGB"
    assert artifact.output_bit_depth == 8
    assert artifact.output_channel_count == 3
    assert artifact.output_format == "PNG"
    assert artifact.compositing_background == "#FFFFFF"
    assert artifact.normalization_implementation == (
        "archivetrust.htr.preprocessing.rgb_normalization"
    )
    assert artifact.normalization_version == "1.0.0"
    assert artifact.configuration_hash == RgbNormalizationConfig().configuration_hash
    assert artifact.executed_at, "executed_at must be a real timestamp"
    assert "NormalizedPageArtifact" not in str(artifact.source_artifact_id)


# -- Failure handling ---------------------------------------------------------------------------


def test_an_undecodable_image_is_recorded_as_a_failure_and_raises(tmp_path):
    """Never a silently swallowed exception, and never a fallback to the original."""
    sink = InMemoryTelemetrySink()
    service, store = _service(tmp_path, sink=sink)

    with pytest.raises(NormalizationError) as excinfo:
        service.normalize_and_record(_images.undecodable(), page_id=PAGE_ID)

    assert excinfo.value.failure.category is NormalizationFailureCategory.UNDECODABLE_IMAGE
    failures = store.normalization_failures(page_id=PAGE_ID)
    assert len(failures) == 1
    assert failures[0].page_id == PAGE_ID
    assert failures[0].occurred_at, "a recorded failure must carry a real timestamp"
    assert failures[0].configuration_hash == RgbNormalizationConfig().configuration_hash


def test_no_normalized_artifact_exists_after_a_failure(tmp_path):
    """The load-bearing consequence: there is nothing for an export path to mistake for a success."""
    service, store = _service(tmp_path)

    with pytest.raises(NormalizationError):
        service.normalize_and_record(_images.undecodable(), page_id=PAGE_ID)

    assert store.normalized_page_artifacts(page_id=PAGE_ID) == ()


def test_a_missing_file_is_a_recorded_failure_not_a_crash(tmp_path):
    service, _ = _service(tmp_path)

    with pytest.raises(NormalizationError) as excinfo:
        service.normalize_file_and_record(tmp_path / "nope.tif", page_id=PAGE_ID)

    assert excinfo.value.failure.category is NormalizationFailureCategory.UNDECODABLE_IMAGE


def test_a_truncated_image_is_a_recorded_failure(tmp_path):
    """A real-world corruption case: a valid PNG header with the pixel data cut off."""
    sink = InMemoryTelemetrySink()
    service, store = _service(tmp_path, sink=sink)
    truncated = _images.rgb()[:20]

    with pytest.raises(NormalizationError):
        service.normalize_and_record(truncated, page_id=PAGE_ID)

    assert store.normalization_failures(page_id=PAGE_ID)[0].category is (
        NormalizationFailureCategory.UNDECODABLE_IMAGE
    )


# -- Durability and reconstruction ---------------------------------------------------------------


def test_normalization_is_written_to_a_real_file_on_disk(tmp_path):
    path = tmp_path / "telemetry" / "htr_research_events.jsonl"
    service, store = _service(tmp_path, sink=FileTelemetrySink(path))
    assert store.is_durable

    service.normalize_and_record(_images.cmyk(), page_id=PAGE_ID)

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [json.loads(line)["kind"] for line in lines] == [
        "ImageNormalizationStarted",
        "DerivedImageArtifactCreated",
        "ImageNormalizationCompleted",
    ]


def test_the_read_model_survives_destruction_of_the_store(tmp_path):
    """The destroy-and-reconstruct proof this codebase requires of every new HTR entity.

    The store, its sink, and the service are deleted; a brand-new `FileTelemetrySink` over the same
    path is replayed; the reconstructed artifacts must be equal to the originals field for field.
    """
    path = tmp_path / "telemetry" / "htr_research_events.jsonl"
    service, store = _service(tmp_path, sink=FileTelemetrySink(path))

    succeeded = service.normalize_and_record(_images.cmyk(), page_id=PAGE_ID)
    with pytest.raises(NormalizationError):
        service.normalize_and_record(_images.undecodable(), page_id="page_broken")

    original_artifact = succeeded.normalized_artifact
    original_source = succeeded.source_artifact
    original_failure = store.normalization_failures(page_id="page_broken")[0]

    del service, store
    gc.collect()

    reconstructed = HtrJournal().replay(FileTelemetrySink(path).all_events())

    assert reconstructed.normalized_page_artifacts(page_id=PAGE_ID) == (original_artifact,)
    assert reconstructed.page_image_artifacts(page_id=PAGE_ID) == (original_source,)
    assert reconstructed.normalization_failures(page_id="page_broken") == (original_failure,)
    # And the by-hash lookup the import-association workflow depends on still resolves.
    assert (
        reconstructed.normalized_artifact_by_hash(original_artifact.normalized_content_hash)
        == original_artifact
    )


def test_a_replayed_artifact_still_points_at_retrievable_bytes(tmp_path):
    """Replay reconstructs provenance; the content-addressed store still holds the bytes, and reading
    them back verifies the hash the replayed record claims."""
    path = tmp_path / "telemetry" / "events.jsonl"
    blobs = ContentAddressedBlobStore(tmp_path / "blobs")
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    service = NormalizationService(blob_store=blobs, store=store)

    recorded = service.normalize_and_record(_images.rgba(), page_id=PAGE_ID)
    del service, store
    gc.collect()

    reconstructed = HtrJournal().replay(FileTelemetrySink(path).all_events())
    artifact = reconstructed.normalized_page_artifacts(page_id=PAGE_ID)[0]
    digest = artifact.normalized_content_hash[len("normalized_page_") :]

    assert ContentAddressedBlobStore(tmp_path / "blobs").get_bytes(digest) == (
        recorded_bytes := blobs.get_bytes(digest)
    )
    assert recorded_bytes.startswith(b"\x89PNG")
    assert recorded.normalized_artifact.byte_size == len(recorded_bytes)


# -- The binary blob store itself ----------------------------------------------------------------


def test_blob_store_round_trips_bytes_and_verifies_integrity(tmp_path):
    blobs = ContentAddressedBlobStore(tmp_path / "blobs")
    data = _images.cmyk()

    digest, size = blobs.put_bytes(data)

    assert size == len(data)
    assert blobs.get_bytes(digest) == data
    # Identical bytes are never stored twice.
    assert blobs.put_bytes(data) == (digest, size)


def test_blob_store_detects_a_corrupted_blob(tmp_path):
    """The read-side integrity check is what makes a stored hash an identity rather than a hope."""
    blobs = ContentAddressedBlobStore(tmp_path / "blobs")
    digest, _ = blobs.put_bytes(_images.cmyk())

    blobs.path_for(digest).write_bytes(b"tampered")

    with pytest.raises(BlobIntegrityError):
        blobs.get_bytes(digest)


def test_text_and_bytes_share_one_addressing_scheme(tmp_path):
    """`put_text` is a UTF-8 wrapper over `put_bytes`, not a second mechanism -- so a string and its
    encoded bytes land at the same address."""
    blobs = ContentAddressedBlobStore(tmp_path / "blobs")

    text_digest, _ = blobs.put_text("Anno 1712 den 3 Januarii")
    bytes_digest, _ = blobs.put_bytes("Anno 1712 den 3 Januarii".encode("utf-8"))

    assert text_digest == bytes_digest
    assert blobs.get_text(text_digest) == "Anno 1712 den 3 Januarii"
