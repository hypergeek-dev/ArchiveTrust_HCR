"""The concrete proof this follow-up asks for: destroy the in-memory store, reconstruct from disk.

`docs/htr-telemetry-knowledge-gap-analysis.md` §2 found that HTR research entities lived only in one
Python process's `dict`s -- "no telemetry emission ... no file/disk backing whatsoever" -- and §4
found that the baseline experiment's entities "only ever lived in one Python process's memory".
These tests are what makes the fix falsifiable rather than asserted.

The load-bearing discipline in `test_read_model_survives_destruction_of_the_store`: the reconstructed
store is built from a **brand-new `FileTelemetrySink` over the same path**, after the original store
object and its sink are deleted. Nothing is carried across in memory, and in particular nothing is
serialized-and-reloaded as a whole-store blob -- the follow-up brief is explicit that
"serialize the whole store as one blob and reload it" does not satisfy the requirement.
"""

from __future__ import annotations

import gc
import json

import pytest

from archivetrust.application.htr_journal import HtrJournal
from archivetrust.htr.persistence import DurableHtrResearchStore, HtrCoarseEntitySnapshot
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.infrastructure.storage.telemetry_sink import (
    FileTelemetrySink,
    InMemoryTelemetrySink,
)
from tests.htr.persistence._fixtures import ARCHIVE_OBJECT_REF, register_small_corpus


def test_registrations_are_written_to_a_real_file_on_disk(tmp_path):
    path = tmp_path / "telemetry" / "htr_research_events.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    assert store.is_durable

    register_small_corpus(store)

    assert path.exists(), "the durable store must write a real file, not just fill a dict"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 18, f"expected one event per registration, got {len(lines)}"

    # Every line is a self-describing telemetry record, not an opaque store dump.
    kinds = [json.loads(line)["kind"] for line in lines]
    for expected in (
        "ResearchProjectCreated",
        "DatasetCreated",
        "CollectionCreated",
        "DocumentRegistered",
        "DatasetVersionCreated",
        "PageRegistered",
        "RegionDetected",
        "TextLineDetected",
        "InputCropCreated",
        "ExperimentCreated",
        "ExperimentVersionCreated",
        "ExperimentRunStarted",
        "MethodRunStarted",
        "MethodRunCompleted",
        "RawMethodResultRecorded",
        "ParsedMethodResultRecorded",
        "NormalizedMethodResultRecorded",
        "GroundTruthTextRecorded",
        "MetricDefinitionRegistered",
        "MetricCalculated",
        "ReproducibilityManifestRecorded",
        "ExperimentRunCompleted",
    ):
        assert expected in kinds, f"{expected} was never written to the durable stream"

    # The hash-chain sidecar `FileTelemetrySink` maintains applies to this stream too, unmodified.
    assert (path.parent / "htr_research_events.jsonl.chain.jsonl").exists()


def test_read_model_survives_destruction_of_the_store(tmp_path):
    """Register -> confirm on disk -> destroy every in-memory object -> replay -> compare."""
    path = tmp_path / "telemetry" / "htr_research_events.jsonl"

    original_store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(original_store)

    # What the live store answered, captured as plain values before it is destroyed.
    assert original_store.experiment(corpus.experiment.experiment_id) == corpus.experiment
    assert original_store.method_run(corpus.method_run.method_run_id) == corpus.method_run

    # 3. Destroy the in-memory store entirely. Nothing below may reach it.
    del original_store
    gc.collect()

    # 4./5. A brand-new sink over the same file, and a fresh projection replayed from it alone.
    reopened_sink = FileTelemetrySink(path)
    reconstructed = HtrJournal().replay(reopened_sink.all_events())
    # Deliberately *not* an `id()` comparison against the destroyed store: once it is freed, CPython
    # may hand the same address to the replacement, so that check passes or fails by allocator luck.
    # `HtrJournal.replay` builds a plain `HtrResearchStore`, never the `DurableHtrResearchStore`
    # subclass that was registered through -- which proves this is a different object *by type*, and
    # cannot flake.
    assert isinstance(reconstructed, HtrResearchStore)
    assert not isinstance(reconstructed, DurableHtrResearchStore)

    # 6. Field-for-field equality, not merely "an object with the same id exists".
    assert reconstructed.project(corpus.project.project_id) == corpus.project
    assert reconstructed.dataset(corpus.dataset.dataset_id) == corpus.dataset
    assert reconstructed.collection(corpus.collection.collection_id) == corpus.collection
    assert (
        reconstructed.dataset_version(corpus.dataset_version.dataset_version_id)
        == corpus.dataset_version
    )
    assert reconstructed.page(corpus.page.page_id) == corpus.page
    assert reconstructed.region(corpus.region.region_id) == corpus.region
    assert reconstructed.text_line(corpus.text_line.text_line_id) == corpus.text_line
    assert reconstructed.experiment(corpus.experiment.experiment_id) == corpus.experiment
    assert (
        reconstructed.experiment_version(corpus.experiment_version.experiment_version_id)
        == corpus.experiment_version
    )
    assert reconstructed.method_run(corpus.method_run.method_run_id) == corpus.method_run
    assert reconstructed.metric_results(method_run_id=corpus.method_run.method_run_id) == (
        corpus.metric_result,
    )
    assert (
        reconstructed.metric_definition(corpus.metric_definition.metric_definition_id)
        == corpus.metric_definition
    )
    assert reconstructed.manifests(
        experiment_run_id=corpus.experiment_run.experiment_run_id
    ) == (corpus.manifest,)
    assert reconstructed.transcript(corpus.method_run.method_run_id) == corpus.transcript
    assert (
        reconstructed.ground_truth_for_line(corpus.text_line.text_line_id)
        == "Anno 1841 den 3 Martii"
    )


def test_reconstructed_input_crop_keeps_its_original_content_address(tmp_path):
    """The crop's hash is `InputCrop`'s own and survives the round trip unchanged -- no second
    hashing scheme is introduced anywhere on the persistence path."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    reconstructed = HtrJournal().replay(FileTelemetrySink(path).all_events())
    crop = reconstructed.input_crop(corpus.crop.crop_id)
    assert crop == corpus.crop
    assert crop.hash == corpus.crop.hash
    from archivetrust.htr.corpus.models import InputCrop

    assert crop.hash == InputCrop.compute_hash(b"synthetic-crop-bytes")


def test_terminal_experiment_run_state_is_reconstructed_not_the_started_one(tmp_path):
    """`ExperimentRunCompleted` advances the projection past `ExperimentRunStarted`, so a replay
    yields the run as it finished -- with `completed_at` -- not as it began."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    reconstructed = HtrJournal().replay(FileTelemetrySink(path).all_events())
    run = reconstructed.experiment_run(corpus.experiment_run.experiment_run_id)
    assert corpus.experiment_run.completed_at is None
    assert run.completed_at == "2026-07-30T09:01:00+00:00"
    assert run == corpus.completed_experiment_run


def test_open_rebuilds_a_durable_store_ready_to_keep_appending(tmp_path):
    """The restart path `composition.py` actually uses: `DurableHtrResearchStore.open` returns a
    store that already contains prior history *and* can record new events onto the same stream."""
    path = tmp_path / "htr.jsonl"
    first = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(first)
    events_after_first = len(list(first.sink.all_events()))
    del first

    second = DurableHtrResearchStore.open(FileTelemetrySink(path))
    assert second.experiment(corpus.experiment.experiment_id) == corpus.experiment
    assert second.method_run(corpus.method_run.method_run_id) == corpus.method_run

    # Re-registering an already-recorded entity is still refused: append-only survives the restart.
    from archivetrust.htr.research_store import DuplicateRegistrationError

    with pytest.raises(DuplicateRegistrationError):
        second.register_experiment(corpus.experiment)

    # And a genuinely new registration appends to the same durable stream.
    from archivetrust.htr.corpus.models import ResearchProject

    second.register_project(ResearchProject.create(name="Second project", created_at="2026-07-30T10:00:00+00:00"))
    assert len(list(second.sink.all_events())) == events_after_first + 1

    third = HtrJournal().replay(FileTelemetrySink(path).all_events())
    assert len(third.projects()) == 2


def test_page_events_stay_scoped_to_their_real_archive_object(tmp_path):
    """`PageRegistered` carries the genuine `archive_object_ref` as its `document_ref`, so
    per-document replay of page-scoped HTR history still works."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    sink = FileTelemetrySink(path)
    per_document = list(sink.events_for_document(ARCHIVE_OBJECT_REF))
    kinds = {event.kind.value for event in per_document}
    assert "PageRegistered" in kinds
    assert "DocumentRegistered" in kinds
    page_event = next(e for e in per_document if e.kind.value == "PageRegistered")
    assert page_event.page == corpus.page
    assert page_event.document_ref == ARCHIVE_OBJECT_REF


def test_coarse_snapshot_is_a_derived_cache_and_deleting_it_loses_nothing(tmp_path):
    """The `WorkspaceStore`-idiom JSON snapshot is a fast-path lookup, never a source of truth."""
    path = tmp_path / "htr.jsonl"
    snapshot_path = tmp_path / "htr_coarse_entities.json"
    snapshot = HtrCoarseEntitySnapshot(snapshot_path)
    store = DurableHtrResearchStore(FileTelemetrySink(path), snapshot=snapshot)
    corpus = register_small_corpus(store)

    assert snapshot_path.exists()
    project_ids = {record["project_id"] for record in snapshot.records("projects")}
    assert corpus.project.project_id in project_ids

    # Delete the cache outright. Every entity must still be reconstructable from the event log.
    snapshot_path.unlink()
    del store
    reconstructed = HtrJournal().replay(FileTelemetrySink(path).all_events())
    assert reconstructed.project(corpus.project.project_id) == corpus.project
    assert reconstructed.dataset(corpus.dataset.dataset_id) == corpus.dataset

    # And `rebuild` regenerates the cache from that replayed projection.
    rebuilt = HtrCoarseEntitySnapshot(snapshot_path)
    rebuilt.rebuild(reconstructed)
    assert snapshot_path.exists()
    assert {record["project_id"] for record in rebuilt.records("projects")} == {
        corpus.project.project_id
    }
    assert {record["dataset_id"] for record in rebuilt.records("datasets")} == {
        corpus.dataset.dataset_id
    }


def test_an_in_memory_sink_reports_itself_as_not_durable(tmp_path):
    """Event sourcing without a file is legitimate for a test or a higher-layer-persisting
    deployment, but must never be mistaken for durability."""
    store = DurableHtrResearchStore(InMemoryTelemetrySink())
    assert store.is_durable is False
    corpus = register_small_corpus(store)
    # Still fully replayable within the process -- the mechanism is the same, only the medium differs.
    reconstructed = HtrJournal().replay(store.sink.all_events())
    assert reconstructed.experiment(corpus.experiment.experiment_id) == corpus.experiment


def test_replay_refuses_to_invent_an_entity_a_terminal_event_advances(tmp_path):
    """An `ExperimentRunCompleted` with no preceding `ExperimentRunStarted` means the log is
    incomplete. That must surface, not be silently turned into a first registration."""
    from archivetrust.domain.telemetry.events import HTR_RESEARCH_SCOPE, ExperimentRunCompleted
    from archivetrust.htr.experiment.models import ExperimentRun
    from archivetrust.htr.research_store import UnknownEntityError

    orphan = ExperimentRun.create(
        experiment_version_id="experiment_version_missing",
        is_end_to_end=False,
        started_at="2026-07-30T09:00:00+00:00",
        completed_at="2026-07-30T09:01:00+00:00",
    )
    event = ExperimentRunCompleted(
        event_id="event_orphan",
        document_ref=HTR_RESEARCH_SCOPE,
        experiment_run=orphan,
    )
    with pytest.raises(UnknownEntityError):
        HtrJournal().replay([event])
