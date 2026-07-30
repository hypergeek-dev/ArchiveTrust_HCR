"""Correlation and causation as a real, walkable DAG -- not decorative fields.

`docs/htr-telemetry-knowledge-gap-analysis.md` §1 found that "no `correlation_id`/`causation_id`
concept exists anywhere in the codebase (zero grep hits)" and that correlation was implicit, inferred
from shared domain ids embedded per-event. `docs/architecture/htr-event-model.md` §4 replaces that
inference with explicit edges.

The load-bearing discipline in `test_causation_chain_of_four_or_more_links`: the chain is walked
**purely via `causation_id` -> `event_id` pointers**. It never consults `recorded_at`, never uses
append order, and never joins on a shared domain id -- so a chain that only reproduced the order
events happened to be written in would not pass it.
"""

from __future__ import annotations

import pytest

from archivetrust.application.htr_journal import causation_chain
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink
from tests.htr.persistence._fixtures import register_small_corpus


def test_causation_chain_of_four_or_more_links(tmp_path):
    """Walks >= 4 causally-linked events by pointer alone and asserts the chain is correct."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    events = list(FileTelemetrySink(path).all_events())
    chain = causation_chain(events, from_event_id=corpus.experiment_run_started_event_id)

    assert len(chain) >= 4, f"expected a chain of at least 4 links, got {len(chain)}"

    # The expected causal path, in order. Each link is an explicit pointer, asserted below.
    assert [event.kind.value for event in chain] == [
        "ExperimentRunStarted",
        "MethodRunStarted",
        "RawMethodResultRecorded",
        "ParsedMethodResultRecorded",
        "NormalizedMethodResultRecorded",
        "MetricCalculated",
        "ReproducibilityManifestRecorded",
        "ExperimentRunCompleted",
    ]

    # Every consecutive pair is joined by causation_id -> event_id, and by nothing else.
    for parent, child in zip(chain, chain[1:]):
        assert child.causation_id == parent.event_id, (
            f"{child.kind.value} does not point at {parent.kind.value} via causation_id"
        )

    # The root of the chain is caused by the ExperimentVersionCreated that preceded it, and is not
    # itself a causal orphan.
    assert chain[0].causation_id is not None
    by_id = {event.event_id: event for event in events}
    assert by_id[chain[0].causation_id].kind.value == "ExperimentVersionCreated"


def test_the_chain_is_not_merely_append_order(tmp_path):
    """Guards the test above from passing trivially: the causal chain and the file's append order
    are genuinely different sequences, so following pointers is doing real work."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    events = list(FileTelemetrySink(path).all_events())
    chain = causation_chain(events, from_event_id=corpus.experiment_run_started_event_id)

    chain_ids = [event.event_id for event in chain]
    append_order_ids = [event.event_id for event in events]
    start = append_order_ids.index(chain_ids[0])
    assert chain_ids != append_order_ids[start : start + len(chain_ids)], (
        "the causal chain reproduces raw append order, so this proves nothing about causation"
    )
    # Concretely: MethodRunCompleted is appended between MethodRunStarted and the raw result, but is
    # a causal *sibling* of the raw result, not a link in this chain.
    assert "MethodRunCompleted" in {event.kind.value for event in events}
    assert "MethodRunCompleted" not in {event.kind.value for event in chain}


def test_method_run_completed_causes_from_method_run_started(tmp_path):
    """Event-model doc §4's worked example, asserted directly."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    events = list(FileTelemetrySink(path).all_events())
    completed = next(e for e in events if e.kind.value == "MethodRunCompleted")
    assert completed.causation_id == corpus.method_run_started_event_id


def test_every_event_in_one_run_shares_that_runs_correlation_id(tmp_path):
    """`correlation_id` is the `ExperimentRun.id` (this pass's documented choice), shared by every
    event belonging to that run's execution."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    events = list(FileTelemetrySink(path).all_events())
    run_id = corpus.experiment_run.experiment_run_id

    execution_kinds = {
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
    }
    execution_events = [e for e in events if e.kind.value in execution_kinds]
    assert execution_events
    for event in execution_events:
        assert event.correlation_id == run_id, (
            f"{event.kind.value} carries correlation_id {event.correlation_id!r}, not {run_id!r}"
        )

    # Registrations that genuinely precede any run carry an honest None rather than a fabricated
    # correlation.
    project_event = next(e for e in events if e.kind.value == "ResearchProjectCreated")
    assert project_event.correlation_id is None


def test_correlation_scope_nests_and_restores(tmp_path):
    """An inner unit of work cannot leak its correlation into the enclosing one."""
    from archivetrust.htr.corpus.models import ResearchProject

    store = DurableHtrResearchStore(FileTelemetrySink(tmp_path / "htr.jsonl"))
    assert store.correlation_id is None
    with store.correlated_to("outer"):
        assert store.correlation_id == "outer"
        with store.correlated_to("inner"):
            assert store.correlation_id == "inner"
            store.register_project(
                ResearchProject.create(name="inner", created_at="2026-07-30T09:00:00+00:00")
            )
        assert store.correlation_id == "outer"
    assert store.correlation_id is None

    events = list(store.sink.all_events())
    assert events[0].correlation_id == "inner"


def test_correlation_survives_replay_because_it_is_stored_not_derived(tmp_path):
    """The correlation/causation edges are durable data, not something recomputed on read."""
    path = tmp_path / "htr.jsonl"
    store = DurableHtrResearchStore(FileTelemetrySink(path))
    corpus = register_small_corpus(store)
    del store

    reread = list(FileTelemetrySink(path).all_events())
    metric = next(e for e in reread if e.kind.value == "MetricCalculated")
    assert metric.causation_id == corpus.transcript_last_event_id
    assert metric.correlation_id == corpus.experiment_run.experiment_run_id


def test_causation_chain_rejects_an_unknown_root(tmp_path):
    store = DurableHtrResearchStore(FileTelemetrySink(tmp_path / "htr.jsonl"))
    register_small_corpus(store)
    with pytest.raises(KeyError):
        causation_chain(store.sink.all_events(), from_event_id="event_does_not_exist")


def test_pre_existing_event_kinds_keep_working_without_correlation_fields():
    """Adding the two fields to the shared base is purely additive: an OCR-era event constructed the
    way it always was still validates, and honestly reports no correlation."""
    from archivetrust.domain.telemetry.events import ProviderObservationAttempted

    event = ProviderObservationAttempted(
        event_id="event_1",
        document_ref="doc-1",
        provider_id="satrn",
        provider_version="1.0",
        invocation_id="invocation_1",
    )
    assert event.correlation_id is None
    assert event.causation_id is None
