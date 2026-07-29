"""`TelemetryAggregate` (Production Hardening Review, 2026-07-13) — the incremental aggregation
that replaced `ProcessingCenterViewModel`'s per-refresh full rescans of the telemetry stream (the
quadratic GUI-thread cost that hung the 2026-07-13 production run at document 184).

The contract under test: a shared, incrementally-updated aggregate produces *identical* ViewModel
results to a fresh full scan of the same append-only stream, no matter how reads interleave with
appends — it is a cache over the stream, never a second source of truth.
"""

from __future__ import annotations

from archivetrust.clients.desktop.demo_data import seed_realistic_archive
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
from archivetrust.learning.review.sink import InMemoryReviewInteractionSink
from archivetrust.presentation.operational_stats import TelemetryAggregate
from archivetrust.presentation.operations_viewmodel import ProcessingCenterViewModel


def _seeded_sink() -> InMemoryTelemetrySink:
    sink = InMemoryTelemetrySink()
    seed_realistic_archive(sink, InMemoryReviewInteractionSink())
    return sink


def _assert_equivalent(shared_vm: ProcessingCenterViewModel, sink: InMemoryTelemetrySink) -> None:
    """A ViewModel over a long-lived shared aggregate must agree with a freshly-constructed one
    (which aggregates the same stream from index 0)."""
    fresh_vm = ProcessingCenterViewModel(sink)
    assert shared_vm.overview() == fresh_vm.overview()
    assert shared_vm.queue() == fresh_vm.queue()
    assert shared_vm.recent_activity() == fresh_vm.recent_activity()
    assert shared_vm.throughput() == fresh_vm.throughput()


def test_incremental_reads_match_full_rescans_as_the_stream_grows() -> None:
    sink = InMemoryTelemetrySink()
    aggregate = TelemetryAggregate()
    shared_vm = ProcessingCenterViewModel(sink, aggregate=aggregate)

    # Read an empty stream first — the cursor must start correctly at zero events.
    assert shared_vm.overview().documents_processed == 0

    # Append the realistic corpus *after* the first read, then read again: only the new events
    # are consumed, and every aggregate matches a from-scratch scan.
    seed_realistic_archive(sink, InMemoryReviewInteractionSink())
    _assert_equivalent(shared_vm, sink)

    # A second read with no new events must be a no-op (cursor at end), still identical.
    _assert_equivalent(shared_vm, sink)


def test_shared_aggregate_across_viewmodel_constructions_matches_private_aggregation() -> None:
    # `AppContext.processing_viewmodel()` constructs a fresh ViewModel per refresh around one
    # shared aggregate — interleaved constructions must not corrupt or double-count anything.
    sink = _seeded_sink()
    aggregate = TelemetryAggregate()

    first = ProcessingCenterViewModel(sink, aggregate=aggregate)
    overview_before = first.overview()
    second = ProcessingCenterViewModel(sink, aggregate=aggregate)
    assert second.overview() == overview_before
    _assert_equivalent(second, sink)


def test_replaced_stream_resets_the_aggregate_instead_of_double_counting() -> None:
    # Defensive contract: if the aggregate is ever pointed at a *shorter* stream (a different
    # Workspace's sink), it rebuilds from scratch rather than reporting a blend of two histories.
    big_sink = _seeded_sink()
    aggregate = TelemetryAggregate()
    aggregate.update(tuple(big_sink.all_events()))

    small_sink = InMemoryTelemetrySink()
    aggregate.update(tuple(small_sink.all_events()))
    assert len(aggregate.documents) == 0
    assert aggregate.provider_invocations == 0


def test_provider_execution_live_reports_the_cheap_columns_honestly() -> None:
    sink = _seeded_sink()
    vm = ProcessingCenterViewModel(sink)
    live = vm.provider_execution_live()
    full = {(h.provider_id, h.provider_version): h for h in vm.provider_execution()}

    assert live, "the seeded archive records provider attempts"
    for row in live:
        counterpart = full[(row.provider_id, row.provider_version)]
        assert row.invocation_count == counterpart.invocation_count
        assert row.rejection_count == counterpart.rejection_count
        assert row.no_observation_invocations == counterpart.no_observation_invocations
        assert row.failed_invocations == counterpart.failed_invocations
        # Contribution/correction data is *not* computed live — reported as no-data, never faked.
        assert row.correction_rate is None
