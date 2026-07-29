"""Headless tests for the DI/composition root — no PySide6 import (the framework-independent half
of the client). Verifies the wiring the Qt widgets rely on is correct before any Qt is involved.
"""

from __future__ import annotations

from archivetrust.composition import AppContext, build_demo_context
from archivetrust.application.journal import Journal
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.log import FileSamplingLogSink, InMemorySamplingLogSink
from archivetrust.review.service import ReviewAction
from archivetrust.learning.review.sink import FileReviewInteractionSink
from archivetrust.review.triage import triage_review_queue
from archivetrust.presentation.review_viewmodel import ReviewStatus
from archivetrust.presentation.shell_viewmodel import ShellViewModel
from tests.review._helpers import emit_document_snapshot, emit_slot, heading


def test_context_wires_service_and_shell(tmp_path) -> None:
    # `deployment_root` must always be a tmp dir in tests (Production Hardening Review,
    # 2026-07-13): a bare `AppContext()` roots at `Path.cwd()/archivetrust_data` -- the *real*
    # deployment when pytest runs from the repo -- opening real Workspaces (loading their full
    # telemetry streams) and appending test runtime events into the production
    # `runtime_events.jsonl`, which contaminated the 2026-07-13 incident's forensic record.
    context = AppContext(deployment_root=tmp_path)
    assert isinstance(context.shell, ShellViewModel)
    assert context.review_service is not None
    assert context.sampling_log is not None
    assert context.adaptive_review_coordinator is not None


def test_persistent_context_uses_workspace_sampling_log(tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    assert isinstance(context.sampling_log, FileSamplingLogSink)
    assert isinstance(context.interaction_sink, FileReviewInteractionSink)


def test_in_memory_context_uses_in_memory_sampling_log(tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path, persistent_telemetry=False)
    assert isinstance(context.sampling_log, InMemorySamplingLogSink)


def test_adaptive_queue_with_zero_budget_matches_operational_review_in_production_context(
    tmp_path,
) -> None:
    context = AppContext(deployment_root=tmp_path)
    for i in range(3):
        emit_slot(
            context.telemetry,  # type: ignore[arg-type]
            document_ref="doc1",
            canonical_payload=heading(f"single {i}"),
            provider_payloads=(("docling", heading(f"single {i}")),),
            classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
        )

    state = Journal().replay(context.telemetry.events_for_document("doc1"))  # type: ignore[attr-defined]
    direct = triage_review_queue(state)
    adaptive = context.adaptive_review_coordinator.build_queue(k=0)

    assert [entry.semantic_slot_id for entry in adaptive] == [item.semantic_slot_id for item in direct]
    assert all(entry.intent is ReviewIntent.OPERATIONAL for entry in adaptive)


def test_adaptive_review_correlation_is_wired_in_production_context(tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path, persistent_telemetry=False)
    emit_slot(
        context.telemetry,  # type: ignore[arg-type]
        document_ref="doc1",
        canonical_payload=heading("single"),
        provider_payloads=(("docling", heading("single")),),
        classification=ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE,
    )
    emit_document_snapshot(
        context.telemetry, document_ref="doc1", archive_object_ref="obj1"  # type: ignore[arg-type]
    )

    queue = context.adaptive_review_coordinator.build_queue(
        intent=ReviewIntent.CALIBRATION, strategy_name="random_sampling", k=1, seed=1
    )
    entry = next(item for item in queue if item.intent is ReviewIntent.CALIBRATION)
    packet = context.adaptive_review_coordinator.open_packet(entry, archive_object_ref="obj1")

    result = context.adaptive_review_coordinator.submit_decision(
        entry=entry, packet=packet, action=ReviewAction.ACCEPT_PROVIDER
    )

    assert result.correction_id is not None
    decisions = list(context.sampling_log.all_decisions())
    assert decisions
    assert decisions[-1].correction_id == result.correction_id


def test_runtime_telemetry_is_durable_even_when_persistent_telemetry_is_false(tmp_path) -> None:
    """Production Investigation (2026-07-13): `--demo` passes `persistent_telemetry=False` so its
    *seeded document data* doesn't accrete into domain telemetry on every launch -- but that flag
    was also (wrongly) gating `runtime_telemetry`, silently discarding real `RUNTIME_START_FAILED`/
    `WARMUP_TIME` events from a real, runtime-backed provider's actual Docker/GPU activity, which
    has nothing to do with demo document seeding and is just as real in `--demo` mode as anywhere
    else. `runtime_telemetry` must be durable (`FileRuntimeTelemetrySink`) regardless.
    """
    from archivetrust.runtime.runtime_telemetry import FileRuntimeTelemetrySink

    context = AppContext(
        deployment_root=tmp_path / "app", auto_create_default_workspace=False,
        persistent_telemetry=False,
    )
    assert isinstance(context.runtime_telemetry, FileRuntimeTelemetrySink)


def test_demo_context_seeds_a_realistic_reviewable_archive(tmp_path) -> None:
    context, documents = build_demo_context(deployment_root=tmp_path)
    assert len(documents) >= 3
    # Realistic content, not a programming example.
    assert any("Kommunfullmäktige" in d.title for d in documents)
    assert all("stormy night" not in d.title for d in documents)

    first = documents[0]
    vm = context.review_viewmodel(
        document_ref=first.document_ref, archive_object_ref=first.archive_object_ref
    )
    vm.load()
    assert vm.status.value is ReviewStatus.REVIEWING
    assert vm.current_packet.value is not None
    assert vm.current_packet.value.primary_bounding_box is not None


def test_all_center_viewmodels_are_constructible_from_context(tmp_path) -> None:
    context, _documents = build_demo_context(deployment_root=tmp_path)
    # Every Center's ViewModel wires from the same context and reads the seeded telemetry.
    assert context.processing_viewmodel().overview().documents_processed >= 3
    assert context.quality_viewmodel().human_effort().session_count >= 1
    assert len(context.evolution_viewmodel().candidates()) >= 1
    assert len(context.evidence_explorer_viewmodel().documents()) >= 3


def test_open_workspace_switches_every_per_workspace_object(tmp_path) -> None:
    # ROADMAP.md §5.13, decision 4: switching Workspaces rebuilds telemetry/provider/model state
    # without any change to what a Center ViewModel factory method itself does.
    context = AppContext(deployment_root=tmp_path)
    first_workspace = context.current_workspace
    first_telemetry = context.telemetry

    second_workspace = context.create_workspace("Second Workspace")
    assert context.current_workspace.id == second_workspace.id
    assert context.telemetry is not first_telemetry  # a fresh, isolated telemetry sink

    context.open_workspace(first_workspace.id)
    assert context.current_workspace.id == first_workspace.id


def test_each_workspace_has_independent_provider_configuration(tmp_path) -> None:
    context = AppContext(deployment_root=tmp_path)
    context.provider_config_store.set(
        context.provider_config_store.get("docling").model_copy(update={"enabled": False})
    )
    assert context.provider_config_store.get("docling").enabled is False

    context.create_workspace("Independent Workspace")
    # A fresh Workspace's ProviderConfigurationStore starts from defaults, never inherits a sibling
    # Workspace's configuration (ROADMAP.md §5.13.1: "No global singleton configuration").
    assert context.provider_config_store.get("docling").enabled is True
