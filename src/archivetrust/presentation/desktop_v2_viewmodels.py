"""Presentation models for the first Desktop v2 operator workflow.

These DTOs translate existing backend services into operator-facing language. They do not create
new domain facts and they do not duplicate comparison/review logic.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.acquisition.events import (
    AcquisitionCompleted,
    AcquisitionEvent,
    AcquisitionFailed,
    AcquisitionStarted,
    ArchiveObjectDiscovered,
    ArchiveObjectRegistered,
)
from archivetrust.presentation.display_names import document_label, short_ref
from archivetrust.presentation.operations_viewmodel import ProcessingCenterViewModel
from archivetrust.review.packet import ReviewPacket
from archivetrust.review.sampling.intent import ReviewIntent
from archivetrust.review.sampling.queue import AdaptiveQueueEntry


class ImpactLabel(str, Enum):
    HIGH = "High review impact"
    MEDIUM = "Medium review impact"
    LOW = "Low review impact"


class SourceActivityRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    detail: str
    outcome: str


class SourceWorkflowSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured_sources: int
    pending_documents: int
    imported_documents: int
    duplicate_discoveries: int
    failures: int
    recent_activity: tuple[SourceActivityRow, ...]
    source_configuration_note: str | None


class DocumentLifecycleSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    display_name: str
    archive_object_ref: str | None
    original_archived: bool
    processing_state: str
    evidence_created: bool
    canonical_snapshot_available: bool
    review_state: str
    output_state: str


class WorkQueueItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry: AdaptiveQueueEntry
    archive_object_ref: str
    document_label: str
    intent_label: str
    explanation: str
    document_context: str
    observation_type: str
    impact_label: str | None
    status: str


class WorkQueuePacket(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    item: WorkQueueItem
    packet: ReviewPacket


def adaptive_review_enabled() -> bool:
    return os.environ.get("ARCHIVETRUST_ADAPTIVE_REVIEW") == "1"


def calibration_budget() -> int:
    if not adaptive_review_enabled():
        return 0
    raw = os.environ.get("ARCHIVETRUST_CALIBRATION_SAMPLE_BUDGET", "0")
    try:
        return max(int(raw), 0)
    except ValueError:
        return 0


def source_workflow_summary(*, manager, viewmodel, acquisition_events: tuple[AcquisitionEvent, ...]) -> SourceWorkflowSummary:
    discovered_by_hash: dict[str, int] = {}
    registered_hashes: set[str] = set()
    failures = 0
    recent: list[SourceActivityRow] = []

    for event in acquisition_events:
        if isinstance(event, ArchiveObjectDiscovered):
            discovered_by_hash[event.content_hash] = discovered_by_hash.get(event.content_hash, 0) + 1
            recent.append(
                SourceActivityRow(
                    kind="Discovered",
                    detail=event.original_filename,
                    outcome="Checking whether this file is new.",
                )
            )
        elif isinstance(event, ArchiveObjectRegistered):
            registered_hashes.add(event.archive_object.content_hash)
            recent.append(
                SourceActivityRow(
                    kind="Imported",
                    detail=event.archive_object.original_filename,
                    outcome="Archived and queued for processing.",
                )
            )
        elif isinstance(event, AcquisitionStarted):
            recent.append(
                SourceActivityRow(kind="Scan started", detail=event.source_id, outcome="Looking for new documents.")
            )
        elif isinstance(event, AcquisitionCompleted):
            recent.append(
                SourceActivityRow(
                    kind="Scan complete",
                    detail=event.source_id,
                    outcome=f"{event.registered_count} new document(s) queued.",
                )
            )
        elif isinstance(event, AcquisitionFailed):
            failures += 1
            recent.append(SourceActivityRow(kind="Failed", detail=event.source_id, outcome=event.reason))

    duplicates = sum(max(count - 1, 0) for count in discovered_by_hash.values())
    return SourceWorkflowSummary(
        configured_sources=len(viewmodel.configured_sources()),
        pending_documents=len(manager.pending),
        imported_documents=len(registered_hashes),
        duplicate_discoveries=duplicates,
        failures=failures,
        recent_activity=tuple(recent[-12:][::-1]),
        source_configuration_note=(
            "Configured sources are session-local in this build; acquisition history is durable, "
            "but source definitions are not yet saved as workspace configuration."
        ),
    )


def document_lifecycle_summaries(
    *,
    processing: ProcessingCenterViewModel,
    acquisition_manager,
    telemetry_source=None,
) -> tuple[DocumentLifecycleSummary, ...]:
    queue_by_doc = {item.document_ref: item for item in processing.queue()}
    review_docs = {item.document_ref for item in processing.review_queue()}
    canonical_docs = processing.documents_with_canonical_snapshots()
    rows: list[DocumentLifecycleSummary] = []
    seen: set[str] = set()

    for archive_object in acquisition_manager.pending:
        seen.add(archive_object.id)
        rows.append(
            DocumentLifecycleSummary(
                document_ref=archive_object.id,
                display_name=document_label(archive_object.id, archive_object=archive_object),
                archive_object_ref=archive_object.id,
                original_archived=True,
                processing_state="Waiting",
                evidence_created=False,
                canonical_snapshot_available=False,
                review_state="Not ready",
                output_state="Pending processing",
            )
        )

    for document_ref, item in queue_by_doc.items():
        seen.add(document_ref)
        needs_review = document_ref in review_docs
        has_snapshot = document_ref in canonical_docs
        archive_object = (
            acquisition_manager.archive_object_by_ref(item.archive_object_ref)
            if item.archive_object_ref is not None
            else None
        )
        rows.append(
            DocumentLifecycleSummary(
                document_ref=document_ref,
                display_name=document_label(document_ref, archive_object=archive_object),
                archive_object_ref=item.archive_object_ref,
                original_archived=item.archive_object_ref is not None,
                processing_state="Needs attention" if needs_review else "Complete",
                evidence_created=item.canonical_facts > 0,
                canonical_snapshot_available=has_snapshot,
                review_state="Review required" if needs_review else "Resolved",
                output_state="Canonical snapshot available" if has_snapshot else "No export available",
            )
        )

    return tuple(rows)


def work_queue_items(
    *,
    coordinator,
    processing: ProcessingCenterViewModel,
    acquisition_manager=None,
) -> tuple[WorkQueueItem, ...]:
    budget = calibration_budget()
    queue = coordinator.build_queue(
        intent=ReviewIntent.CALIBRATION,
        strategy_name="random_sampling",
        k=budget,
        seed=1,
    )
    archive_refs = {item.document_ref: (item.archive_object_ref or item.document_ref) for item in processing.queue()}
    labels: dict[str, str] = {}
    for item in processing.queue():
        archive_object = None
        if acquisition_manager is not None and item.archive_object_ref is not None:
            archive_object = acquisition_manager.archive_object_by_ref(item.archive_object_ref)
        labels[item.document_ref] = document_label(item.document_ref, archive_object=archive_object)
    return tuple(
        _work_item(
            entry,
            archive_refs.get(entry.document_ref, entry.document_ref),
            labels.get(entry.document_ref, short_ref(entry.document_ref)),
        )
        for entry in queue
    )


def _work_item(entry: AdaptiveQueueEntry, archive_object_ref: str, label: str) -> WorkQueueItem:
    return WorkQueueItem(
        entry=entry,
        archive_object_ref=archive_object_ref,
        document_label=label,
        intent_label=_intent_label(entry.intent),
        explanation=_explanation(entry),
        document_context=entry.document_ref,
        observation_type=entry.observation_type.value.replace("_", " ").title(),
        impact_label=_impact_label(entry.value_score) if entry.intent != ReviewIntent.OPERATIONAL else None,
        status="Ready for review",
    )


def _intent_label(intent: ReviewIntent) -> str:
    return {
        ReviewIntent.OPERATIONAL: "Operational review",
        ReviewIntent.CALIBRATION: "Calibration review",
        ReviewIntent.RESEARCH: "Research review",
    }[intent]


def _explanation(entry: AdaptiveQueueEntry) -> str:
    if entry.intent is ReviewIntent.OPERATIONAL:
        return "This result needs a decision because the available evidence is uncertain."
    if entry.intent is ReviewIntent.CALIBRATION:
        return (
            "This example was selected because reviewing it is expected to improve confidence "
            "estimates for similar material."
        )
    return "This example was selected for a defined research or evaluation task."


def _impact_label(value_score: float | None) -> str | None:
    if value_score is None:
        return None
    if value_score >= 0.66:
        return ImpactLabel.HIGH.value
    if value_score >= 0.33:
        return ImpactLabel.MEDIUM.value
    return ImpactLabel.LOW.value
