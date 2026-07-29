"""Review packet assembly (HUMAN_REVIEW_SPECIFICATION.md §8.2).

Builds a `ReviewPacket` for one semantic slot from a replayed `JournalState` — read-only over the
Trust Engine telemetry stream (LP-2). Everything the packet contains, it contains by reading what
the Journal reconstructed (Evidence, Observations, Canonical Observations); it fabricates nothing
and invents no provenance the Trust Engine did not record (`ROADMAP.md` GP 5).
"""

from __future__ import annotations

from archivetrust.application.journal import JournalState
from archivetrust.domain.confidence.models import ComparisonClassification
from archivetrust.domain.ontology.payloads import TablePayload
from archivetrust.review.packet import (
    AgreementView,
    CandidateView,
    DisclosureTier,
    EvidenceView,
    ReviewPacket,
    ReviewReason,
)
from archivetrust.review.triage import TriageItem


class ReviewAssemblyError(KeyError):
    """Raised when a packet cannot be assembled because telemetry the Journal replayed is missing a
    referenced object (a contributing Observation or its Evidence). Fails loudly rather than
    assembling a partial packet with silently dropped candidates — the same Article-18 discipline
    ("silence must be distinguishable from failure") applied at the review layer.
    """


def _value_of(payload: object) -> str | None:
    """The display text for a payload, where it is text-bearing; else None. Uses the same
    `hasattr(payload, "text")` test the Comparison and Feedback engines use, so a packet's notion
    of "the value" matches what an EDIT correction can actually replace.
    """
    text = getattr(payload, "text", None)
    return text if isinstance(text, str) else None


def _structure_summary_of(payload: object) -> dict[str, str | int | float | bool | None]:
    """Neutral payload shape for structural/table review tooling.

    This is deliberately ontology-level, not provider-level: the reviewer may see "3 rows x 2
    columns" or "cell row 1, column 2", but never native provider labels or ids.
    """
    if isinstance(payload, TablePayload):
        return {
            "kind": "table",
            "rows": payload.row_count,
            "columns": payload.column_count,
            "caption": payload.caption_text,
        }
    # `TableCellPayload` (docs/htr-migration-plan.md Stage 5 -- EXECUTED) was deleted along with
    # `ObservationType.TABLE_CELL`; table-cell review-tooling summarization is gone with it.
    return {}


def assemble_packet(
    journal_state: JournalState,
    item: TriageItem,
    *,
    document_ref: str,
    archive_object_ref: str,
) -> ReviewPacket:
    """Assemble the packet for one triaged uncertainty.

    `document_ref` and `archive_object_ref` are supplied by the caller (the `ReviewService`, which
    replayed this document's telemetry and therefore knows both) rather than re-derived here, so
    assembly stays a pure projection of the slot's reconstructed objects.
    """
    canonical = journal_state.canonical_observation(item.canonical_observation_id)

    candidates: list[CandidateView] = []
    for ref in canonical.contributing_observations:
        try:
            observation = journal_state.observation(ref.observation_id)
        except KeyError as exc:
            raise ReviewAssemblyError(
                f"contributing Observation {ref.observation_id!r} for slot "
                f"{item.semantic_slot_id!r} is not present in replayed telemetry"
            ) from exc

        evidence_views: list[EvidenceView] = []
        for evidence_id in observation.evidence_ids:
            try:
                evidence = journal_state.evidence(evidence_id)
            except KeyError as exc:
                raise ReviewAssemblyError(
                    f"Evidence {evidence_id!r} backing Observation {ref.observation_id!r} is not "
                    "present in replayed telemetry"
                ) from exc
            evidence_views.append(
                EvidenceView(
                    evidence_id=evidence.evidence_id,
                    provider_id=evidence.provider,
                    provider_version=evidence.provider_version,
                    page=evidence.page,
                    bounding_box=evidence.bounding_box,
                    raw_output=evidence.raw_output,
                    provider_confidence=evidence.provider_confidence,
                    coordinate_metadata=dict(evidence.supporting_metadata),
                )
            )

        mapping = journal_state.mapping_used_for(observation.observation_id)
        candidates.append(
            CandidateView(
                observation_id=observation.observation_id,
                provider_id=observation.provider_id,
                provider_version=observation.provider_version,
                value=_value_of(observation.payload),
                provider_confidence=(
                    observation.provider_confidence.value
                    if observation.provider_confidence is not None
                    else None
                ),
                evidence=tuple(evidence_views),
                structure_summary=_structure_summary_of(observation.payload),
                mapping_table_entry_id=mapping.mapping_table_entry_id if mapping is not None else None,
            )
        )

    independent_sources = len({c.provider_id for c in candidates})
    agreement = AgreementView(
        classification=canonical.comparison_confidence.classification,
        independent_source_count=independent_sources,
    )

    return ReviewPacket(
        semantic_slot_id=item.semantic_slot_id,
        document_ref=document_ref,
        canonical_observation_id=canonical.canonical_observation_id,
        observation_type=canonical.observation_type,
        archive_object_ref=archive_object_ref,
        current_value=_value_of(canonical.payload),
        candidates=tuple(candidates),
        agreement=agreement,
        comparison_confidence=canonical.comparison_confidence,
        canonical_confidence=canonical.canonical_confidence,
        disclosure_tier=item.disclosure_tier,
        review_reason=item.reason,
        reconciliation_basis_code=canonical.reconciliation_basis_code,
    )
