"""The Review Packet and its presentation-facing views (HUMAN_REVIEW_SPECIFICATION.md §8.2).

A `ReviewPacket` is the immutable, read-only assembly handed to the interface for *one* uncertainty
(one semantic slot, §9 HR-9). It contains everything the reviewer could need — the current
canonical value, the competing candidates with their backing evidence and bounding boxes, the
agreement picture, and the disclosure tier — so the interface never has to fetch anything
mid-review (UX-INV-2). It is the *same* packet regardless of which client renders it (§16, HR-11):
presentation varies, the packet does not.

**Provider independence and §14.** These views carry `provider_id` / `provider_version` because the
correction contract and provider-evaluation analytics need attribution and provenance. That is not
a licence to *display* them: the reviewer-facing client (ViewModels / Qt) must mask provider
implementation detail per anti-goal §14 (no version strings, sources shown as neutral attributions).
The data model stays complete; the presentation layer above is responsible for what a reviewer
actually sees.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.confidence.models import (
    CanonicalConfidence,
    ComparisonClassification,
    ComparisonConfidence,
)
from archivetrust.domain.evidence.models import BoundingBox
from archivetrust.domain.ontology.types import ObservationType


class DisclosureTier(str, Enum):
    """How much review apparatus this uncertainty warrants (HUMAN_REVIEW_SPECIFICATION.md §5).

    Assigned by triage (§8.1) before the reviewer sees the packet, so the interface opens directly
    at the right amount of complexity and never confronts a reviewer with maximum apparatus for a
    simple problem (UX-INV-8). The reviewer may escalate on demand; triage never starts at maximum.
    """

    TIER_1_SIMPLE = "tier_1_simple"
    """A straightforward value disagreement — minimal apparatus, accept/choose."""
    TIER_2_MODERATE = "tier_2_moderate"
    """A contested value with real divergence — overlays and character/word diffs appear."""
    TIER_3_COMPLEX = "tier_3_complex"
    """A structural / table / reading-order disagreement — full apparatus."""


class ReviewReason(str, Enum):
    """Why triage placed this uncertainty in front of a reviewer (HUMAN_REVIEW_SPECIFICATION.md
    §6 "why am I reviewing this?", §8.1). Carried so the interface can explain itself in the
    reviewer's terms (UX-INV-9), never as a raw score.
    """

    SOURCES_DISAGREE = "sources_disagree"
    """ComparisonClassification.CONTESTED — independent sources conflict."""
    SINGLE_SOURCE = "single_source"
    """ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE — no corroboration possible."""
    LOW_CONFIDENCE = "low_confidence"
    """Corroborated but weak — low Canonical Confidence with no offsetting agreement."""


class EvidenceView(BaseModel):
    """One backing Evidence record, projected for display (HUMAN_REVIEW_SPECIFICATION.md §9.1,
    §10.2). Carries the bounding box that drives auto-scroll/zoom/highlight (UX-INV-1) and the
    raw captured text for the OCR/provider overlay.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    provider_id: str
    provider_version: str
    page: int | None
    bounding_box: BoundingBox | None
    raw_output: str
    provider_confidence: float | None
    coordinate_metadata: dict = {}
    """The bounding box's coordinate contract (units/origin/page extent — Operational Hardening
    milestone, Priority 4/14), copied from `Evidence.supporting_metadata`. What lets a viewer map
    `bounding_box` back onto a rendered page deterministically instead of guessing a convention."""


class CandidateView(BaseModel):
    """One provider's competing claim about this slot (HUMAN_REVIEW_SPECIFICATION.md §10). The
    reviewer adjudicates *between* candidates against the original document (§10.2, HR-7).
    `value` is the candidate's display text where the payload is text-bearing, else `None`
    (structural candidates are compared via §10.3 tooling, not a text value).
    """

    model_config = ConfigDict(frozen=True)

    observation_id: str
    provider_id: str
    provider_version: str
    value: str | None
    provider_confidence: float | None
    evidence: tuple[EvidenceView, ...]
    structure_summary: dict[str, str | int | float | bool | None] = {}
    """Provider-masked structural facts extracted from the payload for reviewer tooling.

    Text-bearing candidates already have `value`; structural/table candidates often do not. This
    field carries only neutral, ontology-level shape such as table dimensions or cell coordinates,
    never provider identity or native labels.
    """
    mapping_table_entry_id: str | None = None
    """Constitution Article 28 -- which `MappingTableEntry` (`domain/ontology/mapping.py`) produced
    this candidate's payload, if recorded. Carried on the data model for provenance/debugging (a
    future non-reviewer-facing tool, e.g. Phase 10's visualization work), but **never rendered in
    the reviewer-facing Qt view** -- the id embeds the provider's own identity verbatim
    (`f"{provider_id}:{native_label}"`), and anti-goal §14 requires provider identity be masked
    from reviewers (`_source_label`'s "Source A"/"Source B" scheme). `None` when not recorded."""


class AgreementView(BaseModel):
    """The agreement picture for this slot (HUMAN_REVIEW_SPECIFICATION.md §10.4). The three
    classifications are *categorically* different and must render as distinct states, never as
    points on one bar (UX-INV-6, `ROADMAP.md` §5.3.1).
    """

    model_config = ConfigDict(frozen=True)

    classification: ComparisonClassification
    independent_source_count: int
    """How many distinct providers contributed a candidate to this slot."""


class ReviewPacket(BaseModel):
    """Everything the interface needs to review one uncertainty (HUMAN_REVIEW_SPECIFICATION.md
    §8.2). Immutable and read-only; assembled from the Trust Engine telemetry stream (LP-2),
    never by querying live pipeline state.
    """

    model_config = ConfigDict(frozen=True)

    semantic_slot_id: str
    document_ref: str
    canonical_observation_id: str
    observation_type: ObservationType
    archive_object_ref: str

    current_value: str | None
    """The current canonical value, for text-bearing slots; `None` for structural slots."""
    candidates: tuple[CandidateView, ...]
    agreement: AgreementView
    comparison_confidence: ComparisonConfidence
    canonical_confidence: CanonicalConfidence | None

    disclosure_tier: DisclosureTier
    review_reason: ReviewReason
    reconciliation_basis_code: str | None = None
    """Constitution Article 26/28 -- the structured reason code
    (`domain.comparison.text_reconciliation.ReconciliationBasisCode`) behind this slot's current
    canonical value, in addition to (never instead of) `review_reason`. Rendered in the reviewer-
    facing view translated to plain English (mirroring how `review_reason`'s own enum values are
    never shown to a reviewer verbatim) -- never as the raw code string, which is architecture
    jargon Constitution §14 and this UI's own established discipline both keep out of the reviewer-
    facing surface."""

    @property
    def primary_bounding_box(self) -> BoundingBox | None:
        """The box the Evidence Viewer should auto-navigate to first (UX-INV-1): the first
        candidate's first bounding box. `None` only when no candidate carries geometry — a
        degraded state the interface surfaces honestly (§13), never papers over.
        """
        for candidate in self.candidates:
            for evidence in candidate.evidence:
                if evidence.bounding_box is not None:
                    return evidence.bounding_box
        return None
