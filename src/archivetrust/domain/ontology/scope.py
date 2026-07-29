"""Observation Scope (Phase 19/20, 2026-07-15): how much semantic content one Observation's claim
covers -- orthogonal to `ObservationType`, which answers "what kind of thing is this."

Phase 19 (`docs/PHASE_19_OBSERVATION_SCOPE_ONTOLOGY_ASSESSMENT_2026-07-15.md`) concluded, from
Phase 18's corpus-wide measurement (`paddleocr-vl` emits ~1 page-scale `PARAGRAPH` observation per
page with zero bounding boxes; `docling`/`tesseract_layoutparser` emit paragraph/line-scale
observations with real geometry), that Observation Scope is a genuine, previously-unrepresented
domain concept. Per Constitution Article 7 (graph structure/facts are not duplicated as payload)
it is a *property* every typed Observation can carry, never a competing `ObservationType`; per
Article 20 it is measured from the Observation's own content and geometry, never provider-asserted
or provider-identity-branched.

Facts and policy interpretation are deliberately two different objects, mirroring
`ComparisonConfidence`'s `magnitude` (fact) vs. `review/triage.py`'s `TriagePolicy` ->
`review_reason_for` (a live, versioned, never-persisted decision computed fresh from stable facts):
`ObservationScopeMeasurement` carries no policy dependency and is stored once, at Observation
construction, and never invalidated by a later policy revision. `classify_observation_scope`
is a pure function of a measurement plus a policy, never stored -- if
`ObservationScopePolicy`'s thresholds are later revised (the 40/500-character values below are
explicit placeholders, exactly like `TriagePolicy`'s own documented thresholds), every existing
measurement's classification updates for free, with no data migration and no historical ambiguity
about which policy version produced which stored value (Constitution Article 28's discipline,
applied here).
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.evidence.models import Evidence
from archivetrust.domain.ontology.payloads.base import ObservationPayload

OBSERVATION_SCOPE_POLICY_VERSION = 1

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


class ObservationScopeClassification(str, Enum):
    """Named to match `ComparisonClassification`'s existing vocabulary in this codebase (both are
    policy-derived categories over a measured fact), and deliberately not "Tier" -- `DisclosureTier`
    (`review/packet.py`) is an unrelated review-UI-complexity concept; reusing "Tier" here would
    have made the two easy to confuse.
    """

    FRAGMENT = "fragment"
    """Much shorter than a typical single semantic unit (e.g. a short label or list item)."""
    UNIT = "unit"
    """Ordinary single-semantic-unit scale -- the common case for `docling`/`tesseract_layoutparser`
    observations today (Phase 18), and the honest default for non-text-bearing payloads, which
    have no character-count signal to classify by."""
    EXTENDED = "extended"
    """Substantially larger than a typical single semantic unit -- e.g. `paddleocr-vl`'s
    whole-page transcription blocks (Phase 18)."""


class ObservationScopeMeasurement(BaseModel):
    """Raw, policy-independent facts about one Observation's scope. Computed once at
    `Observation.from_evidence()` time (the only point a payload and its real, resolved `Evidence`
    records -- not just `evidence_ids` references -- are both in hand) and stored on `Observation`.
    Carries no `policy_version` field on purpose: nothing here depends on interpretation, so
    nothing here can go stale when a policy changes.
    """

    model_config = ConfigDict(frozen=True)

    character_count: int | None
    """`None` for non-text-bearing payloads (image, table, layout_region, ...) -- an honest
    not-applicable marker, never a fabricated zero."""
    word_count: int | None
    sentence_count: int | None
    has_bounding_box: bool
    page_area_fraction: float | None
    """Bounding-box area / page area, when both are measurable from the contributing Evidence's
    geometry and `supporting_metadata` (`pixel_width`/`pixel_height`, as observed in real
    telemetry during Phase 16). Retained for future research (Phase 19 Step 9's anticipated
    scope-mismatch comparison feature) -- not classification-decisive today (see
    `classify_observation_scope`)."""


class ObservationScopePolicy(BaseModel):
    """Versioned exactly like `TriagePolicy`/`ConfidencePolicy` -- thresholds are recorded so a
    later, better-calibrated version supersedes rather than silently overwrites them.
    """

    model_config = ConfigDict(frozen=True)

    version: int = OBSERVATION_SCOPE_POLICY_VERSION

    fragment_max_characters: int = 40
    """A text-bearing observation at or below this length classifies as FRAGMENT."""
    unit_max_characters: int = 500
    """Deliberate placeholders pending corpus validation (mirrors `TriagePolicy`'s own documented
    convention), not arbitrary: Phase 18 measured `docling`/`tesseract_layoutparser` paragraph
    means of 120-182 characters and `paddleocr-vl`'s mean of 1,727 characters -- 500 sits inside
    that empirical gap without being fit to any one provider's identity."""


def _extract_text(payload: ObservationPayload) -> str | None:
    """Text-bearing ontology payloads (paragraph, heading, caption, footnote, named_entity,
    table_cell) carry a `text` field; `metadata` carries `value` instead. Structural payloads
    (image, table, layout_region, ...) carry neither -- `None` is a real, expected outcome for
    those, not a parsing failure.
    """
    text = getattr(payload, "text", None)
    if isinstance(text, str) and text:
        return text
    value = getattr(payload, "value", None)
    if isinstance(value, str) and value:
        return value
    return None


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def _bbox_area(bbox) -> float:
    return max(0.0, bbox.x1 - bbox.x0) * max(0.0, bbox.y1 - bbox.y0)


def measure_observation_scope(
    payload: ObservationPayload, evidence: tuple[Evidence, ...]
) -> ObservationScopeMeasurement:
    """Pure and provider-independent: reads only `payload`'s own content and `evidence`'s own
    geometry, never `provider_id` or any provider-specific branch.
    """
    text = _extract_text(payload)
    character_count = word_count = sentence_count = None
    if text is not None:
        normalized = _norm(text)
        character_count = len(text)
        word_count = len(_WORD_RE.findall(normalized.lower()))
        sentences = [p for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
        sentence_count = max(len(sentences), 1)

    has_bounding_box = any(e.bounding_box is not None for e in evidence)

    page_area_fraction = None
    for record in evidence:
        if record.bounding_box is None:
            continue
        page_width = record.supporting_metadata.get("pixel_width")
        page_height = record.supporting_metadata.get("pixel_height")
        if not page_width or not page_height:
            continue
        page_area = float(page_width) * float(page_height)
        if page_area <= 0:
            continue
        page_area_fraction = _bbox_area(record.bounding_box) / page_area
        break

    return ObservationScopeMeasurement(
        character_count=character_count,
        word_count=word_count,
        sentence_count=sentence_count,
        has_bounding_box=has_bounding_box,
        page_area_fraction=page_area_fraction,
    )


def classify_observation_scope(
    measurement: ObservationScopeMeasurement, policy: ObservationScopePolicy | None = None
) -> ObservationScopeClassification:
    """The live, policy-derived interpretation of an already-measured, policy-independent fact --
    never stored (Constitution Article 33: a deterministic projection is not a telemetry event).
    Decided only from `character_count`; a measurement with no text signal (`character_count is
    None`) classifies as UNIT by default -- an honest "insufficient signal," not an invented
    area-based threshold no phase in this investigation series has validated.
    """
    policy = policy or ObservationScopePolicy()
    if measurement.character_count is None:
        return ObservationScopeClassification.UNIT
    if measurement.character_count <= policy.fragment_max_characters:
        return ObservationScopeClassification.FRAGMENT
    if measurement.character_count <= policy.unit_max_characters:
        return ObservationScopeClassification.UNIT
    return ObservationScopeClassification.EXTENDED
