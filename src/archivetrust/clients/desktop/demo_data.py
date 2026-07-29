"""Realistic demo data for the desktop application.

Seeds a small but *representative* archive — Swedish municipal records (protocols, decisions,
agendas, a voting table, a signature, archive references) — so the operator immediately feels they
are running ArchiveTrust, not a programming example. The data is produced by running the **real**
Comparison and Confidence engines over hand-authored provider Observations and emitting the exact
telemetry a live pipeline emits (Evidence/Observation/Alignment/Comparison/Confidence/Canonical
Document, plus a couple of human corrections and review sessions). Nothing here changes backend
behavior — it only exercises existing domain APIs to populate a telemetry stream the Learning
Platform analytics and every Center can read.

Deliberate disagreements are baked in (OCR `l`/`i` confusions between deterministic providers, a
VLM-only observation with no corroboration) so the Review Center, Quality Center, and Evolution
Center all have real signal to show.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from archivetrust.domain.alignment.telemetry import alignment_events
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.engine import ComparisonEngineResult, run_comparison_engine
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.telemetry import comparison_result_to_events
from archivetrust.domain.confidence.engine import apply_confidence_engine
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.confidence.telemetry import confidence_changed_events
from archivetrust.domain.document.canonical_document import CanonicalDocument
from archivetrust.domain.evidence.models import (
    BoundingBox,
    Evidence,
    Precision,
    ProcessingStage,
)
from archivetrust.domain.feedback.engine import apply_human_correction
from archivetrust.domain.feedback.models import (
    CorrectionAction,
    CorrectionCategory,
    HumanCorrection,
)
from archivetrust.domain.feedback.policy import FeedbackPolicy
from archivetrust.domain.feedback.telemetry import human_correction_events
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import (
    HeadingPayload,
    MetadataPayload,
    ParagraphPayload,
)
from archivetrust.domain.ontology.payloads.base import ObservationPayload
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import (
    CanonicalDocumentCreated,
    EvidenceCreated,
    EvidenceRejected,
    ObservationCreated,
    ProviderObservationAttempted,
)
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.learning.review.interaction import ReviewInteraction, ReviewInteractionKind
from archivetrust.learning.review.sink import ReviewInteractionSink

_RECONCILIATION_POLICY = ReconciliationPolicy(policy_version=1)
_CAPABILITY_MATRIX = CapabilityMatrix(matrix_version=1, entries=())
_CONFIDENCE_POLICY = ConfidencePolicy(confidence_policy_version=1)
_FEEDBACK_POLICY = FeedbackPolicy(feedback_policy_version=1)

# Providers, shown to the operator by these neutral, methodology-describing names elsewhere; here
# they are the real source ids the telemetry carries.
_LAYOUT = "docling"
_OCR = "tesseract"
_VLM = "qwen2.5-vl"


@dataclass(frozen=True)
class DemoDocument:
    document_ref: str
    archive_object_ref: str
    title: str


def _observation(
    provider: str, payload: ObservationPayload, y: float
) -> tuple[Observation, Evidence]:
    bbox = BoundingBox(x0=60.0, y0=y, x1=540.0, y1=y + 26.0, precision=Precision.PIXEL_ACCURATE)
    text = getattr(payload, "text", None) or getattr(payload, "value", payload.observation_type.value)
    ev = Evidence.create(
        provider=provider,
        provider_version="1.0",
        raw_output=str(text),
        processing_stage=ProcessingStage.OCR if provider == _OCR else ProcessingStage.LAYOUT_ANALYSIS,
        page=1,
        bounding_box=bbox,
        provider_confidence=0.9 if provider == _LAYOUT else (0.72 if provider == _OCR else 0.8),
    )
    obs = Observation.from_evidence(
        provider_id=provider, provider_version="1.0", payload=payload, evidence=(ev,)
    )
    return obs, ev


def _slot(
    payloads: tuple[tuple[str, ObservationPayload], ...], y: float
) -> tuple[tuple[Observation, ...], tuple[Evidence, ...]]:
    built = [_observation(p, payload, y) for p, payload in payloads]
    return tuple(o for o, _ in built), tuple(e for _, e in built)


def _emit_document(
    sink: TelemetrySink,
    *,
    document_ref: str,
    archive_object_ref: str,
    slots: tuple[tuple[tuple[str, ObservationPayload], ...], ...],
    rejection: tuple[str, str] | None = None,
) -> tuple[tuple[CanonicalObservation, ...], dict[str, Observation]]:
    """Run the real engines over hand-authored provider Observations and emit the full telemetry
    chain, exactly as `run_pipeline` would. Returns the completed Canonical Observations and the
    Observations-by-id map so the caller can apply demo human corrections to some of them.
    """
    all_obs: list[Observation] = []
    evidence_by_id: dict[str, Evidence] = {}
    obs_by_provider: dict[str, list[Observation]] = {}
    y = 80.0
    for payloads in slots:
        observations, evidences = _slot(payloads, y)
        y += 44.0
        for obs, (provider, _payload) in zip(observations, payloads):
            all_obs.append(obs)
            obs_by_provider.setdefault(provider, []).append(obs)
        for ev in evidences:
            evidence_by_id[ev.evidence_id] = ev

    provider_graphs = tuple(
        ProviderObservationGraph(
            provider_id=provider,
            provider_version="1.0",
            invocation_id=new_id("invocation"),
            observations=tuple(obs_list),
        )
        for provider, obs_list in sorted(obs_by_provider.items())
    )
    observations_by_id = {o.observation_id: o for o in all_obs}

    # -- provider invocation / evidence / observation telemetry --------------------------------
    for graph in provider_graphs:
        sink.append(
            ProviderObservationAttempted(
                event_id=new_id("event"),
                document_ref=document_ref,
                provider_id=graph.provider_id,
                provider_version=graph.provider_version,
                invocation_id=graph.invocation_id,
            )
        )
        for obs in graph.observations:
            for evidence_id in obs.evidence_ids:
                sink.append(
                    EvidenceCreated(
                        event_id=new_id("event"),
                        document_ref=document_ref,
                        invocation_id=graph.invocation_id,
                        evidence=evidence_by_id[evidence_id],
                    )
                )
            sink.append(
                ObservationCreated(
                    event_id=new_id("event"),
                    document_ref=document_ref,
                    invocation_id=graph.invocation_id,
                    observation=obs,
                )
            )
    if rejection is not None:
        provider, reason = rejection
        sink.append(
            EvidenceRejected(
                event_id=new_id("event"),
                document_ref=document_ref,
                provider_id=provider,
                provider_version="1.0",
                invocation_id=new_id("invocation"),
                processing_stage=ProcessingStage.OCR,
                raw_output="<unparseable page region>",
                rejection_reason=reason,
            )
        )

    # -- comparison + confidence (real engines) ------------------------------------------------
    comparison_result = run_comparison_engine(
        provider_graphs, evidence_by_id, _CAPABILITY_MATRIX, _RECONCILIATION_POLICY
    )
    completed_graph = apply_confidence_engine(
        comparison_result.reconciled_graph, observations_by_id, _CONFIDENCE_POLICY
    )
    completed_result = ComparisonEngineResult(
        reconciled_graph=completed_graph,
        bundles=comparison_result.bundles,
        alignment=comparison_result.alignment,
    )

    for event in alignment_events(comparison_result.alignment, document_ref=document_ref):
        sink.append(event)
    for event in comparison_result_to_events(
        completed_result,
        document_ref=document_ref,
        policy=_RECONCILIATION_POLICY,
        capability_matrix=_CAPABILITY_MATRIX,
    ):
        sink.append(event)
    for event in confidence_changed_events(
        completed_graph.canonical_observations, document_ref=document_ref, policy=_CONFIDENCE_POLICY
    ):
        sink.append(event)

    canonical_document = CanonicalDocument.assemble(
        reconciled_graph=completed_graph,
        archive_object_ref=archive_object_ref,
        reassembly_trigger="initial_assembly",
    )
    sink.append(
        CanonicalDocumentCreated(
            event_id=new_id("event"), document_ref=document_ref, canonical_document=canonical_document
        )
    )
    return completed_graph.canonical_observations, observations_by_id


def _apply_correction(
    sink: TelemetrySink,
    *,
    document_ref: str,
    canonical: CanonicalObservation,
    observations_by_id: dict[str, Observation],
    action: CorrectionAction,
    category: CorrectionCategory,
    corrected: str | None,
) -> None:
    contributing = tuple(
        observations_by_id[ref.observation_id]
        for ref in canonical.contributing_observations
        if ref.observation_id in observations_by_id
    )
    correction = HumanCorrection(
        correction_id=new_id("correction"),
        target_canonical_observation_id=canonical.canonical_observation_id,
        category=category,
        action=action,
        raw_ai_output=getattr(canonical.payload, "text", "") or "",
        raw_corrected_output=corrected,
    )
    resulting = apply_human_correction(canonical, correction, contributing, _FEEDBACK_POLICY)
    for event in human_correction_events(
        correction=correction,
        original=canonical,
        resulting=resulting,
        document_ref=document_ref,
        policy=_FEEDBACK_POLICY,
    ):
        sink.append(event)


def seed_realistic_archive(
    sink: TelemetrySink, interaction_sink: ReviewInteractionSink
) -> list[DemoDocument]:
    """Populate the telemetry + review-interaction streams with a small municipal archive.

    Returns the documents, first of which is the one the Review Center opens (it has genuine
    uncertainties to resolve).
    """
    documents: list[DemoDocument] = []

    # -- Document 1: Council minutes, with a contested OCR reading and a VLM-only signature ------
    doc1 = "kf_protokoll_2019_03_14"
    arch1 = "arkiv/KF/2019/protokoll-03-14.tiff"

    canonicals_1, all_obs_1 = _emit_document(
        sink,
        document_ref=doc1,
        archive_object_ref=arch1,
        slots=(
            (
                (_LAYOUT, HeadingPayload(text="Kommunfullmäktige — Sammanträdesprotokoll", level=1)),
                (_OCR, HeadingPayload(text="Kommunfullmäktige — Sammanträdesprotokoll", level=1)),
                (_VLM, HeadingPayload(text="Kommunfullmäktige — Sammanträdesprotokoll", level=1)),
            ),
            (
                (_LAYOUT, HeadingPayload(text="§ 24 Beslut om budgetram 2020", level=2)),
                (_OCR, HeadingPayload(text="§ 24 Beslut om budgetram 2020", level=2)),
            ),
            (
                # Contested: OCR misreads "ekonomichefen" as "ekonomlchefen" (l/i confusion).
                (_LAYOUT, ParagraphPayload(text="Ärendet föredrogs av ekonomichefen.")),
                (_OCR, ParagraphPayload(text="Ärendet föredrogs av ekonomlchefen.")),
            ),
            (
                (_LAYOUT, ParagraphPayload(text="Kommunfullmäktige beslutar att fastställa budgetramen för 2020 enligt kommunstyrelsens förslag.")),
                (_VLM, ParagraphPayload(text="Kommunfullmäktige beslutar att fastställa budgetramen för 2020 enligt kommunstyrelsens förslag.")),
            ),
            (
                (_LAYOUT, MetadataPayload(key="Arkivreferens", value="KF 2019/142")),
            ),
            (
                # VLM-only: a signature line no OCR/layout pass recognized -> Unaligned,
                # single-source. `HandwrittenNotePayload` (docs/htr-migration-plan.md Stage 5 --
                # EXECUTED) was deleted along with `ObservationType.HANDWRITTEN_NOTE`; its
                # paragraph-level "is this handwritten" flag is superseded by line-level
                # `TEXT_LINE`/`RAW_TRANSCRIPTION` (a later phase's concern) -- `ParagraphPayload`
                # is the smallest correct stand-in for this demo scenario's text content.
                (_VLM, ParagraphPayload(text="Justerat: Anna Lindqvist")),
            ),
        ),
    )
    documents.append(DemoDocument(doc1, arch1, "Kommunfullmäktige — Protokoll 2019-03-14"))

    # -- Document 2: Building committee decision, a rejection, an agenda table reference ---------
    doc2 = "byggnadsnamnden_beslut_47"
    arch2 = "arkiv/BN/2020/beslut-47.tiff"
    _emit_document(  # noqa: F841 (telemetry side-effect; return not needed here)
        sink,
        document_ref=doc2,
        archive_object_ref=arch2,
        slots=(
            (
                (_LAYOUT, HeadingPayload(text="Byggnadsnämnden — Protokoll", level=1)),
                (_OCR, HeadingPayload(text="Byggnadsnämnden — Protokoll", level=1)),
                (_VLM, HeadingPayload(text="Byggnadsnämnden — Protokoll", level=1)),
            ),
            (
                (_LAYOUT, HeadingPayload(text="§ 47 Bygglov Kvarteret Almen 3", level=2)),
                (_OCR, HeadingPayload(text="§ 47 Bygglov Kvarteret Almen 3", level=2)),
            ),
            (
                (_LAYOUT, ParagraphPayload(text="Nämnden beviljar bygglov för nybyggnad av flerbostadshus.")),
                (_OCR, ParagraphPayload(text="Nämnden beviljar bygglov för nybyggnad av flerbostadshus.")),
                (_VLM, ParagraphPayload(text="Nämnden beviljar bygglov för nybyggnad av flerbostadshus.")),
            ),
            (
                (_LAYOUT, MetadataPayload(key="Arkivreferens", value="BN 2020/389")),
            ),
        ),
        rejection=(_OCR, "OCR confidence below threshold on a degraded stamp region"),
    )
    documents.append(DemoDocument(doc2, arch2, "Byggnadsnämnden — Beslut § 47"))

    # -- Document 3: Agenda, cleanly corroborated (a 'healthy' document) -------------------------
    doc3 = "ks_dagordning_2020_01_15"
    arch3 = "arkiv/KS/2020/dagordning-01-15.tiff"
    _emit_document(
        sink,
        document_ref=doc3,
        archive_object_ref=arch3,
        slots=(
            (
                (_LAYOUT, HeadingPayload(text="Kommunstyrelsen — Dagordning 2020-01-15", level=1)),
                (_OCR, HeadingPayload(text="Kommunstyrelsen — Dagordning 2020-01-15", level=1)),
                (_VLM, HeadingPayload(text="Kommunstyrelsen — Dagordning 2020-01-15", level=1)),
            ),
            (
                (_LAYOUT, ParagraphPayload(text="1. Mötets öppnande och upprop.")),
                (_OCR, ParagraphPayload(text="1. Mötets öppnande och upprop.")),
                (_VLM, ParagraphPayload(text="1. Mötets öppnande och upprop.")),
            ),
            (
                (_LAYOUT, ParagraphPayload(text="2. Fastställande av dagordning.")),
                (_OCR, ParagraphPayload(text="2. Fastställande av dagordning.")),
            ),
        ),
    )
    documents.append(DemoDocument(doc3, arch3, "Kommunstyrelsen — Dagordning 2020-01-15"))

    # -- A couple of human corrections, so Quality/Evolution have real signal --------------------
    # Fix the contested OCR paragraph in document 1 (a transcription correction).
    contested = next(
        (c for c in canonicals_1 if getattr(c.payload, "text", "") == "Ärendet föredrogs av ekonomichefen."),
        None,
    )
    if contested is not None:
        _apply_correction(
            sink,
            document_ref=doc1,
            canonical=contested,
            observations_by_id=all_obs_1,
            action=CorrectionAction.EDIT,
            category=CorrectionCategory.TRANSCRIPTION_ERROR,
            corrected="Ärendet föredrogs av ekonomichefen.",
        )
    # Confirm the budget decision paragraph (an accept).
    accepted = next(
        (c for c in canonicals_1 if getattr(c.payload, "text", "").startswith("Kommunfullmäktige beslutar")),
        None,
    )
    if accepted is not None:
        _apply_correction(
            sink,
            document_ref=doc1,
            canonical=accepted,
            observations_by_id=all_obs_1,
            action=CorrectionAction.ACCEPT,
            category=CorrectionCategory.OTHER,
            corrected=None,
        )

    _seed_review_sessions(interaction_sink)
    return documents


def _seed_review_sessions(interaction_sink: ReviewInteractionSink) -> None:
    """A handful of completed review sessions so Quality Center human-effort figures are non-zero.

    Passive telemetry only (the Learning Platform's own stream) — this seeds *observed behavior*,
    not a domain change.
    """
    base = time.monotonic()
    reviewer = "reviewer.anna"
    # One accepted review (fast path) and one edited review (slower), so effort/edit-rate are real.
    scripts = (
        ("review_demo_1", "canonical_demo_1", (
            (ReviewInteractionKind.REVIEW_OPENED, 0.0),
            (ReviewInteractionKind.OVERLAY_TOGGLED, 3.0),
            (ReviewInteractionKind.VALUE_ACCEPTED, 6.5),
            (ReviewInteractionKind.REVIEW_COMPLETED, 7.0),
        )),
        ("review_demo_2", "canonical_demo_2", (
            (ReviewInteractionKind.REVIEW_OPENED, 0.0),
            (ReviewInteractionKind.ZOOMED, 4.0),
            (ReviewInteractionKind.EDIT_STARTED, 9.0),
            (ReviewInteractionKind.EDIT_COMMITTED, 22.0),
            (ReviewInteractionKind.REVIEW_COMPLETED, 23.0),
        )),
    )
    for review_id, target, steps in scripts:
        for kind, offset in steps:
            interaction_sink.append(
                ReviewInteraction(
                    interaction_id=new_id("interaction"),
                    review_id=review_id,
                    target_canonical_observation_id=target,
                    reviewer_ref=reviewer,
                    kind=kind,
                    timestamp=base + offset,
                )
            )
