from __future__ import annotations

from archivetrust.domain.comparison.capability_matrix import (
    Capability,
    CapabilityMatrix,
    CapabilityMatrixEntry,
)
from archivetrust.domain.comparison.engine import run_comparison_engine
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.comparison.text_reconciliation import ReconciliationBasisCode
from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision, ProcessingStage
from archivetrust.domain.graph.provider_graph import ProviderObservationGraph
from archivetrust.domain.ontology.base import Observation
from archivetrust.domain.ontology.payloads import (
    HeadingPayload,
    ParagraphPayload,
)
from archivetrust.domain.ontology.types import ObservationType


def _policy() -> ReconciliationPolicy:
    return ReconciliationPolicy(policy_version=1)


def _matrix() -> CapabilityMatrix:
    entries = []
    for provider, version in (("docling", "1.0"), ("tesseract", "1.0")):
        for obs_type in (ObservationType.HEADING, ObservationType.PARAGRAPH, ObservationType.TABLE):
            entries.append(
                CapabilityMatrixEntry(
                    provider_id=provider, provider_version=version, observation_type=obs_type, capability=Capability.NATIVE
                )
            )
    return CapabilityMatrix(matrix_version=1, entries=tuple(entries))


class _Fixture:
    def __init__(self) -> None:
        self.evidence_by_id: dict[str, Evidence] = {}

    def evidence(self, provider: str, text: str, bbox: BoundingBox | None = None, page: int = 1, confidence: float | None = None) -> Evidence:
        ev = Evidence.create(
            provider=provider, provider_version="1.0", raw_output=text, processing_stage=ProcessingStage.OCR,
            page=page, bounding_box=bbox, provider_confidence=confidence,
        )
        self.evidence_by_id[ev.evidence_id] = ev
        return ev


def test_two_provider_document_produces_reconciled_graph_and_bundles():
    fx = _Fixture()
    box_heading_docling = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    box_heading_tess = BoundingBox(x0=1, y0=1, x1=99, y1=19, precision=Precision.PIXEL_ACCURATE)
    box_para_docling = BoundingBox(x0=0, y0=30, x1=100, y1=60, precision=Precision.PIXEL_ACCURATE)
    box_para_tess = BoundingBox(x0=1, y0=31, x1=99, y1=59, precision=Precision.PIXEL_ACCURATE)

    ev_h_d = fx.evidence("docling", "Chapter 1", box_heading_docling, confidence=0.9)
    ev_h_t = fx.evidence("tesseract", "Chapter l", box_heading_tess, confidence=0.7)
    ev_p_d = fx.evidence("docling", "Body text here.", box_para_docling)
    ev_p_t = fx.evidence("tesseract", "Body text here.", box_para_tess)

    heading_docling = Observation.from_evidence(
        provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chapter 1", level=1), evidence=(ev_h_d,)
    )
    heading_tess = Observation.from_evidence(
        provider_id="tesseract", provider_version="1.0", payload=HeadingPayload(text="Chapter l", level=1), evidence=(ev_h_t,)
    )
    para_docling = Observation.from_evidence(
        provider_id="docling", provider_version="1.0", payload=ParagraphPayload(text="Body text here."), evidence=(ev_p_d,)
    )
    para_tess = Observation.from_evidence(
        provider_id="tesseract", provider_version="1.0", payload=ParagraphPayload(text="Body text here."), evidence=(ev_p_t,)
    )

    docling_graph = ProviderObservationGraph(
        provider_id="docling", provider_version="1.0", invocation_id="inv-docling",
        observations=(heading_docling, para_docling),
    )
    tesseract_graph = ProviderObservationGraph(
        provider_id="tesseract", provider_version="1.0", invocation_id="inv-tesseract",
        observations=(heading_tess, para_tess),
    )

    result = run_comparison_engine((docling_graph, tesseract_graph), fx.evidence_by_id, _matrix(), _policy())

    assert len(result.reconciled_graph.canonical_observations) == 2
    heading_canonical = next(
        co for co in result.reconciled_graph.canonical_observations if co.observation_type == ObservationType.HEADING
    )
    assert heading_canonical.comparison_confidence.classification.value == "corroborated"
    assert len(heading_canonical.contributing_observations) == 2
    # Constitution Article 26: the structured code reaches the Canonical Observation Comparison
    # Engine emits telemetry from, not just the free-text reconciliation_basis.
    assert heading_canonical.reconciliation_basis_code in (
        ReconciliationBasisCode.TEXT_EXACT_EQUALITY_CORROBORATED.value,
        ReconciliationBasisCode.TEXT_MAJORITY_VOTE_CONSENSUS.value,
    )

    bundle = result.bundles[heading_canonical.canonical_observation_id]
    assert bundle.trust_score.magnitude > 0
    confidences = {e.provider_id: e.provider_confidence for e in bundle.evidence_report.supporting}
    assert confidences == {"docling": 0.9, "tesseract": 0.7}


def test_single_provider_document_is_all_uncorroborated():
    fx = _Fixture()
    box = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    ev = fx.evidence("docling", "Chapter 1", box)
    obs = Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chapter 1", level=1), evidence=(ev,))
    graph = ProviderObservationGraph(provider_id="docling", provider_version="1.0", invocation_id="inv-1", observations=(obs,))

    result = run_comparison_engine((graph,), fx.evidence_by_id, _matrix(), _policy())
    canonical = result.reconciled_graph.canonical_observations[0]
    assert canonical.comparison_confidence.classification.value == "uncorroborated_single_source"
    assert canonical.comparison_confidence.magnitude is None
    assert canonical.reconciliation_basis_code == ReconciliationBasisCode.TEXT_SINGLE_SOURCE_ONLY_PROVIDER.value


# `test_table_with_cells_is_reconciled_with_child_canonical_cells` (TableCellPayload/
# ObservationType.TABLE_CELL child-cell reconciliation via `table_reconciliation.py`) was deleted
# in docs/htr-migration-plan.md Stage 5 (EXECUTED) along with that type and reconciliation pass --
# table semantics don't fit line-level HTR research; no in-scope HTR method produces structured
# tables. `ObservationType.TABLE` itself still reconciles (via the generic single-deterministic-
# pick path, covered implicitly by the matrix/engine plumbing the other tests here exercise).


def test_engine_is_deterministic_across_repeated_runs():
    fx = _Fixture()
    box = BoundingBox(x0=0, y0=0, x1=100, y1=20, precision=Precision.PIXEL_ACCURATE)
    ev = fx.evidence("docling", "Chapter 1", box)
    obs = Observation.from_evidence(provider_id="docling", provider_version="1.0", payload=HeadingPayload(text="Chapter 1", level=1), evidence=(ev,))
    graph = ProviderObservationGraph(provider_id="docling", provider_version="1.0", invocation_id="inv-1", observations=(obs,))

    first = run_comparison_engine((graph,), fx.evidence_by_id, _matrix(), _policy())
    second = run_comparison_engine((graph,), fx.evidence_by_id, _matrix(), _policy())
    first_payloads = [co.payload for co in first.reconciled_graph.canonical_observations]
    second_payloads = [co.payload for co in second.reconciled_graph.canonical_observations]
    assert first_payloads == second_payloads
