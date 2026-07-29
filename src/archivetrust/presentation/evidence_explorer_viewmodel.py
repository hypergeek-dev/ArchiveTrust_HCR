"""Evidence Explorer ViewModel (ROADMAP_V2.md §12; `ARCHITECTURE_TELEMETRY_STANDARD.md` §10).

Framework-independent. Reconstructs the full trust chain — Evidence → Observation → Alignment →
Comparison → Canonical Observation → Human Correction — from stored telemetry alone, via the
existing `Journal`. Everything the operator inspects is traceable and reproducible; nothing is
recomputed by re-running a provider. No backend module is modified.

This is the Architect Client's surface (unmasked, unlike the reviewer-facing Review Center, §14) --
exactly where `mapping_table_entry_id` (Article 28) is appropriate to show unmasked, per the
disclosed decision in Phase 7's own report. Extended 2026-07-14 (Phase 10) to surface `basis_code`
(Article 26), `mapping_table_entry_id` (Article 28), structurally-excluded candidate pairs
(Article 27), and a cross-reference to any Research Telemetry finding examining this document's
Workspace (Article 29/34) -- closing the loop this session's Phases 4, 5, 7, and 9 opened.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.application.journal import Journal
from archivetrust.application.current_state import CurrentStateService
from archivetrust.domain.research.events import AuditConducted, BenchmarkExecuted
from archivetrust.domain.telemetry.events import (
    CanonicalDecisionCreated,
    CanonicalDocumentCreated,
    ObservationLeftUnaligned,
    latest_provenance_context,
)
from archivetrust.infrastructure.storage.research_telemetry_sink import FileResearchTelemetrySink
from archivetrust.learning.source import TelemetrySource

_RESEARCH_TELEMETRY_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "research_telemetry.jsonl"
)


class EvidenceTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    provider_id: str
    raw_output: str
    provider_confidence: float | None


class ContributorTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation_id: str
    provider_id: str
    value: str | None
    alignment_outcome: str
    evidence: tuple[EvidenceTrace, ...]
    mapping_table_entry_id: str | None = None
    """Constitution Article 28 -- which `MappingTableEntry` produced this contributor's payload,
    if recorded. Unmasked here (unlike Review Center, §14): this is the Architect Client surface."""


class ExcludedPairTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_observation_id: str
    compared_against_observation_id: str
    excluding_mechanism: str
    basis_code: str
    structural: bool


class CanonicalTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    canonical_observation_id: str
    semantic_slot_id: str
    observation_type: str
    value: str | None
    classification: str
    canonical_confidence: float | None
    was_corrected: bool
    alignment_rationale: str | None
    reconciliation_basis_code: str | None = None
    """Constitution Article 26/28 -- shown as the raw code, not translated to plain English
    (unlike Review Center's `_BASIS_CODE_EXPLANATION`): this is the engineer-facing surface, where
    the code itself is exactly the queryable fact Article 26 exists to make available."""
    contributors: tuple[ContributorTrace, ...]


class DocumentTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_ref: str
    evidence_count: int
    observation_count: int
    unaligned_count: int
    canonical_traces: tuple[CanonicalTrace, ...]
    excluded_pairs: tuple[ExcludedPairTrace, ...] = ()
    """Constitution Article 27 -- structurally-excluded candidate pairs, the exclusion-side
    symmetry `canonical_traces`' `alignment_rationale` alone does not provide."""
    examining_findings: tuple[str, ...] = ()
    """Constitution Article 29/34 -- titles of any registered Research Telemetry finding whose
    `corpus_ref` matches this document's recorded Workspace (via `ProvenanceContextEstablished`).
    Empty, never fabricated, when no `ProvenanceContextEstablished` was recorded for this document
    or no finding examined its Workspace (Article 18's discipline, applied here)."""


def _value(payload: object) -> str | None:
    text = getattr(payload, "text", None)
    if isinstance(text, str):
        return text
    value = getattr(payload, "value", None)
    return value if isinstance(value, str) else None


class EvidenceExplorerViewModel:
    def __init__(
        self,
        source: TelemetrySource,
        journal: Journal | None = None,
        current_state: CurrentStateService | None = None,
    ) -> None:
        self._source = source
        self._journal = journal or Journal()
        self._current_state = current_state or CurrentStateService(source)

    def documents(self) -> tuple[str, ...]:
        """Documents with a Canonical Document assembled — the archive objects there is a full
        trust chain to explore."""
        return tuple(
            state.document_ref
            for state in self._current_state.documents()
            if state.latest_canonical_document is not None or state.slots
        )

    def document_trace(self, document_ref: str) -> DocumentTrace:
        state = self._journal.replay(self._source.events_for_document(document_ref))
        current = self._current_state.document(document_ref)

        traces: list[CanonicalTrace] = []
        for slot in current.slots:
            canonical = slot.current
            attempt = state.alignment_attempt(canonical.semantic_slot_id)

            contributors: list[ContributorTrace] = []
            for ref in canonical.contributing_observations:
                try:
                    observation = state.observation(ref.observation_id)
                except KeyError:
                    continue
                align_state = state.alignment_state(ref.observation_id)
                outcome = (
                    "unaligned"
                    if isinstance(align_state, ObservationLeftUnaligned)
                    else ("aligned" if align_state is not None else "—")
                )
                evidence_traces = []
                for evidence_id in observation.evidence_ids:
                    try:
                        evidence = state.evidence(evidence_id)
                    except KeyError:
                        continue
                    evidence_traces.append(
                        EvidenceTrace(
                            evidence_id=evidence.evidence_id,
                            provider_id=evidence.provider,
                            raw_output=evidence.raw_output,
                            provider_confidence=evidence.provider_confidence,
                        )
                    )
                mapping = state.mapping_used_for(observation.observation_id)
                contributors.append(
                    ContributorTrace(
                        observation_id=observation.observation_id,
                        provider_id=observation.provider_id,
                        value=_value(observation.payload),
                        alignment_outcome=outcome,
                        evidence=tuple(evidence_traces),
                        mapping_table_entry_id=(
                            mapping.mapping_table_entry_id if mapping is not None else None
                        ),
                    )
                )

            traces.append(
                CanonicalTrace(
                    canonical_observation_id=canonical.canonical_observation_id,
                    semantic_slot_id=canonical.semantic_slot_id,
                    observation_type=canonical.observation_type.value,
                    value=_value(canonical.payload),
                    classification=canonical.comparison_confidence.classification.value,
                    canonical_confidence=(
                        canonical.canonical_confidence.value
                        if canonical.canonical_confidence is not None
                        else None
                    ),
                    was_corrected=canonical.human_correction_ref is not None,
                    alignment_rationale=attempt.alignment_rationale if attempt is not None else None,
                    reconciliation_basis_code=canonical.reconciliation_basis_code,
                    contributors=tuple(contributors),
                )
            )

        traces.sort(key=lambda t: (t.observation_type, t.value or ""))

        excluded_pairs = tuple(
            ExcludedPairTrace(
                candidate_observation_id=pair.candidate_observation_id,
                compared_against_observation_id=pair.compared_against_observation_id,
                excluding_mechanism=pair.excluding_mechanism,
                basis_code=pair.basis_code.value,
                structural=pair.structural,
            )
            for pair in state.excluded_candidate_pairs()
        )

        return DocumentTrace(
            document_ref=document_ref,
            evidence_count=len(state.all_evidence()),
            observation_count=len(state.all_observations()),
            unaligned_count=len(state.unaligned_observations()),
            canonical_traces=tuple(traces),
            excluded_pairs=excluded_pairs,
            examining_findings=self._examining_findings(
                list(self._source.events_for_document(document_ref))
            ),
        )

    @staticmethod
    def _examining_findings(events: list) -> tuple[str, ...]:
        """Constitution Article 29/34: cross-references this document's recorded Workspace
        (via `ProvenanceContextEstablished`) against the checked-in Research Telemetry registry.
        Never fabricated: returns `()` if no context was recorded, the registry doesn't exist yet,
        or no finding's `corpus_ref` matches -- each a real, distinguishable absence (Article 18).
        """
        context = latest_provenance_context(events)
        if context is None or not _RESEARCH_TELEMETRY_PATH.exists():
            return ()
        workspace_ref = f"workspace:{context.workspace_identifier}"
        sink = FileResearchTelemetrySink(_RESEARCH_TELEMETRY_PATH)
        titles: list[str] = []
        for finding in sink.all_events():
            if finding.corpus_ref != workspace_ref:
                continue
            if isinstance(finding, (AuditConducted, BenchmarkExecuted)):
                titles.append(finding.title)
        return tuple(titles)
