"""Interactive Reasoning Explorer query library
(`ROADMAP_TELEMETRY_STANDARD.md` Phase 11; `ARCHITECTURE_TELEMETRY_STANDARD.md` §11).

Each function below answers exactly one row of the architecture document's Adversarial
Completeness Audit table, by name, against a replayed `JournalState` or the checked-in Research
Telemetry registry (`benchmarks/research_telemetry.jsonl`) -- never by reading algorithm source or
writing a bespoke script. This directly obsoletes the need for future one-off scripts like
`scripts/adversarial_audit_dump_eliminated_packets.py`.

Every function here is a thin, pure wrapper over machinery Phases 2-10 already built
(`JournalState`, `FileResearchTelemetrySink`, `triage_classification_for`). None of them introduce
new persistence or new event types (Article 8's discipline against inventing unneeded machinery,
applied to query-library scope, per Phase 11's own stated risk mitigation) -- they only compose and
name what already exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from archivetrust.application.journal import JournalState
from archivetrust.domain.canonical.observation import CanonicalObservation
from archivetrust.domain.research.events import AuditConducted, BenchmarkExecuted, ResearchEvent
from archivetrust.domain.telemetry.events import (
    CandidateExcluded,
    KnowledgeDiscarded,
    ObservationLeftUnaligned,
    TelemetryEvent,
)
from archivetrust.infrastructure.storage.research_telemetry_sink import FileResearchTelemetrySink
from archivetrust.review.triage import TriageClassification, TriagePolicy, triage_classification_for

# -- Document-scoped questions (architecture doc §11, first nine rows) -----------------------


def why_was_observation_ignored(state: JournalState, observation_id: str) -> str | None:
    """"Why was this observation ignored?" -- `ObservationLeftUnaligned.reason`, or
    `CandidateExcluded.basis_code` if excluded before alignment ran (Article 27). `None` if the
    observation was neither excluded nor left unaligned (i.e. it was not ignored)."""
    for excluded in state.excluded_candidate_pairs():
        if observation_id in (excluded.candidate_observation_id, excluded.compared_against_observation_id):
            return excluded.basis_code.value
    align_state = state.alignment_state(observation_id)
    if isinstance(align_state, ObservationLeftUnaligned):
        return align_state.reason
    return None


def was_provider_considered(state: JournalState, provider_id: str) -> bool:
    """"Why wasn't this provider considered?" -- answered by absence itself (Article 18): no
    `ProviderObservationAttempted` event for that provider means no invocation was ever recorded,
    not that one was recorded and silently dropped."""
    return any(a.provider_id == provider_id for a in state.provider_observation_attempts())


def why_wasnt_this_clustered(state: JournalState, semantic_slot_id: str) -> str | None:
    """"Why wasn't this clustered?" -- the recorded `AlignmentAttempted.alignment_rationale` for
    this comparison group, or `None` if no alignment was ever attempted for it (distinct from an
    attempt that produced an empty grouping). If a candidate was excluded *before* clustering ran
    at all, that is answered instead by `why_was_observation_ignored`'s `CandidateExcluded` branch
    -- the two are deliberately separate questions (architecture doc §11)."""
    attempt = state.alignment_attempt(semantic_slot_id)
    return attempt.alignment_rationale if attempt is not None else None


def why_did_canonical_change(
    state: JournalState, semantic_slot_id: str
) -> tuple[CanonicalObservation, ...]:
    """"Why did this canonical change?" -- the full `supersedes` chain for this slot, oldest to
    newest; each entry's `reconciliation_basis_code` names the mechanism that produced it."""
    return state.canonical_observation_history(semantic_slot_id)


def why_was_confidence_assigned(state: JournalState, subject_id: str) -> tuple[str, ...]:
    """"Why was this confidence assigned?" -- every recorded `ConfidenceChanged.reason` for this
    subject, in the order they were recorded."""
    return tuple(event.reason for event in state.confidence_evolution(subject_id))


def why_was_provenance_retained(state: JournalState, canonical_observation_id: str) -> tuple[str, ...]:
    """"Why was this provenance retained?" -- the (Article-4-enforced non-empty)
    `contributing_observations` for this canonical fact."""
    canonical = state.canonical_observation(canonical_observation_id)
    return tuple(ref.observation_id for ref in canonical.contributing_observations)


def why_wasnt_review_generated(
    canonical: CanonicalObservation, policy: TriagePolicy | None = None
) -> TriageClassification:
    """"Why wasn't this review generated?" -- re-runs the deterministic triage projection
    (Article 33) against the given `TriagePolicy`; distinguishes `NOT_ELIGIBLE` (genuinely nothing
    to review) from `WITHHELD_BY_POLICY` (would be queued under a more permissive policy). A thin,
    named alias over `review/triage.py::triage_classification_for` -- Phase 11 introduces no new
    logic here, since Phase 6 already built the deterministic answer this question needs."""
    return triage_classification_for(canonical, policy)


def why_was_evidence_discarded(events: Iterable[TelemetryEvent], discarded_id: str) -> str | None:
    """"Why was this evidence discarded?" -- `KnowledgeDiscarded.reason` for the event whose
    `discarded_ids` names `discarded_id`, or `None` if it was never discarded this way. Takes raw
    events, not a replayed `JournalState`: `KnowledgeDiscarded` names already-created ids that are
    later removed from consideration, and is deliberately not folded into `JournalState`'s replay
    model (it is not itself a source of further derivable state, per `application/journal.py`'s own
    documented no-op list) -- querying it is exactly the kind of "search the raw stream" case this
    library exists to name, not to build new replay machinery for."""
    for event in events:
        if isinstance(event, KnowledgeDiscarded) and discarded_id in event.discarded_ids:
            return event.reason
    return None


def why_was_evidence_rejected_at_ingestion(state: JournalState, invocation_id: str) -> str | None:
    """A provider response that never became Evidence at all -- `EvidenceRejected.rejection_reason`
    for the given `invocation_id` (Article 5: rejected raw output is preserved by invocation, since
    no `evidence_id` could validly be minted for it), or `None` if that invocation had no
    rejection."""
    for rejection in state.evidence_rejections():
        if rejection.invocation_id == invocation_id:
            return rejection.rejection_reason
    return None


def structurally_excluded_provider_pairs(state: JournalState) -> tuple[CandidateExcluded, ...]:
    """"Can a future engineer tell that two providers can structurally never be compared, without
    running the corpus and noticing zero packets?" -- every `CandidateExcluded` recorded with
    `structural=True`, a standing queryable fact rather than something requiring re-discovery per
    document (architecture doc §11)."""
    return tuple(e for e in state.excluded_candidate_pairs() if e.structural)


# -- Corpus-scoped questions (architecture doc §11, Research Telemetry rows) ------------------


def _load_registry(registry_path: Path) -> tuple[ResearchEvent, ...]:
    if not registry_path.exists():
        return ()
    return tuple(FileResearchTelemetrySink(registry_path).all_events())


def find_audits_examining(registry_path: Path, keyword: str) -> tuple[AuditConducted, ...]:
    """"Has this exact question already been answered by a prior audit?" -- every registered
    `AuditConducted` whose `claim` or `verdict` mentions `keyword` (case-insensitive), oldest to
    newest. Empty, never fabricated, if the registry doesn't exist or nothing matches (Article 18's
    discipline)."""
    keyword_lower = keyword.lower()
    return tuple(
        event
        for event in _load_registry(registry_path)
        if isinstance(event, AuditConducted)
        and (keyword_lower in event.claim.lower() or keyword_lower in event.verdict.lower())
    )


def supersession_chain(registry_path: Path, audit_id: str) -> tuple[AuditConducted, ...]:
    """The full chain of findings this `audit_id` supersedes, oldest to newest, ending at
    `audit_id` itself -- "is a finding still valid, or was it superseded, and by what reasoning?"
    Raises `KeyError` if `audit_id` isn't registered (a real absence, never silently empty)."""
    by_id = {
        event.audit_id: event
        for event in _load_registry(registry_path)
        if isinstance(event, AuditConducted)
    }
    chain: list[AuditConducted] = [by_id[audit_id]]
    current = by_id[audit_id]
    while current.supersedes_audit_id is not None:
        current = by_id[current.supersedes_audit_id]
        chain.append(current)
    return tuple(reversed(chain))


def is_finding_superseded(registry_path: Path, audit_id: str) -> AuditConducted | None:
    """"Is a finding still valid after a later fix changed the code it was measured against?" --
    the `AuditConducted` that names `audit_id` in its own `supersedes_audit_id`, if any; `None` if
    no later finding supersedes it (still the current word on its claim)."""
    for event in _load_registry(registry_path):
        if isinstance(event, AuditConducted) and event.supersedes_audit_id == audit_id:
            return event
    return None


def latest_finding_for(registry_path: Path, keyword: str) -> AuditConducted | None:
    """The most-superseding (i.e. current) registered finding whose claim mentions `keyword` --
    "what is the current, non-superseded conclusion about X?" `None` if nothing matches."""
    matches = find_audits_examining(registry_path, keyword)
    if not matches:
        return None
    ids = {m.audit_id for m in matches}
    superseded_ids = {m.supersedes_audit_id for m in matches if m.supersedes_audit_id in ids}
    current = [m for m in matches if m.audit_id not in superseded_ids]
    return current[-1] if current else matches[-1]
