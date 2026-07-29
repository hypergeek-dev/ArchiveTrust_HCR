# The ArchiveTrust Telemetry Architecture Standard

**Status:** Canonical architectural specification. Amends `ARCHITECTURAL_CONSTITUTION.md` (adds
Part X, Articles 26–31) and extends ROADMAP.md §12. Does not replace either — Articles 1–25 and
the existing `TelemetryEventKind` set (ROADMAP.md §12, `domain/telemetry/events.py`) remain in
force unchanged. This document exists because six independent investigations in one week
(Review Packet Integrity Audit, Semantic Alignment Audit, Human Reviewability Audit, Semantic
Contract Audit, Observation Typing Prevalence Audit, and the Adversarial Architecture Audit) each
had to hand-reconstruct reasoning the system itself should have been able to produce on request.

**What this is not:** an implementation note, a sprint plan, or a rewrite of the domain model.
**What this is:** the standard every future telemetry-relevant object, transformation, and
decision in ArchiveTrust must satisfy, and the record of exactly which parts of that standard
ArchiveTrust already satisfies today, cited to real code.

---

## 1. Philosophy

**A semantic claim that cannot explain its own existence is an untrustworthy semantic claim.**
This is Article 4 (no canonical fact without traceable evidence) generalized: it is not enough
that a chain of references *exists* from a Canonical Document field back to Evidence. An engineer,
months later, with no memory of this week's investigations, must be able to ask the system itself
*why* that chain looks the way it does — why this Evidence was accepted and that one discarded,
why this alignment was hypothesized, why this confidence value, why this review packet was
generated or wasn't — and get an answer from stored telemetry, not from re-reading source code,
re-running ad hoc scripts, or interviewing whoever happened to be debugging it that week.

Every one of the six audits above was necessary specifically because the telemetry stream, as
specified, could not answer its central question by itself:

- The **Review Packet Integrity Audit** needed to read `clustering.py`'s source to discover the
  ordinal-fallback hash-sort defect — telemetry recorded `ObservationCompared` with a
  `clustering_basis` string, but nothing recorded *which specific geometric fact* (missing vs.
  coarse box) drove the fallback, so the defect was invisible to anyone not reading the algorithm.
- The **Semantic Contract Audit** needed to read `tesseract_layoutparser`'s adapter source to find
  that its `"Text"` label is unconditionally mapped to `PARAGRAPH` — `ObservationMapped` records
  `ontology_version` but not the mapping rule that fired, so a systematic mislabeling was invisible
  to telemetry even though the map from provider vocabulary to Canonical Observation Ontology
  concept is exactly the kind of decision Article 2/20 says should never be provider-private.
- The **Adversarial Architecture Audit** needed a bespoke script
  (`scripts/adversarial_audit_dump_eliminated_packets.py`) and a hand-drawn 30-packet sample to
  measure that a proposed geometry-precision guard — evaluated only as a counterfactual, via an
  alternate `AlignmentService` inside `scripts/rcdp_ordinal_geometry_counterfactual.py`, **never
  merged into `domain/comparison/clustering.py`** — would, if shipped as originally scoped,
  eliminate 332 review packets, 60% of a random sample of which carried unique, high-value content.
  This is itself the case in point for §1.1 below: the counterfactual's own module docstring states
  plainly that "no file under `src/` is modified," yet the finding was informally treated, in prose
  elsewhere, as describing shipped behavior — exactly the conflation §1.1 now forbids. Separately,
  and independently of that counterfactual: production `clustering.py` *does* contain a real,
  currently-shipped exclusion-like decision today (the "ambiguous: a provider contributed more than
  one same-type Observation" case, `clustering.py`'s `_pairwise_affinity`), and *that* decision's
  exclusion has never become a telemetry fact either — this is the actual, present-tense gap Article
  27 addresses.
- **Confidence calibration** (memory: `confidence-calibration-2026-07-14`) is provably not
  computable today: 13,269 flagged review items exist with **zero** recorded human review
  outcomes. `HumanCorrectionSubmitted`/`HumanCorrectionApplied` are specified and implemented, but
  nothing in the architecture requires that every dispatched review packet eventually produce one
  of these events, or makes their absence itself a queryable, alarmable fact.

These are not four unrelated bugs. They are one structural gap, recurring at four different
layers: **telemetry today records that a decision was made and cites the object it produced, but
does not always record the decision's own reasoning as a first-class, independently queryable
fact.** Levels 1–3 below (Operational, Data Lineage, Decision Lineage) are already strong.
Levels 4–5 (Semantic Lineage, Epistemic Provenance) are where every one of these investigations
found itself locked out.

### 1.1 Evidentiary Status: Three Categories, Never Conflated

Added 2026-07-14, after a Phase 3 implementation attempt discovered this document's original
Article 27 and its Universal Transformation Contract entries described a "geometry-precision
guard" in `clustering.py` as though it were shipped, when it exists only as a counterfactual
analysis in `scripts/rcdp_ordinal_geometry_counterfactual.py` — never merged into `src/`. This is
the architecture document's own instance of exactly the failure mode Article 31 exists to catch in
comparison-engine changes: a measured result was allowed to read as a description of production
behavior. The correction is this standing rule, not a one-time fix:

**Every claim in this document, and in `ARCHITECTURAL_CONSTITUTION.md`'s Articles, about what
ArchiveTrust *does* must be classified as exactly one of:**

1. **Current Production Behavior** — verified present in `src/` at the time of writing, citable to
   a real file and line. This is the only category a Constitution Article's normative "shall"/
   "must" language may describe.
2. **Experimentally Validated Counterfactual** — a measured result from code that runs *outside*
   `src/` (a script, a notebook, an alternate implementation swapped in only for the analysis) —
   valid scientific evidence about what *would* happen, cited as exactly that, never phrased as
   though the analyzed mechanism is already active in production.
3. **Future Architectural Candidate** — a design change evidence supports adopting, not yet
   implemented, not yet passed through Article 31's adversarial-completeness check, and not yet
   scheduled on `ROADMAP_TELEMETRY_STANDARD.md`. Recorded in the registry at the end of Part X, so
   a candidate is never simply forgotten, but also never mistaken for something already built.

An Article may *cite* a counterfactual or a candidate as its motivating evidence (exactly as
Article 27, below, cites the RCDP counterfactual as the reason its production mechanism needed
telemetry too) — but the Article's own normative content must describe only category 1. When in
doubt which category a claim belongs to, the test is Article 31's own: could you point at the exact
line in `src/` that does this, right now, on this branch? If not, it is category 2 or 3, and must
be labeled as such.

---

## 2. The Five-Level Telemetry Model, Mapped to What Exists

### Level 1 — Operational Telemetry
*Can we reconstruct execution?*

**Already satisfied**, and satisfied well. `ProviderObservationAttempted`
(`domain/telemetry/events.py:158`) carries `duration_ms`, `retry_count`, `cold_start`,
`init_duration_ms`, `timeout`, `failure_category`, `pages_processed`, `outcome`,
`observation_count`, `failure_reason` — added specifically by the Operational Hardening and
Observability milestones (2026-07-13/14) in direct response to the stalled-run and benchmark-stall
incidents. Every `TelemetryEvent` carries `recorded_at`, `schema_version`,
`reconciliation_policy_version`, `capability_matrix_version`, `confidence_policy_version`,
`feedback_policy_version`, `alignment_algorithm_version`. Acquisition has its own parallel
operational stream (§12.1: `WorkspaceCreated`, `AcquisitionStarted`, etc.).

**Remaining gap:** machine/process/thread/user/build/git-commit/environment/dependency fields are
not present on any event. These are legitimately Level 1 facts the Constitution's own philosophy
(Article 16: telemetry describes knowledge, not execution) would normally exclude — but a narrow
exception already exists (`reviewer_ref`, `submitted_at` on `HumanCorrectionSubmitted`) for facts
needed to explain *why* a knowledge event happened, not just that it happened. The same exception
should extend to a single `ProvenanceContext` record (§4 below) attached at the *document* level,
not per-event, so replay can still ask "what build produced this document's telemetry?" without
polluting every knowledge event with process noise Article 16 was written to keep out.

### Level 2 — Data Lineage
*Can we reconstruct every object transformation?*

**Already satisfied.** `Evidence` is content-addressed (`evidence_id` is a hash of
`provider, provider_version, processing_stage, prompt, raw_output`) so identical raw output is
never duplicated and is always traceable. `Observation` references `Evidence` by id, never embeds
it (Article 5). `CanonicalObservation.contributing_observations` is a non-empty tuple of
`ContributingObservationRef` (Article 4's enforcement point —
`canonical/observation.py:116` raises if empty). `supersedes`/`superseded_by` give the full parent/
child version chain (Article 15). `CanonicalGraphEdge`/`RelatedCanonicalObservationLink` give
document-structure parent/child/related edges, each independently confidence-scored (S18 item 1).

**Remaining gap:** "discarded" and "ignored" objects are the weak members of this level. Article 4
requires *contributing* observations to be recorded; nothing requires *excluded candidate*
observations to be recorded with equal rigor. `KnowledgeDiscarded` exists in the schema
(`events.py:384`) but is not wired into `comparison/clustering.py`'s real, currently-shipped
exclusion-like decision — the "ambiguous: a provider contributed more than one same-type
Observation to this page" case in `_pairwise_affinity`, which drives affinity to `0.0` for a
structural reason — so no `KnowledgeDiscarded`/`ObservationLeftUnaligned` event distinguishes "this
Observation had nothing to compare against" from "this Observation was excluded from a comparison
it should have participated in by a guard condition." (A *separate*, unshipped geometry-precision
guard was evaluated only as a counterfactual by the Adversarial Audit — §1.1 category 2, registry
in Part X — and is not itself a present gap, since it is not present in `src/` to have one.) See
Article 27.

### Level 3 — Decision Lineage
*Can we reconstruct every deterministic decision?*

**Substantially satisfied**, unevenly. `AlignmentAttempted` (`events.py:265`) records
`candidate_observation_ids`, `selected_observation_ids`, and `alignment_rationale` — exactly the
"evidence used / evidence ignored / reason" triad Level 3 requires, and Article 24's own reason for
existing. `ObservationCompared` carries `clustering_basis`. `CanonicalDecisionCreated` carries the
full `CanonicalObservation` including `reconciliation_basis` and `rationale`. `ConfidenceChanged`
requires `previous_value`, `new_value`, `reason`, and which of three levels changed (Article 13).

**Remaining gap, precisely characterized:** `alignment_rationale` and `clustering_basis` are free
text (`str`), not structured. This is why the Review Packet Integrity Audit had to read source code
rather than query telemetry: "ordinal fallback, hash-sort tie-break" is a fact that *exists* in the
code path that produced the string, but the string itself does not commit to a stable, parseable
vocabulary a query can filter on (e.g. "show me every decision where the ambiguous-multi-per-
provider case fired"). Article 26 below requires this to close.

### Level 4 — Semantic Lineage
*Can we reconstruct semantic meaning?*

**The weakest existing level**, and the one the Semantic Contract Audit and Observation Typing
Prevalence Audit exist because of. `ObservationMapped` (`events.py:251`) records that a mapping
happened and which `ontology_version` was current, but not *which rule, of which provider
adapter's mapping table, mapped which raw provider label to which `ObservationType`*. This is
precisely why `tesseract_layoutparser`'s unconditional `"Text"` → `PARAGRAPH` mapping was invisible
to telemetry and had to be found by reading `providers/tesseract_layoutparser/`'s adapter source: no
event exists whose job is to assert "provider label X means ontology concept Y, as of mapping-table
version Z," so there is nothing to audit *in aggregate*, only in one adapter at a time, by a human.
See Article 28.

Every `ObservationPayload` subtype (`domain/ontology/payloads/`) has an implicit semantic contract
— what a `Paragraph` claims to mean, versus a `LayoutRegion`'s catch-all fallback — but that
contract lives only in docstrings and Article 8's "no ontology concept without evidence" discipline
at authoring time. It is not a versioned, machine-checkable, telemetry-visible object a provider
adapter binds to when it emits an `ObservationMapped` event.

### Level 5 — Epistemic Provenance
*Can we reconstruct why knowledge exists?*

**Not currently a domain concern at all.** This is the sharpest, most consequential gap this
document exists to close. Every audit cited in §1, every benchmark (`benchmarks/`), and every
replay this project has ever run exists only as: a markdown report in `docs/`, a script in
`scripts/`, and raw data in `benchmarks/*.jsonl`. None of these — the single most important
knowledge-producing activities in ArchiveTrust's history — emit a telemetry event. There is no
recorded, queryable answer to "which audit established that the RCDP guard over-eliminates
comparisons?", "which benchmark run is Benchmark #1 derived from, and is it still valid after the
observation-typing fix?", or "has this finding been independently reproduced?" *except* by reading
the docs in the order they were written and cross-referencing dates by hand — precisely the
manual reconstruction Article 17 exists to make unnecessary for pipeline decisions, now recurring
one layer up, for decisions *about* the pipeline. See Articles 29–31.

---

## Part X — Constitutional Amendments (Articles 26–31)

### Article 26. Decision Rationale Is Structured, Not Prose
**Every field that records *why* a deterministic decision was made (`clustering_basis`,
`alignment_rationale`, `reconciliation_basis`, `rejection_reason`, and any future equivalent) must
carry a stable, enumerated `basis_code` drawn from a versioned, closed vocabulary specific to the
algorithm that produced it, in addition to (never instead of) a free-text explanation. The free
text may vary; the code may not, until the vocabulary itself is versioned forward.**
Free text is for a human reading one event. A code is for a query across the whole corpus — "how
many decisions this month cited `ORDINAL_POSITION_FALLBACK`" (a real, shipped `clustering.py` code
path — see §1.1: an *example* code must always name a mechanism actually present in `src/`, never
one only a counterfactual makes) is exactly the question every audit in §1 had to answer by reading
source code instead of running a query. Without a closed vocabulary, "the same reason, worded
differently twice" and "two different reasons" are indistinguishable to any aggregate analysis.
*Sources: Review Packet Integrity Audit (clustering_basis opacity); parallels Article 24 (alignment
as recorded hypothesis).*
**Protects:** Observability, Replayability, Scientific validity.

### Article 27. Exclusion From Comparison Is a Recorded Decision, Symmetric With Inclusion
**Any mechanism *currently shipped in `src/`* that removes a candidate Observation from a
comparison group it would otherwise have entered — today, specifically, `_pairwise_affinity`'s
"ambiguous: a provider contributed more than one same-type Observation to this page" case
(`domain/comparison/clustering.py`), which drives affinity to `0.0` for a structural reason rather
than a content-based one — must emit a telemetry event carrying the same rigor `AlignmentAttempted`
already gives to inclusion: which candidates were considered, which were excluded, the structured
`basis_code` (Article 26) for exclusion, and whether the exclusion is per-instance or a structural
property of the mechanism itself (a fact about the *mechanism*, not about any one document — see
Article 29's `AuditConducted` for how that distinction gets surfaced in aggregate). This article
governs exactly the mechanisms present in `src/` today, per §1.1's evidentiary discipline; it does
not, by itself, mandate building a new exclusion mechanism (such as the geometry-precision guard
discussed below), only instrumenting whichever exclusion mechanisms actually exist.**
This was motivated by, but is not a description of, the Adversarial Architecture Audit's finding: a
*proposed* geometry-precision guard, evaluated only as a counterfactual (§1.1, category 2) via an
alternate `AlignmentService` in `scripts/rcdp_ordinal_geometry_counterfactual.py` — never merged
into `clustering.py` — was measured to eliminate every Docling↔PaddleOCR-VL comparison permanently,
if it were ever shipped as originally scoped. That counterfactual result is exactly why *any*
exclusion mechanism, present or future, must be instrumented before or at the moment it ships, not
after: had the guard been shipped without this article already in force, its structural elimination
would have been indistinguishable from "there was genuinely nothing to compare" (Article 18's
provider-silence ambiguity, one layer up, applied to comparison-exclusion) exactly as the
counterfactual predicts. The geometry-precision guard itself remains an unshipped Future
Architectural Candidate (§1.1, category 3; registry at the end of Part X) — this article's
production-scoped mandate would apply to it automatically, without amendment, the day it is
actually merged.
*Sources: Adversarial Architecture Audit ("What Did We Accidentally Remove?", 2026-07-14) and the
RCDP counterfactual it falsifies, both read strictly as §1.1 category 2 evidence, never as
production-behavior description (correction made 2026-07-14 after a Phase 3 implementation attempt
found the original wording conflated the two); parallels Articles 18, 24, 31.*
**Protects:** Trustworthiness (no capability silently disappears once a real mechanism exists),
Observability, Reproducibility.

### Article 28. The Provider-to-Ontology Mapping Is a Versioned, Telemetry-Visible Semantic Contract
**Every rule that maps a provider's native label, field, or output shape to a Canonical Observation
Ontology concept must exist as a named, versioned entry in that provider adapter's mapping table —
never as an inline conditional buried in adapter code — and every `ObservationMapped` event must
cite which mapping-table entry (not merely which `ontology_version`) fired. A mapping-table version
bump is required whenever a rule's *meaning* changes (a label starts mapping to a different
Observation type), even if the ontology itself does not change.**
`ontology_version` alone answers "which set of Observation types existed at the time." It cannot
answer "was `tesseract_layoutparser`'s `Text` label always meant to become a `PARAGRAPH`, or did
that mapping rule silently start doing something different?" — the exact question the Semantic
Contract Audit needed source-reading, not a query, to answer. This is Article 19 (ontology
versioning independent of telemetry versioning) applied one layer inward: the mapping *from*
provider vocabulary *to* ontology concept needs its own version axis, independent of both.
*Sources: Semantic Contract Audit (2026-07-14), Observation Typing Prevalence Audit (2026-07-14,
100% prevalence of the Text→PARAGRAPH mapping across the corpus); parallels Articles 8, 19, 20.*
**Protects:** Trustworthiness, Provider independence, Replayability, Scientific validity.

### Article 29. Epistemic Objects Are Domain Objects, Not External Artifacts
**An Audit, a Benchmark, and a Replay are first-class ArchiveTrust domain objects with their own
identity, telemetry, and lifecycle — never merely a markdown file and a script living outside the
system's own knowledge model. Every audit that establishes, revises, or falsifies a finding about
ArchiveTrust's own behavior must emit `AuditConducted`; every benchmark run must emit
`BenchmarkExecuted`; every deliberate, investigatory replay must emit `ReplayExecuted`. Each
references the specific telemetry events, code commit, and corpus it examined, and the specific
claim it established, revised, or falsified — including a reference to any *prior* Audit/Benchmark
it supersedes, following the identical supersession discipline Article 15 already requires of
Canonical Observations. These events belong to Research Telemetry (Article 34), a separate stream
from document-scoped Trust Engine telemetry — never the per-document `TelemetryEventKind` set.**
Six audits were required this week specifically because none of them could query "has this already
been investigated," "what did the last investigation conclude," or "does this finding still hold
after the fix that was shipped in response to it" — the Adversarial Audit exists *only* because the
RCDP finding it falsifies was itself invisible as anything but a markdown file no later process
could reference. Knowledge about ArchiveTrust's own trustworthiness is exactly the kind of
knowledge Article 4 requires traceability for; there is no principled reason to exempt
self-knowledge from the standard applied to every other canonical fact.
*Sources: all six 2026-07-14 audits, considered as a set; RCDP-to-Adversarial-Audit supersession
relationship specifically; parallels Articles 4, 15, 17.*
**Protects:** Trustworthiness, Replayability, Scientific validity, Observability — recursively,
of the architecture's own claims about itself.

### Article 30. Every Terminal Review Decision Must Emit a Recorded Outcome; Presentation of a Packet Never Does
**Every terminal human review decision — `ACCEPT`, `REJECT`, `EDIT`, `MARK_AMBIGUOUS`,
`REQUEST_FURTHER_REVIEW`, `SKIP`, or any future terminal action — must emit exactly one
`ReviewOutcomeRecorded` event, `SKIP` included: a skip is a human decision, not silence, and
recording nothing for it is a silence violation in its own right. This is distinct from, and never
conflated with, the act of *computing* which uncertainties currently warrant review (triage) or
*presenting* a packet to a reviewer — both of those are deterministic projections over existing
telemetry (Article 33) and must never themselves emit telemetry, no matter how many times the
projection is recomputed. Whether an eligible slot has aged past a configured horizon with no
recorded outcome is likewise a derived fact, computed by comparing the current triage projection
against the recorded `ReviewOutcomeRecorded` population and the eligible slot's own already-recorded
timestamp — never a third persisted event.**
13,269 flagged review items exist with zero recorded outcomes (`confidence-calibration-2026-07-14`
finding) — not because reviewers rejected them, but because `submit_decision`'s `SKIP` path records
literally nothing (`review/service.py`, confirmed 2026-07-14), so its absence from the telemetry
stream is indistinguishable from "not yet reviewed," "reviewed but the event was lost," and "never
actually dispatched." This is Article 18's silence-vs-failure distinction, applied to the one part
of the pipeline where a human, not a provider, is the source being asked a question — and Article
33's projection/knowledge distinction, applied to correct an original scoping of this article that
would have wrongly turned packet generation itself into a telemetry event (see that article).
*Sources: Confidence calibration memory (2026-07-14); Human Reviewability Audit (Accept never
narrows contributing_observations provenance — a related, separately-scoped gap, see the Review
Provenance Narrowing roadmap item); Architecture Review Report, Phase 6 implementation attempt,
2026-07-14 (corrected this article's scope to exclude packet generation); parallels Articles 18, 33.*
**Protects:** Trustworthiness, Scientific validity (calibration is not computable without this),
Observability.

### Article 31. Adversarial Completeness Is a Standing Obligation, Not a One-Time Audit
**Any change to comparison, clustering, alignment, confidence, or triage behavior must be
accompanied, before it ships, by an explicit answer — recorded as an `AuditConducted` event
(Article 29) — to: what capability, if any, does this change remove; for the population it affects,
does it trade recall for precision or eliminate one entirely; and has a random sample been
inspected against ground truth, not merely against the metric the change was designed to optimize.
A change that only reports "the review queue shrank" or "the review queue grew" without this
accompanying audit is incomplete by construction, regardless of test coverage — and, per §1.1, a
counterfactual measurement of a change is not itself a license to ship it; the audit above must
still happen before the change is merged, not only before it is analyzed.**
The RCDP counterfactual — evaluated only via an alternate `AlignmentService`, never merged into
`clustering.py` (§1.1 category 2) — measured that its proposed geometry-precision guard would
reduce 332 contested packets to 0 and, by review-queue-size alone, would have looked like
unambiguous success if shipped; the Adversarial Audit run against that same counterfactual, before
any merge occurred, found 60% of a random sample of what it would eliminate carried unique,
high-value content the guard would have made permanently unreachable. This is this article working
exactly as intended, not a cautionary tale about a shipped mistake: queue size is not a proxy for
information preservation, adversarial review caught the difference *before* the change reached
`src/`, and the guard correctly remains unshipped (a Future Architectural Candidate, §1.1 category
3) precisely because this audit has not yet been satisfied for it. This article makes that same
habit — audit before merge, every time, not just when a counterfactual happens to get built first —
a standing architectural requirement.
*Sources: RCDP counterfactual finding vs. Adversarial Architecture Audit, the paired case this
whole document generalizes from, read strictly as §1.1 category 2 evidence about a candidate that
correctly never shipped; parallels Article 14 (falsifiability as tiebreak), Article 23 (roadmap
amended before violated).*
**Protects:** Scientific validity, Trustworthiness — specifically against optimizing a visible
metric while an invisible one silently degrades, and against a merge occurring before that check.

### Article 32. Operational Context Is Established Once Per Processing Run, Never Duplicated as Per-Event Fields
**Every processing run of a document establishes a single Operational Context — the git commit,
effective configuration, ontology version, and every named policy version in force at that run's
start — recorded as exactly one `ProvenanceContextEstablished` telemetry event, content-addressed
by its own inputs (Article 5's discipline, applied to context rather than raw provider capture).
No other event may re-carry these facts as its own per-event fields merely for convenience; every
event's Operational Context is instead determined positionally, from the nearest preceding
`ProvenanceContextEstablished` event sharing its `document_ref`, using the ordering guarantee
`TelemetrySink` already provides. Replay (Article 17) never creates a new Operational Context; only
a genuinely new processing run does, and only when its content-address differs from every prior
run's for that document.**
This closes a gap this standard's own §2 exposed on first implementation attempt: `ProvenanceContext`
was named in prose without ever being specified as a real object — an omission a less disciplined
implementation could have silently resolved by inventing a schema, exactly what Article 23 forbids.
Making it content-addressed, singular per distinct run, and positionally (not redundantly)
referenced keeps this addition consistent with Article 5 (content addressing), Article 7 (no
duplicated representation), and Article 16 (telemetry describes knowledge, kept separate from
process-execution noise) rather than bolting a new field onto every existing event type.
*Sources: Architecture Review Report, Phase 2 implementation attempt, 2026-07-14 — the first real
implementation pass against this standard, itself evidence the standard needed exactly this kind of
stress-test; parallels Articles 5, 7, 15, 16, 17.*
**Protects:** Replayability, Observability, Maintainability (schema minimality), Trustworthiness.

### Article 33. Deterministic Projections Are Not Telemetry Events
**A computation that reads existing telemetry and deterministically derives a view over it —
asserting no new knowledge, recording no human action, changing no semantic claim — must never
itself emit telemetry, no matter how many times it is invoked or how many callers invoke it.
Telemetry records state *transitions*; it does not record the recomputation of state that has
already transitioned. Replay reconstructs history; it does not create history. A UI action that
merely triggers such a recomputation (opening a document, refreshing a queue, re-running a query)
is not a knowledge-evolution event and must not be treated as one.**
Exposed directly during a Phase 6 implementation attempt: `review/triage.py`'s
`triage_review_queue` is a pure, idempotent projection over replayed `JournalState` — invoked fresh
every time a reviewer opens a document, by explicit design (`triage.py`'s own docstring:
"deterministic and reproducible from stored telemetry... the same journal state always yields the
same queue"). An earlier draft of this standard's Article 30 and roadmap Phase 6 assumed the
opposite — that "a review packet was generated" was itself a discrete event to record
(`ReviewPacketGenerated`) — which would have flooded the telemetry stream with one duplicate event
per UI page-load, recording nothing new each time, the first structural instance of telemetry noise
this standard would have introduced rather than eliminated. This article is the general rule that
correction generalizes to: a "was this the first time" question about a *pure projection's* output
is answered by re-running the projection deterministically at query time, never by persisting a
marker for having run it once already.
*Sources: Architecture Review Report, Phase 6 implementation attempt, 2026-07-14; parallels
Articles 16 (telemetry describes knowledge, not execution — this is that principle's necessary
converse: a query over knowledge is not itself new knowledge), 17, 24.*
**Protects:** Observability (signal-to-noise: persisted events remain a trustworthy, non-inflated
record), Replayability, Trustworthiness, Maintainability (no unbounded event growth from repeated
reads).

### Article 34. Research Telemetry Is a Third, Corpus-Scoped Ontology, Never Folded Into Document-Scoped Telemetry
**Audit, Benchmark, Replay, Calibration, and any future experimental or investigatory activity
describe ArchiveTrust's own self-knowledge — what the system has learned about its own reasoning,
across an arbitrary corpus — never one Archive Object's knowledge evolution. These events form
Research Telemetry: a third stream, independent of both document-scoped Trust Engine telemetry
(`domain/telemetry/events.py`, Article 16) and Workspace-scoped Acquisition telemetry
(`acquisition/events.py`, ROADMAP.md §12.1). No Research Telemetry event may be added to the
per-document `TelemetryEventKind` set, and `TelemetryEvent.document_ref`'s meaning — scoping one
event to the one Archive Object it concerns — is never weakened or overloaded to mean "a corpus" to
accommodate it. Research Telemetry is scoped by `corpus_ref` (a workspace id, a named fixture
corpus, or a documented dataset identifier — never fabricated), persisted only to repository-local,
version-controlled storage (e.g. `benchmarks/`), and never written into a real deployment's
`archivetrust_data/workspaces/*` telemetry store, which may hold live state this standard has no
authority to mutate.**
Exposed directly during a Phase 9 implementation attempt: this standard's own §7 originally listed
`AuditConducted`/`BenchmarkExecuted`/`ReplayExecuted` as additions to the closed, document-scoped
event set — an oversight, not a considered departure from precedent, since ROADMAP.md §12.1 had
already solved the identical problem for Acquisition telemetry by keeping it a separate stream
rather than diluting §12's closure discipline. The six 2026-07-14 audits and two benchmarks that
motivate Article 29 each examined a whole corpus (hundreds of documents), not one Archive Object;
forcing them onto a single `document_ref` would have been exactly the kind of fictional scoping
Article 18's silence-vs-failure discipline exists to prevent one layer down. A **deliberate,
investigatory** replay (an audit's own reconstruction exercise, e.g. Phase 8's demonstration that
`CandidateExcluded` is reconstructable from telemetry) emits `ReplayExecuted` under this stream;
the routine, constant replay calls production code makes on every document open (`ReviewService
.open_document`, triage) remain governed by Article 33 and must never emit anything — the two are
distinguished by intent (a deliberate research act vs. an ordinary query), never by mechanism.
*Sources: Architecture Review Report, Phase 9 implementation attempt, 2026-07-14; ROADMAP.md §12.1
(Acquisition telemetry's identical precedent, applied here a second time); parallels Articles 16,
18, 29, 33.*
**Protects:** Trustworthiness (self-knowledge held to the same evidentiary standard as document
knowledge), Observability, Maintainability (§12's closure discipline preserved, not diluted),
and operational safety (real deployment state is never written to by this standard's own tooling).

### Documented Architectural Candidates (§1.1 Category 3 Registry)

Every Future Architectural Candidate referenced anywhere in this document or the Constitution is
tracked here, so a candidate is never simply forgotten and never mistaken for something already
built (§1.1). This registry is amended whenever a candidate is proposed, implemented (moved to
category 1 and removed from here), or abandoned (moved to a rejected-candidates note, not deleted
silently, per Article 15's supersession-never-erasure spirit applied to design decisions).

| Candidate | Evidence for | Evidence against shipping unmodified | Status | Blocked on |
|---|---|---|---|---|
| Geometry-precision guard in `clustering.py`'s ordinal fallback (distinguish "no bounding box" from "coarse bounding box," never fall through to ordinal-position guessing when either side has no geometry at all) | RCDP counterfactual (`scripts/rcdp_ordinal_geometry_counterfactual.py`, 2026-07-14): the current ordinal fallback conflates the two, fusing PaddleOCR-VL's whole-page transcript into an arbitrary same-type Tesseract/Docling fragment in ~85% of one corpus's contested packets | Adversarial Architecture Audit (2026-07-14): shipping the guard exactly as the counterfactual scoped it would eliminate the entire Docling↔PaddleOCR-VL comparison capability permanently; 60% of a random sample of what it would eliminate carried unique, high-value content | **NOT IMPLEMENTED** — remains a counterfactual only, no file under `src/` modified | A narrower mechanism distinguishing "genuinely redundant merge" (~27–30% of the eliminated population) from "unique content that must stay reviewable" (~60–70%) — explicitly out of scope for both the RCDP finding and the Adversarial Audit, per each audit's own stated charter |

If and when this candidate is implemented, Article 27 already governs it without amendment (its
production-scoped mandate applies automatically to whichever exclusion mechanisms exist in `src/`
at the time); only this registry row needs to move to "implemented" and Phase 3's roadmap scope
would then need a follow-up phase to instrument it, per Article 23's amend-before-continuing rule.

---

## 3. Universal Transformation Contract, Applied to ArchiveTrust's Real Stages

Every arrow in ArchiveTrust's actual pipeline (`ROADMAP.md` §5.1's diagram) is a transformation.
Per the Universal Transformation Contract, each must answer all twelve questions below. The table
states, for each real stage, what already answers each question and what Articles 26–31 add.

### Stage: Provider Invocation → Evidence
*Code: `application/pipeline.py::run_pipeline`, `domain/evidence/models.py`*

| Contract node | Status | Where |
|---|---|---|
| Input | ✅ archive object region/page | `ProviderObservationAttempted.target_region` |
| Operational Context | ✅ duration, retries, cold start, failure category | `events.py:158-216` |
| Question Answered | ✅ "what does this provider observe here?" | implicit in event kind |
| Evidence Used | N/A — this stage produces Evidence, doesn't consume it | — |
| Evidence Ignored | ✅ `EvidenceRejected` for schema-invalid output | `events.py:219` |
| Transformation | ✅ raw output → content-addressed `Evidence` | `Evidence.create` |
| Knowledge Created | ✅ `EvidenceCreated` | `events.py:237` |
| Knowledge Lost | ✅ rejected raw output preserved, not discarded (Article 5) | `EvidenceRejected.raw_output` |
| Decision | ✅ accept/reject against provider contract schema | adapter validation |
| Semantic Contract | ⚠️ implicit per-provider; not versioned | **Article 28 gap** |
| Guarantees | ✅ content-addressed, never mutated (Article 5) | `Evidence._validate_id` |
| Known Limitations | ✅ documented per provider | `PROVIDER_ANALYSIS.md` |
| Output | ✅ `Evidence` | — |

### Stage: Evidence → Observation (Ontology Mapping)
*Code: provider adapters, `domain/ontology/base.py`*

| Contract node | Status |
|---|---|
| Question Answered | ✅ "which ontology concept does this Evidence represent?" |
| Evidence Used | ✅ `ObservationMapped.source_evidence_ids` |
| Transformation | ✅ recorded via `ontology_version` |
| Semantic Contract | ❌ **which mapping-table rule fired is not recorded — Article 28** |
| Decision | ⚠️ the mapping rule itself is inline adapter code, unversioned independently of `ontology_version` — **Article 28** |

This is the exact stage where the Semantic Contract Audit's finding lives: `Evidence` produced by
`tesseract_layoutparser` with native label `"Text"` is mapped to `ObservationType.PARAGRAPH`
unconditionally. `ObservationMapped` is emitted, `ontology_version` is correct, and yet the
telemetry stream gives no way to distinguish this from a mapping that correctly discriminates
`Text` into `PARAGRAPH` vs. `CAPTION` vs. `FOOTNOTE` by context. Article 28 closes this by
requiring a named, versioned mapping-table entry ID on every `ObservationMapped` event.

### Stage: Observation → Alignment (Comparison Grouping)
*Code: `domain/comparison/clustering.py`, `domain/alignment/service.py`*

| Contract node | Status |
|---|---|
| Question Answered | ✅ "do these Observations refer to the same semantic slot?" (Article 24) |
| Evidence Used | ✅ `AlignmentAttempted.candidate_observation_ids`/`selected_observation_ids` |
| Evidence Ignored | ⚠️ candidates excluded *before* reaching `AlignmentAttempted` by `_pairwise_affinity`'s ambiguous-multi-per-provider case (a real, shipped structural exclusion) are not distinguished from candidates that were considered and rejected — **Article 27 gap** |
| Decision | ⚠️ `alignment_rationale`/`clustering_basis` are free text — **Article 26 gap** |
| Known Limitations | ✅ documented in module docstring (bimodal split, cross-page continuity, containment reinforcement all explicitly deferred, `clustering.py:1-18`) but **not telemetry-visible per decision** |

This is the stage the Adversarial Audit examined — but what it examined was a *counterfactual*
alternate `AlignmentService` (§1.1 category 2, never merged into `clustering.py`), not this table's
subject. What this table describes is the real, currently-shipped gap at this stage: the
ambiguous-multi-per-provider case removes candidates from ever reaching `AlignmentAttempted`,
structurally, for that pool, and nothing records that the exclusion happened at all. Articles 26
and 27 close this: the case must emit an event with a `basis_code` (e.g.
`AMBIGUOUS_MULTI_PER_PROVIDER`) and must be queryable in aggregate as "how many candidate pairs,
across the whole corpus, were excluded by this specific mechanism." The separate, unshipped
geometry-precision guard the Adversarial Audit's counterfactual evaluated remains a Future
Architectural Candidate (registry, Part X) — Article 27 will apply to it automatically if and when
it is ever actually merged, but it is not part of this stage's *current* gap.

### Stage: Comparison → Canonical Decision
*Code: `domain/comparison/engine.py`, `domain/canonical/observation.py`*

| Contract node | Status |
|---|---|
| Question Answered | ✅ `ObservationCompared`, `AgreementCalculated` |
| Knowledge Created | ✅ `CanonicalDecisionCreated` carries the full `CanonicalObservation` |
| Knowledge Lost | ✅ `KnowledgeDiscarded`/`KnowledgeMerged` exist in schema |
| Decision | ⚠️ `reconciliation_basis` free text — **Article 26** |
| Guarantees | ✅ non-empty provenance enforced (Article 4), single legal source (Article 12) |

### Stage: Canonical Decision → Confidence
*Code: `domain/confidence/engine.py`*

Fully satisfied at Levels 1–3: `ConfidenceChanged` requires `level`, `previous_value`, `new_value`,
`reason`, and `confidence_policy_version` (Article 13). No Level 4/5 gap identified here — the
open gap is downstream, at whether the *inputs* to this stage (review outcomes, Article 30) exist
in sufficient volume to validate the policy, which is a Level 5 concern about the stage's
consumers, not the stage itself.

### Stage: Canonical Document → Review Packet → Human Review
*Code: `review/packet.py`, `review/triage.py`, `review/service.py`*

| Contract node | Status |
|---|---|
| Question Answered | ⚠️ triage classification (`CONTESTED`/`UNCORROBORATED_SINGLE_SOURCE`) is computed deterministically (`triage_review_queue`, Article 33 — a projection, never persisted), but *why a slot was not classified as needing review* is currently indistinguishable between "genuinely fine" and "excluded only by policy" — e.g. `PARAGRAPH` is absent from `DEFAULT_SINGLE_SOURCE_REVIEW_TYPES` (`review/triage.py:32`), so an unclustered PaddleOCR-VL paragraph is silently unreviewable, a fact the Adversarial Audit found. Closed not by a new persisted event (Article 33 forbids that) but by making `_reason_for`'s two `None`-producing paths explicitly distinct, so a query-time re-projection can tell them apart |
| Knowledge Created | ⚠️ `HumanCorrectionSubmitted`/`Applied` exist, but `submit_decision`'s `SKIP` path records nothing at all — **Article 30 gap**, closed by `ReviewOutcomeRecorded` covering every terminal action including `SKIP` |
| Knowledge Lost | ❌ Accept never narrows `contributing_observations` provenance (Human Reviewability Audit finding) — split into its own, separately-scoped roadmap item ("Review Provenance Narrowing," Phase 6 amendment 2026-07-14) since it requires a review-contract API change, not a telemetry addition |
| Decision | ✅ triage policy thresholds (`TriagePolicy`) are already fully replayable (versioned, applied deterministically); "withheld by policy" vs. "not eligible" is now an explicit classification distinction the same deterministic projection makes, not a new persisted fact |

This stage is where Article 30 is most load-bearing: the review queue is the one place a human,
not code, is the source of ground truth, and it is also the stage with the worst-documented
outcome rate (0 of 13,269) — closed at the *decision* layer (`ReviewOutcomeRecorded`), not the
*projection* layer (packet generation/withholding remain query-time facts, Article 33).

### Stage: Human Correction → Dataset Candidate
*Code: `domain/feedback/engine.py`*

Fully satisfied at Levels 1–3. `DatasetCandidateCreated.archive_object_ref` closes the
correction → canonical → observations → evidence → archive object chain (Operational Hardening,
Priority 2). No Level 4/5 gap beyond Article 30's upstream dependency (a dataset candidate is only
as good as the review-outcome volume feeding it).

### Stage: Audit / Benchmark / Replay → Architectural Finding *(new — Article 29)*

| Contract node | Status before this document | Status required |
|---|---|---|
| Input | a markdown filename and a script path, informally | telemetry-referenced corpus + commit + prior-audit chain |
| Question Answered | stated in the doc's prose | `AuditConducted.claim` |
| Evidence Used/Ignored | described in prose (e.g. "30 packets, seed=42") | `AuditConducted.sample_basis`, `AuditConducted.population_size` |
| Knowledge Created | a markdown file | `AuditConducted` event + the markdown file (the file remains the human-readable artifact; the event makes it queryable) |
| Supersession | informal — "the adversarial audit falsifies the RCDP finding" is stated in prose, not linked | `AuditConducted.supersedes_audit_id` |
| Output | a conclusion a human must read to discover | a structured `finding` + `verdict` any future process can query |

---

## 4. Object Model

| Domain object | L1 Operational | L2 Data Lineage | L3 Decision | L4 Semantic | L5 Epistemic | Gap owner |
|---|---|---|---|---|---|---|
| `Evidence` | ✅ | ✅ (content-addressed) | N/A | ⚠️ (provider contract implicit) | N/A | Art. 28 |
| `Observation` | ✅ | ✅ | ✅ (`ObservationMapped`) | ❌ (mapping rule unversioned) | N/A | Art. 28 |
| `Cluster`/`AffinityEdge` | ✅ | ✅ | ⚠️ (free-text basis) | N/A | N/A | Art. 26, 27 |
| `AlignmentAttempt` | ✅ | ✅ | ✅ | N/A | N/A | — (Art. 24 already closes this) |
| `CanonicalObservation` | ✅ | ✅ (contributing refs, supersession) | ⚠️ (free-text basis) | ⚠️ (payload contract implicit) | N/A | Art. 26, 28 |
| `ComparisonConfidence`/`CanonicalConfidence` | ✅ | ✅ | ✅ | N/A | ⚠️ (calibration blocked, Art. 30) | Art. 30 |
| `CanonicalDocument` | ✅ | ✅ | N/A | N/A | N/A | — |
| `ReviewPacket`/`ReviewAction` | ⚠️ | ⚠️ (provenance not narrowed on Accept) | ⚠️ (non-generation unexplained) | N/A | ❌ (0/13,269 outcomes) | Art. 30 |
| `DatasetCandidate` | ✅ | ✅ | N/A | N/A | ⚠️ (depends on Art. 30) | Art. 30 |
| `Audit` *(new, Research Telemetry, §9)* | ❌ | ❌ | ❌ | ❌ | ❌ | Art. 29, 34 |
| `Benchmark` *(new, Research Telemetry, §9)* | ❌ | ❌ | ❌ | ❌ | ❌ | Art. 29, 34 |
| `Replay` *(new, Research Telemetry, §9 — investigatory only, §9.5)* | ⚠️ (journal exists, `application/journal.py`) | ✅ | ✅ | N/A | ❌ (not self-referential) | Art. 29, 34 |
| `TelemetryEvent`/`TelemetrySink` | ✅ | ✅ (append-only, ordered) | N/A | N/A | ❌ (not itself audited) | §5 Recursive |
| `ProvenanceContextEstablished` *(new)* | ✅ (this is exactly what it establishes — the root L1 fact) | ✅ (content-addressed, Art. 32) | N/A (not a decision, a context) | N/A | N/A | Art. 32 |

---

## 5. Recursive Telemetry

Per the prompt's requirement that "telemetry objects themselves own telemetry," and Article 29:

- **Evidence owns telemetry** — already true (`EvidenceCreated`/`EvidenceRejected`).
- **Observation owns telemetry** — already true.
- **Cluster/Alignment owns telemetry** — already true (Article 24).
- **Canonical/Review/Human Review own telemetry** — already true.
- **Audit, Benchmark, Replay own telemetry** — **new** (Article 29, Article 34): `AuditConducted`,
  `BenchmarkExecuted`, `ReplayExecuted` — a separate, `corpus_ref`-scoped Research Telemetry stream
  (§9), never additions to the document-scoped `TelemetryEventKind` set.
- **Telemetry itself owns telemetry** — **new, and this is where recursion terminates.** The
  `TelemetrySink` is append-only and ordered (Article 16); its own operational health (write
  failures, event-count drift between a live run and its persisted sink, schema-version
  distribution across a corpus) is exactly a Level 1/2 fact about the telemetry system itself, and
  is captured by a `TelemetryIntegrityChecked` event (§9, Research Telemetry) emitted by the
  replay/audit tooling, never by the sink appending an event about its own append (which would
  recurse infinitely and violates nothing in Article 16 only because it is explicitly excluded
  here as the recursion's defined base case).

**Recursion stops at:** the raw archive object (Article 1, immutable by definition, the one thing
in this architecture with no upstream producer to explain it) and the physical machine/process
executing the code (a Level 1 fact recorded, never itself further decomposed — recording *why* a
CPU executed an instruction is outside any domain model's scope).

---

## 6. Flow of Knowledge, End to End

```
Raw Evidence (immutable Archive Object, Article 1)
   │  emits: WorkspaceCreated, ArchiveObjectRegistered (§12.1)
   ▼
Provider Invocation
   │  emits: ProviderObservationAttempted, EvidenceRejected, EvidenceCreated
   │  new:   (Article 26 basis_code on any rejection)
   ▼
Evidence → Observation (ontology mapping)
   │  emits: ObservationCreated, ObservationMapped
   │  new:   mapping_rule_id, mapping_table_version (Article 28)
   ▼
Observation → Alignment (comparison grouping)
   │  emits: AlignmentAttempted, ObservationAligned, ObservationLeftUnaligned
   │  new:   basis_code (Article 26); CandidateExcluded for pre-alignment guards (Article 27)
   ▼
Alignment → Comparison (clustering + reconciliation)
   │  emits: ObservationCompared, AgreementCalculated, ObservationAccepted/Rejected/Merged
   │  new:   basis_code (Article 26)
   ▼
Comparison → Canonical Decision
   │  emits: CanonicalDecisionCreated, KnowledgeMerged, KnowledgeDiscarded
   │  new:   basis_code (Article 26)
   ▼
Canonical Decision → Confidence
   │  emits: ConfidenceChanged (three levels, Article 13)
   ▼
Canonical Observations → Canonical Document
   │  emits: CanonicalDocumentCreated
   ▼
Canonical Document → Review Packet (triage)
   │  emits: nothing (Article 33) — triage/packet assembly is a deterministic projection over
   │         already-persisted telemetry; "why wasn't this reviewed" (§3 Stage 6, Adversarial
   │         Audit finding re: PARAGRAPH) is answered by re-running the same projection at query
   │         time with a policy-vs-not-eligible distinction, never by a persisted event
   ▼
Review Packet → Human Review
   │  emits: HumanCorrectionSubmitted, HumanCorrectionApplied
   │  new:   ReviewOutcomeRecorded — mandatory terminal event per *decision* (Article 30),
   │         covering ACCEPT/REJECT/EDIT/MARK_AMBIGUOUS/REQUEST_FURTHER_REVIEW/SKIP alike
   │  deferred: provenance narrowing on Accept (Human Reviewability Audit finding) — split into
   │         its own future roadmap item ("Review Provenance Narrowing"), not this phase
   ▼
Human Correction → Dataset Candidate
   │  emits: DatasetCandidateCreated
   ▼
Any of the above → Audit / Benchmark / Replay (recursive, Article 29, Article 34)
   │  A SEPARATE stream (Research Telemetry, corpus_ref-scoped, never document_ref) records:
   │  AuditConducted, BenchmarkExecuted, ReplayExecuted, TelemetryIntegrityChecked
   │  each may supersede a prior Audit/Benchmark/Replay (Article 15's discipline, one layer up)
```

Every stage above already satisfies Levels 1–3 (cited in §3); the "new" annotations are exactly
and only the Level 4/5 additions Articles 26–31 require. Nothing above is a reinvention of the
existing pipeline — it is the existing pipeline with its blind spots named.

---

## 7. New Telemetry Events (Additions to ROADMAP.md §12's Canonical Set)

Extending, never replacing, `TelemetryEventKind`. Each is scoped to close exactly one cited gap:

| Event | Closes | Carries |
|---|---|---|
| `CandidateExcluded` | Article 27 | `candidate_observation_id`, `excluding_mechanism`, `basis_code`, `structural` (bool — is this exclusion a permanent property of a provider pair/mechanism, or incidental to this document) |
| `SemanticContractMapped` | Article 28 | `provider_id`, `native_label`, `observation_type`, `mapping_table_entry_id`, `mapping_table_version` |
| ~~`ReviewPacketGenerated` / `ReviewPacketWithheld`~~ | **Not events (Article 33, corrected 2026-07-14).** Packet generation/withholding is a deterministic projection over already-persisted telemetry (`review/triage.py`'s `triage_review_queue`, `_reason_for`), re-derivable at query time from the persisted `TriagePolicy` version and canonical observation state — recomputed however many times a reviewer opens a document, never emitting a new event per computation. See §3 Stage 6. |
| `ReviewOutcomeRecorded` | Article 30 | `outcome_id`, `semantic_slot_id`, `canonical_observation_id`, `action` (`ACCEPT_PROVIDER`/`MANUAL_EDIT`/`REJECT`/`MARK_AMBIGUOUS`/`REQUEST_FURTHER_REVIEW`/`SKIP`), `outcome` (`resolved`/`deferred`), `correction_id` (`None` for `SKIP`) — terminal by construction, emitted for every `submit_decision` call including `SKIP` |
| `ProvenanceNarrowed` | **Deferred, own roadmap item ("Review Provenance Narrowing"), not this phase** — Human Reviewability Audit finding; requires a review-contract API change (which candidate a reviewer accepted), not a telemetry addition alone |
| ~~`AuditConducted` / `BenchmarkExecuted` / `ReplayExecuted` / `TelemetryIntegrityChecked`~~ | **Not additions to this set (Article 34, corrected 2026-07-14).** These describe self-knowledge about ArchiveTrust's own reasoning across a corpus, never one Archive Object — they belong to Research Telemetry, a separate, `corpus_ref`-scoped stream (§9 below), never the per-document `TelemetryEventKind` set this table extends. |
| `ProvenanceContextEstablished` | Article 32 | see §8 below — the one event type this table only summarizes, since its schema required its own normative section |

`basis_code` vocabularies (Article 26) are versioned per-algorithm (e.g.
`ClusteringBasisCode` for `clustering.py`, `ReconciliationBasisCode` for `engine.py`), living
alongside the algorithms they describe — the same co-location discipline `ReconciliationPolicy`
and `CapabilityMatrix` already follow.

---

## 8. Operational Context Events

Normative specification for `ProvenanceContextEstablished` (Article 32). Unlike every other event
in §7, this one required its own section rather than a table row: an earlier implementation attempt
against Phase 2 stopped precisely because this object was named in prose, in an earlier draft of
this document, without a schema — the correct outcome per Article 23, and the reason this section
exists at the rigor it does.

### 8.1 Purpose

To make "what commit, configuration, and policy versions produced this document's telemetry"
answerable from the telemetry stream itself, without inference, backfilling, or access to
out-of-band operational records (deploy logs, CI history) that are not part of the domain model.
This is Level 1 (Operational Telemetry) closed properly: not by adding process facts to every
knowledge event (which Article 16 forbids), but by giving the run itself one recorded root fact.

### 8.2 Semantics: What It Is, and Is Not

A `ProvenanceContextEstablished` event is a **context-establishment event, not a knowledge-
evolution event.** It does not record that anything was observed, compared, decided, or reviewed —
it records the fixed environment *within which* every subsequent knowledge-evolution event for a
document occurred. It is, structurally, the root of that document's reasoning graph: every other
event for the same processing run is a descendant of it, and none of them repeat the facts it
establishes. It participates in `TelemetryEventKind` and `TelemetrySink` exactly as any other event
does (per the accepted Option 1) — there is no second persistence mechanism, no parallel telemetry
system, and no special-cased bypass. Its distinctness is semantic (what kind of fact it records),
never mechanical (how it is stored, replayed, or queried).

### 8.3 Schema

Every field below is justified individually; fields considered and rejected are listed with the
evidence for rejecting them, per Article 8's evidentiary discipline applied to schema design.

**Fields inherited from `TelemetryEvent` (no redefinition needed):**

| Field | Why it's sufficient as inherited |
|---|---|
| `document_ref` | Already required on every event; scopes the context exactly as it scopes everything else. |
| `event_id`, `schema_version`, `recorded_at` | Already universal; `recorded_at` makes a separate `created_at` redundant — rejected below. |
| `reconciliation_policy_version`, `capability_matrix_version`, `confidence_policy_version`, `feedback_policy_version`, `alignment_algorithm_version` | Already defined on the base class, currently populated only by the specific event kind each governs. `ProvenanceContextEstablished` is the **first event required to populate all five unconditionally**, establishing the full policy-version baseline for the run even for a document that happens to trigger zero `ConfidenceChanged`/`AlignmentAttempted` events — closing a real gap where, today, a document with no confidence changes leaves no record of which confidence policy was even loaded. |

**New fields, specific to this event:**

| Field | Justification |
|---|---|
| `ontology_version` | Establishes the expected ontology version for the run as an upfront anchor, so every later `Observation.ontology_version` can be checked for drift against it (Article 19) rather than only discoverable by scanning every Observation individually. |
| `git_commit` | The precise source-code identity that produced every `basis_code` (Article 26) and mapping-table entry (Article 28) for this run — the single fact Article 17's replay guarantee is least able to do without. |
| `configuration_hash` | A hash over the full effective configuration not already covered by the five named policy versions above — critically, this includes the enabled-adapter roster and invocation plan (which providers, `PAGE_IMAGE` vs. `DOCUMENT`, in what order), so a separate `pipeline_definition_hash` is unnecessary (rejected below, folded in here to avoid Article 7's duplicated-representation defect). |
| `configuration_snapshot_reference` | A pointer (path or content-address) to the actual configuration values the hash was computed over. Required because a hash alone can detect drift but cannot reconstruct what changed — Article 17 requires reconstruction, not mere detection. |
| `machine_identifier` | Narrowly justified, not generic ops boilerplate: the `dev-machine-gpu-vram-exhausted` incident (memory, 2026-07-14) is a real, evidenced case where the *same code, same configuration* produced *different provider behavior* (GPU/VRAM-constrained probabilistic-provider degradation) depending on which physical machine ran it. This is a knowledge-relevant fact under Article 16's narrow exception (the same exception that already permits `HumanCorrectionSubmitted.reviewer_ref`), not process logging for its own sake. |
| `workspace_identifier` | Ties this Trust-Engine-side event back to the Acquisition-side Workspace stream (ROADMAP.md §12.1) as an opaque foreign key — the same reference-without-dependency pattern `DatasetCandidateCreated.archive_object_ref` already uses, so this does not violate Article 25 (Trust Engine remains unaware of acquisition mechanism; referencing a Workspace's id is not depending on any acquisition concept). |

**Fields considered and rejected** (per the requirement that an unjustifiable field be removed, not merely deprioritized):

| Rejected field | Why |
|---|---|
| `telemetry_version` | Redundant with `schema_version`, already inherited. |
| `application_version` | No evidence this project has a version distinct from its git commit — no semver/CHANGELOG mechanism exists in the codebase; `git_commit` alone is the precise identity. |
| `build_identifier`, `build_timestamp` | No evidence of a build/CI/packaging pipeline producing artifacts distinct from a source checkout at a given commit (no CI configuration found in the repository). Revisit if/when one exists, per Article 8's evidentiary-deferral pattern (the same treatment Signature/Stamp received). |
| `feature_flag_hash`, `feature_flag_set` | No evidence this project has a feature-flag mechanism distinct from per-Workspace provider/adapter enablement, which `configuration_hash` already covers. |
| `provider_versions` (document-level roster) | Redundant with the per-invocation `provider_id`/`provider_version` already on every `ProviderObservationAttempted`; Article 18 already makes the *absence* of an invocation a recorded fact, so a redundant roster adds no reconstructive power. |
| `pipeline_definition_hash` | Folded into `configuration_hash`'s scope rather than kept as a second field for the same underlying fact (Article 7). |
| `execution_environment` | A vague catch-all superseded by the specific, evidence-backed fields kept or the specific fields rejected next. |
| `operating_system`, `runtime_version` | No evidenced knowledge-relevant defect has ever been traced to either in this project's history, unlike `machine_identifier` (GPU/VRAM). Revisit only if such evidence arises — never speculatively. |
| `created_at` | Redundant with inherited `recorded_at`. |

### 8.4 Identity and Deduplication

**The identity of a `ProvenanceContextEstablished` event is content-addressed**, following Article
5's precedent for `Evidence` exactly: a `context_id` computed as a hash over
`(document_ref, git_commit, configuration_hash, ontology_version, reconciliation_policy_version,
capability_matrix_version, confidence_policy_version, feedback_policy_version,
alignment_algorithm_version, machine_identifier, workspace_identifier)` — every field that
constitutes *what the run's environment actually was*, excluding pure metadata (`event_id`,
`recorded_at`) that carries no identity.

**Deduplication rule, stated normatively:** if `WorkspaceProcessingService.process()` (or any future
processing entry point) is invoked for a `document_ref` whose computed `context_id` already has a
persisted `ProvenanceContextEstablished` event in the sink, no new event is appended — the existing
one is reused, exactly as `Evidence.create()` never duplicates identical raw output. If any input to
the hash differs from every prior context recorded for that `document_ref` (a new commit, a changed
policy, a different machine), a **new, distinct** `ProvenanceContextEstablished` event is appended
for that `document_ref` — never replacing the prior one. Multiple contexts coexisting for one
document is not an error: it is the accurate record of a document having been processed more than
once, under different environments, over its lifetime (Article 15's supersession-never-erasure
principle, applied here as "multiplicity, never erasure," since there is no single current context
to supersede — each run's context remains true of that run, permanently).

### 8.5 Lifecycle

`ProvenanceContextEstablished` is computed and appended at the single point
`WorkspaceProcessingService.process()` already stamps every other event
(`acquisition/processing.py`, immediately before the existing `for event in result.events:
self._telemetry_sink.append(stamp_recorded_at(event))` loop) — computed once, before that loop, and
prepended so it is the first event of the run in append order. It is never updated or corrected
after creation; if the environment was recorded wrong, a later, distinct context (per §8.4) is the
only correction path, never an edit to the original (Article 15's discipline, applied to context
the same way it already applies to Canonical Observations).

### 8.6 Lineage: How Later Events Reference Their Context

**The relationship is positional, not carried by an explicit field on every event.** For any event
`E` with a given `document_ref`, its Operational Context is the `ProvenanceContextEstablished` event
sharing that `document_ref` which most closely precedes `E` in append order — using the ordering
guarantee `TelemetrySink` already documents and already relies on for replay
(`sink.py`: "Must preserve insertion order for events sharing a `document_ref` — `events_for_document`
relies on it for replay ordering"). This is fully deterministic (a total order exists per
`document_ref`) and fully replayable (it uses no information beyond what replay already reads).

This was a deliberate choice against the alternative of adding an explicit `context_ref` field to
the base `TelemetryEvent` class: doing so would touch every one of the ~20 existing event types for
information already recoverable from a guarantee the sink already provides, which is exactly the
duplicated-representation defect Article 7 forbids. For historical events recorded before this
standard existed (no preceding `ProvenanceContextEstablished` exists for their `document_ref`), the
context is `None` — never inferred or backfilled, matching the existing discipline for every other
field added after the fact (`duration_ms`'s own precedent).

### 8.7 Relationship to Replay

**Replay (Article 17, `application/journal.py`) never creates a new `ProvenanceContextEstablished`
event, under any condition.** Replay is defined as a pure reconstruction over already-persisted
telemetry; it does not invoke providers, does not call `WorkspaceProcessingService.process()`, and
therefore has no occasion to compute a new context. It reads whichever `ProvenanceContextEstablished`
event(s) already exist for a `document_ref` (per §8.6's positional rule) and reports them as part of
the reconstructed reasoning trail.

A **genuinely new processing run** — a document reprocessed after a bug fix, under a new commit —
is not replay in Article 17's sense, and is the only case that can produce a new
`ProvenanceContextEstablished` event, governed entirely by §8.4's content-addressed deduplication
rule (a new context is appended if and only if the new run's inputs differ from every prior run's
for that `document_ref`). These two cases — replay (read-only, never creates) and reprocessing
(a real run, creates conditionally) — must never be confused with each other in implementation.

### 8.8 Relationship to Provenance and the Article 4/17 Guarantees

`ProvenanceContextEstablished` does not itself carry evidentiary weight the way `Evidence` or a
`CanonicalObservation` does — it is not part of the Article 4 chain a canonical fact must trace to.
It is the *environment* that chain was produced in, not a link in it. Its guarantee is narrower and
different in kind: not "this fact is true because of this evidence," but "this reasoning trail was
produced under this exact, reconstructible environment" — the fact that makes Article 17's replay
guarantee actually verifiable end-to-end, rather than merely reproducible in the domain-object sense.

### 8.9 Relationship to Document Processing

Fires exactly once per distinct processing run, at the single existing seam
(`WorkspaceProcessingService.process()`) already responsible for stamping and appending every other
event for that run — no new seam, no new service, no new call site beyond the one line computing
`context_id` and conditionally appending before the existing per-event loop.

### 8.10 Context-Specific Invariants

1. Every `ProvenanceContextEstablished` event's `context_id` is a deterministic content-address of
   its own identity fields (§8.4) — never a random or sequential id.
2. No two `ProvenanceContextEstablished` events for the same `document_ref` may share an identical
   `context_id` (the sink must treat re-append of an identical context as a no-op, not a duplicate).
3. No event of any other kind may carry a field duplicating information `ProvenanceContextEstablished`
   already establishes for its run (Article 7).
4. Replay never appends a `ProvenanceContextEstablished` event (§8.7).
5. A `document_ref` with zero `ProvenanceContextEstablished` events predates this standard; every
   later event for it resolves its context to `None`, never inferred (§8.6).

---

## 9. Research Telemetry

Normative specification for `AuditConducted`/`BenchmarkExecuted`/`ReplayExecuted`/
`TelemetryIntegrityChecked` (Article 34). Added 2026-07-14 after a Phase 9 implementation attempt
found this document's own §7 had, by oversight, listed these as additions to the closed,
document-scoped `TelemetryEventKind` set — the exact mistake Article 34 exists to correct.

### 9.1 Purpose

To make ArchiveTrust's self-knowledge — every audit, benchmark, and deliberate investigatory
replay that examines the system's own reasoning — a first-class, queryable, superseding record,
per Article 29, without weakening what `document_ref` means anywhere else in this standard.

### 9.2 Why a Third Stream, Not the Document-Scoped Set

`TelemetryEvent.document_ref` scopes one event to one Archive Object. An audit like the RCDP
finding or the Adversarial Audit examines a whole corpus — hundreds of documents — not one. Forcing
a corpus-wide finding onto a single `document_ref` would be exactly the kind of fictional scoping
Article 18 forbids one layer down (silence must be distinguishable from failure; a corpus-wide fact
mislabeled as one document's fact is a parallel category error). ROADMAP.md §12.1 already
established the precedent for exactly this situation — Acquisition telemetry is Workspace-scoped
and deliberately kept separate from the document-scoped Trust Engine stream, "preserv[ing] §12's
own closure discipline rather than diluting it." Research Telemetry is the same move, applied to
self-knowledge: a third stream, corpus-scoped, never folded into either of the other two.

### 9.3 Schema

```
class ResearchEventKind(str, Enum):
    AUDIT_CONDUCTED = "AuditConducted"
    BENCHMARK_EXECUTED = "BenchmarkExecuted"
    REPLAY_EXECUTED = "ReplayExecuted"
    TELEMETRY_INTEGRITY_CHECKED = "TelemetryIntegrityChecked"
    CALIBRATION_MEASURED = "CalibrationMeasured"  # reserved, see 9.3.1
```

`ResearchEvent` (abstract base, mirrors `TelemetryEvent`'s shape but is never a subclass of it —
the two hierarchies are deliberately unrelated, so a Research Telemetry event can never be
type-confused with a document-scoped one):

| Field | Type | Notes |
|---|---|---|
| `kind` | `ResearchEventKind` | |
| `event_id` | `str` | |
| `corpus_ref` | `str` | What was examined — a Workspace id, a named fixture corpus, or a documented dataset identifier. Never a single `document_ref`; never fabricated when the true scope is unknown (Article 18). |
| `commit_ref` | `str \| None` | The git commit the examined code was at, if known — lets a later query detect a finding's commit predates a change to the file it examined (§11). |
| `recorded_at` | `str \| None` | UTC ISO-8601, same convention as `TelemetryEvent.recorded_at`. |
| `schema_version` | `int` | Independent of the document-scoped stream's `schema_version` counter (Article 19's discipline, applied a third time). |

`AuditConducted` adds: `audit_id`, `title`, `claim`, `verdict`, `population_size: int | None`,
`sample_basis: str | None`, `supersedes_audit_id: str | None`.

`BenchmarkExecuted` adds: `benchmark_id`, `title`, `document_count: int | None`,
`metrics_ref: str | None`, `supersedes_benchmark_id: str | None`.

`ReplayExecuted` adds: `replay_id`, `events_replayed: int | None`, `divergence_found: bool`,
`divergence_detail: str | None`. **Only for a deliberate, investigatory replay** — see 9.5.

`TelemetryIntegrityChecked` adds: `sink_ref`, `event_count: int`,
`schema_version_distribution: dict[str, int]`, `gaps_found: tuple[str, ...]`.

#### 9.3.1 `CalibrationMeasured` — Reserved, Not Yet Wired

Named in this vocabulary because calibration measurements (`domain/calibration/`) are the same
shape of self-knowledge fact as an audit or benchmark, and belong in the identical stream when
wired. Verified 2026-07-14: `domain/calibration/` emits no telemetry of any kind today. Adding the
member now (schema-ready, per the same discipline Article 28 already established for
`ObservationMapped`'s then-unwired fields) costs nothing and avoids a second vocabulary amendment
later; wiring it is explicitly out of Phase 9's scope.

### 9.4 Identity and Supersession

Every `AuditConducted`/`BenchmarkExecuted` carries its own `audit_id`/`benchmark_id` (a random,
non-content-derived identifier, per `domain/shared/ids.py`'s own distinction between content-
addressed and randomly-assigned identities — a finding is not defined by its content alone the way
Evidence is). `supersedes_audit_id`/`supersedes_benchmark_id` chains findings exactly as
`CanonicalObservation.supersedes` chains canonical versions (Article 15): a superseded finding is
never deleted or edited, only ever pointed to by whatever superseded it.

### 9.5 Replay: Routine vs. Investigatory

`ReplayExecuted` is emitted **only** for a deliberate, investigatory replay — an audit's own
reconstruction exercise (e.g. Phase 8's demonstration that `CandidateExcluded` is reconstructable
from stored telemetry alone). It is **never** emitted for the routine, constant replay production
code already performs on every document open (`ReviewService.open_document`, `triage_review_queue`)
— those remain governed by Article 33 and must never emit anything, no matter how many times they
run. The two are distinguished by *intent* (a deliberate research act vs. an ordinary query), never
by mechanism — the underlying `Journal.replay()` call is identical code either way; whether a
`ReplayExecuted` event is emitted is a decision made by the *caller* (audit tooling vs. production
`ReviewService`), never inferred from the call itself.

### 9.6 Persistence

Research Telemetry is persisted only to repository-local, version-controlled storage (e.g.
`benchmarks/research_telemetry.jsonl`) — **never** to a real deployment's
`archivetrust_data/workspaces/*/telemetry` store, which may hold live state this standard has no
authority to mutate (Article 34). `ResearchTelemetrySink` (mirroring `TelemetrySink`'s Protocol
shape: `append`, `all_events`, plus `events_for_corpus(corpus_ref)`) has an in-memory implementation
for tests and a file-backed one for the checked-in registry, exactly paralleling
`InMemoryTelemetrySink`/`FileTelemetrySink`'s existing split.

### 9.7 Recursion, Closed

`TelemetryIntegrityChecked` is Research Telemetry's own self-check — event-count and schema-version
distribution against a named corpus — and is itself where §5's recursion terminates (alongside the
immutable Archive Object, upstream). No event type may reference "the event that recorded this
event's own append"; a `TelemetryIntegrityChecked` event checks the *document-scoped* stream's
integrity, and is not itself re-checked by a further event (Article 16's boundary against infinite
regress, restated here for the third stream).

---

## 10. Visualization Model

An engineer clicks any object and the UI reconstructs its full chain of custody, one hop at a
time, at every hop displaying all five levels:

```
Review Packet
  │ L1: generated_at, triage engine version
  │ L2: source Canonical Observation id
  │ L3: triage_classification, why this packet (not withheld)
  │ L4: which ObservationType's review policy applied
  │ L5: does an AuditConducted event reference this packet's semantic_slot_id?
  ▼
Canonical Observation
  │ L1: reconciliation_sequence, when created
  │ L2: contributing_observations (with any ProvenanceNarrowed applied)
  │ L3: reconciliation_basis + basis_code
  │ L4: payload's semantic contract (ObservationType's documented meaning)
  │ L5: has any Audit/Benchmark examined this slot or its type in aggregate?
  ▼
Comparison / Alignment
  │ L3: AlignmentAttempted's candidates considered vs. selected, basis_code
  │ L2: CandidateExcluded events for this comparison_group_id, if any
  ▼
Cluster
  │ L3: clustering_basis + basis_code; affinity_edges and their scores
  ▼
Observation
  │ L2: source_evidence_ids; L4: mapping_table_entry_id + version (Article 28)
  ▼
Evidence
  │ L1: provider, provider_version, duration, retry/cold-start facts
  │ L2: content address; any EvidenceRejected siblings from the same invocation
```

This is not a new UI framework — every id in this chain (`evidence_id`, `observation_id`,
`canonical_observation_id`, `semantic_slot_id`, `review_packet_id`, `correction_id`, and the new
`audit_id`/`benchmark_id`/`replay_id`) already exists or is a direct, structurally identical
extension of an existing id scheme (`domain/shared/ids.py`). The visualization layer is a query
surface over the existing append-only `TelemetrySink`, not a new store.

---

## 11. Adversarial Completeness Audit

Questions invented against this design, until no further question produced a missing capability:

| Question | Answered by |
|---|---|
| Why was this observation ignored? | `ObservationLeftUnaligned.reason`, or `CandidateExcluded.basis_code` if excluded pre-alignment (Art. 27) |
| Why wasn't this provider considered? | `ProviderObservationAttempted` absence is itself the answer (Article 18) — no invocation event exists for that provider/region |
| Why wasn't this clustered? | `AlignmentAttempted.alignment_rationale` + `basis_code`, or `CandidateExcluded` if excluded before clustering ran |
| Why did this canonical change? | `supersedes` chain + `reconciliation_basis`/`basis_code` on the new version |
| Why was this threshold chosen? | `ReconciliationPolicy`/`TriagePolicy` version cited on the event; the policy's own change history is git-tracked, cited by `reconciliation_policy_version` |
| Why wasn't this review generated? | Re-run `triage_review_queue`'s deterministic projection with the persisted `TriagePolicy` version against the canonical observation's recorded state (Article 33) — the policy-vs-not-eligible distinction is explicit in `_reason_for`'s own classification, not a separately persisted event |
| Why was this confidence assigned? | `ConfidenceChanged.reason` + `confidence_policy_version` |
| Why was this provenance retained? | `contributing_observations` is Article-4-enforced non-empty; `ProvenanceNarrowed` for post-review narrowing |
| Why was this evidence discarded? | `EvidenceRejected.rejection_reason`, or `KnowledgeDiscarded` |
| **Has this exact question already been answered by a prior audit?** | `AuditConducted.claim` is queryable; `supersedes_audit_id` chains prevent re-litigating a falsified finding without citing what falsified it |
| **Is a finding still valid after a later fix changed the code it was measured against?** | `AuditConducted.commit_ref` + `corpus_ref`; a query can detect "this audit's commit predates a change to the file it examined" and flag it stale — not automatically re-validate it, but make staleness a detectable, not silent, fact |
| **Did a benchmark's headline number (e.g. "332→0 packets") measure information preservation, or only queue size?** | Article 31 requires this be recorded on the accompanying `AuditConducted` event, not left implicit in the number itself |
| **Can a future engineer tell that two providers can structurally never be compared, without running the corpus and noticing zero packets?** | `CandidateExcluded.structural=true`, aggregated — this is a standing queryable fact, not something that requires re-discovery per document |
| **What happens if the telemetry sink itself drops or corrupts an event?** | `TelemetryIntegrityChecked` (§5) — event-count and schema-version-distribution checks against the corpus, run by replay tooling; this is the recursion's terminal self-check |
| **Who decided a semantic-contract mapping rule was correct, and can that decision be revisited?** | `SemanticContractMapped.mapping_table_entry_id` is a named, versioned, git-blamable table row (Article 28) — the same discipline `ReconciliationPolicy` already gives thresholds, extended to label mappings |
| **Does closing these gaps risk becoming process-level logging, violating Article 16?** | No: every new event above describes a *decision about knowledge* (what was excluded, what a mapping meant, what an audit concluded) — none describes *which function ran*. `TelemetryIntegrityChecked` is the one event closest to the line, and is deliberately scoped to sink-level facts (event counts, version distributions) rather than process facts (CPU, thread, PID), keeping it a fact about the *knowledge record's* integrity, not about execution. |

No further question, after these, produced a capability not already covered by an existing event,
Articles 26–31, or an explicit, cited exclusion (machine/process/thread facts, deliberately kept at
the document-level `ProvenanceContext` per §2 Level 1, not per-event, to avoid violating Article
16). **Stopping condition met.**

---

## 12. Invariants

1. No `AlignmentAttempted`, `ObservationCompared`, or `CanonicalDecisionCreated` event may carry a
   `*_basis`/`*_rationale` string without an accompanying `basis_code` from that algorithm's
   versioned vocabulary (Article 26).
2. No mechanism may remove a candidate Observation from consideration without emitting
   `CandidateExcluded` or an existing equivalent (`ObservationLeftUnaligned`) — silent exclusion is
   a defect, not an optimization (Article 27).
3. No provider adapter may map a native label to an `ObservationType` via inline conditional logic
   uncited by a named mapping-table entry (Article 28).
4. No terminal review decision (`SKIP` included) may complete without emitting exactly one
   `ReviewOutcomeRecorded` (Article 30). Aging (a currently-eligible slot with no recorded outcome
   past a configured horizon) is a derived, query-time fact, computed from the deterministic triage
   projection plus the recorded `ReviewOutcomeRecorded` population — never its own persisted event
   (Article 33).
5. No deterministic projection over already-persisted telemetry (triage classification, packet
   assembly, or any future query-time view) may itself emit telemetry, regardless of how many
   times it is invoked (Article 33).
6. No architectural finding (audit, benchmark, replay) may be treated as authoritative without a
   corresponding `AuditConducted`/`BenchmarkExecuted`/`ReplayExecuted` event citing its corpus and
   commit (Article 29).
7. No behavior change to comparison/clustering/alignment/confidence/triage may ship without an
   accompanying audit answering Article 31's three questions.
8. Recursion terminates at the immutable Archive Object (upstream) and at
   `TelemetryIntegrityChecked` (self-referential, downstream) — no event type may reference "the
   event that recorded this event's own append" (Article 16's boundary against infinite regress).

## 13. Guarantees

- **Replay guarantee (extended):** replay reconstructs not only Evidence, both Observation Graphs,
  every comparison decision, and confidence evolution (Article 17, unchanged), but also which
  semantic-contract mapping and which structured basis code governed each decision, and whether any
  prior audit already examined the same question.
- **Explainability guarantee:** any canonical field, any exclusion, and any withheld review can be
  explained by a bounded query against the telemetry stream — never by reading algorithm source.
- **Falsifiability guarantee (Article 14, operationalized):** every `AuditConducted` event
  explicitly states what would falsify its claim and cites the sample/population it checked against
  — the exact discipline the Adversarial Audit modeled and this document now requires by default.
- **Debugging guarantee:** the nine categories of question enumerated in the prompt that motivated
  this document, and the further eight discovered in §11's adversarial pass, are each answerable by
  a named event field, not a source-reading exercise.

## 14. Extension Guidelines / Future Provider Integration Rules

- A new provider adapter must ship with a mapping table satisfying Article 28 before its first
  `ObservationMapped` event — no inline-conditional mapping is accepted at review.
- A new comparison/clustering strategy must declare its `basis_code` vocabulary (Article 26) at the
  same time it is introduced, versioned from `1`, never retrofitted after the fact.
- A new exclusion mechanism (a future capability-matrix rule, a future threshold) must emit
  `CandidateExcluded` from its first commit, with `structural` correctly set — determined by asking
  "does this exclude a class of comparison forever, or this one document's data only."
- A new review-triage policy must be accompanied by an `AuditConducted` event measuring information
  preservation against the *previous* policy, per Article 31, before it becomes the default.

## 15. Stopping Condition, Restated

This document stops here because §11's adversarial pass, run to exhaustion against the six real
2026-07-14 investigations that motivated it, produced no further missing telemetry capability —
only the explicit, principled exclusion of process-level facts Article 16 already rules out. Any
future engineer who finds a seventh unanswerable question should amend this document the same way
Articles 26–31 were derived here: cite the specific investigation that found the gap, name the
event or field that closes it, and show which existing article it parallels. This document is not
closed to amendment — it is closed to amendment *without that discipline*.
