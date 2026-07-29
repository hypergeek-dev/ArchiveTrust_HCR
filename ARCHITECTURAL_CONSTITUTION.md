# The ArchiveTrust Architectural Constitution

**Status:** Ratified from existing architecture. Nothing in this document is new. Every article below is extracted from, and cited to, decisions already made in ROADMAP.md, `MILESTONE0_REVIEW.md`, `MILESTONE1_DOMAIN_MODEL.md`, `PROPOSAL_REVIEW_CANONICAL_CHUNK.md`, `ARCHITECTURE_VALIDATION_REPORT.md`, and `docs/investigation/`.

**What this is not:** not a roadmap, not a design document, not an implementation guide. It contains no milestones, no schemas, no module boundaries, no technology choices. Those live in ROADMAP.md and change often. This document is meant to change rarely, if ever.

**What this is:** the set of laws every future design decision, pull request, and milestone must obey. When a proposal is evaluated (as `PROPOSAL_REVIEW_CANONICAL_CHUNK.md` was), it is these articles it is being measured against, whether or not that document says so explicitly.

**Amendment:** this constitution may be amended only when a genuinely new architectural principle is *discovered* through the same rigor Milestone 0 and its subsequent reviews applied — never added because it "sounds good." An amendment must cite the document and finding that established it, exactly as every article below does. If no such citation exists, it is not ready to be an article.

**Contributor governance:** once contributors exist, constitutional amendments require a recorded governance note before merge. The note must name the proposed article or procedural change, cite the audit/incident/benchmark/roadmap finding that established it, describe affected modules and migration or rollback impact, and record reviewer approval. Contributor proposals that lack this evidence remain design discussion, not constitutional amendments. This preserves Article 23's stop-update-explain-continue rule while making the review path explicit for non-maintainer contributors.

---

## Part I — Foundational

### Article 1. The Archive Object Is Immutable
**The original archived document is never modified, annotated, or overwritten by any process. Everything ArchiveTrust produces is a derived, reproducible artifact of it.**
Without a fixed source, "trustworthiness" would be measured against a moving target. This is the precondition every other article depends on.
*Sources: ROADMAP.md §1 (Vision), Guiding Principle 1.*
**Protects:** Trustworthiness, Scientific validity.

### Article 2. Compare Observations, Never Provider Outputs
**All comparison, scoring, and trust reasoning operates exclusively on Observations mapped into the Canonical Observation Ontology. No component may reason about raw provider output directly.**
This is what makes providers substitutable and comparison meaningful, rather than an accumulation of per-provider special cases.
*Sources: ROADMAP.md Guiding Principle 2, §5 Architecture.*
**Protects:** Provider independence, Trustworthiness.

### Article 3. Providers Are Interchangeable Observation Producers
**An OCR engine, a layout parser, and a vision-language model — present or future — are architecturally identical: observation producers behind one shared interface. None is privileged by kind.**
This is ArchiveTrust's stated purpose, not a convenience: it owns the shared observational language through which any producer's output becomes trustworthy knowledge. Scope decisions are measured against this.
*Sources: ROADMAP.md §6 (Architectural Objective), §16.4.*
**Protects:** Provider independence, Extensibility, Scientific validity.

### Article 4. No Canonical Fact Without Traceable Evidence
**Every canonical field must be explainable through an unbroken chain: Evidence → Observation → Comparison → Canonical Observation → Canonical Document field. A canonical fact that cannot be traced to at least one Evidence record is a defect, not an exception.**
The load-bearing trust principle of the project. Everything else in this constitution exists in its service.
*Sources: ROADMAP.md Guiding Principle 5, `MILESTONE1_DOMAIN_MODEL.md` §1.2.*
**Protects:** Trustworthiness, Replayability.

---

## Part II — Evidence and Observation

### Article 5. Evidence and Observation Are Distinct, and Evidence Is Never Mutated
**Evidence (raw provider capture) and Observation (ontology-mapped claim) are separate, independently referenceable entities. Observations reference Evidence by id, never embed or duplicate it. Evidence, once created, is never altered — even if later shown to be erroneous, and even when the response it captured was malformed and no Observation could validly be derived from it.**
The split is empirically forced, not aesthetic: the three investigated providers expose confidence at three incompatible granularities, and one Evidence record legitimately backs multiple Observations. Preserving even rejected/malformed raw output is what makes audit and upstream bug reports possible later.
*Sources: ROADMAP.md §5.2, `MILESTONE0_REVIEW.md` Architecture Strengths #1, `ARCHITECTURE_VALIDATION_REPORT.md` Finding F16.*
**Protects:** Trustworthiness, Replayability, Observability.

### Article 6. Full Exposure Before Canonicalization
**A deterministic provider must surface the richest observation set it is capable of producing. Information a provider actually computes must never be silently discarded before it reaches the Evidence layer. Where a provider computes something valuable internally but doesn't expose it, contribute the change upstream rather than fork.**
Trust claims can only be as strong as the evidence available to support them; discarding signal early forecloses every later analysis.
*Sources: ROADMAP.md Guiding Principle 4, `docs/investigation/PROVIDER_ANALYSIS.md` (upstream contribution sections, all three providers).*
**Protects:** Trustworthiness, Scientific validity.

### Article 7. Graph Structure Is Not Duplicated as Payload Content
**An Observation payload field must never exist solely to record which other Observation it points to and how. That fact belongs on a typed graph edge, not a payload field — reading order, document hierarchy, and captioning/footnoting relationships are graph facts, not content.**
Two representations of one fact is exactly the defect class that produced the Bounding Box duplication, and recurred three more times the moment it was looked for systematically.
*Sources: `MILESTONE1_DOMAIN_MODEL.md` §5.1 (the Graph-Reference Rule) and its four applications (§5.5, §5.6, §5.12, §5.14).*
**Protects:** Maintainability, Trustworthiness.

### Article 8. No Ontology Concept Without Evidence
**A new Observation type enters the Canonical Observation Ontology only once at least one investigated provider is shown to actually produce it. Plausible-sounding concepts without evidentiary support are recorded as open questions or research items — never added speculatively.**
This is Milestone 0's method, reapplied consistently: Handwritten Note and Checkbox were accepted as candidates (real, cited provider support); Signature and Stamp were declined for now (no provider evidence) on the identical standard.
*Sources: ROADMAP.md Milestone 0 and §13 (Research Backlog), `PROPOSAL_REVIEW_CANONICAL_CHUNK.md` §5, `ARCHITECTURE_VALIDATION_REPORT.md` Finding F18.*
**Protects:** Scientific validity, Maintainability.

---

## Part III — Comparison and Reconciliation

### Article 9. Per-Provider Structure and Cross-Provider Structure Are Different Graphs
**A provider's own claimed internal structure (`ProviderObservationGraph`) and the reconciled, cross-provider structure the Comparison Engine produces (`ReconciledObservationGraph`) are distinct types. Neither may substitute for the other, and only the reconciled graph may feed the Canonical Document.**
Conflating them would let one provider's structural claims become canonical by default — precisely the failure this split exists to prevent.
*Sources: `MILESTONE1_DOMAIN_MODEL.md` §2–§3, `MILESTONE0_REVIEW.md` Weakness W2.*
**Protects:** Provider independence, Trustworthiness.

### Article 10. Clustering Must Be Content-Independent
**Grouping Observations that plausibly refer to the same logical document object is decided by ontology-native, position- and type-based signals — never by whether the payload content happens to agree, and never by deferring to one provider's own grouping as authoritative.**
If clustering depended on content agreement, disagreement would be structurally undetectable — three OCR engines misreading one smudged date three different ways would silently become three uncontested facts instead of one contested one.
*Sources: ROADMAP.md §5.9, `MILESTONE0_REVIEW.md` W2, `ARCHITECTURE_VALIDATION_REPORT.md` Finding F4.*
**Protects:** Trustworthiness, Provider independence, Scientific validity.

### Article 11. Uncertainty and Disagreement Are Not the Same Thing
**A canonical field with only one possible corroborating source (Uncertainty) must never be represented identically to a field where independent sources actively conflict (Disagreement). Collapsing both into one "low confidence" number discards information a reviewer needs.**
This distinction directly resolves what was an open question, and is the reason clustering must remain content-independent (Article 10) — without that, this distinction could never even be observed.
*Sources: `MILESTONE0_REVIEW.md` §4 (Trust Model), ROADMAP.md §15 Q9, `ARCHITECTURE_VALIDATION_REPORT.md` Findings F4–F5.*
**Protects:** Trustworthiness, Scientific validity.

### Article 12. The Canonical Document Has Exactly One Legal Source
**The Canonical Document may only be constructed from the `ReconciledObservationGraph`'s Canonical Observations. No provider, adapter, or human-feedback pathway may write to it directly.**
This is the single-authority guarantee that everything upstream — Evidence, Observations, Comparison, Confidence — exists to earn.
*Sources: ROADMAP.md §5.9, `MILESTONE1_DOMAIN_MODEL.md` §1.5.*
**Protects:** Trustworthiness, Provider independence.

---

## Part IV — Confidence and Trust

### Article 13. Confidence Is Derived, Never Asserted, and Has Three Distinct Levels
**Provider Confidence (a provider's own self-report), Comparison Confidence (cross-observation agreement), and Canonical Confidence (final trust in a canonical field) are three distinct, never-conflated concepts. No component may present one as if it were another.**
This is what prevents "the provider is sure" from being mistaken for "we trust the final answer," and is what makes the Confidence Engine's output auditable at all.
*Sources: ROADMAP.md §5.3, `MILESTONE0_REVIEW.md` Architecture Strengths #2.*
**Protects:** Trustworthiness, Scientific validity.

### Article 14. The Scientific Hypothesis Governs Ambiguity
**When two principles appear to conflict, or it is unclear whether a proposed capability belongs in the architecture, resolve in favor of whichever choice keeps the project's core hypothesis — that provider-independent, evidence-based comparison produces more trustworthy results than any single system alone — measurable and falsifiable.**
Stated explicitly as the article every other principle serves; it is the tiebreak of last resort.
*Sources: ROADMAP.md §3 (Scientific Hypothesis), Guiding Principle 0.*
**Protects:** Scientific validity (and serves as tiebreak across all other categories).

---

## Part V — Replayability and Telemetry

### Article 15. Supersession, Never Erasure
**A Canonical Observation, once created, is never edited in place. Re-reconciliation and human correction alike produce a new version, linked to its predecessor; prior versions are retained, never deleted.**
This is what makes "what did we believe about this fact at time T" a well-formed, answerable question — for machine re-reconciliation and for human intervention alike, under one rule.
*Sources: `MILESTONE1_DOMAIN_MODEL.md` §1.2/§1.4, ROADMAP.md telemetry `HumanCorrectionApplied`.*
**Protects:** Replayability, Trustworthiness.

### Article 16. Telemetry Describes Knowledge, Not Execution
**Telemetry events describe changes in what is known, compared, or decided about a document — never which function ran or which process executed. Process-level logging is a separate, non-canonical concern.**
This is what keeps telemetry provider-independent and lets replay work from the knowledge trail alone, not an execution trace.
*Sources: ROADMAP.md §12.*
**Protects:** Replayability, Observability.

### Article 17. Replay Reconstructs the Full Reasoning Trail, Not Just the Output
**Replay must be able to reconstruct Evidence, both Observation Graphs, every comparison decision, and the full confidence evolution — not merely reproduce the final Canonical Document.**
A system that only replays outputs cannot answer "why," which defeats the purpose of an evidence-based trust architecture.
*Sources: ROADMAP.md §5.10.*
**Protects:** Replayability, Observability, Trustworthiness.

### Article 18. Silence Must Be Distinguishable From Failure
**The absence of an Observation from a provider must never be ambiguous between "never invoked here," "invoked and found nothing," and "invoked and failed." Each is a distinguishable, recorded fact.**
Without this, replay cannot explain gaps in the record, and a systematic provider failure could be indistinguishable from a legitimate negative finding.
*Sources: `MILESTONE0_REVIEW.md` W8/#12 (`ProviderObservationAttempted`, `EvidenceRejected`), `ARCHITECTURE_VALIDATION_REPORT.md` Finding F16.*
**Protects:** Replayability, Observability, Trustworthiness.

### Article 19. Ontology Versioning Is Independent of Telemetry Versioning
**The Canonical Observation Ontology carries its own version, independent of telemetry's schema version. Every Observation records both, and migrations between ontology versions are planned for, registered, and required to support replay.**
Without a separate axis, replay cannot distinguish "this predates a type's redefinition" from "this is malformed."
*Sources: ROADMAP.md §5.4.*
**Protects:** Replayability, Extensibility, Maintainability.

---

## Part VI — Provider Independence and Extensibility

### Article 20. Provider Independence
**No downstream component may depend on a provider-specific concept, field, quirk, or vocabulary. Providers are plugins behind one shared interface, and the reconciled structure they feed must contain no provider-native terms.**
Reinforced every time it was tested: Docling's internal document model is structurally tempting to adopt directly and was explicitly rejected as canonical each time the question arose.
*Sources: ROADMAP.md Guiding Principle 3, `docs/investigation/PROVIDER_ANALYSIS.md` §1.9 (Risk Flag), `MILESTONE1_DOMAIN_MODEL.md` §3.6.*
**Protects:** Provider independence, Extensibility.

### Article 21. Deterministic and Probabilistic Providers Are Formally Distinct, Along Independent Axes
**Reproducibility (deterministic/probabilistic), deployment locality (in-process/out-of-process), and introspectability (open/black-box) are three independent attributes of a provider. None may be inferred from another.**
The architecture once implicitly paired these and would have broken on real, named future providers (a deterministic-but-remote Document AI; a deterministic-but-black-box Donut) — corrected specifically so the failure cannot recur.
*Sources: ROADMAP.md §5.5 (Revision 2), `docs/investigation/ARCHITECTURE.md` §6, `MILESTONE0_REVIEW.md` W3/§6.*
**Protects:** Extensibility, Provider independence.

### Article 22. No Unnecessary Forks
**Where an upstream open-source provider computes something valuable internally but does not expose it, ArchiveTrust prefers contributing the change upstream over maintaining a divergent fork.**
Keeps ArchiveTrust a thin trust layer rather than a maintenance burden duplicating providers' own codebases.
*Sources: ROADMAP.md Guiding Principle 11, `docs/investigation/PROVIDER_ANALYSIS.md` (named upstream contribution proposals for all three providers).*
**Protects:** Maintainability, Extensibility.

---

## Part VII — Governance

### Article 23. Milestones Are Sequential, and the Roadmap Is Amended Before It Is Violated
**No milestone's functionality may be implemented before the milestones it depends on are complete. If implementation reveals the roadmap should change, implementation stops, the roadmap is updated first, the reason is recorded, and only then does implementation continue.**
This procedural rule has been followed literally throughout this project's history to date — every gap this constitution's own source documents found in Milestone 1 was corrected in documentation before any Milestone 1 code existed.
*Sources: ROADMAP.md preamble and Guiding Principle 10.*
**Protects:** Maintainability, Trustworthiness (of the process itself).

---

## Part VIII — Alignment (added in the Alignment Observability refinement; topically extends Part III)

### Article 24. Alignment Is a Recorded Hypothesis, Not an Assumption
**The grouping of Observations for comparison — the claim that the Observations entering a comparison group refer to the same semantic object — is a hypothesis, and must be recorded as first-class, replayable evidence, never left as a silent assumption. Every Observation must end in exactly one recorded alignment state, Aligned or Unaligned; none may silently disappear. The mechanism that produces the grouping must be a versioned, observable, replaceable responsibility, so that alignment can evolve the way Confidence did — by first becoming explicit.**
Before this article, clustering was made and consumed as though its output were settled fact; if the wrong Observations were grouped, every downstream trust number could appear valid while describing a comparison that should never have occurred. This is the same discipline Article 13 imposes on Confidence (derived, never asserted) and Article 18 imposes on provider silence (an absence must be a recorded fact, not an ambiguity), applied to the grouping decision itself. It does **not** mandate a better alignment algorithm, an alignment confidence, or any scoring — only that whatever grouping is performed is made observable. The existing clustering (Article 10, content-independent) remains the sole authoritative grouping mechanism; this article governs its *observability*, not its method.
*Sources: ROADMAP.md §5.12 (Revision 4); parallels Articles 10, 13, 16, 17, 18.*
**Protects:** Trustworthiness (no hidden decision behind a trust number), Reproducibility, Observability, Extensibility (a replaceable alignment seam).

---

## Part IX — Acquisition (added in the Workspace & Acquisition Management refinement; sits upstream of Part I)

### Article 25. Acquisition Precedes Trust, and the Trust Engine Remains Unaware of It
**Every archive object entering the Trust Engine is already immutable, hashed, validated, and registered before any provider is invoked. No Trust Engine module (`domain`, `providers`, `application`, `infrastructure`, `runtime`) may depend on any acquisition concept, mechanism, or vocabulary, and no acquisition source may bypass registration to invoke a provider directly.**
This is Article 20's provider-independence discipline applied one stage upstream: Acquisition Sources — Manual Import, Folder Watch, and any future connector — must be as substitutable and as invisible to the Trust Engine as OCR engines and vision-language models already are to each other. It is also Article 1 restated at its true boundary: immutability is not a property the Trust Engine merely assumes, it is a property Acquisition is responsible for establishing before the Trust Engine ever sees a file.
*Sources: ROADMAP.md §5.13 (Revision 5); parallels Articles 1, 20.*
**Protects:** Trustworthiness (a fixed source, established at the true point of entry), Provider independence (extended to acquisition mechanism independence), Extensibility (a replaceable acquisition seam).

### Article 26. Evaluation References Never Produce the Output They Evaluate
**Evaluation reference is independent evidence used to judge outputs. It never participates in producing the output it evaluates. Operational corrections do not automatically become evaluation references; verified references require separate assignment, blinding, independent review, and adjudication where reviewers disagree.**
Allowing a reference to vote in reconciliation contaminates the prediction with its target and makes accuracy and calibration results invalid. Evaluation-reference identities are therefore rejected at the production provider-registration boundary.
*Sources: `audit-artifacts/FINAL_AUDIT_REPORT.md` evaluation-contamination finding; `artifacts/production_closure_workstream_04.md`; `docs/EVALUATION_REFERENCE_MIGRATION.md`.*
**Protects:** Scientific validity, Provider independence, Trustworthiness.

---

## Apparent Conflicts, Resolved

**Article 1 (immutable archive object) vs. Article 15 (supersession, the canonical layer evolves).** Not a conflict: immutability governs the *source* (the archive object) and, by Article 5, *Evidence* (raw capture) — never the *canonical* layer, which is explicitly permitted to evolve, just never destructively. Different layers, different rules.

**Article 6 (full exposure — surface everything a provider computes) vs. Article 8 (no ontology concept without evidence — don't add speculative types).** Not a conflict: Article 6 forbids *discarding* real signal a provider already produces; Article 8 forbids *inventing* ontology structure for signal no provider actually produces. Both pull toward the same discipline — evidence-grounded completeness — from opposite ends.

**Article 4 (no canonical fact without evidence) vs. total disagreement scenarios (Article 11).** Not a conflict, a boundary condition: Article 4 requires at least one contributing Evidence record for a Canonical Observation to exist at all. Where zero evidence exists for a semantic slot, no Canonical Observation is created for it — there is nothing canonical to assert, and none is forced.

**Article 2 (compare observations, never provider outputs) vs. Article 20 (provider independence).** These are not in tension; Article 2 is the mechanism, Article 20 is the property it protects. Listed as separate articles because each has been independently tested and cited by name in the source documents, not because they compete.

---

## Constitutional Audit

Every milestone in ROADMAP.md, checked against this constitution:

- **Milestone 0 (Provider Investigation)** — complete, and compliant by construction: this constitution is extracted from its findings.
- **Milestone 1 (Evidence, Ontology, Canonical Document)** — compliant as specified in `MILESTONE1_DOMAIN_MODEL.md`. Highest implementation-time risk: Articles 5, 7, and 9 — the Evidence/Observation split, the Graph-Reference Rule, and the two-graph separation are exactly the kind of discipline an implementer under time pressure could quietly erode (e.g., by embedding a bounding box directly on an Observation "just this once," or letting an adapter populate a cross-provider edge as a shortcut).
- **Milestone 2 (Telemetry, Journal)** — compliant as scoped. Highest risk: Articles 16 and 17 — telemetry design drifting toward process-logging convenience instead of knowledge-evolution events would violate both at once.
- **Milestone 3 (Provider Adapters)** — the highest-risk milestone in the roadmap for this constitution as a whole. Articles 9, 18, 20, and 21 are all adapter-boundary rules; an adapter that leaks provider vocabulary downstream, fails silently instead of emitting `ProviderObservationAttempted`/`EvidenceRejected`, or hardcodes a deployment-locality assumption tied to determinism would violate the constitution while still "working."
- **Milestone 4 (Comparison Engine)** — Articles 10 and 11 are directly at stake: a clustering implementation that takes the shortcut of matching on content similarity, or a confidence calculation that collapses Uncertainty and Disagreement into one number for simplicity, would violate the constitution without necessarily failing any obvious test.
- **Milestone 5 (Confidence Engine)** — Article 13 is directly at stake: any code path that lets Provider Confidence flow through to a caller labeled as Canonical Confidence, even as a convenience shortcut, is a constitutional violation.
- **Milestone 6 (Human Feedback)** — Article 15 is directly at stake: a "quick edit" correction UI that mutates a Canonical Observation in place instead of producing a superseding version would violate the constitution even if the resulting data looked correct at a glance.
- **Milestone 7 (MVP Integration)** — Article 17 is exercised here for the first time at full scale; this is also where Article 14 matters most in practice — if the hypothesis turns out to be wrong or only weakly supported on the fixture corpus, that result must be reported as a valid finding, not concealed as a shortfall.
- **Future milestones (Semantic Chunking, Embeddings/Vector DB/RAG, Knowledge Graphs, Active Learning, Dataset Generation, Self-Calibrating Confidence, Fine-Tuning)** — currently compliant by virtue of being undeveloped (Article 23). Each will need to independently satisfy Article 4 and Article 12 when built: chunking and embeddings must be derived from the Canonical Document, never from raw Evidence or Observations directly, or they would bypass the entire trust chain this constitution exists to protect.

**No milestone, as currently specified, requires violating this constitution to implement.** Every risk identified above is a risk of *incorrect* implementation departing from what is already documented — not a gap in the documented plan itself.

- **Workspace & Acquisition Management (Revision 5)** — compliant by construction: Article 25 is extracted from its own design, not retrofitted. Highest implementation-time risk: an acquisition source, under time pressure, invoking a provider or writing to the Trust Engine directly instead of going through Archive Object registration first — exactly the shortcut Article 25 forbids, enforced mechanically by extending the existing `tests/review/test_separation.py` dependency-direction check.
