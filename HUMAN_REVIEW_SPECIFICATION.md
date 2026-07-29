# The Human Review System — Architectural Specification

**Status:** Proposed foundational specification for the Learning Platform's Human Review System. This document defines the *interaction architecture, behavioral model, and design philosophy* of Human Review at the same level of rigor as `MILESTONE4_COMPARISON_ENGINE.md` and `docs/investigation/CANONICAL_ONTOLOGY.md`. It is not a UI mockup, a wireframe, or a visual style guide: it defines the behavioral principles from which any future interface — desktop, web, tablet, or otherwise — must be derived, so that all of them behave identically where it matters.

**Authority and relationship to other documents.** `ROADMAP_V2.md` §9 is the authoritative summary of the Human Review System; this document refines it and must not silently diverge from it. Where this specification and `ROADMAP_V2.md` differ, `ROADMAP_V2.md` governs and is amended first (per its own front-matter rule). Two contracts are *frozen* and this document builds on them without altering them:

- The **three-level Confidence model** and its comparability invariant (`ROADMAP.md` §5.3, §5.3.1) — Human Review must display these three levels distinctly and must never render a blended cross-provider confidence.
- The **Human Feedback correction contract** (`ROADMAP.md` Milestone 6; `src/archivetrust/domain/feedback/models.py`) — `CorrectionCategory`, `CorrectionAction`, `HumanCorrection`, `DatasetCandidate`, and the `HumanCorrectionSubmitted` / `HumanCorrectionApplied` / `DatasetCandidateCreated` telemetry events. Human Review is a *client* of this contract; it does not reach around it.

**Maturity placement.** Human Review is the defining capability of maturity level **ML3 — Surfaced** (`ROADMAP_V2.md` §16), and it is the surface through which **ML1 — Observed** passive telemetry is actually generated under real reviewer behavior. This specification is the entry criterion for building either. It is written for a platform expected to evolve, over many years, from a single-operator engineering tool into an organizational platform serving many reviewers through several distinct client applications (§16, §19) — and to remain faithful to ArchiveTrust's core philosophy of trustworthy, evidence-driven document understanding throughout that evolution.

---

## 0. Frozen Constraints This Design Must Obey

Stated first, as `MILESTONE4_COMPARISON_ENGINE.md` §0 does, so every later section can be read against them:

1. **Human Review spans the Dual Architecture boundary, and the two halves obey different rules (§3).** The interface and its passive telemetry are Learning Platform components (ML3, subject to `ROADMAP_V2.md` §15 invariants LP-1…LP-10). The *correction capture* a reviewer's decision produces is a **Trust Engine** operation (Milestone 6), invoked along the evidence chain. The Learning Platform half never writes to canonical knowledge; the Trust Engine half legitimately supersedes a Canonical Observation *as new human evidence*, never as an out-of-band edit.
2. **Two telemetry streams, never merged (`ROADMAP_V2.md` LP-3).** A review action emits *both* a Trust Engine `HumanCorrectionSubmitted` (timeless, frozen §12 schema) *and* Learning Platform passive `ReviewInteraction` records (temporal, `LEARNING_TELEMETRY_SCHEMA_VERSION`). Neither stream absorbs the other; no review-timing event is ever added to the Trust Engine's frozen event set.
3. **The reviewer produces evidence, not authority (§18, HR-1).** A human decision is one more piece of provenance-bearing evidence entering the same chain the Comparison Engine's outputs entered — not a privileged override that bypasses it.
4. **Provider independence holds at the interface (`ROADMAP.md` Guiding Principle 3).** Nothing the reviewer sees or does may special-case a provider by identity. Providers appear as attributed, interchangeable observation sources; the review logic reads capability and agreement, never provider name.
5. **No canonical fact without traceable evidence (`ROADMAP.md` Guiding Principle 5).** Every value the reviewer accepts or supplies must remain traceable through the chain Evidence → Observation → Comparison → Canonical Observation. A reviewer-supplied edit becomes a new `HumanCorrection` that is itself the evidentiary basis for the resulting superseding Canonical Observation.
6. **Human Review belongs to the Learning Platform, not to any one client (§16).** The interaction model specified here is exposed through Learning Platform *service contracts*; the graphical clients that consume it (Reviewer, Architect, Management) are interchangeable implementations, none of which the model may depend on (`ROADMAP_V2.md` LP-1; HR-11).

---

## 1. Purpose

The Human Review System exists for two purposes at once, and the design must serve both without letting either degrade the other.

### 1.1 Operational purpose

Let a reviewer resolve one document uncertainty **quickly, accurately, and with minimal cognitive effort.** Operational success is measured, not asserted (§11): review duration, edit rate, and completion rate are the operational-horizon metrics (`ROADMAP_V2.md` §6.1) this system is optimized for.

### 1.2 Research purpose

Generate **high-quality evidence** that improves, over the scientific horizon (`ROADMAP_V2.md` §6.2): future providers, prompt contracts, the Canonical Observation Ontology, Confidence calibration, quality analytics, and machine-learning datasets.

### 1.3 One instrument, two readings — and one relationship

Human Review is both an operational tool and a scientific instrument. The critical design consequence is that **the research reading must be obtained without taxing the operational reading.** This is why passive telemetry (§11) is preferred to interrogation: the moment the interface asks the reviewer to serve the research purpose (a mandatory "how confident were you?" prompt), it has taxed the operational purpose and corrupted the very measurement it sought (§11, invariant HR-3). The instrument reads the reviewer by watching, not by interviewing.

Underneath both readings is a single relationship, which the entire design expresses: **the reviewer collaborates with ArchiveTrust; they do not repair it.** The Trust Engine prepares evidence, the human adjudicates it, ground truth is produced, and the Learning Platform learns from the whole exchange:

```
Trust Engine  ──►  prepares evidence  ──►  Human reviewer adjudicates
                                                     │
                                                     ▼
                                            Ground truth produced
                                                     │
                                                     ▼
                                          Learning Platform learns
```

The reviewer is not correcting an AI's mistakes on an assembly line. They are the human judgment in a collaboration whose other half is the entire Trust Engine (§3, §4). Every framing in this document follows from that relationship.

---

## 2. The Cognitive Workflow (defined before any screen)

Per the specification's method, the interface is *derived from* the reviewer's cognitive workflow, never the reverse. This section defines that workflow with no reference to layout.

### 2.1 What the reviewer is trying to accomplish

The reviewer is answering exactly one question per uncertainty:

> *"Is the system's current canonical value for this one thing correct, and if not, what is correct?"*

They are **verifying evidence**, not editing OCR. This distinction is the design's north star: the reviewer's default mental posture should be adjudication between already-surfaced candidates, not transcription from scratch. An interface that feels like a text editor has already failed §2 (invariant HR-8: verification is preferred over correction).

### 2.2 The order in which the reviewer needs information

Human attention is a scarce resource and information has a *sequence* of relevance. The workflow requires, in order:

1. **Location first.** *Where is this in the document?* — answered before the reviewer asks, by the Evidence Viewer auto-navigating to the region (§9, invariant HR-2). This must be free; the reviewer never spends attention locating.
2. **The claim under review.** The current canonical value and its `observation_type`.
3. **The nature of the uncertainty.** *Why am I being asked?* — corroborated-but-low, contested between sources, or single-source-uncorroborated (the `ComparisonClassification`, §10.4). This tells the reviewer what *kind* of judgment is needed before they see the details (and it is a question the interface answers explicitly — §6).
4. **The competing candidates.** What each source claims, with character/word/structural differences made immediately legible (§10).
5. **The grounding evidence.** The original document region and each candidate's backing Evidence — always available, consulted on demand, never hunted for.

The interface exposes this information *progressively*, in this order, revealing only as much as the current uncertainty requires (§5).

### 2.3 The questions the reviewer mentally answers

- *Do the sources agree?* → answered by agreement visualization (§10.4), not by reading raw values side by side and inferring.
- *If they disagree, which is right?* → answered by comparing each candidate against the **original document**, which is primary (invariant HR-7).
- *Is any candidate present at all, or is something missing?* → the "missing observation" case (§13.5) must be as legible as a value conflict.
- *Is this genuinely ambiguous?* → a first-class outcome (Mark Ambiguous, §10), not a failure the reviewer must force into an accept/edit.

### 2.4 The always-visible / on-demand split

- **Always visible:** the original document region under review; the current canonical value; the uncertainty classification; the available decision actions.
- **On demand (one gesture, no navigation away):** each candidate's full backing Evidence; provider/OCR overlays; the raw provider output; layer toggles; optional metadata capture.

The split is an architectural rule, not a preference: promoting on-demand detail to always-visible raises cognitive load, and demoting always-visible context to on-demand breaks HR-2. Both are defects. The split is the static form of Progressive Disclosure (§5); the tiers in §5 are its dynamic form.

### 2.5 Automatic vs. explicit actions

- **Automatic (the system does it, the reviewer never requests it):** scrolling to the region, zooming to fit it, highlighting its bounding box, synchronizing every viewer to the same semantic slot, starting/stopping the passive-telemetry clocks.
- **Explicit (a human decides, the system never does it for them):** accepting a value, editing a value, marking ambiguous, skipping, escalating. Per invariant LP-5 and HR-7, no decision is ever taken automatically — the system prepares the decision; the human makes it.

---

## 3. Where Human Review Sits in the Dual Architecture

This is the specification's central subtlety and it is stated precisely, because getting it wrong would either violate `ROADMAP_V2.md` LP-5 (Learning Platform never writes canonical knowledge) or `ROADMAP.md` Guiding Principle 5 (no canonical fact without evidence).

A single review decision fans out into **three distinct effects across two systems** — the concrete mechanics of the collaboration in §1.3:

```
                         Reviewer's explicit decision
                                     │
              ┌──────────────────────┼──────────────────────────┐
              ▼                      ▼                            ▼
   (A) Trust Engine:        (B) Trust Engine:          (C) Learning Platform:
   HumanCorrection          Confidence Evolution        passive ReviewInteraction
   captured as new          updated; superseding        + ReviewSession telemetry
   human EVIDENCE           CanonicalObservation         (its OWN temporal stream)
   (HumanCorrectionSubmitted) produced                   (never in the frozen §12 set)
                            (HumanCorrectionApplied)
```

- **(A) and (B) are Trust Engine operations (Milestone 6).** A correction is *new evidence*, and the resulting Canonical Observation is a **supersession, never an erasure** (`ROADMAP.md` Constitution Article 15). This is not the Learning Platform reaching into canonical knowledge; it is the Trust Engine's own human-feedback contract being exercised by an authorized human decision, fully traceable and replayable. "Human Review never *directly* modifies the Trust Engine" (the collaboration framing, §1.3) means exactly this: it never edits a Canonical Observation out of band — the reviewer contributes evidence, and the Trust Engine's frozen contract produces the new version. The reviewer collaborates; they do not reach in and repair.
- **(C) is a Learning Platform operation (ML3).** The passive interaction stream is derived understanding *about* the review, produces only advisory analytics (§15), and holds no write path to any Trust Engine object (LP-1, LP-2, LP-5).

The interface is a **client of the Trust Engine's Milestone-6 contract and a producer of the Learning Platform's ML1 telemetry, simultaneously and separately.** It is never a fourth thing that edits canonical state directly.

---

## 4. Human Review Philosophy

The behavioral commitments every interface built from this specification must honor. Each is later restated as an enforceable invariant (§18); here they are stated as design values.

### 4.1 The founding assumption

ArchiveTrust assumes, as an explicit architectural premise, that **humans are significantly better at judging evidence than at repeatedly transcribing documents.** Transcription is slow, error-prone, fatiguing, and — critically — it is work a maturing Trust Engine should increasingly do *for* the human. Judgment is where human attention is irreplaceable: deciding which of several well-surfaced candidates is correct, or recognizing that a source is genuinely ambiguous, is exactly the discrimination machines do worst and humans do best.

From that premise the design commits to a direction of travel:

- **Verification is preferred over correction.** The reviewer's primary act is confirming or choosing among prepared candidates, not producing new text.
- **The interface prepares evidence; the reviewer makes judgments.** These are two different jobs, and the interface owns the first so the human can spend all their attention on the second.
- **Manual transcription should become increasingly rare as the Trust Engine improves.** A high and stable manual-edit rate is not a normal operating point; it is a signal that the Trust Engine, an ontology concept, or a prompt contract needs attention — surfaced as such (§15, `ROADMAP_V2.md` §11), never quietly absorbed as "just how review works."
- **Human Review should feel like adjudicating evidence, not editing OCR.** If the reviewer's felt experience is "typing corrections into a form," the design has failed its founding assumption regardless of how accurate the underlying engine is.

### 4.2 The reviewer as collaborator, not repairer

Following directly from §1.3 and §4.1: the reviewer is a **collaborator with ArchiveTrust, not a repair technician for a broken AI.** The Trust Engine prepares the best evidence it can; the human supplies the judgment the engine cannot; the Learning Platform learns from the exchange so the engine prepares better evidence next time. Framing the reviewer as "fixing mistakes" both misdescribes the architecture and, over time, demoralizes the most valuable participant in it. Every label, prompt, and interaction in a conforming client should read as collaboration ("confirm," "choose," "this source suggests…"), never as blame ("fix the error," "the AI got this wrong").

### 4.3 The remaining commitments

- **One uncertainty at a time.** The unit of review is a single semantic slot's uncertainty (§2.1). Batching unrelated uncertainties onto one screen multiplies cognitive load and is prohibited (HR-9).
- **Preserve context.** The reviewer never loses their place in the document (HR-2). Context loss is treated as the single most expensive failure the interface can inflict.
- **Observe more than you ask.** Behavior is measured passively; questions are optional and never gate work (`ROADMAP_V2.md` GP 1; HR-3).
- **The source document is primary.** Every judgment resolves against the original archive object, which outranks any provider's rendering of it (HR-7).
- **Human attention is valuable and never wasted.** Any attention spent on navigation, searching, re-establishing context, or absorbing complexity the current problem does not require is waste the architecture must design out (HR-2, HR-10, §5).

---

## 5. Progressive Disclosure

Progressive Disclosure is a first-class architectural principle of Human Review, co-equal with the philosophy in §4: **the interface exposes only the complexity the current uncertainty requires, and no more.** A reviewer solving a one-character OCR disagreement must never be confronted with the full apparatus needed for a structural table conflict. Complexity is *earned by the problem*, revealed as the problem demands it, and never presented pre-emptively.

This is the dynamic counterpart to the static always-visible/on-demand split (§2.4). The split says *what* is hidden until requested; Progressive Disclosure says *how much of the review apparatus is even present*, scaled to the difficulty the triage stage (§8.1) already measured. Because triage classifies each uncertainty from Comparison/Confidence outputs before the reviewer ever sees it (`ComparisonClassification`, `RiskScore`, `RiskItem` types), the interface knows the appropriate disclosure tier in advance and opens directly at it.

### 5.1 Disclosure tiers

The tiers are behavioral contracts, not screens. A client may render them however suits its medium, provided the *amount of apparatus* matches the tier.

- **Tier 1 — Simple (e.g. a straightforward OCR disagreement).** Minimal interface: the document region, the competing values, and a simple accept/choose action. A small evidence region, no overlays surfaced, no difference tooling beyond a plain highlight of what differs. The overwhelming majority of reviews should resolve here, and resolve fast.
- **Tier 2 — Moderate (a contested value with meaningful divergence).** Additional comparison tools appear *because the problem needs them*: provider overlays, character/word-level difference highlighting (§10.2), the backing Evidence for each candidate one gesture away.
- **Tier 3 — Complex (a structural, table, or reading-order disagreement).** The full apparatus: relationship/hierarchy visualization, table comparison on the provider-neutral lattice, reading-order visualization (§10.3), and additional evidence layers. This tier carries real cognitive cost, and precisely because of that it is reserved for the problems that genuinely require it.

### 5.2 Escalation is available but never forced

A reviewer at Tier 1 who discovers the problem is harder than triage judged may escalate the disclosure themselves (reveal overlays, open the difference tools) with a single non-displacing gesture (UX-INV-2). Disclosure escalates *on demand upward*; it never *starts* at maximum. The governing rule (UX-INV-8): **the reviewer is never overwhelmed with maximum complexity when solving a simple problem.** Over-disclosure is as much a defect as context loss.

### 5.3 Relationship to review effort telemetry

Which tier a reviewer actually operated at, and whether they escalated, is itself passive evidence (§11): frequent manual escalation above the triage-assigned tier means triage is under-classifying difficulty, and is surfaced to the Evolution Center as a candidate (§15) rather than left as private reviewer friction.

---

## 6. Human Trust — the Self-Explaining Interface

The reviewer must trust the system they collaborate with, and trust is earned by transparency. A conforming interface answers, wherever practical and *before the reviewer has to wonder*, four questions:

- **"Why am I reviewing this?"** — because triage (§8.1) flagged this slot on a specific, nameable signal: the sources disagree, or only one source attempted it, or a risk item was raised. The interface states the reason in the reviewer's terms, not "RiskScore magnitude 0.72."
- **"Why is this source highlighted / weighted?"** — because of *agreement and capability*, never provider identity (Frozen Constraint 4). The interface may say "two independent sources agree on this reading" — it never says "Provider X is more reliable," which would both leak provider identity as authority (§14) and violate the comparability invariant (`ROADMAP.md` §5.3.1).
- **"Why is this uncertain?"** — because of a specific `ComparisonClassification` and its disagreements (§10.4): contested between sources, or corroborated but weak, or single-source. The three read as categorically different (UX-INV-6), so *why* is legible, not just *that*.
- **"Why was this candidate the current canonical value?"** — because of the reconciliation that produced it (the cluster's `reconciliation_basis`), explained as "chosen because the most independent sources agreed," not as an exposed algorithm trace.

**Transparency is part of trust, but it is transparency in the reviewer's vocabulary, not the architect's.** This is the tension the design must hold: the interface explains itself *fully* in terms a reviewer understands (sources, agreement, the document), and *never* in terms only an architect understands (policy versions, capability tags, telemetry internals — forbidden by §14). A self-explaining interface that explains itself in architecture jargon has failed both this section and the anti-goals of §14. Explanations are always available, never mandatory, and never interrupt the decision (HR-3).

---

## 7. UX Invariants (architectural rules, not suggestions)

These are binding on every client (§16). They are architectural because violating one changes *what the system is*, not merely how it looks — and because client independence (HR-11) means they must hold identically across every client that presents review.

- **UX-INV-1 — "Where is this?" is never asked.** The reviewer shall never need to determine, by their own effort, where in the document the current uncertainty lies. The interface answers this before the question forms (§9). *(≡ HR-2.)*
- **UX-INV-2 — Context is never lost.** No action the reviewer takes — consulting evidence, toggling an overlay, escalating disclosure, escalating for further review — may cost them their location, their candidates, or their place in the review queue.
- **UX-INV-3 — The original document is always immediately accessible.** Not one navigation away; present, or reachable by a single non-displacing gesture.
- **UX-INV-4 — Navigation is minimized.** The default resolution path for a corroborated uncertainty is zero navigation: look, accept. Every required navigation step is a cost to be justified.
- **UX-INV-5 — Every interaction reduces cognitive effort.** Each affordance must lower the reviewer's load or be removed. Affordances that merely add capability at the cost of attention are defects.
- **UX-INV-6 — The three confidence levels are never conflated in display.** Provider, Comparison, and Canonical Confidence (`ROADMAP.md` §5.3) are shown as distinct signals; the interface never renders a single "confidence" number that silently blends them, and never renders a cross-provider blend of raw Provider Confidence (`ROADMAP.md` §5.3.1; UX-INV-6 is the display-layer enforcement of that invariant).
- **UX-INV-7 — Meaning never depends on color alone.** Agreement, disagreement, and confidence are conveyed by shape, text, and position as well as color (§12), so the instrument is legible to color-blind reviewers and in high-contrast modes without loss of information.
- **UX-INV-8 — Complexity is disclosed progressively.** The interface presents only the apparatus the current uncertainty requires (§5); a reviewer is never confronted with maximum complexity for a simple problem, and disclosure escalates on demand, never pre-emptively.
- **UX-INV-9 — The interface explains itself in the reviewer's language.** Wherever practical, the reasons a slot is under review, a source is highlighted, or a value is uncertain are available to the reviewer (§6), stated in terms of documents, sources, and agreement — never in architecture, ontology, or telemetry terms.
- **UX-INV-10 — Behavioral consistency across clients.** Every client that exposes review (§16) must behave identically with respect to these invariants and the decision model (§10), however different its presentation and audience. A judgment reached in one client is indistinguishable, as evidence, from the same judgment reached in another (HR-11).

---

## 8. Review Lifecycle

The complete path of one uncertainty through the system. Each transition names the artifact it produces. This is the collaboration (§1.3) traced end to end.

```
Document enters review
        │   (a Canonical Document snapshot exists; its Canonical Observations carry
        │    ComparisonConfidence + Canonical Confidence from Milestones 4–5)
        ▼
Uncertainty detected
        │   (a triage pass, §8.1, selects slots whose ComparisonClassification /
        │    RiskScore warrant human attention — never every slot — and assigns a
        │    Progressive Disclosure tier, §5)
        ▼
Evidence prepared
        │   (the slot's contributing Observations, their Evidence, bounding boxes,
        │    Agreement/Evidence Reports are assembled into a review packet, §8.2)
        ▼
Reviewer presented with ONE issue
        │   (Evidence Viewer auto-navigates; interface opens at the assigned disclosure
        │    tier and can explain itself, §6; passive REVIEW_OPENED recorded, clock starts)
        ▼
Decision
        │   (explicit human adjudication, §10 — accept / edit / skip / ambiguous / escalate)
        ▼
Ground truth generated
        │   (Trust Engine: HumanCorrectionSubmitted → HumanCorrectionApplied →
        │    superseding CanonicalObservation + ConfidenceChanged, §3 effects A/B)
        ▼
Passive telemetry collected
        │   (Learning Platform: ReviewInteraction stream folded into a ReviewSession,
        │    §3 effect C; REVIEW_COMPLETED stops the clock)
        ▼
Learning Platform updated
        │   (Quality Center, Provider Health, Evolution Center recompute from the
        │    now-larger telemetry stream — reproducibly, §15; no production change)
        ▼
Next issue
```

### 8.1 Uncertainty detection (triage)

Triage decides *which* slots enter review, in what order, and *at what disclosure tier* (§5). It reads only Comparison/Confidence outputs already present on each Canonical Observation — never re-runs providers. Selection signals, in decreasing urgency:

- `ComparisonClassification.CONTESTED` — sources actively disagree.
- High `RiskScore` magnitude / severe `RiskItem`s (`VALUE_CONFLICT`, `STRUCTURAL_CONFLICT`, `SEGMENTATION_CONFLICT`, `MISSING_EXPECTED`).
- `ComparisonClassification.UNCORROBORATED_SINGLE_SOURCE` on high-value observation types.
- Low Canonical Confidence with no offsetting corroboration.

The signal that selects a slot is also the material the interface uses to answer "why am I reviewing this?" (§6) and to choose the disclosure tier (a value conflict → Tier 1/2; a structural or table conflict → Tier 3). Triage is deterministic and reproducible from stored telemetry (it is itself a Learning Platform read, LP-4). Corroborated, high-confidence slots are *not* queued — reviewer attention is spent where uncertainty is, not uniformly (UX-INV-5).

### 8.2 The review packet

An immutable, read-only assembly handed to the interface for one uncertainty: the semantic slot id; the current Canonical Observation; each contributing Observation with its provider attribution and backing Evidence (including bounding boxes and, for probabilistic providers, the originating prompt); the `AgreementReport`, `EvidenceReport`, `TrustScore`, and `RiskScore` for the cluster; the assigned disclosure tier; and the original archive-object region reference. The packet contains everything the reviewer could need so the interface never has to fetch mid-review (UX-INV-2), and it is the same packet regardless of which client renders it (§16, HR-11).

---

## 9. Screen Architecture (logical regions, not pixels)

Defined only *after* the cognitive workflow (§2) and the disclosure model (§5), and expressed as logical regions with defined behavior — never coordinates. A client may arrange these for its medium and audience (§16) provided every region's behavioral contract below is met.

| Region | Responsibility | Always visible? |
|---|---|---|
| **Evidence Viewer** | Renders the original document region and its overlays; owns auto-navigation (§9.1). | Yes |
| **Difference Viewer** | Renders competing candidates and their differences; owns agreement/confidence visualization (§10). | Yes |
| **Decision Panel** | Presents the explicit decision actions and optional metadata (§10.5–§10.6). | Yes |
| **Metadata** | Observation type, ontology version, provenance summary. | On demand |
| **Navigation** | Movement within the review *queue* (not within the document). | Minimal, peripheral |
| **Review Progress** | Position in the current batch; nothing that pressures speed over accuracy. | Peripheral |

The three always-visible regions (Evidence Viewer, Difference Viewer, Decision Panel) form the irreducible core: a reviewer must be able to resolve a corroborated uncertainty using only these three, with zero navigation (UX-INV-4). At Tier 1 (§5) the Difference Viewer and Decision Panel are at their most minimal; higher tiers enrich them without displacing the core.

### 9.1 Evidence Viewer

The Evidence Viewer's contract is **the reviewer never manually searches for the current evidence** (UX-INV-1). Behavior:

- **Automatic scrolling.** On presenting an uncertainty, the viewer scrolls the original document to the region under review. The reviewer never scrolls to find it.
- **Automatic zoom.** The viewer zooms to frame the region legibly — tight enough to read, wide enough to keep surrounding context (§2.2 step 1). Zoom level is derived from the bounding box, not fixed.
- **Bounding-box highlighting.** The exact region (`ROADMAP.md` §5.2 Bounding Box, §9 Capability Matrix) is highlighted so the reviewer's eye lands on it without search.
- **Semantic zoom.** The detail rendered adapts to zoom level — word-level glyphs when close, region/structure when far — rather than a single fixed rendering (`ROADMAP_V2.md` §9.3).
- **OCR overlays / Provider overlays.** Each provider's recognized text can be overlaid *in place* over the source, attributed, toggled on demand (surfaced by default only at Tier 2+, §5).
- **Layer visualization.** The reviewer can toggle between raw scan, per-provider overlays, and the reconciled canonical view — the same underlying evidence seen through different layers, never separate screens.
- **Evidence synchronization.** Every viewer region shows the *same semantic slot* at all times. Selecting a candidate in the Difference Viewer moves the Evidence Viewer's highlight to that candidate's Evidence automatically. The reviewer's context is never split across mismatched slots (UX-INV-2).

---

## 10. Difference Viewer and the Decision Model

### 10.1 Purpose

Make disagreement **immediately comprehensible** (`ROADMAP_V2.md` §9.3). The reviewer should grasp *what differs and whether it matters* at a glance, before reading any raw value in full. How much of the tooling below is present is governed by the disclosure tier (§5).

### 10.2 Value differences (text-bearing observations)

Grounded in the Comparison Engine's text reconciliation ladder (`MILESTONE4_COMPARISON_ENGINE.md` §5); surfaced at Tier 2+:

- **Character-level differences** for near-identical candidates (a single transposed glyph), so a one-character OCR error is not disguised as two "different" strings.
- **Word-level differences** for larger divergences, so the reviewer reads changed words, not changed lines.
- The diff is presented against the **original document** (HR-7), not merely candidate-against-candidate, so the reviewer adjudicates against ground truth rather than choosing the more plausible-looking string.

### 10.3 Structural, table, and reading-order differences

The Tier 3 apparatus (§5), present only when the problem is genuinely structural:

- **Structural differences** (parent/child hierarchy) from structural reconciliation (`MILESTONE4_COMPARISON_ENGINE.md` §4) — e.g. one source nests a caption under a figure, another under a paragraph — shown as competing hierarchy fragments, not raw edge lists.
- **Table differences** from the provider-neutral separator lattice (`MILESTONE4_COMPARISON_ENGINE.md` §6) — merged cells, missing rows, shifted columns, differing segmentation — surfaced on the lattice so the reviewer sees *where* the tables disagree geometrically, never as two opaque grids to mentally align.
- **Reading-order differences** shown as competing sequences over the same regions, so an ordering conflict is legible without the reviewer re-deriving order by eye.

### 10.4 Agreement and confidence visualization

- **Agreement visualization** renders the `AgreementReport`: how many *independent* sources corroborate each candidate (`independent_corroborating_count`) and the `ComparisonClassification`. The three classifications look **categorically different**, never like three points on one bar (`ROADMAP.md` §5.3.1; UX-INV-6): `CORROBORATED`, `CONTESTED`, and `UNCORROBORATED_SINGLE_SOURCE` are distinct visual states, because "corroborated and scored low" must never look like "could not be corroborated at all." This visualization is also the answer to "why is this uncertain?" (§6).
- **Confidence visualization** keeps Provider, Comparison, and Canonical Confidence **visually distinct** (UX-INV-6). Provider Confidence is shown *only within* a single provider's own values, never as a cross-provider bar (`ROADMAP.md` §5.3.1). The reviewer sees *why* something is uncertain (no corroboration vs. active conflict vs. weak agreement), not just *that* it is.

### 10.5 Decision actions

The explicit actions a reviewer may take, and their precise mapping onto the **frozen** `CorrectionAction` / `CorrectionCategory` taxonomy (`src/archivetrust/domain/feedback/models.py`). Where a reviewer-facing action has no exact taxonomy member, this specification *proposes* an extension rather than silently inventing one — the taxonomy is authoritative and any addition follows the stop-update-explain rule. In keeping with §4.2, these are phrased as collaboration, not repair.

| Reviewer action | Meaning | Maps to | Produces ground truth? |
|---|---|---|---|
| **Accept Provider** | Confirm a presented candidate as canonical (the fast path). | `CorrectionAction.ACCEPT` | Yes — confirmation is evidence (§15) |
| **Manual Edit** | Supply a value no source presented (the exceptional path, §4.1). | `CorrectionAction.EDIT` + `raw_corrected_output` | Yes |
| **Reject** | Reject the current value without supplying a replacement. | `CorrectionAction.REJECT` | Yes |
| **Mark Ambiguous** | The archive itself admits multiple valid readings (§13.4). | `CorrectionAction.FLAG_FOR_REVIEW` + rationale; *proposed:* a distinct `AMBIGUOUS` category | Yes — ambiguity is a finding |
| **Request Further Review** | Escalate beyond this reviewer's authority/expertise. | `CorrectionAction.FLAG_FOR_REVIEW` | No canonical change; escalation recorded |
| **Skip** | Defer without judging. | *No* `HumanCorrection`; passive telemetry only | No — a skip is never an acceptance (§10.7) |

`MERGE` / `SPLIT` (segmentation corrections) exist in the taxonomy but their structural execution is out of Milestone 6 scope (`feedback/models.py`); the interface may *record* a merge/split proposal as a `FLAG_FOR_REVIEW` with category `SEGMENTATION_ERROR` until that execution path exists.

**Mark Ambiguous and Request Further Review currently collapse onto `FLAG_FOR_REVIEW`.** This specification flags that as a taxonomy gap: they are semantically distinct (an intrinsic property of the document vs. a routing decision about the reviewer) and a future revision should distinguish them, either via `CorrectionCategory` or a proposed `CorrectionAction` extension. Recorded here so it is decided deliberately, not discovered later.

### 10.6 Optional metadata (never interrupts flow)

Modeled by `ReviewerMetadata` (`src/archivetrust/learning/review/interaction.py`), all nullable, none ever required (HR-3, `ROADMAP_V2.md` §9.5):

- **Reviewer confidence** — one lightweight rating, not a form.
- **Failure category** — reusing `CorrectionCategory`'s values (transcription / classification / structural / segmentation / missing / spurious / metadata), not a parallel taxonomy.
- **Free-text rationale** — optional, and a first-class Evolution Center input once clustered (§15, `ROADMAP_V2.md` §11).

No decision action may be gated on metadata. An interface that blocks Accept until a category is chosen has violated HR-3 and is defective (`ROADMAP_V2.md` LP-7).

### 10.7 Skip is not an outcome, and that is deliberate

A Skip produces **no** `HumanCorrection` — no ground truth, no supersession — because the reviewer rendered no judgment. It produces only passive telemetry (a review that opened and did not complete → `ReviewOutcome.INCOMPLETE`, `src/archivetrust/learning/review/session.py`). This preserves honesty: the same discipline the Trust Engine applies with `ProviderObservationAttempted` (`ROADMAP.md` §12 — "attempted, whether or not it produced anything") applies to reviewers. A skipped uncertainty returns to triage; it is never silently counted as agreement.

---

## 11. Passive Telemetry — precise specification

The single most important research mechanism, specified to the level of *what, when, why*. All of it is the Learning Platform's own stream (LP-3), modeled by `ReviewInteraction` / `ReviewInteractionKind` (`src/archivetrust/learning/review/interaction.py`) and folded into a `ReviewSession` (`session.py`).

### 11.1 What is collected, when, and why

| Measure | Interaction kind(s) | Collected when | Why it is research data |
|---|---|---|---|
| **Review duration** | `REVIEW_OPENED` → `REVIEW_COMPLETED` | On present / on decision | Primary difficulty signal; feeds review-duration-outlier detection (`ROADMAP_V2.md` §11) |
| **Edit duration** | `EDIT_STARTED` → `EDIT_COMMITTED` | On edit start / commit | Separates *deciding* cost from *typing* cost; a slot cheap to judge but expensive to type indicates an interface, not an understanding, problem |
| **Accept vs. Edit** | `VALUE_ACCEPTED` / `EDIT_COMMITTED` | On decision | The concrete measure behind "editing is the exception" (HR-8, §4.1); a rising edit rate is provider/ontology pressure surfaced to §15 |
| **Number of edits** | `EDIT_COMMITTED` count | Per session | Distinguishes a one-touch fix from a struggled-with slot |
| **Undo count** | `EDIT_UNDONE` | On undo | Friction/uncertainty signal — the reviewer changed their mind, which a final value alone hides |
| **Zoom usage** | `ZOOMED` | On zoom | How hard the reviewer had to look; concentrated zoom on a type suggests illegible evidence or too-tight auto-zoom |
| **Pan usage** | `PANNED` | On pan | Context-seeking beyond the auto-framed region — a possible UX-INV-1 leak to investigate |
| **Overlay usage** | `OVERLAY_TOGGLED` | On toggle | Which layers a reviewer needed before deciding — reveals what evidence actually drove the judgment |
| **Disclosure escalation** | `OVERLAY_TOGGLED` / *proposed* `DISCLOSURE_ESCALATED` | On manual escalation above the assigned tier | Under-classified difficulty: triage assigned Tier N, the reviewer needed Tier N+1 (§5.3) |
| **Navigation behavior** | `NAVIGATED` | On navigate | Any navigation is a cost (UX-INV-4); its frequency measures how well the packet (§8.2) pre-assembled what was needed |
| **Keyboard vs. mouse** | *proposed* `INPUT_MODALITY` (extension) | Per interaction | Long-session ergonomics (§12); keyboard-first reviewers should not be forced to the mouse. Not in the current enum — added via the `OTHER`/extension seam, per `ReviewInteractionKind`'s documented extensibility |

### 11.2 How measurements become research data

Interactions are inert until folded. `ReviewSession.from_interactions` (pure, reproducible — LP-4) derives per-review effort/outcome; `sessions_from_interactions` groups a stream; `human_effort_summary` (`analytics/quality.py`) aggregates. From there:

- **Quality Center** (`ROADMAP_V2.md` §10) rolls effort up per observation-type, provider, and document-type over time.
- **Evolution Center** (`ROADMAP_V2.md` §11) reads review-duration outliers, rising edit rates, and disclosure-escalation frequency as *evidence-backed evolution candidates* — advisory only (LP-5).
- **Confidence calibration** (future, `ROADMAP.md` §14) may use accept/edit outcomes as ground-truth signal, at *use time*, never by rewriting stored Provider Confidence (`ROADMAP.md` §5.3.1).

Because every step is a pure function of the stored interaction stream, every derived research figure is reproducible (HR-6, LP-4) and versioned by `LEARNING_TELEMETRY_SCHEMA_VERSION` (HR-6, LP-8).

### 11.3 Why passive beats interrogation (design justification)

A post-hoc questionnaire asks the reviewer to introspect and self-report, which is subject to recall bias, fatigue, and inconsistent per-reviewer calibration, and — decisively — it *taxes the operational purpose to serve the research purpose* (§1.3). Behavior recorded live requires zero reviewer effort, is comparable across reviewers and time, and is captured at the instant it happens. Observation is preferred to inquiry wherever observation suffices (`ROADMAP_V2.md` GP 1).

---

## 12. Accessibility (designed for long sessions)

Accessibility here is partly a human-factors requirement and partly a *correctness* requirement of the instrument — some items below are not optional courtesies but consequences of §7's invariants.

- **Keyboard-first workflow.** The entire accept/edit/skip/escalate loop is operable without a mouse. This lowers cognitive and physical load over long sessions and interacts with the keyboard-vs-mouse telemetry (§11.1): the interface must not *force* the mouse and then measure mouse dependence.
- **High contrast.** A high-contrast mode is a first-class rendering, not an afterthought — long review sessions are visually taxing.
- **Large / scalable fonts.** Text scales without breaking region layout or the semantic-zoom contract (§9.1).
- **Color-independent indicators.** *Required by UX-INV-7.* Agreement, disagreement, and confidence must be distinguishable by shape/text/position, not color alone — otherwise the agreement visualization (§10.4) is illegible to a color-blind reviewer, which is an information-loss defect, not merely an accessibility shortfall.
- **Screen-reader compatibility.** Regions, candidates, and classifications are exposed with semantic labels, so the workflow is navigable non-visually.
- **Low cognitive load.** The overriding accessibility principle (UX-INV-5), and the reason Progressive Disclosure (§5) is itself an accessibility feature: not showing complexity a reviewer does not need is the most effective load reduction available.

---

## 13. Error and Edge States

The reviewer's experience must be honest and specific in each degraded case — the interface never fabricates a candidate to fill a gap, and never hides that something is wrong. Each state defines what the reviewer sees and what is recorded.

### 13.1 Missing provider output

A provider was invoked but produced nothing for this region. Distinguished — via `ProviderObservationAttempted` with no resulting Evidence (`ROADMAP.md` §12, Article 18) — from "never invoked here." The Difference Viewer shows the source as *attempted, no observation*, never as a blank candidate that could be mistaken for an empty value. This is itself evidence (a coverage gap), surfaced, not smoothed over.

### 13.2 Corrupted or unreadable document region

The archive object or region cannot be rendered. The Evidence Viewer states this explicitly and offers escalation (Request Further Review); it never presents provider text as if the source were verifiable when it is not. The reviewer is not asked to adjudicate against evidence they cannot see.

### 13.3 Conflicting evidence

The `CONTESTED` classification (§10.4). Fully supported as the *normal* hard case: competing candidates, differences highlighted, adjudicated against the original. This is the workflow's core competency, not an error, and it opens at Tier 2 or 3 (§5) as its structure demands.

### 13.4 Multiple valid interpretations

The archive genuinely admits more than one correct reading (an ambiguous abbreviation, a smudged digit). **Mark Ambiguous** (§10.5) makes this a first-class finding rather than forcing a false accept/edit. Ambiguity recorded here is a research signal about the *document*, and eventually about ontology expressiveness.

### 13.5 Incomplete observations

A candidate exists but is partial (a table detected with missing cells, a heading without its level). Rendered *as* partial — the missing aspect is shown as missing (§10.3), never auto-completed. The reviewer decides whether the partial reading is acceptable or an edit is warranted.

### 13.6 Timeouts

A review left open indefinitely. The session remains `ReviewOutcome.INCOMPLETE` (§10.7); no ground truth is fabricated from an abandoned review. On timeout the uncertainty returns to triage. Passive telemetry records the abandonment honestly (an opened, never-completed review), which is itself a difficulty signal.

---

## 14. What the Interface Should Not Do (Anti-Goals)

Stated as explicitly as the goals, because a specification that only says what to build invites the wrong things to creep in. These anti-goals follow directly from Human Trust (§6, explanations in the reviewer's vocabulary) and from client independence (§16): the reviewer's job is bounded, and the interface must not push architecture work onto them.

**The review interface shall never:**

- **Expose provider implementation details.** No provider names as authority, no version strings, no library internals. Sources appear as attributed, interchangeable observation producers (Frozen Constraint 4).
- **Expose telemetry internals.** No event names, no schema versions, no `RiskScore` magnitudes as raw numbers. The reviewer sees *reasons* (§6), never the instrument's plumbing.
- **Require ontology knowledge.** A reviewer need not know what an `observation_type`, a `semantic_slot_id`, or the Canonical Observation Ontology is to review a document.
- **Require AI or machine-learning expertise.** Nothing about the workflow presumes the reviewer understands how a probabilistic provider works.
- **Overwhelm reviewers with diagnostics.** Policy versions, capability tags, reconciliation traces, and comparison internals are diagnostic material for the Architect Client (§16), not the review surface (§5, UX-INV-8).
- **Require architectural understanding.** A reviewer should never need to understand the Trust Engine / Learning Platform separation, the evidence chain, or this document to do their job well.

The principle in one line:

> **The reviewer reviews documents. The architect reviews architecture.**

Each audience gets the surface appropriate to it (§16). Collapsing them — showing a reviewer the architect's diagnostics, or asking an architect to review documents to reach the platform's health — is a design failure, not a convenience.

---

## 15. Relationship to the Learning Platform

Human Review is the richest single producer of Learning Platform evidence. From ordinary review work — the collaboration of §1.3 — it yields:

- **Ground truth** — human-confirmed canonical values (via the Trust Engine correction contract, §3).
- **Human effort metrics** — durations and counts (§11), aggregated by the Quality Center.
- **Difficulty metrics** — review-duration, undo/zoom patterns, and disclosure-escalation frequency marking which slots and types are hard.
- **Provider evaluation** — which source the reviewer trusted, feeding Provider Health (`analytics/provider_health.py`) — computed per `(provider, provider_version)`, never blended (LP-6).
- **Ontology evolution signals** — repeated corrections of a type, Mark-Ambiguous clusters, and missing-observation patterns feeding the Evolution Center (`ROADMAP_V2.md` §11).
- **Prompt evolution signals** — corrections concentrated on a probabilistic provider's output feeding Prompt Contract Health (`ROADMAP_V2.md` §11.1).
- **Quality metrics** — observation-type survival (`analytics/quality.py`).
- **Machine-learning datasets** — via `DatasetCandidate` (§17).

**Human Review never directly modifies the Trust Engine** in the Learning-Platform sense (LP-5): its analytics half produces only advisory evidence for future *architectural decisions* made by humans (`ROADMAP_V2.md` §7). The one path by which a review changes canonical state is the Trust Engine's own Milestone-6 correction contract (§3), which is evidence-backed and replayable — not an analytics write-back. The reviewer collaborates; the Learning Platform learns; the architecture changes only when a human decides it should.

---

## 16. Human Review as a Service — Multiple Clients

Human Review is **not tied to any particular graphical interface.** It is a set of behaviors and contracts that belong to the Learning Platform and are exposed through Learning Platform *service interfaces*; graphical clients consume those services (Frozen Constraint 6). This is what lets the same review model outlive any one application and serve different audiences at once.

```
Learning Platform
        │   (Quality Center, Evolution Center, Human Review, Provider Health, …)
        ▼
Learning Services
        │   (versioned service contracts: review-packet delivery, decision submission,
        │    passive-telemetry ingest, analytics queries — all honoring §15's invariants)
        ▼
Multiple Clients
   ┌──────────────┬──────────────────┬──────────────────┐
   ▼              ▼                  ▼
Reviewer Client   Architect Client    Management Client
```

The three client roles below are **architectural responsibilities, not implementation choices.** They define *what each audience is entitled to see and do*, from which any number of concrete applications — desktop, web, tablet — can be built as future implementations of the same service contracts.

### 16.1 Reviewer Client

**Purpose:** focused production review. The surface this entire specification is primarily written for.

- **Includes:** an assigned review queue, the review interface (§2–§13), and progress tracking.
- **Excludes:** provider management, analytics, architecture, and any setting unrelated to review — everything §14 names as an anti-goal for the review surface.

The Reviewer Client is deliberately narrow: it is the embodiment of "the reviewer reviews documents" (§14).

### 16.2 Architect Client

**Purpose:** the complete operational and evolutionary platform — "the architect reviews architecture" (§14).

- **Includes:** the Processing Center, Quality Center, Evolution Center, and Evidence Explorer (`ROADMAP_V2.md` §8, §10, §11, §12); provider management; prompt contracts and Prompt Contract Health (`ROADMAP_V2.md` §11.1); and research tools. It may also contain a review surface, but as one capability among many, not its reason for existing.

### 16.3 Management Client

**Purpose:** operational oversight, without touching individual reviews.

- **Includes:** a dashboard of quality metrics, review workload, throughput, human effort, and historical trends (the operational- and scientific-horizon rollups, `ROADMAP_V2.md` §6, §10).
- **Has no direct review interface.** A manager observes the health and throughput of review; they do not adjudicate uncertainties. This keeps oversight and adjudication as separate responsibilities.

### 16.4 Consistency across clients

Every client that exposes review must behave identically with respect to the invariants (§7) and the decision model (§10), while presenting a user experience appropriate to its audience (UX-INV-10). A judgment reached in the Reviewer Client and the same judgment reached through a review surface in the Architect Client are indistinguishable as evidence: same packet (§8.2), same decision actions (§10.5), same telemetry (§11). Presentation varies; behavior and evidence do not (HR-11).

---

## 17. Machine-Learning Perspective

Human Review naturally produces research-quality datasets, but the framing is a load-bearing part of ArchiveTrust's philosophy and is stated as flatly as possible:

> **ArchiveTrust does not collect datasets. ArchiveTrust collects trustworthy evidence. Datasets are reproducible derived artifacts generated from that evidence.**

The distinction is not rhetorical. A system whose goal is "collect training data" optimizes for volume and is tempted to cut provenance corners to get it. A system whose goal is "collect trustworthy evidence" optimizes for provenance, reproducibility, and traceability — and *gets* datasets as a derived consequence, for free and without compromise. Machine-learning datasets are therefore **derived artifacts**, reproducibly generated from primary, immutable inputs:

```
Immutable Archive Objects  ┐
Telemetry (both streams)   ├──►  Dataset Candidate  ──►  (future) ML Dataset
Human Review evidence      │      (DatasetCandidateCreated,      (ROADMAP.md §14,
Canonical Observations     │       ROADMAP.md Milestone 6)        deferred)
Provenance chains          ┘
```

This is the same discipline the Trust Engine applies to *every* derived artifact (`ROADMAP.md` Guiding Principle 1: everything derived from the immutable archive is a reproducible artifact, never a mutation of the source). Framing datasets this way is what **preserves provenance, replayability, and scientific reproducibility** as the platform scales:

- **Provenance preserved.** Every example carries its lineage back through Canonical Observation → contributing Observations → Evidence (`ROADMAP.md` GP 5); no example exists whose origin cannot be named.
- **Replayable.** A dataset is regenerable from the same immutable inputs and stored telemetry (HR-6, LP-4), never a hand-curated pile that cannot be rebuilt from first principles.
- **Scientifically reproducible.** Each dataset is bound to the ontology, policy, and telemetry-schema versions in effect (LP-8), so a result obtained on a dataset built under ontology v1 is never silently conflated with one built under v2 — the difference the Trust Engine already tracks for its own knowledge is preserved for its research outputs too.

`DatasetCandidate` (`feedback/models.py`) is the seam Milestone 6 already leaves for this; dataset *generation* itself is documented-not-built (`ROADMAP.md` §14). Human Review's role is to make the evidence trustworthy at the point of capture, not to train anything.

---

## 18. Human Review Architectural Invariants

Enforceable rules protecting the system, in the style of `ROADMAP_V2.md` §15's LP-invariants. A violation is a defect, not a preference. Each names how it is protected.

- **HR-1 — Human Review produces evidence, not authority.** A reviewer decision enters the evidence chain as provenance-bearing human evidence (§3), never as an override that bypasses it. *Protected by:* routing all canonical change through the Milestone-6 `HumanCorrection` contract; no direct Canonical Observation edit path exists.
- **HR-2 — The reviewer never loses document context.** *Protected by:* UX-INV-1/2/3 and the Evidence Viewer's auto-navigation and synchronization (§9.1).
- **HR-3 — Passive observation is preferred over questionnaires.** No optional metadata gates a decision (§10.6). *Protected by:* modeling metadata as nullable and workflow-independent (`ReviewerMetadata`); `ROADMAP_V2.md` LP-7.
- **HR-4 — Every review action is replayable.** Both effects — the Trust Engine correction and the Learning Platform interaction stream — are append-only, ordered, and replayable (`ROADMAP.md` §5.10; `ROADMAP_V2.md` LP-4). *Protected by:* emitting through existing append-only telemetry sinks only.
- **HR-5 — Every review result is traceable.** A canonical value produced by review traces to its `HumanCorrection` and thence through the full chain (`ROADMAP.md` GP 5). *Protected by:* the correction contract's provenance fields.
- **HR-6 — Every review metric is reproducible.** All effort/difficulty/quality figures are pure functions of stored telemetry, versioned by schema (LP-4, LP-8). *Protected by:* the pure `ReviewSession` / analytics functions.
- **HR-7 — The source document is primary; research never alters production automatically.** Judgments resolve against the original archive object; the Learning Platform's analytics never change production (LP-5). *Protected by:* the §3 boundary and the separation tests (`ROADMAP_V2.md` LP-1).
- **HR-8 — Verification over correction; editing is the exception.** *Protected by:* making Accept the zero-navigation fast path (UX-INV-4), presenting the reviewer with prepared candidates (§4.1), and treating rising edit rate as a surfaced signal (§15), not a normal state.
- **HR-9 — One uncertainty at a time.** The unit of review is a single semantic slot (§2.1, §4.3). *Protected by:* the review-packet contract (§8.2), which scopes to one slot.
- **HR-10 — Learning never bypasses evidence.** No dataset, metric, or evolution candidate is produced from anything but the immutable archive, the telemetry, and traceable review evidence (§17). *Protected by:* LP-4 reproducibility and GP-5 traceability, jointly.
- **HR-11 — Client independence.** Human Review shall never depend on a specific client implementation. All review functionality is exposed through Learning Platform service interfaces (§16); different clients must behave consistently with respect to the invariants and the decision model while presenting user experiences appropriate to their audiences (UX-INV-10). *Protected by:* the service-contract boundary and the single shared review packet (§8.2).
- **HR-12 — The reviewer's surface stays a reviewer's surface.** The review interface never requires provider, telemetry, ontology, AI, or architectural knowledge of the reviewer (§14). *Protected by:* Progressive Disclosure (§5), self-explanation in the reviewer's vocabulary (§6), and the role separation of §16.

---

## 19. Future Directions

Marked as future work, not built at ML3, and — per `ROADMAP_V2.md` GP 10 — documented rather than implemented until their maturity level.

### 19.1 Interaction and visualization

- **Collaborative review** — multiple reviewers on one uncertainty, inter-reviewer agreement as a second-order trust signal.
- **AI review assistants** — a provider-independent assistant that *pre-summarizes* an uncertainty, strictly as another advisory observation source, never as a decision-maker (must remain subject to HR-1/HR-7 and provider independence).
- **Heatmaps** — document- and corpus-level visualizations of where disagreement and human effort concentrate.
- **Reviewer expertise modeling** — routing uncertainties to reviewers by demonstrated strength (interacts with Request Further Review, §10.5).
- **Adaptive interfaces / workflow personalization** — per-reviewer ergonomics informed by their own passive telemetry — permissible only while it never compromises reproducibility (HR-6), behavioral consistency across clients (UX-INV-10, HR-11), or the shared contract of §7.
- **Advanced visualizations** — richer structural/table/reading-order renderings beyond §10.3.

### 19.2 Organizational adoption

ArchiveTrust is expected to evolve from a single-operator engineering platform into an **organizational platform** serving many people at once. The following capabilities are anticipated and reserved:

- Multiple reviewers working concurrently.
- Assigned review queues, per reviewer or team.
- Authentication and authorization.
- Audit trails of who reviewed what, and when.
- Review ownership and hand-off.
- Per-reviewer and per-document review history.

**These capabilities belong to future *client implementations* and *deployment* concerns (§16), not to the Human Review interaction model itself.** The model specified in §2–§14 — one uncertainty at a time, evidence prepared, the human adjudicates, passive telemetry, minimal cognitive load — is invariant under the transition from one reviewer to many. Authentication decides *who* may open a review packet and *which* queue is theirs; it does not change what a review *is*. Keeping organizational features in the client/service layer, and out of the interaction model, is what allows ArchiveTrust to grow to serve organizations without renegotiating its core philosophy (HR-9, HR-11) — the same discipline `ROADMAP_V2.md` applies in keeping learning strictly separate from production.

---

## 20. Acceptance Criteria

Standing criteria for the Human Review System, comparable to the acceptance criteria of the other foundational specifications and consistent with `ROADMAP_V2.md` §17's **ML3 — Surfaced** level. These are *held* while true, not passed once (LP-9).

- ✓ **The review workflow minimizes cognitive effort** — a corroborated uncertainty is resolvable with look-then-accept, zero navigation, at the minimal disclosure tier (UX-INV-4, UX-INV-5, UX-INV-8).
- ✓ **The reviewer never loses document context** — auto-scroll, auto-zoom, bounding-box highlight, and evidence synchronization satisfy UX-INV-1/2/3 (HR-2).
- ✓ **One-click acceptance is possible** — Accept Provider is a single explicit action mapping to `CorrectionAction.ACCEPT` (§10.5).
- ✓ **Editing is exceptional, not default** — the interface prepares candidates for verification (§4.1); accept-vs-edit passive telemetry surfaces a rising edit rate as a signal, never normalized (§11, HR-8).
- ✓ **Complexity is disclosed progressively** — simple problems present minimal apparatus; the reviewer is never overwhelmed solving a simple uncertainty (§5, UX-INV-8).
- ✓ **The interface explains itself** — why-this-is-under-review, why-a-source-is-highlighted, and why-this-is-uncertain are answerable in the reviewer's own vocabulary (§6, UX-INV-9).
- ✓ **Passive telemetry is comprehensive** — every measure in §11.1 is captured without gating workflow (HR-3), folded reproducibly into `ReviewSession`.
- ✓ **Human Review produces reproducible evidence** — all effort/quality metrics are pure functions of stored, versioned telemetry (HR-6, LP-4/LP-8).
- ✓ **Machine-learning datasets are reproducible derived artifacts** — regenerable from immutable inputs + telemetry, fully traceable and versioned; the platform collects evidence, not datasets (§17, HR-10).
- ✓ **Human Review is completely independent of provider implementations** — no interface behavior or review logic branches on provider identity (Frozen Constraint 4, `ROADMAP.md` GP 3).
- ✓ **Human Review is independent of any specific client** — all functionality is exposed through Learning Platform service interfaces; clients behave consistently while serving different audiences (§16, HR-11, UX-INV-10).
- ✓ **The reviewer's surface requires no architectural knowledge** — provider, telemetry, ontology, AI, and architecture details are absent from the review interface (§14, HR-12).
- ✓ **The three confidence levels are never conflated, and no cross-provider confidence is ever blended in display** (UX-INV-6, `ROADMAP.md` §5.3.1).
- ✓ **Canonical change flows only through the evidence-backed correction contract** — Human Review's analytics half holds no write path to canonical knowledge (§3, HR-1, HR-7, `ROADMAP_V2.md` LP-5).

---

*This specification is a foundational architectural document of the Learning Platform, comparable in standing to `MILESTONE4_COMPARISON_ENGINE.md` and the Canonical Ontology. Like them, it is a living contract: any change must be made deliberately, with the reasoning recorded, before implementation proceeds on the affected area — and any change to the frozen contracts it depends on (§0) must be reconciled here in the same revision, never left to drift. It is written to remain faithful, as ArchiveTrust grows to serve many reviewers, organizations, and client applications, to one unchanging idea: the reviewer collaborates with a system that prepares trustworthy evidence, and the system learns from every judgment the reviewer makes.*
