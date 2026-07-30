# The Knowledge Lifecycle

How a telemetry event becomes an observation, an observation becomes a candidate finding, and a
candidate finding becomes — or fails to become — accepted research knowledge. Referenced by
[`docs/architecture/htr-event-model.md`](architecture/htr-event-model.md) §1, which states the rule this
document implements:

> **Hard rule enforced by construction, not convention**: a telemetry event (layer 3) never directly
> becomes a `ResearchObservation` (layer 10) … A `ResearchObservation` never auto-promotes to a
> `ResearchFinding` … and a finding never auto-promotes past `Candidate` status without a … event
> recording a human actor.

Entity details: [`docs/research-observations.md`](research-observations.md),
[`docs/research-findings.md`](research-findings.md).

## The three gates

| Arrow | What blocks it | Where |
|---|---|---|
| event → observation | `supporting_evidence` is non-empty by construction and typed, so extraction always names what it read. No constructor accepts a `TelemetryEvent`, and no code infers which event caused an observation. | `models.py`, `durable_store.register_research_observation` |
| observation → finding | `supporting_observations` is non-empty by construction; `create` refuses any status past `Draft`/`Candidate`; the persistence layer refuses to *register* one past it either. | `models.py`, `durable_store.register_candidate_finding` |
| candidate → accepted | Every status past `Candidate` requires a non-`None` `reviewer` **and** a `FindingRevision`, both validated on the model. The only path that produces them is `transition_finding_status`. | `models.py`, `lifecycle.py` |

## The state machine

`src/archivetrust/htr/knowledge/lifecycle.py::ALLOWED_TRANSITIONS`.

```
Draft ──────────► Candidate ──────► Under review ──┬──► Provisionally supported ──► Supported
  │                   │                  │         ├──► Disputed                      │
  └──► Rejected       └──► Rejected      └──────────┴──► Rejected                      │
                                                                                       │
                                          Disputed ◄───────────────────────────────────┘
                                             ├──► Under review   (re-examination)
                                             └──► Rejected

any status ──► Superseded   (requires a pointer to the superseding finding)
Superseded ──► (terminal)
```

| From | Permitted targets |
|---|---|
| `Draft` | `Candidate`, `Rejected`, `Superseded` |
| `Candidate` | `Under review`, `Rejected`, `Superseded` |
| `Under review` | `Provisionally supported`, `Disputed`, `Rejected`, `Superseded` |
| `Provisionally supported` | `Supported`, `Disputed`, `Rejected`, `Superseded` |
| `Supported` | `Disputed`, `Superseded` |
| `Disputed` | `Under review`, `Rejected`, `Superseded` |
| `Rejected` | `Superseded` |
| `Superseded` | *(terminal)* |

Two absences are deliberate. **`Candidate → Provisionally supported` does not exist**: a candidate must
pass through `Under review`, which is the step that records a human actually looking. **`Supported →
Provisionally supported` does not exist**: walking a claim back is a dispute, recorded as one, not a
quiet downgrade.

`test_every_status_has_an_explicit_edge_set` asserts the table is exhaustive over `FindingStatus`, so
adding a status without deciding its transitions fails a test rather than silently producing a dead end.

## Rules beyond the edge list

1. **Non-empty `reasoning` for every transition.** The requirement is any transition away from
   `Candidate`; this is stricter deliberately — a `Draft → Candidate` promotion with no stated reason is
   no more auditable than a silent `→ Supported`.
2. **Non-empty `reviewer` for every transition.** No finding changes status without an attributable
   actor.
3. **`Provisionally supported → Supported` requires reproduction evidence from a *different run*.** Not
   merely a non-empty list: `reproduction_evidence` must contain at least one `EvidenceReference` of
   kind `experiment_run` naming a run the finding's own `scope.experiment_run_ids` does not already
   cover. `Provisionally supported` already means "the evidence in scope supports this"; `Supported` has
   to mean something more, and for a scoped empirical claim the only thing it can honestly mean is that
   it held up somewhere else. Evidence recycled from the finding's own run would make the two statuses
   synonyms.
4. **`→ Superseded` requires `superseded_by`.** A superseded finding always points at its successor, so
   the chain stays walkable forwards.
5. **`→ Disputed` requires a `ContradictoryEvidence` entry**, appended to whatever the finding already
   carried.

`transition_finding_status` is pure: it computes the next frozen `ResearchFinding` and raises
`InvalidFindingTransitionError`. It emits nothing. `DurableHtrResearchStore.record_finding_transition`
wraps it and emits — the same computation/persistence split `htr/evaluation/*` already has. The
`InvalidFindingTransitionError` propagates **before** anything is appended, so a refused transition
leaves no trace in the log.

## History is append-only, and that is asserted internally

Every transition appends exactly one `FindingRevision` (`revision_id`, `revised_at`, `from_status`,
`to_status`, `actor`, `reasoning`, `evidence_refs`, `superseding_finding_id`,
`contradicting_finding_id`) and preserves every prior one, in order, unmodified.

`_assert_history_only_grew` checks that inside the function — not only in a test — because "this entity
must never lose its history" is the requirement the whole revision mechanism exists to satisfy, and a
future edit that replaced instead of extended would otherwise be caught only by whichever test happened
to look. It raises if the new history is not the old one plus exactly one entry, or if any prior
`ContradictoryEvidence` entry's prefix changed.

The input finding is frozen and is never mutated: `test_the_original_finding_object_is_never_mutated`
holds the pre-transition object and asserts it is still `Candidate` with an empty history afterwards.

## Two telemetry events per transition, in causal order

`record_finding_transition` emits two, because they are two different facts and conflating them would
lose one:

1. **`FindingReviewed`** — a named human looked at this finding and reached a verdict. `actor_type` is
   `HtrActorType.HUMAN`.
2. **`FindingStatusChanged`**, `causation_id` = the `FindingReviewed`'s `event_id` — the status therefore
   changed. Carries the whole post-transition finding.

A review that reached a verdict but changed nothing would emit only (1). A status change with no (1)
before it cannot happen, because this is the only path that emits one.

`HtrJournal` projects `FindingStatusChanged` via `advance_finding` and names `FindingReviewed` an
explicit no-op: the reviewed state is the outcome, and applying one state twice would be the bug.

## The real demonstration, on real data

Performed by `htr/knowledge/registration.py::demonstrate_review_workflow`, committed in
`docs/experiments/baseline-comparison/htr_knowledge_events.jsonl` (18 events: 5
`ResearchObservationCreated`, 5 `CandidateFindingCreated`, 4 `FindingReviewed`, 4
`FindingStatusChanged`).

**Reviewer attribution, stated plainly.** The reviewer on all four transitions is `hypergeek-dev`, the
repository maintainer who ran this phase. There is **no independent second reviewer and no domain expert
in the loop**. `reviewer` is a required argument with no default on both the transition function and
`demonstrate_review_workflow` precisely so this had to be a decision rather than something a default
quietly supplied. Any future reading of these two findings should weigh that: a single-actor review is
what the record says happened, and the record does not dress it up as more.

### `Provisionally supported` — `transkribus_not_comparable`

`research_finding_839388b4d62642359c535ea4e4b29226`

| Step | Events |
|---|---|
| `Candidate → Under review` | `FindingReviewed` `event_51636b97a0e2443eac5072c1d121571f` → `FindingStatusChanged` `event_94d2db768ea14ea89ed9c39fd39a8680` |
| `Under review → Provisionally supported` | `FindingReviewed` `event_030d743ee96a4306934b7e917cce80ef` → `FindingStatusChanged` `event_0123b4eb081849a699ecc35b72a69655` |

Reasoning for the first, verbatim from the log:

> Reviewed against the committed log. The claim is verifiable by inspection rather than by measurement:
> the fixture's own content is an unrelated passage, the method run carries `input_crop_id = null`, and
> no `MetricCalculated` for CER or WER exists for it anywhere in the log (asserted by
> `tests/htr/persistence/test_real_baseline_reconstruction.py::test_no_invalid_transkribus_cer_or_wer_exists_anywhere_in_the_log`
> against every recorded metric *and* every `MetricCalculated` event). The exclusion was also declared in
> advance by the experiment's own `exclusion_criteria`, so it is not a post-hoc rationalisation of an
> inconvenient result. Advancing to Under review to record that a human has examined it.

And for the second:

> Provisionally supported, and deliberately not Supported. The evidence in scope is as strong as this
> kind of claim gets — the fixture's non-correspondence is a fact about a file, not an estimate, and it
> is independently asserted by a test — but 'Supported' in this lifecycle requires reproduction in an
> experiment run outside the finding's own scope, and only one end-to-end run exists. The honest ceiling
> is therefore Provisionally supported. It would reach Supported the moment a second run over the same
> fixture recorded the same absence, which costs no GPU time; nobody has run it.

### `Disputed` — `florence2_environment_reproducible`

`research_finding_4b7b1aff39d3426f87ec5e4cc4b7cbe5`

| Step | Events |
|---|---|
| `Candidate → Under review` | `FindingReviewed` `event_bfb45e8914914f83ab55b206286fbac4` → `FindingStatusChanged` `event_38c89f17a13542949cc076153a056806` |
| `Under review → Disputed` | `FindingReviewed` `event_57e6a242627e47c88d241714faaf05be` → `FindingStatusChanged` `event_ff00ae77bed545c7850a961fb0feac9f` |

Reasoning, verbatim:

> Disputed. The two measurements differ by a factor of ~3.3 (3983 MB documented vs. 1210.64 MiB
> measured), so the reproducibility this finding claims is contradicted by the repository's own records.
> Disputed rather than Rejected because which figure is representative is genuinely unknown: the earlier
> session left no durable record of its torch version or device state, so there is no basis for
> declaring either measurement wrong. The SATRN control reproduced to within 0.3 MB across the same pair
> of sessions, which rules out 'this repository cannot measure GPU memory' as the explanation and is why
> the dispute is specific to Florence-2. The candidate finding, its supporting observation, and the
> contradiction all remain readable — nothing was deleted to resolve this.

The attached `ContradictoryEvidence` has `source_kind = research_observation`, `source_id =
research_observation_fdddf9997a9f456b964ef04694f0f118` (the GPU-memory observation), and four
`evidence_refs`: the measured metric result `metric_result_6b3a191d07074ddebf21e2bf9ef27894`, the
adapter README, the SATRN control metric result `metric_result_11954d828f2742059dfd81a324347008`, and the
observation itself.

**Every record involved is real and committed.** No hypothetical second run was invented. The dispute is
between two measurements this repository already contained.

### `Supported` — not achieved, and why

**No finding in the committed artifact is `Supported`**, and this is not an omission.

Rule 3 above makes `Supported` require reproduction evidence naming an experiment run outside the
finding's scope. This repository has exactly two `ExperimentRun`s — one controlled, one end-to-end — and
each finding is scoped to the one that produced its evidence. There is no second run of either to
reproduce anything in. Forcing a `Supported` example would have meant either relaxing the rule for the
demonstration or fabricating a run, and the follow-up's own limitation ("no general ranking of methods
can be inferred") is the same point from the other side.

`Provisionally supported` on `transkribus_not_comparable` is the honest ceiling, and it is the strongest
of the five findings on purpose: it is a statement about a file's contents, not an estimate from a
sample, and it is independently asserted by an existing test. It still stops short of `Supported`,
because a claim scoped to one run's fixture has not been shown to hold anywhere else.

`test_the_committed_findings_carry_the_statuses_the_documentation_quotes` asserts the committed statuses
are exactly `["Candidate", "Candidate", "Candidate", "Disputed", "Provisionally supported"]` and that
`findings(review_status="Supported")` is empty. The mechanism for reaching `Supported` is proven
separately by `test_supported_accepts_reproduction_from_a_different_run`, alongside the two tests that
prove it cannot be reached dishonestly (`test_supported_requires_reproduction_evidence` and
`test_supported_refuses_evidence_from_the_findings_own_run`).

### `Superseded` — no real example exists, only a proven mechanism

**There is no real superseded finding in this repository, and there cannot yet be one.** Superseding a
scoped empirical claim means replacing it with a better-scoped successor, and a better-scoped successor
needs a second experiment run. There is one baseline run. Fabricating a second run to manufacture an
example is exactly what must not happen, so it did not.

The *mechanism* is proven instead, on synthetic findings, by
`test_superseding_preserves_both_findings_and_their_history`: it creates a one-run finding, creates a
two-run successor, supersedes the first pointing at the second, replays both off disk, and asserts the
superseded finding still exists with its full history, `superseded_by` names the successor, the
revision's `superseding_finding_id` matches — and that the successor is **still `Candidate`**, because
superseding an old finding does not promote the new one; the successor still has to be reviewed on its
own merits. `test_superseding_requires_a_pointer_to_the_superseding_finding` and
`test_a_superseded_finding_cannot_change_status_again` cover the two rules around it.

`test_no_finding_in_the_committed_artifact_is_superseded` asserts the absence, so if a second run ever
produces a superseding finding that test fails and this section gets updated with it. That is the point
of writing the gap as a test rather than only as prose.

## Reproducing the workflow

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/register_baseline_knowledge.py --fresh
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/htr/knowledge -q
```

Timestamps are the fixed `EXTRACTED_AT` constant rather than a clock read, so the artifact is
reproducible apart from freshly-minted entity ids (`observation_id`, `finding_id`, `revision_id`,
`contradiction_id`, `event_id`). Regenerating it changes those ids, and the id-quoting sections of these
three documents would need updating with them.

## Known gaps for the next phase

* **No `Supported` and no `Superseded` finding exists**, for the honest reasons above. Both need a
  second experiment run; neither needs new code.
* **Single-actor review.** One reviewer, the repository maintainer, on all four transitions. No blind
  dual review, no adjudication, no independent domain expert.
* **`ObservationReviewStatus` has no transition function.** All five observations are `Unreviewed`.
  Observations have no state machine because accepting one means only "yes, this is really what the
  records show" — but there is currently no producer that moves one to `Accepted` either, so the field
  is presently write-once at construction.
* **No `ResearchObservation` supersession path.** `docs/architecture/htr-event-model.md` §1's layer-10
  row says new observations supersede scope rather than overwriting, which the append-only bucket
  honours — but there is no `supersedes` pointer on the entity, so a scope-refining observation cannot
  yet name the one it refines.
* **`ResearchReportGenerated` still has no producer** (deliberate; see its docstring).
* **The knowledge stream is not exported.** No knowledge export format and no retention policy — Phase
  12, explicitly out of scope here.
* **No frontend.** The Research Knowledge ViewModel is Phase 10, and the observation → question →
  experiment-draft feedback loop is Phase 11. `HtrResearchStore` now has the query surface both will
  need (`research_observations`, `findings`, `findings_contradicting`), but nothing consumes it yet.
