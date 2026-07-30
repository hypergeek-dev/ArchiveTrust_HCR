# Research Findings

Layers 11–12 of [`docs/architecture/htr-event-model.md`](architecture/htr-event-model.md) §1, as built.
Read [`docs/research-observations.md`](research-observations.md) first — a finding rests on
observations and never re-derives their evidence. The state machine and the human-approval workflow are
in [`docs/knowledge-lifecycle.md`](knowledge-lifecycle.md).

A **`ResearchFinding`** is a scoped claim someone is (or is not yet) willing to stand behind. Layer 11
is `Candidate`; layer 12 is everything past it, and reaching layer 12 always requires a named human.

## Fields

`src/archivetrust/htr/knowledge/models.py`.

| Field | Type | Notes |
|---|---|---|
| `finding_id` | `str` | |
| `statement` | `str` | non-empty |
| `scope` | `ResearchScope` | the same structured type observations use — see the observations doc |
| `research_question` | `str \| None` | |
| `hypothesis_relationship` | `HypothesisRelationship \| None` | |
| `supporting_experiments` | `tuple[str, ...]` | |
| `supporting_observations` | `tuple[str, ...]` | **non-empty by construction** |
| `supporting_metrics` | `tuple[str, ...]` | may legitimately be empty |
| `affected_datasets` | `tuple[str, ...]` | |
| `affected_dataset_versions` | `tuple[str, ...]` | |
| `affected_document_types` | `tuple[str, ...]` | |
| `affected_handwriting_periods` | `tuple[str, ...]` | |
| `affected_methods` | `tuple[str, ...]` | |
| `affected_model_versions` | `tuple[str, ...]` | |
| `transcription_convention` | `str \| None` | **nullable, and `None` for every real finding here** |
| `confidence_level` | `FindingConfidence` | `low`/`moderate`/`high` |
| `limitations` | `tuple[str, ...]` | **non-empty by construction** |
| `contradictory_evidence` | `tuple[ContradictoryEvidence, ...]` | append-only, may be empty |
| `author` | `str` | non-empty |
| `reviewer` | `str \| None` | required for any status past `Candidate`/`Draft` |
| `review_status` | `FindingStatus` | the 8-member enum below |
| `creation_date` | `str` | |
| `revision_history` | `tuple[FindingRevision, ...]` | append-only, never rewritten |
| `superseded_by` | `str \| None` | required iff `review_status is Superseded` |

`HypothesisRelationship`: `supports`, `contradicts`, `refines`, `untested`, `no_hypothesis_asserted`.
The last member exists because the real baseline experiment's own `hypothesis` field says exactly that —
"No directional hypothesis is asserted about which method performs better" — and forcing that into
`supports`/`contradicts` would invent a hypothesis the experiment declined to make. All five real
findings use it.

### `FindingStatus`

`Draft`, `Candidate`, `Under review`, `Provisionally supported`, `Supported`, `Disputed`, `Superseded`,
`Rejected`. Values carry their display spelling (`"Under review"`, not `under_review`) because these are
read by humans in a research context and a second vocabulary to translate between adds nothing.

### Two fields that are non-empty by construction, and why

* **`supporting_observations`** — a finding always says which observations it rests on. This is the
  structural block on "an observation auto-promotes to a finding": producing one is always an explicit
  step that names its sources.
* **`limitations`** — a scoped empirical claim with zero stated limitations is either not scoped or not
  honest. On an N=1 run the limitations are most of the content.

### `transcription_convention` is nullable, and it is `None`

No versioned `TranscriptionConvention` record was ever created for the 2026-07-30 run's ground truth
(`docs/experiments/baseline-comparison/README.md`, "Honest gaps in this run"). A non-nullable field here
would have forced a fabricated convention id.
`test_every_finding_states_the_n_equals_1_limitation_and_no_transcription_convention` asserts it is
`None` on all five.

## A finding cannot be created already accepted

Two independent guards, because one would be a suggestion:

1. **`ResearchFinding.create(...)` refuses any `status` outside `{Draft, Candidate}`** and raises
   `ValueError: ResearchFinding.create refuses status 'Supported': …`.
2. **A model validator refuses it too**, so `ResearchFinding(review_status=SUPPORTED, …)` with an empty
   `revision_history` fails as well — bypassing `create` does not help.

The validator also enforces:

* any status past `Candidate`/`Draft` requires a non-`None` `reviewer`;
* `Superseded` requires `superseded_by`, and nothing else may set it;
* `Disputed` requires at least one `ContradictoryEvidence` entry — a dispute with no recorded
  contradiction is an unexplained status.

`tests/htr/knowledge/test_models.py` parametrises the first over all six forbidden statuses.

## The five real candidate findings

All created at `Candidate`, registered via `CandidateFindingCreated` (its first real producer), each
caused by the `ResearchObservationCreated` of the observation it rests on. Durable record:
`docs/experiments/baseline-comparison/htr_knowledge_events.jsonl`.

Every one shares five limitations, stated once in `baseline_knowledge.SHARED_LIMITATIONS` rather than
paraphrased five times and drifting:

1. N=1: one line crop, one experiment run, one checkpoint per method. No aggregate over more than this
   crop exists in this repository.
2. **No general ranking of methods can be inferred from this scope.**
3. The ground truth is a single external annotation from the upstream published dataset — no blind dual
   annotation and no adjudication was performed for it.
4. No versioned `TranscriptionConvention` record exists for this ground truth.
5. Metrics hold only under the normalization profile named in scope.

### 1. `florence2_lower_error_rates` — `Candidate`

`research_finding_0a8f60fd19444221bdfcd89c32ad58ef` · event
`event_1ab747f9dfc5429e90221cb283c3385b` · caused by `event_2b12ab73c92d4dbc8cb5de232d35f402` ·
`confidence_level` = `low`

> On the single controlled baseline line, Florence-2 produced lower CER and WER than SATRN (CER
> 0.43103448275862066 vs. 0.7931034482758621; WER 0.9 vs. 1.0), on input crop
> `input_crop_8f36c0e77f914386ad917c09d5ef9947` in experiment version
> `experiment_version_088916c194ad4aecbb5b8578cb405dc3`, experiment run
> `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`, at model revisions
> `994f47e8a0e8d77cb2e11528665efd07a855c3af` and `a40c7093232eaa47a83ce6469fc4abd033486bdc`
> respectively, under `evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run
> collapse, no case folding)`.

`supporting_observations` = the relative-result observation
`research_observation_9716f54778994a3fb868f8b40beea064` **and** the SATRN omitted-text observation
`research_observation_5ae3115e51a64d67816c6f6567b63fd6` (it cites SATRN's numbers, so it names the
observation that recorded them rather than re-deriving them).
`supporting_metrics` = the four real CER/WER metric-result ids.

Extra limitations: that it says nothing about which method is better at Swedish historical HTR — two
error rates on one crop are two measurements, not a ranking — and that the two methods' reported
confidences are not comparable to each other (SATRN's is the model's own scalar, Florence-2's a proxy
`exp(sequences_scores)`), so this finding rests on CER/WER only.

### 2. `satrn_omitted_reference_text` — `Candidate`

`research_finding_49e83783e393448abd9e6661b65a3cf9` · event
`event_7b2a8361174e4750a19fd218a4714371` · `confidence_level` = `moderate`

> On the single controlled baseline line, SATRN omitted a substantial portion of the reference text: 6
> of 10 reference words and 38 characters were deleted, on input crop
> `input_crop_8f36c0e77f914386ad917c09d5ef9947` in experiment run
> `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` at model revision
> `a40c7093232eaa47a83ce6469fc4abd033486bdc`, flagged `omitted_text` by
> `htr/evaluation/failures.py::classify_reliability`.

An extra limitation notes that the output has no lexical overlap with the line at all, so "omitted"
understates it — but `omitted_text` is the classification the evaluation engine actually recorded, and
the finding does not substitute a different word for it.

### 3. `satrn_confidence_not_aligned` — `Candidate`

`research_finding_0f77a4f5443e4e79b429d8076596c4eb` · event
`event_aa9d4394ff08491ab513465e1b75d3ed` · `confidence_level` = `moderate`

> The SATRN confidence value on this sample was not aligned with the observed transcription accuracy:
> reported confidence 0.6666051723062992 against measured CER 0.7931034482758621 and WER 1.0 on input
> crop `input_crop_8f36c0e77f914386ad917c09d5ef9947` in experiment run
> `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`, flagged
> `confidence_calibration_disagreement`.

Extra limitations: calibration is a distributional property and one sample cannot measure it; and the
0.35 disagreement threshold is `classify_reliability`'s own parameter — a different threshold would not
have flagged the run.

### 4. `transkribus_not_comparable` — `Candidate` → `Provisionally supported`

`research_finding_839388b4d62642359c535ea4e4b29226` · event
`event_2bd9573b0b164b469e635d361a7caf91` · `confidence_level` = `high`

> The current Transkribus fixture cannot be included in the controlled recognizer comparison because it
> does not correspond to the shared ground-truth line: method run
> `method_run_fa544c4adb9d44d89439c8e85d1b8565` ran page-level with `input_crop_id = null` on
> `tests/fixtures/transkribus/sample_page.xml`, whose content is an unrelated passage, and no CER or WER
> was computed for it anywhere in the run's log.

`supporting_metrics` = `()`. The evidence is the *absence* of a metric.

This is the one finding taken through the review workflow to a supported-family status — see
[`docs/knowledge-lifecycle.md`](knowledge-lifecycle.md) for the transitions, the verbatim reasoning, and
why `Provisionally supported` is the honest ceiling rather than `Supported`.

### 5. `florence2_environment_reproducible` — `Candidate` → `Disputed`

`research_finding_4b7b1aff39d3426f87ec5e4cc4b7cbe5` · event
`event_5d12b327abc84d9a899bc2fcc1f2f819` · `confidence_level` = `low`

> Florence-2's peak-GPU-memory measurement on the shared baseline line crop is reproducible across
> sessions: the adapter README documents ~3983 MB for this fixture, and the controlled run
> `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` re-measures the same quantity for method run
> `method_run_e3209c63bb90458ea8d7a0673dd5c706` at model revision
> `994f47e8a0e8d77cb2e11528665efd07a855c3af`.

**This is a fifth finding, beyond the four specified, and is labelled as one rather than passed off as
part of the set.** It exists because demonstrating a `Disputed` status honestly requires a genuinely
disputable claim, and fabricating a second experiment run to manufacture one is exactly what must not
happen. This claim is real — it is the reproducibility premise implicit in documenting a resource
measurement in a README at all — and it is genuinely contradicted by another committed record in the
same repository. One of its own limitations states that it makes that implicit premise explicit so it
can be examined rather than assumed.

## Contradiction preservation

`ContradictoryEvidence` is a real, populated list type, not a schema field:

| Field | Notes |
|---|---|
| `contradiction_id` | |
| `source_kind` | `research_finding`/`research_observation`/`metric_result`/`telemetry_event`/`external_document` |
| `source_id` | non-empty |
| `description` | non-empty |
| `recorded_by` | non-empty |
| `recorded_at` | |
| `evidence_refs` | `tuple[EvidenceReference, ...]` |

**Nothing is ever removed from it.** `transition_finding_status` only appends, and
`_assert_history_only_grew` raises if a prior entry's prefix changed. Disputing finding A does not
delete, edit, or downgrade the record that disputed it; both sides stay independently readable.

The relationship is queryable in **both** directions:

* `store.finding(a).contradictory_evidence` — what disputes A;
* `store.findings_contradicting(b_id)` — what B disputes.

Neither is derived from the other by deletion.

`tests/htr/knowledge/test_lifecycle.py::test_a_contradiction_leaves_both_findings_independently_readable`
is the follow-up's required proof and passes. It takes finding A all the way to `Supported`, creates
finding B whose evidence contradicts A, transitions A to `Disputed` referencing B, then **replays both
off disk** and asserts: both exist; A's statement and `supporting_observations` are unchanged; A's
revision history is `["Under review", "Provisionally supported", "Supported", "Disputed"]` — every state
it passed through, `Supported` included, still readable; the reproduction evidence that earned
`Supported` survives; B is untouched at its own status with its own history and is *not* retroactively
marked as contradicted; and both are listed independently by `store.findings()`.

## Durable persistence

`CandidateFindingCreated`, `FindingReviewed` and `FindingStatusChanged` all got their first real
producers on `DurableHtrResearchStore`:

* `register_candidate_finding(finding, *, caused_by, correlation_id)` — refuses a finding past
  `Draft`/`Candidate`, so the log can never contain a `CandidateFindingCreated` announcing an already
  `Supported` claim. A caller holding a legitimately-transitioned finding cannot re-register it as new
  and lose the review trail.
* `record_finding_transition(...)` — emits `FindingReviewed`, then `FindingStatusChanged` caused by it,
  then advances the projection. See the lifecycle doc.

`CandidateFindingCreated.finding` and `FindingStatusChanged.finding` are new optional typed fields
carrying the whole entity, including the full `revision_history` and every `ContradictoryEvidence`
entry. That is what lets replay rebuild a finding's complete lifecycle from the log alone, and why a
disputed finding's prior support is never lost: each state is its own durable event, and the latest one
still contains every revision that produced it.

`FindingReviewed` deliberately carries no finding — the reviewed *state* is the outcome, which
`FindingStatusChanged` carries; duplicating it would put two copies of one entity state in the log with
nothing saying which is authoritative. `HtrJournal` names it as an explicit projection no-op for that
reason.

`HtrResearchStore` gained a `_findings` bucket, `findings(review_status=…)`, `finding(id)` and
`findings_contradicting(id)`, plus `advance_finding` — the same projection-advancement pattern as
`advance_experiment_run`, and legitimate for the same reason: a transition is a later state of the same
entity, every intermediate state stays in the log as its own event, and a fresh replay reaches the same
state deterministically.

`ResearchReportGenerated` remains the one knowledge kind with no producer, for the reason already in its
docstring: announcing a report as a *published research artifact* is a different act from generating
one, and Article 33 forbids a projection emitting telemetry about itself.
