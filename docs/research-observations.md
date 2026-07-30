# Research Observations

Layer 10 of [`docs/architecture/htr-event-model.md`](architecture/htr-event-model.md) §1, as built.
Companion documents: [`docs/research-findings.md`](research-findings.md) (layers 11–12) and
[`docs/knowledge-lifecycle.md`](knowledge-lifecycle.md) (the state machine and the human-approval
workflow).

A **`ResearchObservation`** records one scoped, evidence-backed fact about what durable records show.
It is not a finding and it is not a conclusion. The distinction is the whole reason this layer exists:

> a telemetry event is not automatically a research observation; a research observation is not
> automatically a finding; a finding is not automatically accepted knowledge.

## Where it lives

`src/archivetrust/htr/knowledge/models.py`, a new package sibling to `htr/corpus/` and
`htr/experiment/`. That placement follows this codebase's existing one-package-per-concern convention
(each owns its own `models.py`) and keeps a real seam visible: `htr/experiment/models.py` models what a
machine *did*, `htr/knowledge/models.py` models what a person is willing to *claim*, and the layering
above says the second never follows automatically from the first. Folding findings into the experiment
package would have put an entity whose defining property is *not* being auto-derived from execution
into the module that models execution.

The package imports nothing but `pydantic` and `domain.shared.ids`, which is the condition
[`docs/architecture/htr-telemetry.md`](architecture/htr-telemetry.md) §5 sets for
`domain/telemetry/events.py` to carry an entity as a **typed embedded object** rather than a `record`
dict. That is what lets `HtrJournal.replay` reconstruct an observation from the event log alone.

## Fields

| Field | Type | Notes |
|---|---|---|
| `observation_id` | `str` | `new_id("research_observation")` |
| `observation_type` | `ObservationType` | 14 members, closed enum — see below |
| `title` | `str` | non-empty |
| `description` | `str` | non-empty; **facts only** |
| `scope` | `ResearchScope` | structured, never free text — see below |
| `supporting_evidence` | `tuple[EvidenceReference, ...]` | **non-empty by construction**, typed refs |
| `source_experiment_id` | `str` | must equal `scope.experiment_id` |
| `source_experiment_run_id` | `str` | must appear in `scope.experiment_run_ids` |
| `affected_method` | `str \| None` | |
| `affected_model_version` | `str \| None` | |
| `affected_dataset_id` | `str \| None` | |
| `affected_dataset_version_id` | `str \| None` | |
| `affected_document_or_segment_ids` | `tuple[str, ...]` | |
| `author_or_source_component` | `str` | non-empty |
| `creation_timestamp` | `str` | passed in; the domain layer owns no clock |
| `tags` | `tuple[str, ...]` | |
| `observation_confidence` | `ObservationConfidence` | `low`/`moderate`/`high` |
| `review_status` | `ObservationReviewStatus` | `Unreviewed`/`Under review`/`Accepted`/`Rejected` |
| `unverified_hypothesis` | `str \| None` | **not a factual field** — see below |

### `observation_confidence`, not `confidence`

The field is named `observation_confidence` deliberately. It is the **observer's epistemic confidence
in the observation**, and it has nothing to do with `Evidence.provider_confidence`,
`RecognitionResult.confidence`, or the `RecognitionMetrics` family — all of which describe how sure a
*model* was about a transcription. A bare `confidence` attribute on an entity sitting one layer above
recognition output would be read as the model's number by anyone arriving from adapter code.
`ResearchObservation` has no attribute called `confidence` at all, and
`test_observation_confidence_is_named_unambiguously_and_is_not_a_recognition_confidence` asserts that
rather than leaving it to review.

It is ordinal with three values rather than a float: an observation's epistemic standing is not
measured, and a `0.72` would imply a calibration that does not exist.

### `unverified_hypothesis` — structurally separate from every fact

One field on this entity is explicitly *not* a fact. Everything in `description` is quoted from a
durable record; anything in `unverified_hypothesis` is what someone thinks might explain it, held apart
so an exporter, a UI, or a reader can tell the two apart without parsing prose. The GPU-memory
observation below is the reason it exists.

### `ObservationType`

`successful_recognition_behavior`, `recurring_recognition_failure`, `segmentation_problem`,
`handwriting_feature`, `document_layout_feature`, `model_limitation`, `confidence_anomaly`,
`performance_bottleneck`, `environment_issue`, `reviewer_observation`, `possible_hypothesis`,
`unexpected_method_disagreement`, `experiment_validity_boundary`, `reproducibility_anomaly`.

`experiment_validity_boundary` is load-bearing: it makes "this fixture is outside the controlled set" a
statement about the *experiment*, permanently un-confusable with a claim about the method that ran on
it.

## `ResearchScope`: why a scoped result cannot be read as a general one

Scope is a structured type, not a sentence. Four mechanisms make specificity structural rather than a
matter of authorial care, and none is a naming convention:

1. **`experiment_id`, `experiment_version_id`, and at least one `experiment_run_id` are required.**
   There is no way to describe a result without naming the configuration that produced it.
2. **`covered_unit_ids` must be non-empty and must enumerate real ids.** Placeholders (`"*"`, `"all"`,
   `""`, `"any"`) are refused by a validator. A scope can only cover units it can name.
3. **`sample_size` is a derived property, not a field.** It is `len(covered_unit_ids)`. `extra="forbid"`
   on the model means `ResearchScope(..., sample_size=500)` *raises* rather than being silently ignored
   (Pydantic's default would ignore it, letting a caller believe they had set it). So `N=1` appears in
   every rendering of a one-crop scope whether the author thought to mention it or not.
4. **Every named method carries its exact model revision.** `method_ids` and `model_version_ids` are
   validated to equal length and paired positionally, so a relative result cannot be scoped without
   stating *which checkpoints*.

Plus: `dataset_id` without `dataset_version_id` is refused — a `Dataset` is a moving target and only its
immutable `DatasetVersion` snapshot scopes a result.

`scope.describe()` renders the whole thing as one sentence. This is the real output for the
Florence-2/SATRN comparison observation:

```
1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947);
experiment experiment_fa667e22afe241b8b27f9aa3998edb0f;
version experiment_version_088916c194ad4aecbb5b8578cb405dc3;
run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8;
method(s) satrn@a40c7093232eaa47a83ce6469fc4abd033486bdc,
          florence2_htr@994f47e8a0e8d77cb2e11528665efd07a855c3af;
dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e;
normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse,
              no case folding);
ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
```

## `EvidenceReference`: typed pointers, not notes

`supporting_evidence` is a list of typed references, never free-text notes. `kind` and `reference_id`
are both required; `note` is optional annotation *on* a reference and can never stand in for one.

`EvidenceReferenceKind`: `telemetry_event`, `experiment_run`, `method_run`, `metric_result`,
`reliability_classification`, `evidence_record`, `input_crop`, `text_line`, `ground_truth_text`,
`research_observation`, `research_finding`, `reproducibility_manifest`, `external_document`.

`external_document` is the only member whose target is not in an event stream (an adapter README, a
fixture). It is named separately rather than passed off as a `telemetry_event`.

## The five real observations

Registered for real from the committed 2026-07-30 baseline run. Source:
`src/archivetrust/htr/knowledge/baseline_knowledge.py`; durable record:
`docs/experiments/baseline-comparison/htr_knowledge_events.jsonl` (18 events, 4 kinds).
`tests/htr/knowledge/test_baseline_knowledge.py` re-verifies every id and every quoted number against
`htr_research_events.jsonl`, so a drift fails a test.

Entity ids are minted per registration pass; the ids below are the ones in the committed artifact.

### 1. SATRN omitted text — `model_limitation`

`research_observation_5ae3115e51a64d67816c6f6567b63fd6`
· event `event_76c4a917c9044f05bc43af35edfb6062`
· `causation_id` = `event_aca5554290d149fe85d9332ec32e3802` (the `ReliabilityIssueClassified`)
· `correlation_id` = `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`
· `observation_confidence` = `high` · 13 evidence references

> SATRN omitted 6 of 10 reference words on input crop `input_crop_8f36c0e77f914386ad917c09d5ef9947`
> in experiment run `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`

Links: the crop (hash `crop_e59f301d0763fab60e0141b8d984e88adc764b955c34e6b9490a316cf3a48261`), the
raw output event `event_b53e520f7a8745c4ab7c32772e07a888` (`'till den 23 Januarii'`), the normalized
output event `event_5c24fe2d39da4dfd9e8feb1f0772816a`, the ground truth event
`event_cc9e4261bca841c69565b4e30552c977` (`'bekiendt. Säger och deth hon Minnes hoon Tuå gånger
waritt'`), CER `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` = `0.7931034482758621`, WER
`metric_result_16be0b552e5844c7b1a9f1db0fffd031` = `1.0`, character deletions
`metric_result_fc94177ea53e4a969b8a1fc5907aabc1` = `38.0`, word deletions
`metric_result_3b006aa6bce2425c8826cc9f260b7eda` = `6.0`, the `omitted_text` classification
`failure_record_c17788ac55564906921a9d3a393183b8`, and `Evidence`
`evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7`.

**Honest note about "parsed output".** There is no `ParsedMethodResultRecorded` for SATRN in the run's
log — the adapter has no separate parsed stage. The observation says so and links the two text stages
that exist instead of pointing at a stage that does not.
`test_satrn_really_has_no_parsed_stage_in_the_committed_log` asserts the absence against the log, so
if one ever appears the observation must be updated.

Typed `model_limitation`, not `recurring_recognition_failure`: exactly one occurrence is recorded, and
"recurring" would assert a recurrence nothing in this repository establishes.

### 2. SATRN confidence disagreement — `confidence_anomaly`

`research_observation_067cd6a4ae80439289aa8256efffa835`
· event `event_6f74b2b1d13d45a0b6dd93b5c5a9310b`
· `causation_id` = `event_b8cb85b68ae94a41811c737c85470e6e`
· `observation_confidence` = `high` · 10 evidence references
· tags include `this_run_and_this_sample_only`, `not_a_calibration_claim`

> SATRN reported confidence 0.6666 on a transcription with CER 0.7931 and WER 1.0 — one crop, one run

Links: `Evidence.provider_confidence` = `0.6666051723062992`, CER `0.7931034482758621`, WER `1.0`,
exact word accuracy `metric_result_d462f57778f84cbc97cacc3805aa56a0` = `0.0`, and the
`confidence_calibration_disagreement` classification
`failure_record_544c359478124a9cb401b57798641c14`, whose recorded detail is verbatim:

```
reported confidence 0.667 vs. measured similarity 0.207 against ground truth
(disagreement 0.460 > threshold 0.35)
```

Reading this as a general claim about SATRN's calibration is not a matter of the reader's care: the
`scope` type cannot express one. `covered_unit_ids` is one crop, `experiment_run_ids` is one run,
`method_ids`/`model_version_ids` name one checkpoint, and `sample_size` is `1` because it is computed
from the enumeration.

### 3. Florence-2 relative result — `unexpected_method_disagreement`

`research_observation_9716f54778994a3fb868f8b40beea064`
· event `event_2b12ab73c92d4dbc8cb5de232d35f402`
· `causation_id` = `event_ee1c4a31acef40849146a053a9120182` (Florence-2's CER `MetricCalculated`)
· `observation_confidence` = `high` · 13 evidence references

Title, verbatim:

> Florence-2 produced lower CER and WER than SATRN on one shared line crop
> (`input_crop_8f36c0e77f914386ad917c09d5ef9947`) in baseline experiment version
> `experiment_version_088916c194ad4aecbb5b8578cb405dc3` under configuration
> `experiment_version_088916c194ad4aecbb5b8578cb405dc3`

Florence-2 `metric_result_25048768c1024dbf8bf9b0e1c91f598e` CER `0.43103448275862066` and
`metric_result_ddbf925f834f4c1483f5708384e9a0b7` WER `0.9`, against SATRN's `0.7931034482758621` and
`1.0`, on the same hash-verified crop.

The word "better" appears nowhere in any title, description, statement, or limitation across all five
observations and all five findings.
`test_no_observation_or_finding_claims_one_method_is_better_than_another` checks that mechanically
against a list of ranking phrases, because that wording is exactly what creeps back during an edit.

### 4. GPU-memory variability — `reproducibility_anomaly`

`research_observation_fdddf9997a9f456b964ef04694f0f118`
· event `event_050ccacfb1024e0d94cd753c2dfd74da`
· `causation_id` = `event_2301532626da4c4da50acb91957ad9ea`
· `observation_confidence` = `moderate` · 7 evidence references
· **the only observation with a populated `unverified_hypothesis`**

Facts, in `description`: `src/archivetrust/providers/florence2_htr/README.md` documents ~3983 MB from
an earlier CUDA session; this run measured `1210.64111328125` MiB
(`metric_result_6b3a191d07074ddebf21e2bf9ef27894`). The earlier session has **no durable telemetry
record at all** — it predates this persistence — which is itself part of the observation. The SATRN
control did *not* diverge: ~439 MB documented against `438.78173828125` MiB measured
(`metric_result_11954d828f2742059dfd81a324347008`), so the divergence is specific to Florence-2 rather
than a general property of how this repository measures GPU memory. Both methods' text outputs
reproduced exactly.

Hypothesis, in `unverified_hypothesis` and nowhere else:

> UNVERIFIED. One candidate explanation is CUDA caching-allocator state: `torch.cuda
> .max_memory_allocated` (or reserved) reflects allocator history within a process … THIS PROJECT HAS
> NOT MEASURED THAT. No controlled experiment varying allocator state exists, the earlier session's
> torch version and process history were never captured, and this explanation is therefore a
> hypothesis to test, not a cause to cite.

`test_the_gpu_memory_explanation_is_held_as_an_unverified_hypothesis_not_a_fact` asserts that the word
"allocator" does not appear in `description` — because next to two measured figures in the same
paragraph, a guess is indistinguishable from a measurement.

### 5. Transkribus comparability boundary — `experiment_validity_boundary`

`research_observation_c4cac4cbd6274815a17d7123b8cf4482`
· event `event_f0551bb1d8174acd9191d12f82799e1a`
· `causation_id` = `event_ec65ed1b1a1746959b91965739a2a1a7`
· `correlation_id` = `experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73` (**the end-to-end run**)
· `observation_confidence` = `high` · 9 evidence references
· tags include `not_a_method_performance_observation`, `no_ground_truth_exists`

> The Transkribus fixture `tests/fixtures/transkribus/sample_page.xml` does not correspond to the
> controlled ground-truth line and was correctly excluded from CER/WER

`method_run_fa544c4adb9d44d89439c8e85d1b8565` ran page-level with `input_crop_id = null` on a fixture
whose first line is `'Anno 1712 den 3 Januarii holltes ting'` — a different, unrelated passage. No CER
and no WER exists for it anywhere in the log: not a null over a computed value, but a metric never
computed. The exclusion was declared in advance by the experiment's own `exclusion_criteria`.

`supporting_metrics` on the corresponding finding is **empty**, deliberately: the evidence is the
*absence* of a metric, and there is no id for a metric that was never computed.
`test_no_transkribus_cer_or_wer_is_quoted_anywhere_in_the_knowledge` asserts both that the committed log
has no error-rate metric owned by the Transkribus run and that this observation references no metric
result at all.

`scope.ground_truth_ref` is `None` here — the honest value, and the substance of the observation.

## Durable persistence

`DurableHtrResearchStore.register_research_observation(observation, *, caused_by, correlation_id)` is
the first real producer of `RESEARCH_OBSERVATION_CREATED`, one of the five kinds
[`docs/architecture/htr-telemetry.md`](architecture/htr-telemetry.md) §6 disclosed as schema-only.
`ResearchObservationCreated` gained an optional `observation: ResearchObservation | None` field —
optional with a default, exactly as `CanonicalResultCreated.canonical_result` was added, so every
pre-existing id-only construction still validates.

* **Correlation** defaults to the observation's own `source_experiment_run_id`, the same rule
  `register_experiment_run` applies: an observation extracted from a run's records belongs to that
  run's unit of work by definition.
* **Causation** is the run event that supplied the decisive evidence — the
  `ReliabilityIssueClassified` for the two reliability observations, the `MetricCalculated` for the
  comparison and GPU-memory ones, and for Transkribus the `NormalizedMethodResultRecorded` that is its
  last text stage (the absence of any `MetricCalculated` after it *is* the evidence). The mapping is a
  table in `baseline_knowledge.OBSERVATION_CAUSING_EVENT` rather than buried at a call site, because
  choosing one decisive event out of a dozen is a judgement and it should be reviewable.
* There is deliberately **no** logic that infers which event caused an observation. §1's hard rule is
  that no automatic step turns an event into an observation, so the caller must say.

`HtrJournal._apply` gained a branch reconstructing each observation from
`ResearchObservationCreated.observation`, and `HtrResearchStore` gained an `_observations` bucket with
`research_observations(...)` / `research_observation(id)` accessors. Observations are registered and
never advanced — a new observation supersedes an old one's *scope*, it does not overwrite it, which is
what §1's layer-10 row requires.

### Two streams, one mechanism

Knowledge events go to `htr_knowledge_events.jsonl`, not into the run's `htr_research_events.jsonl`.
That log is a committed research artifact whose README documents it as exactly one run's 100 events and
which `tests/htr/persistence/test_real_baseline_reconstruction.py` reads as that run's evidence;
appending would falsify its own description. Same `FileTelemetrySink` class, same append-only
guarantees, same hash-chain sidecar — the "two streams, one mechanism" precedent
[`htr-telemetry.md`](architecture/htr-telemetry.md) §7 already set.

`causation_id`s therefore point *across* files, which every `EvidenceReference` records in its `stream`
field. Replaying the two concatenated reconstructs one graph:
`test_every_run_scoped_evidence_reference_resolves_against_the_replayed_run` resolves 25+ references
from the knowledge log against entities only the run log carries, and
`test_a_causation_chain_walks_from_a_run_event_into_the_knowledge_stream` walks
`ReliabilityIssueClassified → ResearchObservationCreated → CandidateFindingCreated` by pointer alone.

## Regenerating

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/register_baseline_knowledge.py --fresh
```

No model runs and no metric is recomputed — every number is quoted from the committed run log.
`--fresh` is required when the log exists, for the same reason `run_baseline_comparison.py` requires
it: two sets of observations about one run in one file, with nothing saying which the documentation
quotes, is not a state the artifact should be able to reach.
