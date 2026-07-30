# Swedish Historical HTR Baseline Comparison — executed run of 2026-07-30

**N=1 pipeline demonstration. Not a general-performance claim about any method.** Read
["What this run does and does not show"](#what-this-run-does-and-does-not-show) before quoting any
number below.

## Read this first: this is a fresh run, not a recovered one

This directory records **one real execution, performed on 2026-07-30**. It is not a migration,
recovery, or re-publication of an earlier run's data.

An earlier execution of this same experiment template did happen, during the HTR transformation
(Stage 12/13). **Its durable evidence does not exist.** The report files it is described as having
produced were never committed to git and are not on disk; `README.md`'s and
`docs/htr-repository-cleanup.md`'s descriptions of this directory as "a real executed example"
already in the repository were, at the time they were written, describing files that had not
survived. The full evidence for that finding — including why "lost to a filesystem glitch" cannot be
distinguished from "never survived its worktree copy-back" — is
[`docs/htr-telemetry-knowledge-gap-analysis.md` §0](../../htr-telemetry-knowledge-gap-analysis.md).

So there was nothing to migrate. The baseline was **re-executed exactly once**, this time through the
durable persistence path built in the preceding phase
([`docs/architecture/htr-telemetry.md`](../../architecture/htr-telemetry.md)). Every number on this
page is from that 2026-07-30 run and was measured by it.

**The difference that matters:** the prior run's results existed only in one Python process's memory
and in three export files. This run's results exist as an append-only, hash-chained telemetry log —
`htr_research_events.jsonl` — from which the entire entity graph is reconstructable without re-running
any model. `research_report.json` is *derived from* that log, not the other way round. If every file
in this directory except the log were deleted, the report could be regenerated; if the log were
deleted, it could not. That inversion is the whole point of the change.

## What is in this directory

| File | Role |
|---|---|
| `htr_research_events.jsonl` | **The source of truth.** 100 append-only telemetry events, 25 distinct kinds. |
| `htr_research_events.jsonl.chain.jsonl` | `HashChainAppender` tamper-evidence sidecar for the above. |
| `research_report.json` | The generated `ResearchReport`, sourced entirely from the log. |
| `research_report_summary.csv` | The same report, one row per experiment run (lossy, spreadsheet-friendly). |
| `metric_results.csv` | One row per (method, metric, value) — 26 rows, all from the log. |
| `htr_coarse_entities.json` | A **derived cache** (`WorkspaceStore` idiom) of the 4 coarse registration entities. Deleting it loses nothing. |
| `blobs/` | `FileTelemetrySink`'s content-addressed store for any `raw_output` over 4 KiB. **Legitimately empty** at line scale — nothing here exceeded the threshold — so git does not track it. |
| `htr_knowledge_events.jsonl` | A **second, separate** append-only log: 18 events recording the 5 `ResearchObservation`s and 5 candidate `ResearchFinding`s extracted *from* the run log above, added 2026-07-30. See below. |
| `htr_knowledge_events.jsonl.chain.jsonl` | `HashChainAppender` sidecar for the knowledge log. |

### Why the knowledge records are a separate file

`htr_research_events.jsonl` above is **exactly this run's 100 events and nothing else**, and the counts
and kind tables on this page describe it as such. The research knowledge extracted from it — observations
and findings, [`docs/architecture/htr-event-model.md`](../../architecture/htr-event-model.md) §1's layers
10–12 — is written to `htr_knowledge_events.jsonl` beside it rather than appended, so that description
stays true and so `tests/htr/persistence/test_real_baseline_reconstruction.py` keeps reading one run's
evidence. Same `FileTelemetrySink` class, same append-only guarantees, same hash-chain sidecar:
[`docs/architecture/htr-telemetry.md`](../../architecture/htr-telemetry.md) §7's "two streams, one
mechanism".

The knowledge events' `causation_id`s point **into** this run's log — an observation is caused by the
`MetricCalculated` or `ReliabilityIssueClassified` event that supplied its decisive evidence — and their
`correlation_id`s are these same two `ExperimentRun` ids. Replaying the two files concatenated
reconstructs one graph; `tests/htr/knowledge/test_baseline_knowledge.py` proves it, and also re-verifies
every id and every number those records quote against this log. Full documentation:
[`docs/research-observations.md`](../../research-observations.md),
[`docs/research-findings.md`](../../research-findings.md),
[`docs/knowledge-lifecycle.md`](../../knowledge-lifecycle.md).

Nothing on this page was recomputed to produce them: every figure in the knowledge records is quoted from
the `MetricCalculated` events already in this log.

Regenerate with real inference:

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/run_baseline_comparison.py --fresh
```

`--fresh` is required when a log already exists: the script refuses to silently append a second run's
events to a committed artifact. See that script's module docstring.

## Run identity

| | |
|---|---|
| Executed | `2026-07-30T00:39:30.777840+00:00` → `2026-07-30T00:39:48.109810+00:00` (17.3 s wall) |
| Application commit | `8a45f3201e65b2afbf0926b690669d1498349e96` |
| Controlled `ExperimentRun` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` (`is_end_to_end=false`) |
| End-to-end `ExperimentRun` | `experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73` (`is_end_to_end=true`) |
| `ReproducibilityManifest` | `reproducibility_manifest_16ce1fd2b1724c8c9e00cf9d2d230d9e` |
| Environment | Python 3.13.13, Windows-11-10.0.26200-SP0, torch 2.13.0+cu130, CUDA 13.0, transformers 4.49.0 |
| GPU | NVIDIA GeForce RTX 3070 (8191 MiB) — real CUDA inference, `cuda_available=True` |

## The hash-verified shared input crop

Both local recognizers read **byte-identical** input, verified by content hash rather than assumed
because the source code pointed them at the same path:

```
InputCrop.hash    = crop_e59f301d0763fab60e0141b8d984e88adc764b955c34e6b9490a316cf3a48261
InputCrop.crop_id = input_crop_8f36c0e77f914386ad917c09d5ef9947
```

This hash is **unchanged** from the value the pre-existing code and tests were written against — the
fixture (`tests/fixtures/htr/trolldomskommissionen_sample_line.jpg`, 2568×231 px, one line from
Riksarkivet's `trolldomskommissionen_lines`) and `InputCrop.compute_hash` are both untouched. It was
recomputed from the fixture's bytes during this run, not copied forward.

Both controlled `MethodRun`s reference that one `crop_id`, and
`tests/htr/persistence/test_real_baseline_reconstruction.py` re-asserts this **from the committed log
alone**, recomputing the hash from the fixture on disk each time it runs.

## Ground truth

```
bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt
```

Verbatim from the upstream published dataset (UTF-8; the special-character glyphs are the upstream
transcription convention for uncertain characters, not corruption). Single external annotation — no
blind dual-annotation or adjudication was performed for it.

## Measured results

### Controlled line-level comparison (hash-matched set)

| | SATRN | Florence-2 |
|---|---|---|
| Raw output | `till den 23 Januarii` | `</s><s>Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff</s>` |
| Parsed output | *(none — no separate parsed stage)* | `Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff` |
| Normalized output | `till den 23 Januarii` | `Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff` |
| CER (raw / normalized) | **0.7931 / 0.7931** | **0.4310 / 0.4310** |
| WER (raw / normalized) | **1.0000 / 1.0000** | **0.9000 / 0.9000** |
| Exact word accuracy | 0.0000 | 0.2000 |
| Char ins / del / subst | 0 / 38 / 8 | 4 / 2 / 19 |
| Word ins / del / subst | 0 / 6 / 4 | 1 / 0 / 8 |
| Reported confidence | 0.66661 (model's own scalar) | 0.24904 (**proxy**: `exp(sequences_scores)`) |
| Device | `cuda` | `cuda` |
| Execution time | 1131.28 ms | 646.73 ms |
| Peak GPU memory | 438.78 MiB | 1210.64 MiB |
| Reliability flags | `confidence_calibration_disagreement`, `omitted_text` | *(none)* |

The two confidence figures **are not comparable to each other** — see each adapter's README. CER/WER
raw and normalized coincide here because normalization (NFC + whitespace-run collapse, no case
folding) does not alter either string.

### End-to-end result (explicitly NOT in the controlled set)

Transkribus Swedish Lion I, page-level only, `input_crop_id = null`, on the separate end-to-end
`ExperimentRun`:

```
Anno 1712 den 3 Januarii holltes ting
medh allmogen aff Sochnen
NB dombook
```

Reported (vendor) confidence 0.86333. **No CER or WER exists for it, anywhere in the log** — not a
null written over a computed value, but a metric that was never computed. Its fixture
(`tests/fixtures/transkribus/sample_page.xml`, hand-authored, not genuine vendor output) transcribes
a *different, unrelated* Swedish court-record passage with no line-to-line correspondence to the
ground truth above; a CER between them would compare two unrelated texts rather than measure
recognition accuracy. See `baseline_template.py`'s `exclusion_criteria`.
`test_no_invalid_transkribus_cer_or_wer_exists_anywhere_in_the_log` asserts this against every
recorded metric *and* every `MetricCalculated` event, so the absence cannot be a projection artifact.

## Reproducibility against previously documented values — and one finding that needs stating plainly

**What reproduced exactly.** Both adapters' own READMEs document a measured CUDA run against this
same fixture from an earlier session. This run matches them on every value that is not a wall-clock
measurement:

| | Previously documented (adapter README) | This run |
|---|---|---|
| SATRN text | `till den 23 Januarii` | `till den 23 Januarii` ✔ |
| SATRN confidence | 0.6666 | 0.66661 ✔ |
| SATRN elapsed | 1.31 s | 1.13 s — differs, as timing legitimately does |
| Florence-2 text | `Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff` | identical ✔ |
| Florence-2 proxy confidence | 0.249 | 0.24904 ✔ |
| Florence-2 elapsed | 0.67 s | 0.65 s — differs |

So the *model outputs* are deterministic and confirmed. The task brief asked whether this run
reproduces the lost run's documented SATRN output "till den 23 Januarii" — **it does, exactly.**

**What was never documented, and is reported here for the first time: the CER those outputs actually
earn.** No CER/WER figure for this fixture survives from the earlier run, so the numbers in the table
above are newly measured and are compared against nothing. They are worth reading carefully:

> SATRN's output, `till den 23 Januarii`, has **no relationship to the content of the image it was
> given**. The line reads `bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt`; SATRN's
> CER is 0.7931 and its WER is 1.0000 — every word wrong. Florence-2's output,
> `Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff`, is visibly a (poor) attempt at the
> actual line, and scores CER 0.4310.

This is a real, reproducible property of the checkpoint on this input, not a wiring error: the crop
hash is verified identical for both methods, and Florence-2 — reading the same bytes — clearly
tracks the real text. SATRN's own reliability classification independently flagged the run
(`confidence_calibration_disagreement`: it reported 0.667 confidence on an almost entirely wrong
line; `omitted_text`: 38 character deletions).

**A related trap worth flagging.** Several *synthetic UI test fixtures* in this repository use the
string `till den 23 Januarii` as a stand-in **ground truth** (e.g.
`tests/presentation/_htr_fixtures.py::GROUND_TRUTH_LINE_0`). Those are ViewModel fixtures and are not
wrong for their purpose, but a reader moving between them and this experiment could easily conclude
SATRN transcribes this line correctly. It does not. The real ground truth for the real fixture is the
`bekiendt. Säger …` line above, and this is the first executed run to state SATRN's error rate
against it.

Nothing here should be read as SATRN being worse than Florence-2 in general: **N=1**, one line, one
checkpoint each.

## Durable telemetry: what was recorded and how to verify it

100 events, 25 kinds:

| Group | Kinds |
|---|---|
| Corpus registration | `ResearchProjectCreated`, `DatasetCreated`, `CollectionCreated` (+2 `DocumentRegistered`), `DatasetVersionCreated`, `PageRegistered` ×2, `RegionDetected` ×3, `TextLineDetected` ×4, `InputCropCreated`, `GroundTruthTextRecorded` |
| Experiment | `ExperimentCreated`, `ExperimentVersionCreated`, `ExperimentRunStarted` ×2, `ExperimentRunCompleted` ×2 |
| Method execution | `MethodRunStarted` ×3, `MethodRunCompleted` ×3, `EvidenceCreated` ×3 |
| Result stages | `RawMethodResultRecorded` ×2, `ParsedMethodResultRecorded` ×2, `NormalizedMethodResultRecorded` ×3 |
| Evaluation | `MetricDefinitionRegistered` ×30, `MetricCalculated` ×26, `ReliabilityIssueClassified` ×2, `MethodRunFailed` ×2 |
| Reproducibility | `ReproducibilityManifestRecorded` |

Asymmetries in those counts are all real and intentional: SATRN emits no
`ParsedMethodResultRecorded` because it has no separate parsed stage; Transkribus emits no
`RawMethodResultRecorded` because a PAGE XML export has no pre-parse model text; only SATRN was
flagged, hence 2 of each reliability event.

### Correlation and causation

`correlation_id` is the `ExperimentRun.id`. The two runs therefore carry **two different**
correlations, and no event belongs to both:

- 44 events correlated to the controlled run
- 13 events correlated to the end-to-end run
- 43 events with an honest `null` — the 30 metric-definition registrations (a module-import fact of
  the evaluation engine, not something this experiment caused) and the 13 pre-run corpus
  registrations, which genuinely precede any run

`causation_id` gives an explicit, walkable DAG. The full chain this follow-up asked to be
demonstrable, walked by pointer alone (no timestamps, no append order, no shared ids):

```
ExperimentVersionCreated → ExperimentRunStarted → MethodRunStarted
  → RawMethodResultRecorded → ParsedMethodResultRecorded → NormalizedMethodResultRecorded
  → MetricCalculated → ReproducibilityManifestRecorded → ExperimentRunCompleted
```

SATRN's chain is honestly one link shorter (`MethodRunStarted → Raw → Normalized → MetricCalculated`)
because it has no parsed stage. No event was invented to make the two methods' chains look alike.

### Reconstruction

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/htr/persistence/test_real_baseline_reconstruction.py -q
```

That suite copies this log to a temp directory, opens a brand-new `FileTelemetrySink` over it, calls
`HtrJournal().replay(...)`, and — from the replayed projection alone — asserts both controlled
`MethodRun`s are present, share one hash-verified `InputCrop`, carry their four distinct text stages,
carry the exact CER/WER above, carry their reliability classifications, and that the committed
`research_report.json` regenerates from it field for field. No GPU required. The live counterpart,
which runs real inference and *then* destroys and reconstructs, is
`tests/htr/experiment/test_baseline_durable_telemetry.py` (`-m real_model`).

## What this run does and does not show

**Does**: that the pipeline executes end to end for real — real GPU inference from two independent
local recognizers on byte-identical, hash-verified input; real CER/WER against real ground truth;
real reliability classification; real environment capture; and durable, replayable provenance for all
of it.

**Does not**: say anything about which method is better at Swedish historical HTR. N=1 line for the
controlled comparison, N=1 hand-authored page for the end-to-end result. See
`baseline_template.py`'s `BaselineExperimentDefinition.scope_caveats`.

## Honest gaps in this run

- **No blind dual review, no adjudication, no `CanonicalResult`.** Steps the design supports but this
  run did not exercise. The ground truth is a single external annotation.
- **No `TranscriptionConvention` record** was created for that ground truth.
- **No segmentation run was recorded**, deliberately: the controlled fixture is already a pre-cropped
  line, and Transkribus's segmentation ran externally. Emitting `SegmentationRunCompleted` for either
  would claim an execution that did not happen.
- **No `ExternalImport` record** for the Transkribus fixture, though the store and event kind support
  one.
- **Reliability classifications are recorded twice**, as `ReliabilityIssueClassified` (the correct
  evaluation-layer kind, first produced here) *and* as `MethodRunFailed` carrying the `FailureRecord`
  (which is what `HtrJournal` actually projects into a queryable surface). `MethodRunFailed` is
  documented as "a MethodRun that could not complete", and applying it to a reliability flag on a run
  whose `outcome` is `"succeeded"` stretches that. The stretch is `htr/evaluation/failures.py`'s own
  modelling choice — `classify_reliability` returns `FailureRecord`s — and predates this phase.
- **`blobs/` is empty**, so git does not track the directory; a future run with output over 4 KiB
  would populate it and it would then need committing alongside the log.
- **This is the only run, which caps what the research knowledge derived from it can claim.** No finding
  in `htr_knowledge_events.jsonl` is `Supported` or `Superseded`: both require a second experiment run to
  reproduce or refine a claim, and there is one. The honest ceiling reached is `Provisionally supported`.
  See [`docs/knowledge-lifecycle.md`](../../knowledge-lifecycle.md).
