# SATRN output-repetition diagnostic

**Date:** 2026-07-30
**Scope:** bounded investigation of one Checkpoint 3 smoke-test signal. Not a benchmark, not a
redesign, not an accuracy evaluation.
**Verdict:** **genuine model behaviour, not an integration defect.** No code was changed. A
`ResearchObservation` was registered; SATRN **remains in scope** for the full benchmark.

---

## 1. The finding under investigation

The Checkpoint 3 smoke test ran SATRN and Florence-2 on 15 real, byte-identical line crops from 5
real pages of `dataset-rgb/` (the 766-page Swedish witchcraft-trial court-record corpus). SATRN
returned the identical string `staden den 27 dennes` for **6 of 15** crops — 7 distinct outputs over
15 distinct inputs. Florence-2 produced **15/15** distinct outputs on the exact same crops.

The user's stated concern was a **stale tensor, cached result, or reused temporary filename**, or
**subprocess communication reusing an earlier response**. That hypothesis class was the primary
target of this diagnostic and is addressed in §4.

Reproduction artifacts:

| Artifact | Path |
|---|---|
| Diagnostic runner | `scripts/run_satrn_repetition_diagnostic.py` |
| Machine-readable results | `docs/experiments/technical-reliability-screening/diagnostic/satrn_repetition_diagnostic.json` |
| Registration script | `scripts/register_satrn_repetition_observation.py` |
| Registered observation stream | `docs/experiments/technical-reliability-screening/diagnostic/htr_knowledge_events.jsonl` |
| Knowledge entity | `src/archivetrust/htr/knowledge/satrn_repetition_knowledge.py` |

---

## 2. The 15 crops — identity, dimensions, outputs

All SHA-256 digests below were computed by this diagnostic **from the PNG files themselves**, not
trusted from filenames or from the smoke test's own records. The smoke test's recorded `crop_...`
hashes were independently confirmed to be these same digests, and every file's on-disk size matches
the `byte_size` its `InputCropCreated` event recorded.

`**R**` = one of the 6 repeating `staden den 27 dennes` cases. `r2` = the second repeat group
(`talan att`, 4 crops). `-` = distinct-output control.

| # | crop_id | W×H | aspect | bytes | SHA-256 (PNG bytes) | SATRN output | confidence |
|---|---|---|---|---|---|---|---|
| 1 **R** | `input_crop_1d9a3a4f709a4472ac488189f154b5c4` | 1353×170 | 7.959 | 155338 | `3e840fc5bc49e708fa219cedb22b7f3aab36dd994fdc135f9ca0f328dc9da97f` | `staden den 27 dennes` | 0.5366043906658888 |
| 2 **R** | `input_crop_50dc8a1c4cbb493c9af29979b467f7dc` | 1303×153 | 8.516 | 157601 | `4a2061ba3756e73e9ea2c5ad4e4b791367f05db1b658e90f1a2b8938aee07f75` | `staden den 27 dennes` | 0.536318052560091 |
| 3 **R** | `input_crop_377f189a2c5f491b81ec4964709a5a40` | 1291×189 | 6.831 | 183861 | `021e75f0e849d80e8536fe963d3835e1f4e04cb452b0b658302da705fb7f0528` | `staden den 27 dennes` | 0.5346601057797671 |
| 4 - | `input_crop_4f034275f85f42d8bdbf6659998939f7` | 222×208 | 1.067 | 45309 | `7186b8cdb1cbc2c7e35d3627f81703b364a66b87e0bc1c6343b2a175577121c8` | `staden den 22 dennes` | 0.5230167284607887 |
| 5 - | `input_crop_0a37475cf52942bba20b31918530feea` | 2190×273 | 8.022 | 560254 | `3707f5d3e8de4a771aad7e92041fe10c447cf2f704d07a9dfc98bf3321a97c49` | `tala till den 22 dennes` | 0.5659244267836862 |
| 6 **R** | `input_crop_1af212a49c2145b98c36a2cd7f98049e` | 2131×257 | 8.292 | 560684 | `3952b1461a1233dacb9965d3d77c109a1456d258413e5dc29f79740760f4384e` | `staden den 27 dennes` | 0.5660757340490818 |
| 7 - | `input_crop_0c89a9e029fb443eb4c368b47ece51f8` | 129×162 | 0.796 | 16592 | `3c9ab5c936faf728265ff7337e29096c087c6f1fa6613d7f3ea39cd5634a1864` | `stånd till` | 0.5310136713087559 |
| 8 - | `input_crop_2aab3b7726bc4fa5a1531687ace3f10a` | 1815×242 | 7.500 | 286145 | `4127622bef544eb5e6c7633a848cb987b95ad6d4ffc65b143a4428bdfa405f5a` | `stånd till den till denna` | 0.55534986525774 |
| 9 - | `input_crop_9dc3a7f35d71438391d8f93c32efc49e` | 1828×197 | 9.279 | 271334 | `5f10ba5a9952cc4f128165ffcb3c4aaf54c0a5f44fa6d54795700fba5d9e869e` | `stånd till den 27 dennes` | 0.5613289028406143 |
| 10 **R** | `input_crop_c1c74279326c4fa89c030fa55bb592af` | 124×177 | 0.701 | 14831 | `24c3a196f02caf735978b76dc1c66c1a67fda8a69acf3ae83e2b9f388f9085bd` | `staden den 27 dennes` | 0.5236199896782636 |
| 11 r2 | `input_crop_c2253d155b8a4ae8acaccdb7587a2574` | 1820×190 | 9.579 | 145425 | `733691069a06284a1d72d4877c04b243d1befb7300d0df86890de78b7aea46b5` | `talan att` | 0.3814932828148206 |
| 12 r2 | `input_crop_91595c063b0b4842b8f61d664d71f99c` | 1809×194 | 9.325 | 162970 | `d300591d4a4fb3375b3145a8ad6da9ae3e92d37a523f03ceaf09e024db2ea468` | `talan att` | 0.3798314208785693 |
| 13 r2 | `input_crop_368077e5975c47bc9e4c1661442eab93` | 701×260 | 2.696 | 46624 | `ed628bbb6d682a4465ad23f51434db600b65ef62bded37bcc3bf8fc248a45705` | `talan att` | 0.38571450610955554 |
| 14 r2 | `input_crop_5ea401d261f345de8ec874e634e925f6` | 1860×308 | 6.039 | 149891 | `12f88319f5cae206f7611126288c0e53a30890fa50ab8384110c3274fbed9131` | `talan att` | 0.3832441709107823 |
| 15 **R** | `input_crop_fb0310f8ad5a481da6e2f9e2a19ac9ad` | 1877×222 | 8.455 | 136593 | `c1fe5627ee76eecf1edae8c6a01fa32fd4b54e5e349ef36d8ff56b3762399059` | `staden den 27 dennes` | 0.5265959832817316 |

**Crop file paths** follow the pattern
`docs/experiments/technical-reliability-screening/smoke-test/crops/page_<page_id>/text_line_<text_line_id>.png`;
the exact per-crop path is recorded in `satrn_repetition_diagnostic.json`'s `crop_manifest[].path`
and originates from each crop's `InputCropCreated` telemetry event.

### The crops are genuinely different images

| Check | Result |
|---|---|
| All 15 files present | yes |
| On-disk size == smoke test's recorded `byte_size` | 15/15 |
| Smoke test's `crop_...` hash == independently computed SHA-256 | 15/15 |
| **Distinct SHA-256 of PNG bytes** | **15/15** |
| **Distinct SHA-256 of decoded pixel data** | **15/15** |
| Any blank / near-blank crop (grayscale σ < 5) | none — σ ranges 29.8 … 84.3 |
| Distinct gray levels per crop | 227 … 256 |
| Colour mode | RGB for all 15 |

The decoded-pixel check matters independently of the byte check: two PNGs could differ in bytes
while decoding to the same image. They do not.

### Orientation and dimensions are valid, but wildly heterogeneous

Aspect ratios span **0.701 to 9.579**. Crops 7 and 10 (0.796, 0.701) are taller than they are wide —
they are not line-shaped at all, and crop 4 (1.067) is essentially square. This is a **segmentation
quality problem inherited from the Florence-2 line detector**, and it is recorded here as context,
not as this diagnostic's subject. It becomes relevant to the hypothesis in §7.

---

## 3. Checkpoint, revision and configuration — verified per invocation

The revision string was **logged back from the worker on every one of the 33 adapter invocations**,
not assumed from `get_metadata()`.

| Property | Value | How verified |
|---|---|---|
| `model_revision` | `a40c7093232eaa47a83ce6469fc4abd033486bdc` | worker-reported, 1 distinct value across all calls |
| `config_revision` | `a40c7093232eaa47a83ce6469fc4abd033486bdc` | worker-reported, 1 distinct value across all calls |
| Device | `cuda`, NVIDIA GeForce RTX 3070 | worker-reported per call |
| Test-pipeline resize | `Resize(scale=(400, 64), keep_ratio=False)` | read from the downloaded `config.py` |
| Train-pipeline resize | `Resize(scale=(400, 64), keep_ratio=False)` | read from the downloaded `config.py` |
| Normalization mean | `[123.675, 116.28, 103.53]` | `config.py` `data_preprocessor` |
| Normalization std | `[58.395, 57.12, 57.375]` | `config.py` `data_preprocessor` |
| Backbone input channels | 3 (RGB) | `config.py` `ShallowCNN` |
| Decoder `max_seq_len` | 100 | `config.py` `NRTRDecoder` |

**Preprocessing is exactly what the checkpoint documents**, and train and test pipelines are
identical to each other. The crops are RGB and the model takes 3-channel RGB input with ImageNet
normalization. **No setting was found to be inconsistent with the checkpoint's documented inference
requirements, so nothing was changed.** Decoding configuration is the checkpoint's own
`AttentionPostprocessor`, identical across all calls by construction (each call re-reads the same
pinned config).

---

## 4. The integration-defect hypotheses — the prime suspect, investigated

### 4.1 What the code actually does

`providers/satrn/facade.py::_SubprocessSatrnWorkerFacade.run_inference` calls `subprocess.run(...)`
once per invocation, with argv:

```
[<.venv-satrn python>, <_worker.py>, <image_path>, "--device", <d>, "--model-id", <id>, "--revision", <rev>]
```

`<image_path>` is **the caller's crop path, passed straight through**. The facade writes no
temporary file, creates no scratch copy, and holds no state between calls (`_timeout_seconds` is its
only instance attribute). `_worker.py` is a fresh interpreter each time: it loads the model, runs one
inference, prints one JSON object and exits. There is no warm worker, no pool and no queue —
`providers/satrn/README.md` explicitly documents the absence of a persistent warm process as a known
cost.

### 4.2 What was measured

Instrumentation wrapped `subprocess.run` **inside the facade module** to capture argv and the child
PID for every real launch, plus the SHA-256 of the bytes at `image_path` immediately before and
immediately after each call.

| Hypothesis | Check | Result |
|---|---|---|
| Reused subprocess / stale process state | distinct child PIDs vs. number of calls | **15 distinct PIDs for 15 calls** |
| **Reused temporary filename** | is argv's image path ever a temp file? | **never — argv path == source crop path, 15/15** |
| Temp-file collision between crops | are any two calls handed the same path? | **no — 15 distinct paths** |
| Stale / mutated input | crop SHA-256 before vs. after its own call | **unchanged, 15/15** |
| Wrong bytes handed to worker | crop SHA-256 at call time vs. manifest digest | **matches, 15/15** |
| Wrong checkpoint on some call | worker-reported revision per call | **1 distinct value, the pinned one** |
| **Cached / reused response** | distinct confidence values across 15 calls | **15 distinct** |
| **Cached response within the repeat group** | distinct confidences among the 6 identical strings | **6 distinct** |
| Cross-call model state (KV cache, hidden state) | fresh process per call ⇒ structurally impossible | confirmed by PID check |
| Parsing fallback to last-known-good | code path inspected; no fallback exists | see §4.4 |
| IPC response reuse | `subprocess.run(capture_output=True)` returns only that child's stdout | see §4.4 |

### 4.3 The single most decisive number

The six crops that returned the **identical string** returned **six different confidence scores**:

```
input_crop_1d9a3a4f709a4472ac488189f154b5c4   0.5366043906658888
input_crop_50dc8a1c4cbb493c9af29979b467f7dc   0.536318052560091
input_crop_377f189a2c5f491b81ec4964709a5a40   0.5346601057797671
input_crop_1af212a49c2145b98c36a2cd7f98049e   0.5660757340490818
input_crop_c1c74279326c4fa89c030fa55bb592af   0.5236199896782636
input_crop_fb0310f8ad5a481da6e2f9e2a19ac9ad   0.5265959832817316
```

A cached result, a stale tensor, a reused temp file or a mis-matched IPC response would necessarily
have carried **the same score too** — the score travels in the same JSON object as the text. Six
distinct scores mean six genuinely distinct forward passes over six genuinely distinct inputs, which
happened to decode to the same argmax string. Confidence and raw text are captured separately and
correctly for all six (raw worker stdout is retained per call in the diagnostic JSON).

### 4.4 Output parsing and IPC — inspected

`facade.py` parses `json.loads(stdout_lines[-1])` where `stdout_lines` is derived from *this call's*
`CompletedProcess.stdout`. There is no module-level buffer, no response cache, no "last known good"
variable, and no code path that substitutes a previous result: a non-`ok` payload returns
`RecognitionResult(text=None, ...)` and `adapter.py::build_failure_record` turns it into a
`FailureRecord`. `subprocess.run(capture_output=True)` reads that child's own pipes to EOF, so an
earlier response cannot be delivered to a later call.

**One real (non-causal) contract deviation was found and is recorded, not fixed.** `_worker.py`'s
docstring states "stdout is exactly one JSON object, all diagnostic output goes to stderr". In
reality `mmengine` writes **4 lines to stdout** before the JSON:

```
Loads checkpoint by local backend from path: C:\...\model.pth
07/30 20:18:37 - mmengine - WARNING - Failed to search registry with scope "mmocr" ...
07/30 20:18:37 - mmengine - WARNING - "FileClient" will be deprecated in future. ...
07/30 20:18:37 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" ...
{"ok": true, "text": "staden den 27 dennes", "score": 0.5366043906658888, ...}
```

Measured stdout line count was 5 on every call. The facade's `[-1]` parse handles this correctly
because `mmengine` emits nothing after inference completes. This is a documentation/robustness gap,
**not the cause of the repetition** and not in scope to change here; it is pinned by a regression
test (§8) so a future "parse stdout as a whole" refactor cannot break it silently.

### 4.5 Preprocessing independence

Each crop is preprocessed inside its own fresh interpreter by mmocr's own pipeline, constructed
per-process from the pinned config. There is no shared mutable state across calls because there is
no shared process. (One hygiene note: `_worker.py::_load_mmlabs` calls `cfg.dump(config_path)`,
rewriting the config file inside the Hugging Face cache snapshot on every call. It is idempotent and
sequential here, so it is harmless in this run, but it would be a race under concurrency. Recorded
for a later task; not touched.)

---

## 5. Reruns

Every pass re-ran the real model on the real crops.

| Pass | What it isolates | Result |
|---|---|---|
| **A** — all 15, original order, via `SatrnAdapter` | baseline reproduction | **All 15 outputs byte-identical to the smoke test**, all confidences bit-identical. Constancy 7/15 (ratio 0.4667), most-repeated `staden den 27 dennes` ×6 |
| **B** — the 6 repeated crops, same order, same host process | determinism | **All 6 repeated identically**, same confidences |
| **C** — the 6 repeated crops, **reversed order** | temp-file / order-dependent state | **The same 6 crops repeated the same text**, same confidences. Which crops repeat does **not** change with order |
| **D** — the 6 crops, each in its own **isolated top-level Python process** | any cross-call state in the host | **All 6 repeated identically**, same confidences |
| **E** — all 15 via `mmocr.apis.TextRecInferencer` **directly in `.venv-satrn`**, one process, one model load, bypassing `SatrnAdapter`/`facade.py`/`_worker.py` | ArchiveTrust integration entirely | **Identical 7/15 constancy, identical text for all 15, identical confidences to full float precision** |

Pass E was feasible quickly (the worker already uses this API, so no new infrastructure was built)
and is the decisive control: it deliberately inverts the adapter's fresh-process-per-crop pattern by
reusing **one** inferencer instance for all 15 crops. If cross-call state were the cause, pass E
would have made the repetition *worse* or *different*. It reproduced it exactly.

**Is the repeated output deterministic across reruns?** **Yes** — all five passes returned
byte-identical text and bit-identical confidence for every crop.

**Does isolating each crop in a fresh process change the result?** **No** — pass D is identical to
pass A, and pass A is identical to the original smoke test.

---

## 6. Control group — distinct-output crops

Same measurements, for the 5 crops whose output was unique in the sample (rows 4, 5, 7, 8, 9 above):
content hashes distinct, dimensions valid, confidences 0.5230 … 0.5659, per-call timing ~9.6–10.0 s
wall (dominated by subprocess start + model load; worker-reported inference time is ~1.1 s). Their
behaviour is indistinguishable from the repeated group on every integration dimension — same fresh
process, same pinned revision, same argv shape, same parse path. **The only thing that differs is the
decoded string.**

Note that even the "distinct" outputs are lexically adjacent to the repeated ones:
`staden den 22 dennes`, `tala till den 22 dennes`, `stånd till`, `stånd till den till denna`,
`stånd till den 27 dennes`. All 7 distinct outputs in the entire sample occupy one narrow
neighbourhood — `staden`/`stånd`/`tala(n)` + `den` + `22`/`27` + `dennes`/`att`. This is itself part
of what was observed, and it is what makes "the model is not reading the input" a poor description
and "the model is collapsing onto a phrase prior" a better one.

---

## 7. Classification and hypothesis

### Classification: **genuine SATRN reliability observation. No integration defect found.**

Every integration-defect hypothesis was tested and excluded with evidence. The behaviour survives
complete removal of ArchiveTrust's code from the path (pass E), is fully deterministic, and is
order-independent. **No code was changed, because no defect was found and no configuration was
inconsistent with the checkpoint's documented requirements.**

### Corroboration from an independent, already-committed run

`htr/knowledge/baseline_knowledge.py` records this same checkpoint returning
`till den 23 Januarii` for a line whose reference text is
`bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt` — a short, generic, date-shaped phrase
with no lexical overlap with the actual line, produced on a **different image** by a **different
pipeline**, months before any repetition question arose. The same behaviour was already in the
record.

### Unverified hypothesis (explicitly not established)

Held separately from every factual claim above, and stored in the observation's
`unverified_hypothesis` field rather than its `description`:

The checkpoint's own `test_pipeline` applies `Resize(scale=(400, 64), keep_ratio=False)`, forcing
every input to a 6.25 width/height ratio. This sample spans 0.701 … 9.579, so some crops are
stretched roughly 9× horizontally and others compressed. Combined with the model card's statement
that training used **binarized** line images while these crops are unbinarized RGB, this is a
plausible input-domain mismatch under which the decoder falls back on a high-frequency phrase prior
from its training distribution — which would also explain the single narrow lexical neighbourhood in
§6.

**This is a guess.** Confirming or refuting it requires a controlled preprocessing experiment that
has not been run. It is recorded so it can be tested later, not acted on now.

### What this does **not** say

This diagnostic makes **no** claim about SATRN's suitability for the corpus. The sample is 15 crops
from 5 pages under one configuration, several of those crops are not line-shaped (§2), and the
segmentation confound flagged at Checkpoint 1 still applies: the crops both methods read were
detected by a **Florence-2-family** detector, so one screened method's own model family controlled
the input the other was judged on.

**SATRN remains in scope for the full ~60-page benchmark.** The operational verdict belongs to the
`ScreeningPolicy` / `ScreeningDecision` entities that `design-audit.md` §3.3 records as **not yet
existing**, whose thresholds must be fixed *before* results are seen. Producing a verdict here would
be exactly the post-hoc rationalization that audit warns against.

---

## 8. What was recorded and what was added

### Registered observation

`research_observation_b359224960cf49219c2de4541dd714b3` — type `model_limitation`, observer
confidence `high`, review status `Unreviewed`, 15 typed evidence references, scope `sample_size` 15
(derived from the enumerated crop ids), both methods named with exact revisions, dataset version
pinned. Written to a **new** stream,
`docs/experiments/technical-reliability-screening/diagnostic/htr_knowledge_events.jsonl` (9 events),
and verified to round-trip through `HtrJournal().replay(...)`.

Type rationale: `MODEL_LIMITATION`, not `REPRODUCIBILITY_ANOMALY` (nothing here is irreproducible —
the opposite is true, and that reproducibility *is* the evidence) and not `ENVIRONMENT_ISSUE` (pass
E removes the environment-specific integration and the behaviour persists).

### Supersede-not-overwrite

The smoke-test evidence is **completely untouched**. Verified by size + mtime comparison across the
whole registration, and re-confirmed after the fact:

| File | Bytes | Unchanged |
|---|---|---|
| `smoke_test_events.jsonl` | 456192 | yes |
| `smoke_test_events.jsonl.chain.jsonl` | 96463 | yes |
| `smoke_test_results.json` | 52549 | yes |

The registration creates version 1 of a **new** `Experiment` for the diagnostic and supersedes
nothing — the smoke test never created an `Experiment`, so there is nothing to mutate.
`assert_experiment_mutable` is nonetheless called before the version is created, so the discipline is
enforced by the same mechanism the rest of the codebase uses. Re-running the registration script is
refused if the stream already exists.

### Regression tests added

No defect was found, so there is no defect-fix test. What was added instead pins the four properties
that **carried the exclusion**, so a future warm-worker optimization cannot silently reintroduce the
bug class this diagnostic ruled out:

`tests/providers/satrn/test_worker_call_isolation.py` (5 tests, no GPU/model/venv required)
- each `run_inference` launches its own child process
- the crop path is passed through verbatim and **no temporary file is created**
- each call returns its own worker's response (distinct responses stay distinct and in order)
- a failed call never falls back to the previous successful result
- mmengine-polluted stdout still parses (pins §4.4)

`tests/htr/knowledge/test_satrn_repetition_knowledge.py` (13 tests)
- the knowledge module has not drifted from the diagnostic artifact (full recomputation)
- the 6 repeated crops resolve against the **smoke test's own** results, independently of the diagnostic
- the repeat group carries one distinct confidence per crop
- all five passes reproduced the repetition; the reference path reproduced it without ArchiveTrust
- the observation is bounded, attributed to the pinned checkpoint, and **does not recommend excluding SATRN**
- the causal explanation stays quarantined in `unverified_hypothesis`
- the registered observation replays from its durable stream

### Files added

```
scripts/run_satrn_repetition_diagnostic.py
scripts/register_satrn_repetition_observation.py
src/archivetrust/htr/knowledge/satrn_repetition_knowledge.py
tests/htr/knowledge/test_satrn_repetition_knowledge.py
tests/providers/satrn/test_worker_call_isolation.py
docs/experiments/technical-reliability-screening/satrn-repetition-diagnostic.md
docs/experiments/technical-reliability-screening/diagnostic/satrn_repetition_diagnostic.json
docs/experiments/technical-reliability-screening/diagnostic/htr_knowledge_events.jsonl
docs/experiments/technical-reliability-screening/diagnostic/_reference_inference.py
```

**No existing file was modified.**
