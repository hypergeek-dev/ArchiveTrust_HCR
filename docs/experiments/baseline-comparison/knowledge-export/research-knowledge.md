# Research Knowledge Export

* Export schema: `archivetrust.research_knowledge.v1`
* Generated at: `2026-07-30T03:30:00+00:00`
* Source: Replayed from the committed append-only logs docs/experiments/baseline-comparison/{htr_research_events, htr_knowledge_events, htr_knowledge_feedback_events}.jsonl (2026-07-30 baseline run)
* Contents: 5 finding(s), 5 observation(s), 1 research question(s)

> **Read the status on every finding below.** A `Candidate` or `Disputed` finding is not an established conclusion. Statuses are carried in each finding's heading, in a badge under it, and inside the quoted statement itself -- none of the three is a footnote, and none may be dropped when quoting.

## Status summary

| Status | Findings |
|---|---|
| **Candidate** | 3 |
| **Disputed** | 1 |
| **Provisionally supported** | 1 |

## Findings

### 1. Candidate — `research_finding_0a8f60fd19444221bdfcd89c32ad58ef`

**Status: Candidate**

> [Candidate] On the single controlled baseline line, Florence-2 produced lower CER and WER than SATRN (CER 0.43103448275862066 vs. 0.7931034482758621; WER 0.9 vs. 1.0), on input crop input_crop_8f36c0e77f914386ad917c09d5ef9947 in experiment version experiment_version_088916c194ad4aecbb5b8578cb405dc3, experiment run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8, at model revisions 994f47e8a0e8d77cb2e11528665efd07a855c3af and a40c7093232eaa47a83ce6469fc4abd033486bdc respectively, under evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding).

* **Status**: `Candidate`
* Confidence in the claim: `low`
* Hypothesis relationship: `no_hypothesis_asserted`
* Research question: How do SATRN (Riksarkivet), Florence-2 (fine-tuned OCR checkpoint) and Transkribus Swedish Lion I compare on Swedish historical handwriting recognition, under a controlled line-level comparison on byte-identical input crops and an end-to-end page-level comparison?
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) satrn@a40c7093232eaa47a83ce6469fc4abd033486bdc, florence2_htr@994f47e8a0e8d77cb2e11528665efd07a855c3af; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Sample size (derived, `len(covered_unit_ids)`): **N=1**
* Methods @ model versions: `florence2_htr` @ `994f47e8a0e8d77cb2e11528665efd07a855c3af`, `satrn` @ `a40c7093232eaa47a83ce6469fc4abd033486bdc`
* Datasets @ dataset versions: `dataset_014079548fb34741b8b3334bbb7587f7` @ `dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e`
* Transcription convention: **none recorded** — no versioned `TranscriptionConvention` exists for this ground truth
* Supporting experiments: `experiment_fa667e22afe241b8b27f9aa3998edb0f`
* Supporting metrics: `metric_result_25048768c1024dbf8bf9b0e1c91f598e`, `metric_result_ddbf925f834f4c1483f5708384e9a0b7`, `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a`, `metric_result_16be0b552e5844c7b1a9f1db0fffd031`
* Supporting observations: `research_observation_9716f54778994a3fb868f8b40beea064`, `research_observation_5ae3115e51a64d67816c6f6567b63fd6`

**Evidence references** (26, resolved via supporting observations)

| Kind | Reference | Stream | Via observation |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `method_run` | `method_run_e3209c63bb90458ea8d7a0673dd5c706` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `method_run` | `method_run_5bd842b5b850442daa0d701b63c3e545` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `metric_result` | `metric_result_25048768c1024dbf8bf9b0e1c91f598e` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `metric_result` | `metric_result_ddbf925f834f4c1483f5708384e9a0b7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `metric_result` | `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `metric_result` | `metric_result_16be0b552e5844c7b1a9f1db0fffd031` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `telemetry_event` | `event_f76ae5e0525e4d51ae9624a757135cb4` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `telemetry_event` | `event_5c24fe2d39da4dfd9e8feb1f0772816a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `evidence_record` | `evidence_6f80d565e3bf614b65066302960d35615f2e3df2e529987ba8f888fb28c76cbc` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `evidence_record` | `evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_9716f54778994a3fb868f8b40beea064` |
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `method_run` | `method_run_5bd842b5b850442daa0d701b63c3e545` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `telemetry_event` | `event_b53e520f7a8745c4ab7c32772e07a888` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `telemetry_event` | `event_5c24fe2d39da4dfd9e8feb1f0772816a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_16be0b552e5844c7b1a9f1db0fffd031` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_fc94177ea53e4a969b8a1fc5907aabc1` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_3b006aa6bce2425c8826cc9f260b7eda` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `reliability_classification` | `failure_record_c17788ac55564906921a9d3a393183b8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `telemetry_event` | `event_aca5554290d149fe85d9332ec32e3802` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `evidence_record` | `evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |

**Limitations**

* N=1: one line crop, one experiment run, one checkpoint per method. No aggregate over more than this crop exists in this repository.
* No general ranking of methods can be inferred from this scope.
* The ground truth is a single external annotation from the upstream published dataset -- no blind dual annotation and no adjudication was performed for it.
* No versioned TranscriptionConvention record exists for this ground truth, so the transcription rules the reference was authored under are not themselves versioned here.
* Metrics hold only under the normalization profile named in scope; a different profile could produce different values from the same texts.
* Says nothing about which method is better at Swedish historical HTR. Two error rates on one crop are two measurements, not a ranking.
* The two methods' reported confidence values are not comparable to each other: SATRN's is the model's own scalar, Florence-2's is a proxy exp(sequences_scores). This finding rests on CER/WER only, not on either confidence.

**Contradictions** (0)

*None recorded.*

**Revision history** (0, append-only)

*None — this finding is still at the status it was created at, which is why it has no reviewer.*

*End of finding `research_finding_0a8f60fd19444221bdfcd89c32ad58ef` — status `Candidate`.*

### 2. Candidate — `research_finding_0f77a4f5443e4e79b429d8076596c4eb`

**Status: Candidate**

> [Candidate] The SATRN confidence value on this sample was not aligned with the observed transcription accuracy: reported confidence 0.6666051723062992 against measured CER 0.7931034482758621 and WER 1.0 on input crop input_crop_8f36c0e77f914386ad917c09d5ef9947 in experiment run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8, flagged confidence_calibration_disagreement.

* **Status**: `Candidate`
* Confidence in the claim: `moderate`
* Hypothesis relationship: `no_hypothesis_asserted`
* Research question: How do SATRN (Riksarkivet), Florence-2 (fine-tuned OCR checkpoint) and Transkribus Swedish Lion I compare on Swedish historical handwriting recognition, under a controlled line-level comparison on byte-identical input crops and an end-to-end page-level comparison?
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) satrn@a40c7093232eaa47a83ce6469fc4abd033486bdc; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Sample size (derived, `len(covered_unit_ids)`): **N=1**
* Methods @ model versions: `satrn` @ `a40c7093232eaa47a83ce6469fc4abd033486bdc`
* Datasets @ dataset versions: `dataset_014079548fb34741b8b3334bbb7587f7` @ `dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e`
* Transcription convention: **none recorded** — no versioned `TranscriptionConvention` exists for this ground truth
* Supporting experiments: `experiment_fa667e22afe241b8b27f9aa3998edb0f`
* Supporting metrics: `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a`, `metric_result_16be0b552e5844c7b1a9f1db0fffd031`, `metric_result_d462f57778f84cbc97cacc3805aa56a0`
* Supporting observations: `research_observation_067cd6a4ae80439289aa8256efffa835`

**Evidence references** (10, resolved via supporting observations)

| Kind | Reference | Stream | Via observation |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `method_run` | `method_run_5bd842b5b850442daa0d701b63c3e545` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `evidence_record` | `evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `metric_result` | `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `metric_result` | `metric_result_16be0b552e5844c7b1a9f1db0fffd031` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `metric_result` | `metric_result_d462f57778f84cbc97cacc3805aa56a0` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `reliability_classification` | `failure_record_544c359478124a9cb401b57798641c14` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `telemetry_event` | `event_b8cb85b68ae94a41811c737c85470e6e` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_067cd6a4ae80439289aa8256efffa835` |

**Limitations**

* N=1: one line crop, one experiment run, one checkpoint per method. No aggregate over more than this crop exists in this repository.
* No general ranking of methods can be inferred from this scope.
* The ground truth is a single external annotation from the upstream published dataset -- no blind dual annotation and no adjudication was performed for it.
* No versioned TranscriptionConvention record exists for this ground truth, so the transcription rules the reference was authored under are not themselves versioned here.
* Metrics hold only under the normalization profile named in scope; a different profile could produce different values from the same texts.
* A statement about one confidence value on one sample. Calibration is a distributional property and one sample cannot measure it.
* The disagreement threshold (0.35) and the similarity measure are classify_reliability's own parameters; a different threshold would not have flagged the run.

**Contradictions** (0)

*None recorded.*

**Revision history** (0, append-only)

*None — this finding is still at the status it was created at, which is why it has no reviewer.*

*End of finding `research_finding_0f77a4f5443e4e79b429d8076596c4eb` — status `Candidate`.*

### 3. Candidate — `research_finding_49e83783e393448abd9e6661b65a3cf9`

**Status: Candidate**

> [Candidate] On the single controlled baseline line, SATRN omitted a substantial portion of the reference text: 6 of 10 reference words and 38 characters were deleted, on input crop input_crop_8f36c0e77f914386ad917c09d5ef9947 in experiment run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8 at model revision a40c7093232eaa47a83ce6469fc4abd033486bdc, flagged omitted_text by htr/evaluation/failures.py::classify_reliability.

* **Status**: `Candidate`
* Confidence in the claim: `moderate`
* Hypothesis relationship: `no_hypothesis_asserted`
* Research question: How do SATRN (Riksarkivet), Florence-2 (fine-tuned OCR checkpoint) and Transkribus Swedish Lion I compare on Swedish historical handwriting recognition, under a controlled line-level comparison on byte-identical input crops and an end-to-end page-level comparison?
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) satrn@a40c7093232eaa47a83ce6469fc4abd033486bdc; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Sample size (derived, `len(covered_unit_ids)`): **N=1**
* Methods @ model versions: `satrn` @ `a40c7093232eaa47a83ce6469fc4abd033486bdc`
* Datasets @ dataset versions: `dataset_014079548fb34741b8b3334bbb7587f7` @ `dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e`
* Transcription convention: **none recorded** — no versioned `TranscriptionConvention` exists for this ground truth
* Supporting experiments: `experiment_fa667e22afe241b8b27f9aa3998edb0f`
* Supporting metrics: `metric_result_fc94177ea53e4a969b8a1fc5907aabc1`, `metric_result_3b006aa6bce2425c8826cc9f260b7eda`, `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a`, `metric_result_16be0b552e5844c7b1a9f1db0fffd031`
* Supporting observations: `research_observation_5ae3115e51a64d67816c6f6567b63fd6`

**Evidence references** (13, resolved via supporting observations)

| Kind | Reference | Stream | Via observation |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `method_run` | `method_run_5bd842b5b850442daa0d701b63c3e545` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `telemetry_event` | `event_b53e520f7a8745c4ab7c32772e07a888` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `telemetry_event` | `event_5c24fe2d39da4dfd9e8feb1f0772816a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_16be0b552e5844c7b1a9f1db0fffd031` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_fc94177ea53e4a969b8a1fc5907aabc1` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `metric_result` | `metric_result_3b006aa6bce2425c8826cc9f260b7eda` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `reliability_classification` | `failure_record_c17788ac55564906921a9d3a393183b8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `telemetry_event` | `event_aca5554290d149fe85d9332ec32e3802` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |
| `evidence_record` | `evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_5ae3115e51a64d67816c6f6567b63fd6` |

**Limitations**

* N=1: one line crop, one experiment run, one checkpoint per method. No aggregate over more than this crop exists in this repository.
* No general ranking of methods can be inferred from this scope.
* The ground truth is a single external annotation from the upstream published dataset -- no blind dual annotation and no adjudication was performed for it.
* No versioned TranscriptionConvention record exists for this ground truth, so the transcription rules the reference was authored under are not themselves versioned here.
* Metrics hold only under the normalization profile named in scope; a different profile could produce different values from the same texts.
* One occurrence. 'Omission' here is the measured edit-distance category on this sample, not an established behaviour of the checkpoint.
* The output has no lexical overlap with the line, so 'omitted' understates it -- but 'omitted_text' is the classification the evaluation engine actually recorded, and this finding does not substitute a different word for it.

**Contradictions** (0)

*None recorded.*

**Revision history** (0, append-only)

*None — this finding is still at the status it was created at, which is why it has no reviewer.*

*End of finding `research_finding_49e83783e393448abd9e6661b65a3cf9` — status `Candidate`.*

### 4. Disputed — `research_finding_4b7b1aff39d3426f87ec5e4cc4b7cbe5`

**Status: Disputed**

> [Disputed] Florence-2's peak-GPU-memory measurement on the shared baseline line crop is reproducible across sessions: the adapter README documents ~3983 MB for this fixture, and the controlled run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8 re-measures the same quantity for method run method_run_e3209c63bb90458ea8d7a0673dd5c706 at model revision 994f47e8a0e8d77cb2e11528665efd07a855c3af.

* **Status**: `Disputed`
* Confidence in the claim: `low`
* Hypothesis relationship: `no_hypothesis_asserted`
* Research question: How do SATRN (Riksarkivet), Florence-2 (fine-tuned OCR checkpoint) and Transkribus Swedish Lion I compare on Swedish historical handwriting recognition, under a controlled line-level comparison on byte-identical input crops and an end-to-end page-level comparison?
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) florence2_htr@994f47e8a0e8d77cb2e11528665efd07a855c3af; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Sample size (derived, `len(covered_unit_ids)`): **N=1**
* Methods @ model versions: `florence2_htr` @ `994f47e8a0e8d77cb2e11528665efd07a855c3af`
* Datasets @ dataset versions: `dataset_014079548fb34741b8b3334bbb7587f7` @ `dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e`
* Transcription convention: **none recorded** — no versioned `TranscriptionConvention` exists for this ground truth
* Supporting experiments: `experiment_fa667e22afe241b8b27f9aa3998edb0f`
* Supporting metrics: `metric_result_6b3a191d07074ddebf21e2bf9ef27894`
* Supporting observations: `research_observation_fdddf9997a9f456b964ef04694f0f118`

**Evidence references** (7, resolved via supporting observations)

| Kind | Reference | Stream | Via observation |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_fdddf9997a9f456b964ef04694f0f118` |
| `method_run` | `method_run_e3209c63bb90458ea8d7a0673dd5c706` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_fdddf9997a9f456b964ef04694f0f118` |
| `metric_result` | `metric_result_6b3a191d07074ddebf21e2bf9ef27894` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_fdddf9997a9f456b964ef04694f0f118` |
| `telemetry_event` | `event_2301532626da4c4da50acb91957ad9ea` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_fdddf9997a9f456b964ef04694f0f118` |
| `evidence_record` | `evidence_6f80d565e3bf614b65066302960d35615f2e3df2e529987ba8f888fb28c76cbc` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_fdddf9997a9f456b964ef04694f0f118` |
| `external_document` | `src/archivetrust/providers/florence2_htr/README.md#peak-gpu-memory` | `(this stream)` | `research_observation_fdddf9997a9f456b964ef04694f0f118` |
| `metric_result` | `metric_result_11954d828f2742059dfd81a324347008` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_fdddf9997a9f456b964ef04694f0f118` |

**Limitations**

* N=1: one line crop, one experiment run, one checkpoint per method. No aggregate over more than this crop exists in this repository.
* No general ranking of methods can be inferred from this scope.
* The ground truth is a single external annotation from the upstream published dataset -- no blind dual annotation and no adjudication was performed for it.
* No versioned TranscriptionConvention record exists for this ground truth, so the transcription rules the reference was authored under are not themselves versioned here.
* Metrics hold only under the normalization profile named in scope; a different profile could produce different values from the same texts.
* The earlier session has no durable telemetry record, so its torch version, device state and process history are unknown and the two measurements cannot be compared under matched conditions.
* This is the reproducibility claim implicit in documenting a resource measurement in a README at all, made explicit here so it can be examined rather than assumed.

**Contradictions** (1)

* `contradiction_b3f70dd3b3d94687b560ee0f3e467a31` — research_observation `research_observation_fdddf9997a9f456b964ef04694f0f118`, recorded by hypergeek-dev at 2026-07-30T03:30:00+00:00
  * The two figures differ by a factor of ~3.3 and are therefore not reproductions of one another: 3983 MB documented in src/archivetrust/providers/florence2_htr/README.md#peak-gpu-memory against 1210.64111328125 MiB measured as metric result metric_result_6b3a191d07074ddebf21e2bf9ef27894 in run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8. The SATRN control reproduced to within 0.3 MB over the same pair of sessions (439 MB documented vs. 438.78173828125 MiB measured, metric result metric_result_11954d828f2742059dfd81a324347008), so the divergence cannot be attributed to the measurement method in general. Whether the earlier figure, the later figure, or both are environment-dependent is not established -- which is precisely why the finding is Disputed rather than Rejected.

**Revision history** (2, append-only)

| Revised at | From | To | Actor | Reasoning |
|---|---|---|---|---|
| 2026-07-30T03:30:00+00:00 | `Candidate` | `Under review` | hypergeek-dev | Reviewed against the two committed records this claim rests on: the Florence-2 adapter README's documented peak GPU memory from an earlier session, and this run's measured figure. They do not agree. Advancing to Under review so the disagreement is examined rather than left implicit in a README. |
| 2026-07-30T03:30:00+00:00 | `Under review` | `Disputed` | hypergeek-dev | Disputed. The two measurements differ by a factor of ~3.3 (3983 MB documented vs. 1210.64 MiB measured), so the reproducibility this finding claims is contradicted by the repository's own records. Disputed rather than Rejected because which figure is representative is genuinely unknown: the earlier session left no durable record of its torch version or device state, so there is no basis for declaring either measurement wrong. The SATRN control reproduced to within 0.3 MB across the same pair of sessions, which rules out 'this repository cannot measure GPU memory' as the explanation and is why the dispute is specific to Florence-2. The candidate finding, its supporting observation, and the contradiction all remain readable -- nothing was deleted to resolve this. |

*End of finding `research_finding_4b7b1aff39d3426f87ec5e4cc4b7cbe5` — status `Disputed`.*

### 5. Provisionally supported — `research_finding_839388b4d62642359c535ea4e4b29226`

**Status: Provisionally supported**

> [Provisionally supported] The current Transkribus fixture cannot be included in the controlled recognizer comparison because it does not correspond to the shared ground-truth line: method run method_run_fa544c4adb9d44d89439c8e85d1b8565 ran page-level with input_crop_id = null on tests/fixtures/transkribus/sample_page.xml, whose content is an unrelated passage, and no CER or WER was computed for it anywhere in the run's log.

* **Status**: `Provisionally supported`
* Confidence in the claim: `high`
* Hypothesis relationship: `no_hypothesis_asserted`
* Research question: How do SATRN (Riksarkivet), Florence-2 (fine-tuned OCR checkpoint) and Transkribus Swedish Lion I compare on Swedish historical handwriting recognition, under a controlled line-level comparison on byte-identical input crops and an end-to-end page-level comparison?
* Scope: 1 page (page_68aa5147bfd544d9816ecb5e9f3f0d74); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73; method(s) transkribus_swedish_lion_1@Swedish Lion I - v3; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding)
* Sample size (derived, `len(covered_unit_ids)`): **N=1**
* Methods @ model versions: `transkribus_swedish_lion_1` @ `Swedish Lion I - v3`
* Datasets @ dataset versions: `dataset_014079548fb34741b8b3334bbb7587f7` @ `dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e`
* Transcription convention: **none recorded** — no versioned `TranscriptionConvention` exists for this ground truth
* Supporting experiments: `experiment_fa667e22afe241b8b27f9aa3998edb0f`
* Supporting metrics: *(none — see limitations)*
* Supporting observations: `research_observation_c4cac4cbd6274815a17d7123b8cf4482`

**Evidence references** (9, resolved via supporting observations)

| Kind | Reference | Stream | Via observation |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `method_run` | `method_run_fa544c4adb9d44d89439c8e85d1b8565` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `telemetry_event` | `event_ee5448b140e14320b256028fb86fee03` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `telemetry_event` | `event_cd394580328a4189a005cf8aea842da5` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `telemetry_event` | `event_ec65ed1b1a1746959b91965739a2a1a7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `external_document` | `src/archivetrust/htr/experiment/baseline_template.py#exclusion_criteria` | `(this stream)` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |
| `external_document` | `tests/fixtures/transkribus/sample_page.xml` | `(this stream)` | `research_observation_c4cac4cbd6274815a17d7123b8cf4482` |

**Limitations**

* N=1: one line crop, one experiment run, one checkpoint per method. No aggregate over more than this crop exists in this repository.
* No general ranking of methods can be inferred from this scope.
* The ground truth is a single external annotation from the upstream published dataset -- no blind dual annotation and no adjudication was performed for it.
* No versioned TranscriptionConvention record exists for this ground truth, so the transcription rules the reference was authored under are not themselves versioned here.
* Metrics hold only under the normalization profile named in scope; a different profile could produce different values from the same texts.
* A statement about experiment validity, not about method performance. It says nothing about Transkribus's accuracy, which this run did not and could not measure.
* Scoped to the *current* fixture. A genuine vendor export of the controlled line would be a different fixture and this finding would not apply to it.
* supporting_metrics is deliberately empty: the evidence is the *absence* of a CER/WER metric, and there is no metric id for a metric that was never computed.

**Contradictions** (0)

*None recorded.*

**Revision history** (2, append-only)

| Revised at | From | To | Actor | Reasoning |
|---|---|---|---|---|
| 2026-07-30T03:30:00+00:00 | `Candidate` | `Under review` | hypergeek-dev | Reviewed against the committed log. The claim is verifiable by inspection rather than by measurement: the fixture's own content is an unrelated passage, the method run carries input_crop_id = null, and no MetricCalculated for CER or WER exists for it anywhere in the log (asserted by tests/htr/persistence/test_real_baseline_reconstruction.py::test_no_invalid_transkribus_cer_or_wer_exists_anywhere_in_the_log against every recorded metric *and* every MetricCalculated event). The exclusion was also declared in advance by the experiment's own exclusion_criteria, so it is not a post-hoc rationalisation of an inconvenient result. Advancing to Under review to record that a human has examined it. |
| 2026-07-30T03:30:00+00:00 | `Under review` | `Provisionally supported` | hypergeek-dev | Provisionally supported, and deliberately not Supported. The evidence in scope is as strong as this kind of claim gets -- the fixture's non-correspondence is a fact about a file, not an estimate, and it is independently asserted by a test -- but 'Supported' in this lifecycle requires reproduction in an experiment run outside the finding's own scope, and only one end-to-end run exists. The honest ceiling is therefore Provisionally supported. It would reach Supported the moment a second run over the same fixture recorded the same absence, which costs no GPU time; nobody has run it. |

*End of finding `research_finding_839388b4d62642359c535ea4e4b29226` — status `Provisionally supported`.*

## Observations

An observation claims only *what the records for its scope show*, never how a method behaves in general. `unverified_hypothesis`, where present, is a candidate explanation that has **not** been measured and is held separately from every factual field.

### O1. SATRN reported confidence 0.6666 on a transcription with CER 0.7931 and WER 1.0 -- one crop, one run — `confidence_anomaly` (review: Unreviewed)

`research_observation_067cd6a4ae80439289aa8256efffa835`

> For method run method_run_5bd842b5b850442daa0d701b63c3e545 in experiment run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8, on the single crop input_crop_8f36c0e77f914386ad917c09d5ef9947, SATRN's own reported confidence scalar was 0.6666051723062992 (recorded as Evidence.provider_confidence on evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7) while the measured accuracy of the same output against ground truth was CER 0.7931034482758621 and WER 1.0. classify_reliability flagged confidence_calibration_disagreement: reported confidence 0.667 vs. measured similarity 0.207 against ground truth (disagreement 0.460 > threshold 0.35). This is a statement about this run and this sample. It is not a claim about SATRN's confidence calibration, which one sample cannot measure -- and it must not be read as one: the scope covers exactly one line crop in exactly one experiment run at exactly one model revision, and no record in this repository extends it further.

* Observer confidence in the observation: `high`
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) satrn@a40c7093232eaa47a83ce6469fc4abd033486bdc; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Source: experiment `experiment_fa667e22afe241b8b27f9aa3998edb0f`, run `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`
* Method @ model version: `satrn` @ `a40c7093232eaa47a83ce6469fc4abd033486bdc`
* Tags: `confidence_calibration_disagreement`, `n=1`, `this_run_and_this_sample_only`, `not_a_calibration_claim`
* Recorded by: htr.knowledge.baseline_knowledge at 2026-07-30T03:30:00+00:00

**Supporting evidence** (10 typed references)

| Kind | Reference | Stream | Note |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `method_run` | `method_run_5bd842b5b850442daa0d701b63c3e545` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `evidence_record` | `evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | provider_confidence = 0.6666051723062992 (the model's own scalar) |
| `metric_result` | `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | character_error_rate_normalized = 0.7931034482758621 |
| `metric_result` | `metric_result_16be0b552e5844c7b1a9f1db0fffd031` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | word_error_rate_normalized = 1.0 |
| `metric_result` | `metric_result_d462f57778f84cbc97cacc3805aa56a0` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | exact_word_accuracy = 0.0 |
| `reliability_classification` | `failure_record_544c359478124a9cb401b57798641c14` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | confidence_calibration_disagreement -- reported confidence 0.667 vs. measured similarity 0.207 against ground truth (disagreement 0.460 > threshold 0.35) |
| `telemetry_event` | `event_b8cb85b68ae94a41811c737c85470e6e` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | ReliabilityIssueClassified carrying the confidence_calibration_disagreement classification |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |

### O2. SATRN omitted 6 of 10 reference words on input crop input_crop_8f36c0e77f914386ad917c09d5ef9947 in experiment run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8 — `model_limitation` (review: Unreviewed)

`research_observation_5ae3115e51a64d67816c6f6567b63fd6`

> On the one hash-verified shared line crop (input_crop_8f36c0e77f914386ad917c09d5ef9947, content hash crop_e59f301d0763fab60e0141b8d984e88adc764b955c34e6b9490a316cf3a48261), SATRN at model revision a40c7093232eaa47a83ce6469fc4abd033486bdc produced 'till den 23 Januarii' against the reference 'bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt'. Measured against that reference under evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding): CER 0.7931034482758621, WER 1.0 (every word wrong), exact word accuracy 0.0, 38 character deletions and 6 word deletions. htr/evaluation/failures.py::classify_reliability independently flagged the run omitted_text: '6 deleted words out of 10 reference words'. The output is not a partial or degraded reading of the line -- it has no lexical overlap with it. Recorded for this run and this sample only; no recurrence is established by any record in this repository. Note that SATRN emitted no ParsedMethodResultRecorded event: the adapter has no separate parsed stage, so the raw and normalized stages are the only two text stages that exist to link.

* Observer confidence in the observation: `high`
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) satrn@a40c7093232eaa47a83ce6469fc4abd033486bdc; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Source: experiment `experiment_fa667e22afe241b8b27f9aa3998edb0f`, run `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`
* Method @ model version: `satrn` @ `a40c7093232eaa47a83ce6469fc4abd033486bdc`
* Tags: `omitted_text`, `n=1`, `single_occurrence`, `controlled_comparison`, `swedish_17c`
* Recorded by: htr.knowledge.baseline_knowledge at 2026-07-30T03:30:00+00:00

**Supporting evidence** (13 typed references)

| Kind | Reference | Stream | Note |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `method_run` | `method_run_5bd842b5b850442daa0d701b63c3e545` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | content hash crop_e59f301d0763fab60e0141b8d984e88adc764b955c34e6b9490a316cf3a48261 |
| `telemetry_event` | `event_b53e520f7a8745c4ab7c32772e07a888` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | RawMethodResultRecorded: 'till den 23 Januarii' |
| `telemetry_event` | `event_5c24fe2d39da4dfd9e8feb1f0772816a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | NormalizedMethodResultRecorded -- SATRN's only other text stage; there is no ParsedMethodResultRecorded for this method run |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | GroundTruthTextRecorded for text_line_aa805ef122b54d1e961e0f6ec11e266e: 'bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt' |
| `metric_result` | `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | character_error_rate_normalized = 0.7931034482758621 |
| `metric_result` | `metric_result_16be0b552e5844c7b1a9f1db0fffd031` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | word_error_rate_normalized = 1.0 |
| `metric_result` | `metric_result_fc94177ea53e4a969b8a1fc5907aabc1` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | character_deletions = 38.0 |
| `metric_result` | `metric_result_3b006aa6bce2425c8826cc9f260b7eda` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | word_deletions = 6.0 |
| `reliability_classification` | `failure_record_c17788ac55564906921a9d3a393183b8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | omitted_text -- 6 deleted words out of 10 reference words |
| `telemetry_event` | `event_aca5554290d149fe85d9332ec32e3802` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | ReliabilityIssueClassified carrying the omitted_text classification |
| `evidence_record` | `evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |

### O3. Florence-2 produced lower CER and WER than SATRN on one shared line crop (input_crop_8f36c0e77f914386ad917c09d5ef9947) in baseline experiment version experiment_version_088916c194ad4aecbb5b8578cb405dc3 under configuration experiment_version_088916c194ad4aecbb5b8578cb405dc3 — `unexpected_method_disagreement` (review: Unreviewed)

`research_observation_9716f54778994a3fb868f8b40beea064`

> On the one shared, byte-identical, hash-verified line crop input_crop_8f36c0e77f914386ad917c09d5ef9947 (content hash crop_e59f301d0763fab60e0141b8d984e88adc764b955c34e6b9490a316cf3a48261), in experiment run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8 of experiment version experiment_version_088916c194ad4aecbb5b8578cb405dc3, against the reference 'bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt' and under evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding): Florence-2 at model revision 994f47e8a0e8d77cb2e11528665efd07a855c3af produced 'Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff' with CER 0.43103448275862066 and WER 0.9, while SATRN at model revision a40c7093232eaa47a83ce6469fc4abd033486bdc produced 'till den 23 Januarii' with CER 0.7931034482758621 and WER 1.0. Both values are lower for Florence-2 on this crop under these exact model revisions, this exact ground truth, this exact experiment configuration and these exact normalization rules. Change any one of those five and this observation says nothing about the result. It is a measurement on one sample, not a ranking of the two methods, and no aggregate over more than this one crop exists in this repository.

* Observer confidence in the observation: `high`
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) satrn@a40c7093232eaa47a83ce6469fc4abd033486bdc, florence2_htr@994f47e8a0e8d77cb2e11528665efd07a855c3af; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Source: experiment `experiment_fa667e22afe241b8b27f9aa3998edb0f`, run `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`
* Tags: `n=1`, `one_shared_line_crop`, `not_a_method_ranking`, `controlled_comparison`, `hash_verified_input`
* Recorded by: htr.knowledge.baseline_knowledge at 2026-07-30T03:30:00+00:00

**Supporting evidence** (13 typed references)

| Kind | Reference | Stream | Note |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | one shared crop, content hash crop_e59f301d0763fab60e0141b8d984e88adc764b955c34e6b9490a316cf3a48261 verified identical for both method runs |
| `method_run` | `method_run_e3209c63bb90458ea8d7a0673dd5c706` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `method_run` | `method_run_5bd842b5b850442daa0d701b63c3e545` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `metric_result` | `metric_result_25048768c1024dbf8bf9b0e1c91f598e` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | Florence-2 character_error_rate_normalized = 0.43103448275862066 |
| `metric_result` | `metric_result_ddbf925f834f4c1483f5708384e9a0b7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | Florence-2 word_error_rate_normalized = 0.9 |
| `metric_result` | `metric_result_40f23f7723b54c6f92ab4d2d14c6b85a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | SATRN character_error_rate_normalized = 0.7931034482758621 |
| `metric_result` | `metric_result_16be0b552e5844c7b1a9f1db0fffd031` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | SATRN word_error_rate_normalized = 1.0 |
| `telemetry_event` | `event_f76ae5e0525e4d51ae9624a757135cb4` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | ParsedMethodResultRecorded: 'Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff' |
| `telemetry_event` | `event_5c24fe2d39da4dfd9e8feb1f0772816a` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | NormalizedMethodResultRecorded: 'till den 23 Januarii' |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `evidence_record` | `evidence_6f80d565e3bf614b65066302960d35615f2e3df2e529987ba8f888fb28c76cbc` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `evidence_record` | `evidence_82cd7c1a5005828f106c305d381ab97ba735941e56219ff47abdb345697b1aa7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |

### O4. The Transkribus fixture tests/fixtures/transkribus/sample_page.xml does not correspond to the controlled ground-truth line and was correctly excluded from CER/WER — `experiment_validity_boundary` (review: Unreviewed)

`research_observation_c4cac4cbd6274815a17d7123b8cf4482`

> Method run method_run_fa544c4adb9d44d89439c8e85d1b8565 (Transkribus Swedish Lion I - v3) ran on the separate end-to-end experiment run experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73 with input_crop_id = null, page-level only, parsing tests/fixtures/transkribus/sample_page.xml. That fixture transcribes a different, unrelated Swedish court-record passage -- its first line is 'Anno 1712 den 3 Januarii holltes ting' -- with no established line-to-line correspondence to the controlled reference 'bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt'. No CER and no WER exists for it anywhere in the run's log: not a null written over a computed value, but a metric that was never computed, because a CER between two unrelated texts would measure nothing. The exclusion is declared in advance by the experiment's own exclusion_criteria (src/archivetrust/htr/experiment/baseline_template.py#exclusion_criteria) rather than applied after seeing results. THIS IS NOT A METHOD-PERFORMANCE OBSERVATION. It records where the controlled comparison's boundary lies. Nothing here says anything about Transkribus's recognition accuracy, and nothing could: the fixture it read has no ground truth at all, and is itself hand-authored rather than genuine vendor output (tests/fixtures/transkribus/README.md).

* Observer confidence in the observation: `high`
* Scope: 1 page (page_68aa5147bfd544d9816ecb5e9f3f0d74); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73; method(s) transkribus_swedish_lion_1@Swedish Lion I - v3; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding)
* Source: experiment `experiment_fa667e22afe241b8b27f9aa3998edb0f`, run `experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73`
* Method @ model version: `transkribus_swedish_lion_1` @ `Swedish Lion I - v3`
* Tags: `experiment_validity`, `comparison_boundary`, `not_a_method_performance_observation`, `excluded_from_controlled_set`, `no_ground_truth_exists`
* Recorded by: htr.knowledge.baseline_knowledge at 2026-07-30T03:30:00+00:00

**Supporting evidence** (9 typed references)

| Kind | Reference | Stream | Note |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ef72f1ca494a8e8eebf0b4fde33d73` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | is_end_to_end = true |
| `method_run` | `method_run_fa544c4adb9d44d89439c8e85d1b8565` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | input_crop_id = null -- page-level, never bound to the shared crop |
| `telemetry_event` | `event_ee5448b140e14320b256028fb86fee03` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | MethodRunStarted carrying the MethodRun with input_crop_id = null |
| `telemetry_event` | `event_cd394580328a4189a005cf8aea842da5` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | ParsedMethodResultRecorded, first line 'Anno 1712 den 3 Januarii holltes ting' |
| `telemetry_event` | `event_ec65ed1b1a1746959b91965739a2a1a7` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | NormalizedMethodResultRecorded -- the last text stage; no MetricCalculated follows it for CER or WER |
| `ground_truth_text` | `event_cc9e4261bca841c69565b4e30552c977` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | the controlled reference this fixture does not correspond to |
| `input_crop` | `input_crop_8f36c0e77f914386ad917c09d5ef9947` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | the controlled crop, not read by this method run |
| `external_document` | `src/archivetrust/htr/experiment/baseline_template.py#exclusion_criteria` | `(this stream)` | the pre-declared exclusion criterion |
| `external_document` | `tests/fixtures/transkribus/sample_page.xml` | `(this stream)` | the fixture itself -- hand-authored, not genuine vendor output |

### O5. Florence-2 peak GPU memory on the shared baseline crop differs by ~3.3x between two documented sessions (3983 MB documented vs. 1210.64 MiB measured) — `reproducibility_anomaly` (review: Unreviewed)

`research_observation_fdddf9997a9f456b964ef04694f0f118`

> Two committed records in this repository state different peak-GPU-memory figures for Florence-2 on the same fixture. src/archivetrust/providers/florence2_htr/README.md documents ~3983 MB from an earlier CUDA session. The 2026-07-30 run (experiment run experiment_run_30ba2bcc18a14c06af0f9ca291442cf8, method run method_run_e3209c63bb90458ea8d7a0673dd5c706) measured 1210.64111328125 MiB, recorded as metric result metric_result_6b3a191d07074ddebf21e2bf9ef27894 and as Evidence.gpu_memory_mb on evidence_6f80d565e3bf614b65066302960d35615f2e3df2e529987ba8f888fb28c76cbc. Both figures are real measurements; the earlier session has no durable telemetry record at all, which is itself part of this observation -- it predates the persistence this project now has, so there is no way to compare the two runs' environments beyond what each wrote down. The same comparison for SATRN did NOT diverge: its README documents ~439 MB and this run measured 438.78173828125 MiB (metric result metric_result_11954d828f2742059dfd81a324347008), so the divergence is specific to Florence-2 rather than a general property of how this repository measures GPU memory. Both methods' text outputs reproduced exactly across the two sessions, so this is an operational/reproducibility observation about resource measurement, not about recognition output.

* Observer confidence in the observation: `moderate`
* Scope: 1 line_crop (input_crop_8f36c0e77f914386ad917c09d5ef9947); experiment experiment_fa667e22afe241b8b27f9aa3998edb0f; version experiment_version_088916c194ad4aecbb5b8578cb405dc3; run(s) experiment_run_30ba2bcc18a14c06af0f9ca291442cf8; method(s) florence2_htr@994f47e8a0e8d77cb2e11528665efd07a855c3af; dataset version dataset_version_35afef34f88d4d9a8a8bc4ed0295dc6e; normalization evaluation.metrics.normalize_text v1 (Unicode NFC composition + whitespace-run collapse, no case folding); ground truth text_line_aa805ef122b54d1e961e0f6ec11e266e
* Source: experiment `experiment_fa667e22afe241b8b27f9aa3998edb0f`, run `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8`
* Method @ model version: `florence2_htr` @ `994f47e8a0e8d77cb2e11528665efd07a855c3af`
* Tags: `reproducibility`, `operational`, `gpu_memory`, `environment_not_fully_captured`, `outputs_reproduced_exactly`
* Recorded by: htr.knowledge.baseline_knowledge at 2026-07-30T03:30:00+00:00

**UNVERIFIED HYPOTHESIS — not a measurement, not part of this observation's factual content:**

> UNVERIFIED. One candidate explanation is CUDA caching-allocator state: torch.cuda.max_memory_allocated (or reserved) reflects allocator history within a process, so a session that had already run other models, or that used a different torch build (2.13.0+cu130 here vs. whatever the earlier session used -- not recorded), can report a substantially different peak for identical work. THIS PROJECT HAS NOT MEASURED THAT. No controlled experiment varying allocator state exists, the earlier session's torch version and process history were never captured, and this explanation is therefore a hypothesis to test, not a cause to cite. Testing it would need two runs in this repository's own durable persistence with the environment fields populated for both.

**Supporting evidence** (7 typed references)

| Kind | Reference | Stream | Note |
|---|---|---|---|
| `experiment_run` | `experiment_run_30ba2bcc18a14c06af0f9ca291442cf8` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `method_run` | `method_run_e3209c63bb90458ea8d7a0673dd5c706` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` |  |
| `metric_result` | `metric_result_6b3a191d07074ddebf21e2bf9ef27894` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | gpu_memory_mb = 1210.64111328125 (2026-07-30 run) |
| `telemetry_event` | `event_2301532626da4c4da50acb91957ad9ea` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | MetricCalculated carrying the measured figure |
| `evidence_record` | `evidence_6f80d565e3bf614b65066302960d35615f2e3df2e529987ba8f888fb28c76cbc` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | Evidence.gpu_memory_mb = 1210.64111328125, execution_device='cuda', torch 2.13.0+cu130 |
| `external_document` | `src/archivetrust/providers/florence2_htr/README.md#peak-gpu-memory` | `(this stream)` | 'Peak GPU memory: ~3983 MB' -- the earlier session, which has no durable telemetry record |
| `metric_result` | `metric_result_11954d828f2742059dfd81a324347008` | `docs/experiments/baseline-comparison/htr_research_events.jsonl` | SATRN control: gpu_memory_mb = 438.78173828125, against a documented ~439 MB -- reproduced |

## Research questions

### Q1. Being investigated — `research_question_0ce2e1c1f1284af093512384a36c9047`

> Does SATRN's reported confidence track its measured transcription accuracy on a sample larger than one line crop -- i.e. is the confidence/accuracy disagreement recorded on the single controlled baseline line a property of this checkpoint's calibration, or a property of that one crop?

* **Status**: `Being investigated`
* Motivation: The observation records reported confidence 0.6666051723062992 against measured CER 0.7931034482758621 and WER 1.0 on one crop in one run, and states explicitly that this is not a claim about SATRN's confidence calibration because one sample cannot measure a distributional property. htr/evaluation/failures.py::classify_reliability flagged the run only because the disagreement (0.460) exceeded its own 0.35 threshold, and the candidate finding derived from the observation carries that threshold sensitivity as a stated limitation. The gap is therefore named in the record rather than inferred: what is unknown is whether the disagreement recurs.
* Originating observation: `research_observation_067cd6a4ae80439289aa8256efffa835`
* Originating finding: `research_finding_0f77a4f5443e4e79b429d8076596c4eb`
* Drafted experiment: `experiment_f590cff12db94b29bfe1588a691ef832` version `experiment_version_1be05a170466424b966b2c66458470cf` — **drafted, not executed**
* Raised by: hypergeek-dev at 2026-07-30T03:30:00+00:00

**Hypothesis** `hypothesis_cdf572be0a4c4794836ba2e7564dd260`

> SATRN's reported confidence is not monotonically related to its measured character error rate on 17th-century Swedish court-record lines at this checkpoint.

* Falsification criterion: Refuted if, over a sample of ground-truthed lines large enough to estimate a rank correlation, reported confidence and measured CER are monotonically related (higher confidence accompanying lower CER) with the relationship holding across the sample rather than only in aggregate. Refuted equally by the disagreement failing to recur at all: a single non-recurring disagreement is evidence about one crop, which is what the originating observation already says.
* Author: hypergeek-dev at 2026-07-30T03:30:00+00:00
