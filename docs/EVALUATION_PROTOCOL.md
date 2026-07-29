# ArchiveTrust Evaluation Protocol

Status: Current protocol; real pilot not executed  
Scope: Independent accuracy/calibration evaluation  
Governs: Assignment, annotation, adjudication, and claims  
Applies to version: evaluation reference schema v2  
Supersedes: ground-truth-as-provider and contaminated B5 paths  
Superseded by: **extended, not replaced, by `src/archivetrust/htr/evaluation/` (Migration Plan Stage 9) and `src/archivetrust/review/blind_review/` (Stage 10)**  
Last verified against code: 2026-07-17

> **HTR transformation notice:** The blinded double-annotation/adjudication workflow described
> below is still the retained mechanism (`evaluation/ground_truth.py`), now extended with
> `TranscriptionConvention`/`GroundTruthGranularity` and a mechanically-enforced blind-isolation
> layer (`review/blind_review/`). CER/WER computation is extended, not replaced, in
> `src/archivetrust/htr/evaluation/recognition.py` (raw+normalized CER/WER, edit-operation
> classification, segmentation/historical-feature/reliability metrics). This document's protocol
> description remains largely applicable; provider-specific examples referring to the deleted OCR
> providers should be read as historical.

Operational review resolves the current document and may show candidates/confidence/history.
Evaluation judges one source crop/region and initially hides provider/canonical identity,
confidences, agreement, expected answer, and the other reviewer.

Each chunk has a campaign/stratum, source hash, geometry, task/type, legal basis, and sampling basis.
Two distinct humans independently Accept, Correct, Illegible, Reject segment, Boundary incorrect,
or Cannot determine. Agreement creates a new verified record. Disagreement routes to a third
adjudicator whose new record cites both answers. Nothing is overwritten. AI assistance is labeled
and excluded by default.

Only verified human-independent references enter default metrics. They never enter provider
registration, agreement counts, confidence, or reconciliation. Reports must state sampling, counts,
agreement/adjudication, text and supported layout/table metrics, intervals, calibration, correlated
errors, exclusions, biases, and legal scope. Automatic acceptance is prohibited; no superiority
claim is allowed before the controlled pilot is completed.
