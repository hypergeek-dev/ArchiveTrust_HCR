# ArchiveTrust — Swedish Historical HTR Research Center

ArchiveTrust is a research environment for the systematic, evidence-first evaluation of
**Historical Handwritten Text Recognition (HTR)**, with a primary focus on Swedish historical
handwriting.

> **ArchiveTrust does not determine historical truth automatically. It preserves evidence,
> compares recognition methods and supports transparent human evaluation.**

It is not a transcription product and does not claim any method's output is correct. It exists to
make HTR method comparison reproducible, traceable back to raw model output, and honest about
disagreement, failure, and human correction effort.

## Why this is hard

Swedish historical handwriting (17th–19th century court records, parish registers, administrative
documents) combines archaic spelling, obsolete characters (e.g. the long s, ſ), abbreviations,
inconsistent letterforms, ink bleed-through and damage, marginalia, crossed-out and interlinear
text, and — unlike printed text — no fixed character-shape vocabulary a model can be perfectly
calibrated against. General-purpose OCR (built for printed, modern-language text) performs poorly
on this material; this project studies dedicated historical-HTR methods instead.

## The three methods under study

| Method | What it is | Execution |
|---|---|---|
| **SATRN** (`Riksarkivet/satrn_htr`) | A dedicated CTC/attention OCR decoder, trained by Riksarkivet (the Swedish National Archives) specifically for historical Swedish handwriting. Consumes pre-segmented line-image crops only. | Local, GPU/CPU, via `mmocr` in an isolated Python environment ([`src/archivetrust/providers/satrn/README.md`](src/archivetrust/providers/satrn/README.md)) |
| **Florence-2 HTR pipeline** (fine-tuned checkpoint from [`hoanghapham/vlm-htr`](https://github.com/hoanghapham/vlm-htr)) | A general vision-language model (Microsoft Florence-2), fine-tuned for Swedish historical line-level OCR via task-token prompting (`<OCR>`). Architecturally different from SATRN — a causal-LM decoder, not a dedicated OCR head. | Local, GPU/CPU, plain `transformers` ([`src/archivetrust/providers/florence2_htr/README.md`](src/archivetrust/providers/florence2_htr/README.md)) |
| **Transkribus Swedish Lion I** | A production HTR model hosted by READ-COOP's Transkribus service. Only **manual import mode** is implemented: a researcher runs recognition in Transkribus themselves and exports PAGE XML / ALTO XML / plain text, which ArchiveTrust parses. | External service, **manual import only** — see privacy notice below ([`src/archivetrust/providers/transkribus/README.md`](src/archivetrust/providers/transkribus/README.md)) |

All three implement one shared contract, `HtrMethodAdapter`
([`src/archivetrust/providers/htr_adapter.py`](src/archivetrust/providers/htr_adapter.py)), which
represents each method's capabilities explicitly rather than assuming feature parity — e.g. SATRN
reports a real decoder confidence, Florence-2 reports an explicitly-labeled *proxy* confidence
derived from beam-search log-probabilities, and Transkribus's confidence (if present) comes from
whatever the export file itself states. Nothing is silently upgraded to look more comparable than
it is.

## Controlled vs. end-to-end comparison

These measure different things and are never presented as interchangeable:

- **Controlled line-level comparison**: SATRN and Florence-2 are given the byte-identical input
  crop (hash-verified — see `docs/experiments/baseline-comparison/`) so differences in output
  reflect only the recognizer, not segmentation. Transkribus participates in a controlled
  comparison only where its exported geometry gives reliable line-level correspondence; otherwise
  its result is explicitly labeled **page-level only** and excluded from the controlled set.
- **End-to-end comparison**: each method uses its own full pipeline (its own segmentation,
  preprocessing, reading order). Differences may come from any pipeline stage, not recognition
  alone — this is stated explicitly wherever end-to-end results are reported.

## Segmentation vs. recognition

The pipeline is deliberately staged so segmentation and recognition are independently replaceable
and independently evaluable:

```
Document → page extraction → preprocessing → region detection → line detection →
line ordering → line crops → recognition → parsing → normalization →
evaluation → human review → adjudication → canonical result
```

See [`docs/htr-domain-design.md`](docs/htr-domain-design.md) §7 for the segmentation-as-independent-stage
design.

## Evidence and provenance

Every result is traceable back through a content-addressed chain: `CanonicalResult` → reviewed →
normalized → parsed → **raw, unmodified model output** → method run → model version/revision →
input crop (hashed) → segmentation result → page → document → dataset → research project. Raw
model output is never silently cleaned, corrected, or discarded — parsing and normalization are
separate, explicitly stored steps, not in-place edits. See
[`docs/htr-domain-design.md`](docs/htr-domain-design.md) §2–4 for the full mechanism (built on
ArchiveTrust's original Evidence/Observation/Comparison substrate, extended rather than replaced —
see [`docs/htr-transformation-audit.md`](docs/htr-transformation-audit.md)).

## Ground truth, blind dual review, and adjudication

Ground truth is versioned against a named, immutable `TranscriptionConvention` — once a convention
version has been used in an experiment it cannot silently change. Two reviewers transcribe or
validate the same ground-truth item independently; a mechanically-enforced isolation guard
(`src/archivetrust/review/blind_review/`) prevents either reviewer from seeing the other's
submission before both are finalized. Agreement is computed from real CER/WER (not eyeballed) and
classified as Agreed / Minor disagreement / Material disagreement / Requires adjudication /
Excluded from benchmark — disagreements are never silently discarded, and exclusion requires a
recorded reason. A third reviewer can adjudicate material disagreements; both original submissions
and the adjudicated result are preserved separately, never overwritten.

## Metrics

Character/Word Error Rate (raw and normalized), exact-line/word accuracy, and a full
insertion/deletion/substitution breakdown at both character and word level
(`src/archivetrust/htr/evaluation/recognition.py`); historical-feature evaluators for dates,
abbreviations, and unusual/archaic characters, with a documented plugin interface for more
(`historical_features.py`); geometric segmentation metrics — precision/recall/IoU, missed/
duplicate/merged/split-line rates, reading-order accuracy (`segmentation.py`); and a reliability
classifier that preserves method failures (OOM, malformed output, empty output, model-loading
failure) as first-class research results rather than silently dropping them from aggregates
(`failures.py`). No unexplained composite score exists — component metrics are the reported unit.

## Reproducibility

Every local execution records, where available: OS, Python/torch/transformers versions, GPU name,
CUDA availability, application commit, adapter version, model identifier and revision, and start/
completion timestamps, assembled into a `ReproducibilityManifest` per experiment run
(`src/archivetrust/htr/experiment/models.py`). See
[`docs/experiments/baseline-comparison/`](docs/experiments/baseline-comparison/) for a real example
manifest and research report generated by an actual executed run.

## Privacy: Transkribus is an external service

Transkribus is a third-party, external, network-based service. ArchiveTrust **never** uploads a
document to Transkribus automatically, and the current implementation only supports **manual
import** — a researcher runs recognition in Transkribus themselves, under their own account, and
exports the result to local disk; ArchiveTrust only parses that already-exported file. No
Transkribus API credentials are hardcoded anywhere in this codebase; a future API-mode integration
(out of scope for the current implementation) would read credentials from environment/config at
runtime, never from source. Vendor-reported accuracy figures found in an export file are stored as
a distinctly-labeled field and are never merged into ArchiveTrust's own computed CER/WER — they are
not directly comparable (different ground truth, different measurement methodology).

## Known limitations (honestly, as of this writing)

- **Corpus scale**: the only real, shared, ground-truthed asset currently in the repository is one
  line image (`tests/fixtures/htr/trolldomskommissionen_sample_line.jpg`, 17th-century Swedish
  court-record handwriting from Riksarkivet's `trolldomskommissionen_lines` dataset) plus one
  hand-authored PAGE XML page fixture. The executed baseline experiment
  (`docs/experiments/baseline-comparison/`) is explicitly an **N=1 pipeline demonstration**, not a
  general performance claim about any method.
- **SATRN** requires a second, isolated Python environment (`mmocr`/`mmcv`/`mmdet` have hard
  version constraints incompatible with this project's main environment) — see its README for
  setup. Its confidence is a single per-line scalar, not per-character.
- **Florence-2**'s confidence is an explicitly-labeled *proxy* (beam-search log-probability), not a
  calibrated OCR confidence signal, and its fine-tuned checkpoint was trained on line crops from
  its own detector's boundaries — crop-boundary mismatches with a different segmentation source are
  a real, observed source of error (documented in its README, not hidden).
- **Transkribus** API mode does not exist yet — manual import only, per the project's explicit
  phasing requirement. No genuine Transkribus export was available during development; its parser
  fixtures are hand-authored, schema-conformant PAGE/ALTO XML, not real vendor output.
- **First-launch provider wizard**: `clients/desktop/` (legacy) still offers now-nonexistent vision
  provider choices in one first-run flow not yet redesigned for HTR — flagged in
  `presentation/first_launch_viewmodel.py`, not yet fixed.
- The Qt research-interface ViewModels were verified headless (`QT_QPA_PLATFORM=offscreen`,
  construction/binding/navigation all tested); no visual/layout QA has been done — no display was
  available during development.

## Repository structure

```
src/archivetrust/
  domain/               Evidence, Observation ontology, Comparison/Confidence Engines, telemetry
                         (retained substrate, extended for line-level HTR — see htr-domain-design.md)
  htr/
    corpus/              ResearchProject, Dataset, Collection, Page, Region, TextLine, InputCrop
    experiment/          Experiment, ExperimentVersion, ExperimentRun, MethodRun, ReproducibilityManifest
    evaluation/           recognition/segmentation/historical-feature/reliability/operational metrics
  providers/
    htr_adapter.py       The shared HtrMethodAdapter contract
    satrn/                SATRN adapter
    florence2_htr/         Florence-2 HTR adapter
    transkribus/           Transkribus manual-import adapter
  evaluation/            Ground truth store (blinded double-annotation, TranscriptionConvention)
  review/
    blind_review/         Blind dual-review isolation, agreement, adjudication, exclusion
  presentation/          Qt-independent ViewModels, incl. the HTR research-interface ViewModels
  clients/desktop_v2/    The supported Qt desktop client (research dashboard, methods, dataset
                         explorer, experiment builder, comparison view, evidence view, review center)
docs/
  htr-transformation-audit.md   What the pre-HTR system looked like and what changed
  htr-domain-design.md          Entity map, versioning, evidence-chain design
  htr-repository-cleanup.md     Full record of what was deleted/replaced/archived and why
  htr-migration-plan.md         The staged rollout this transformation followed
  experiments/baseline-comparison/  A real executed example experiment run
  archive/                      Superseded planning/audit documents from the pre-HTR project
tests/                   Mirrors src/ package-for-package; fixtures under tests/fixtures/htr/,
                         tests/fixtures/transkribus/
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate    # .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
pytest -q
```

The full test suite runs without any model download, GPU, or network access — real-model inference
tests are marked `real_model` and excluded unless explicitly selected. Run `pytest -m real_model`
to exercise them, which does require the setup below.

### Model download / environment setup

- **SATRN**: no download step alone — needs a second isolated Python 3.10 environment
  (`.venv-satrn`) with `mmocr`/`mmcv`/`mmdet`; the model (`Riksarkivet/satrn_htr`, revision
  `a40c7093232eaa47a83ce6469fc4abd033486bdc`) downloads automatically on first real inference call.
  Full setup steps: [`src/archivetrust/providers/satrn/README.md`](src/archivetrust/providers/satrn/README.md).
- **Florence-2**: `pip install -e ".[transformers]"` (pins `transformers>=4.40,<5.0` — verified
  `>=5.0` breaks Florence-2's remote config code). Base model `microsoft/Florence-2-base-ft`
  and fine-tuned checkpoint `nazounoryuu/florence_base__mixed__line_bbox__ocr` both download
  automatically on first use. Details: [`src/archivetrust/providers/florence2_htr/README.md`](src/archivetrust/providers/florence2_htr/README.md).
- **Transkribus**: no download — point the adapter at a PAGE XML / ALTO XML / plain-text file you
  already exported from Transkribus yourself.

## Experiment workflow

1. Register a `Dataset`/`DatasetVersion` and the documents/pages/lines it covers
   (`src/archivetrust/htr/corpus/`).
2. Define an `Experiment` (research question, hypothesis, methods, metrics, controlled variables —
   see [`docs/htr-domain-design.md`](docs/htr-domain-design.md)); it becomes immutable once an
   `ExperimentRun` exists — later edits create a new `ExperimentVersion`.
3. Run methods against shared (controlled) or method-preferred (end-to-end) input, producing
   `MethodRun`s with raw/parsed/normalized results kept separate.
4. Run the evaluation engine to produce `MetricResult`s and `FailureRecord`s.
5. Route ambiguous or disagreeing items through blind dual review and, if needed, adjudication.
6. Export a `ResearchReport` (JSON and CSV — see `src/archivetrust/research/reports/export.py`).

`docs/experiments/baseline-comparison/` is a real, executed worked example of steps 1–6 (the
"Swedish Historical HTR Baseline Comparison" template) — `scripts/run_baseline_comparison.py` runs
it end to end.

## Testing

```bash
pytest -q                                  # full suite (no GPU/network required)
pytest -m real_model -q                    # real model download + inference (GPU recommended)
pytest tests/htr -q                        # HTR domain/experiment/evaluation
pytest tests/providers/satrn tests/providers/florence2_htr tests/providers/transkribus -q
pytest tests/review/blind_review -q        # blind dual-review workflow
QT_QPA_PLATFORM=offscreen pytest tests/presentation tests/clients -q   # headless Qt
```

## Project history

ArchiveTrust began as a general, provider-independent OCR-comparison trust layer. Its evidence-
first architecture — Evidence, Observation ontology, Comparison Engine, append-only telemetry,
supersede-not-overwrite corrections — proved to be exactly the right substrate for rigorous HTR
method comparison, and was retained and extended rather than rewritten. The five general OCR
providers it originally compared (Docling, Tesseract+LayoutParser, Qwen2.5-VL, PaddleOCR-VL, Surya)
were removed, as none were specialized for historical handwriting; see
[`docs/htr-transformation-audit.md`](docs/htr-transformation-audit.md) and
[`docs/htr-repository-cleanup.md`](docs/htr-repository-cleanup.md) for the full record of what
changed and why (the superseded planning documents from that earlier phase were archived and then,
by explicit later decision, deleted outright — see that document's closing section). Architectural
invariants that still govern the project are in
[`ARCHITECTURAL_CONSTITUTION.md`](ARCHITECTURAL_CONSTITUTION.md).
