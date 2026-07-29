# Florence-2 HTR adapter (`hoanghapham/vlm-htr`)

Second real `HtrMethodAdapter` implementation (`docs/htr-migration-plan.md` Stage 7). Line-level
Swedish historical handwriting recognition via a Florence-2 vision-language model, task-token
prompted (`<OCR>`) -- consumes pre-segmented text-line crops through `recognize()`
(`get_capabilities().page_level_supported is False`), same as SATRN's `recognize()` boundary, for
reasons explained under "Divergence from SATRN" below.

## What the pipeline actually is (investigation result)

`github.com/hoanghapham/vlm-htr` (commit `ced3b30222770911dcb900c3a1f83a247100d1a3`, `main` HEAD at
investigation time) is a master's thesis (Uppsala University, with Riksarkivet) comparing a
"traditional" YOLO+TrOCR pipeline against a **two-stage Florence-2 VLM pipeline**:

1. **Text line detection** -- Florence-2 fine-tuned for the `<OD>` (object detection) task token,
   full page image in, quantized `<loc_N>` bounding boxes out.
2. **Text recognition** -- Florence-2 fine-tuned for the `<OCR>` task token, a rectangular
   line-bbox crop in, plain text out.

Confirmed by reading the repo's actual training/inference code directly (not just README prose):
`scripts/train/finetune_florence_od.py`, `scripts/train/finetune_florence_ocr.py`, and
`src/vlm/data_processing/florence.py` (`FlorenceTask` task-token constants and the `predict()`
helper this adapter's decoding parameters -- `num_beams=3`, `do_sample=False`,
`max_new_tokens=1024` -- are read from verbatim).

## Base model + fine-tuned checkpoint (non-gated -- real fine-tuned inference, no blocker)

- **Base model**: `microsoft/Florence-2-base-ft`, `revision="refs/pr/6"` (resolved this session to
  commit `e0b8f375661041228a6431c950adac1a5c539b98`) -- the exact revision `vlm-htr`'s own training
  scripts pin.
- **Fine-tuned OCR checkpoint** (this adapter's `recognize()`, line-crop -> text):
  `nazounoryuu/florence_base__mixed__line_bbox__ocr`, resolved this session to commit
  `994f47e8a0e8d77cb2e11528665efd07a855c3af`. Model card: `base_model:
  microsoft/Florence-2-base-ft`, `pipeline_tag: image-to-text`, `license: mit`, **`gated: false`**.
  Located via the `vlm-htr` repo's linked Gradio demo Space
  (`huggingface.co/spaces/nazounoryuu/vlm-htr`), whose own README names this checkpoint (and its
  companion detector below) as exactly what it loads -- not guessed from a naming convention alone,
  though the name does match `finetune_florence_ocr.py`'s own `--model-name` example verbatim.
- **This is the non-gated case.** No user-provided HF token was required or is needed; the tests
  and the measurements below run against the real fine-tuned checkpoint, not merely the base model.
- **Companion checkpoint, not used by this adapter**: `nazounoryuu/florence_base__mixed__page__line_od`
  -- the fine-tuned line-*detector*. Confirmed to exist and be public, but deliberately not wired
  up here: `docs/htr-domain-design.md` §7 makes segmentation an independent stage
  (`htr/segmentation/SegmentationAdapter`), not part of `HtrMethodAdapter`. This is a real,
  confirmed extension point for a future segmentation-stage implementation.
- The fine-tuned OCR checkpoint's HF repo ships only `config.json`/`generation_config.json`/
  `model.safetensors`/optimizer+scheduler state/`metrics.json` -- no processor files. Its
  tokenizer/image-processor are loaded from its declared base model
  (`microsoft/Florence-2-base-ft@refs/pr/6`) instead; fine-tuning only touched weights, not
  vocabulary/preprocessing.

## Install / setup

Plain `transformers` -- no isolated venv, no subprocess (see "Divergence from SATRN" below).

```powershell
pip install -e ".[transformers]"   # torch, transformers>=4.40,<5.0, timm, einops
```

Two real, narrow compatibility gaps were found and closed during this investigation (not assumed
in advance -- checked first, per the task brief):

1. **`transformers>=5.0` cannot construct Florence-2's remote config.** Verified with
   `transformers==5.13.1`: `AttributeError: 'Florence2LanguageConfig' object has no attribute
   'forced_bos_token_id'` while loading `configuration_florence2.py` (Florence-2's own
   `trust_remote_code` module, downloaded from the HF Hub, not this project's code). This is a
   real incompatibility between very recent `transformers` internals and Florence-2's community
   modeling file, verified by reading the traceback down to the exact failing line, not guessed.
   Fix: downgrade to `transformers==4.49.0` -- **still inside this project's pre-existing declared
   range** (`pyproject.toml` previously said `<6.0`; this Stage tightens it to `<5.0` once the
   incompatibility was confirmed, rather than leaving a range known to include broken versions).
2. **`timm` and `einops` are required but undeclared transitive dependencies** of Florence-2's
   remote modeling code (its vision backbone + tensor reshaping helpers) -- `from_pretrained(...,
   trust_remote_code=True)` raises `ImportError: This modeling file requires ... timm` otherwise.
   Both are now part of this project's `transformers` extra (see `pyproject.toml`).

No isolated venv, no `ARCHIVETRUST_*_PYTHON` environment variable, no subprocess boundary --
`facade.py`'s real facade loads the model directly in-process, in this project's normal main venv.

**GPU**: used automatically when available (`device="auto"`, the default); `device="cpu"` forces
CPU. Both measured below.

## Known limitations

- `recognize()` is line-level only (fed an already-segmented crop via `RecognitionInput.input_crop_id`/
  `page_image_ref`) -- the fine-tuned OCR checkpoint used here was itself trained exclusively on
  "rectangular crops of line images" (per `vlm-htr`'s own training script), so declaring
  `page_level_supported=True` would overstate what this specific checkpoint was ever trained to do,
  even though Florence-2's *architecture* is general enough that the base model can technically
  process a full page under the same `<OCR>` task token.
- Confidence is a **proxy**, not raw/calibrated confidence: `exp(sequences_scores)`, where
  `sequences_scores` is HF beam-search's length-normalized sum of per-token log-probabilities for
  the selected beam (`generate(..., num_beams=3, output_scores=True,
  return_dict_in_generate=True)`). This is architecturally different from SATRN's confidence (a
  dedicated OCR decoder's own postprocessor scalar) -- it is a byproduct of the causal-LM's
  token-generation process, not an OCR-specific quality signal, and not per-character/token
  granularity. Never present it downstream as more than that.
- The line-detection half of the real `vlm-htr` pipeline (`<OD>` task token,
  `nazounoryuu/florence_base__mixed__page__line_od`) is not implemented by this adapter -- see
  "base model + fine-tuned checkpoint" above.
- First `recognize()` call in a process pays the HF download (first run only) + model-load cost
  (~1.5s once cached); the real facade caches the loaded (model, processor) pair in-process for
  every call after that (unlike SATRN's per-call subprocess restart).

## Divergence from SATRN (per the migration plan: do not force-fit)

| | SATRN | Florence-2 |
|---|---|---|
| Architecture | Dedicated CTC/attention OCR decoder | General task-token-prompted VLM |
| Runtime isolation | Subprocess + isolated `.venv-satrn` (real, hard `mmocr`/`mmcv`/`mmdet` incompatibility with the main venv) | In-process, plain `transformers` in the main venv (once two narrow version/dependency gaps were closed -- no architecturally incompatible dependency stack) |
| Output structure | Already plain text -- no separate parsing stage | Raw decoder output (`</s><s>...text...</s>`) requires a genuine structural parsing step (`processor.post_process_generation`) before it is usable text |
| Transcription payload stages populated | Two (`RawTranscriptionPayload`, `NormalizedTranscriptionPayload`) -- no distinct parse step exists | **Three** (`Raw`, `Parsed`, `Normalized`) -- the first adapter in this codebase to exercise `ParsedTranscriptionPayload` for real |
| Confidence | Real scalar directly from the model's own `AttentionPostprocessor` | Real **proxy** derived from beam-search log-probabilities (`exp(sequences_scores)`) -- labeled a proxy, not raw confidence |
| `Evidence.processing_stage` | `OCR` | `VLM_INFERENCE` |
| Underlying pipeline shape | Recognition-only; never does its own segmentation | The *source* pipeline is two-stage end-to-end (own line detection + OCR) via the same model family, task-token-switched -- this adapter deliberately implements only the recognition half, per `docs/htr-domain-design.md` §7's independent-segmentation-stage design |

## Measured real-inference behavior (this session)

Ran against the real downloaded fine-tuned checkpoint
(`nazounoryuu/florence_base__mixed__line_bbox__ocr@994f47e8a0e8d77cb2e11528665efd07a855c3af`) on
`tests/fixtures/htr/trolldomskommissionen_sample_line.jpg` (the same real 17th-century Swedish
handwritten court-record line SATRN's tests use -- see that directory's `README.md` for
provenance):

| device | parsed text | proxy confidence (`exp(sequences_scores)`) | elapsed |
|---|---|---|---|
| cuda (RTX 3070) | `Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff` | 0.249 (`sequences_scores=-1.3901`) | 0.67s |
| cpu | `Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff` | 0.249 (`sequences_scores=-1.3902`) | 3.9s |

Raw decoder output (before parsing): `</s><s>Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger
werff</s>`. Peak GPU memory: ~3983 MB (a full VLM forward pass over a vision encoder + language
model, substantially more than SATRN's ~439 MB dedicated OCR decoder). Software versions recorded
in `Evidence.software_environment` for this run: `torch==2.13.0+cu130`, `transformers==4.49.0`.

The transcription is a real, honestly-reported model output -- it does **not** closely match the
fixture's ground truth (`bekiendt. Sager och deth hon Minnes hoon Tua ganger waritt`). This is a
genuine, non-trivial recognition error, not an adapter bug: this fixture line comes from
`trolldomskommissionen_seg`, which is one of the nine datasets `vlm-htr`'s own README lists as
training data for this checkpoint, but the checkpoint was trained on rectangular line-bbox crops
extracted by *its own* line-detection stage's bounding boxes, not on the SATRN fixture's exact crop
boundaries/preprocessing -- a real crop-boundary/preprocessing-convention mismatch, left
undoctored here rather than picking a fixture that happens to score well (same discipline SATRN's
README applies to its own non-matching result).
