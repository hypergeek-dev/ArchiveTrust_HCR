# Swedish Lion adapter (`Riksarkivet/trocr-base-handwritten-hist-swe-2`)

Third real `HtrMethodAdapter` implementation. Line-level Swedish historical handwriting
recognition via a standard TrOCR (`VisionEncoderDecoderModel`) checkpoint fine-tuned by the Swedish
National Archives -- consumes pre-segmented text-line crops through `recognize()`
(`get_capabilities().page_level_supported is False`), the same boundary SATRN and Florence-2 use.

## Why this adapter exists (the correction that led to it)

Earlier in this project, "Transkribus Swedish Lion I" was treated as cloud-only: an
external-upload-required method with no local execution path
(`docs/methods/transkribus-swedish-lion-1.md`, `providers/transkribus/`). That assumption was
wrong for the specific model documented here. The user's own reference repository,
`github.com/hypergeek-dev/rigsarkivet_hcr_test`, runs a real, fully local HTRFlow-style pipeline
(YOLO region + line detection, then TrOCR recognition), confirmed by reading its
`htrflow-swedish-htr/pipeline.yaml` directly -- the recognition step is
`Riksarkivet/trocr-base-handwritten-hist-swe-2`.

This model's own Hugging Face card names it **"Swedish Lion Libre"**: "An HTR model for historical
Swedish developed by the Swedish National Archives", a `trocr-base-handwritten` architecture
fine-tuned on Swedish running-text handwriting from the 17th-19th centuries across eleven public
datasets plus non-public archive material, developed with Stockholm City Archives, the Finnish
National Archives, and Jämtlands Fornskriftsällskap. **No Transkribus/READ-COOP reference appears
anywhere on the model card.** The naming similarity to Transkribus's own "Swedish Lion I" branding
is real but unconfirmed as identical lineage -- this adapter is therefore registered under its own
`method_id` (`swedish_lion`), never merged into or presented as the `transkribus` provider. The
existing `transkribus` package and its cloud-upload assumption are unmodified: both remain true
statements about the distinct thing each actually describes.

## Architecture: closer to SATRN than to Florence-2

Confirmed by reading the model card and HTRflow's own `TrOCR`/`WordLevelTrOCR` classes
(`AI-Riksarkivet/htrflow`, `src/htrflow/models/huggingface/trocr.py`) directly:

- **Input granularity is line-level**, identical to SATRN and Florence-2 -- the model card states
  plainly "the image has to be a single text line". `WordLevelTrOCR` (what HTRFlow's own pipeline
  actually instantiates) is a thin subclass of the base `TrOCR` class that additionally derives
  word-level bounding boxes from the model's attention weights; it takes the exact same line-crop
  input as the base class. This adapter talks to the plain `VisionEncoderDecoderModel`/
  `TrOCRProcessor` HF classes directly, not HTRFlow's own wrapper or its dependency chain -- no
  word-level bounding boxes are needed for `recognize()`'s text-out contract
  (`geometry_supported=False`).
- **Decode genuinely has a raw/parsed distinction**, like Florence-2's, unlike SATRN's:
  `processor.batch_decode` differs between `skip_special_tokens=False` (`"<s>...text...</s>"`) and
  `=True` (clean text) -- confirmed by calling both against the real facade, not assumed. This
  adapter therefore populates **three** transcription payload stages (raw, parsed, normalized),
  not SATRN's two.
- **No confidence signal is requested or exposed.** `model.generate(...)` here is called without
  `output_scores=True`/`return_dict_in_generate=True` (unlike Florence-2's facade).
  `MethodCapabilities.confidence_supported=False` -- an honest `False`, not a fabricated or
  silently-proxied value. A future version could add a beam-search log-probability proxy the same
  way Florence-2's facade does; until then, none is invented.
- `Evidence.processing_stage` is `OCR` (a dedicated encoder-decoder OCR model, not a VLM prompted
  with a task token) -- same as SATRN, not Florence-2's `VLM_INFERENCE`.

## Install / setup

Plain `transformers` -- no isolated venv, no subprocess, no `trust_remote_code`, no extra
dependencies beyond what Florence-2's `transformers` extra already installs (no `timm`, no
`einops`): `VisionEncoderDecoderModel` and `TrOCRProcessor` are both core `transformers` classes.

```powershell
pip install -e ".[transformers]"
```

**GPU**: used automatically when available (`device="auto"`, the default); `device="cpu"` forces
CPU. Both measured below.

## Model + processor (the brief's required disclosure)

- **Fine-tuned checkpoint**: `Riksarkivet/trocr-base-handwritten-hist-swe-2`, resolved this session
  to commit `aa79fcb1850bf3155ebc442570d6c6bfc0ac8100` (via the HF model API, not a floating
  `main`). License: Apache 2.0. Not gated.
- **Processor**: `microsoft/trocr-base-handwritten`, resolved this session to commit
  `eaacaf452b06415df8f10bb6fad3a4c11e609406`. The fine-tuned repo does ship its own
  `preprocessor_config.json`/tokenizer files (verified via the HF API), so it does not strictly
  need the base model's processor -- but the model card's own documented usage example loads it
  from `microsoft/trocr-base-handwritten` regardless, and this facade follows that documented
  example exactly rather than substituting an assumption of equivalence.
- **Decoding**: `num_beams=1`, matching `hypergeek-dev/rigsarkivet_hcr_test`'s
  `htrflow-swedish-htr/pipeline.yaml` `WordLevelTrOCR` step exactly (verified by reading it
  directly).

## Known limitations

- `recognize()` is line-level only, same reasoning as SATRN and Florence-2.
- No confidence signal (see above) -- `confidence_supported=False`.
- `num_beams=1` (greedy decoding) is inherited from the source pipeline's own configuration, not
  independently tuned by this adapter.
- First `recognize()` call in a process pays the HF download (first run only) + model-load cost;
  the real facade caches the loaded (model, processor) pair in-process for every call after that.

## Measured real-inference behavior (this session)

Ran against the real downloaded checkpoint
(`Riksarkivet/trocr-base-handwritten-hist-swe-2@aa79fcb1850bf3155ebc442570d6c6bfc0ac8100`) on
`tests/fixtures/htr/trolldomskommissionen_sample_line.jpg` -- the same real 17th-century Swedish
handwritten court-record line SATRN's and Florence-2's tests use:

| device | text | elapsed |
|---|---|---|
| cuda (RTX 3070) | `bekiendt. Säger och deth hoon Minnes hoon Tuå gånger waritt` | 0.83s |
| cpu | `bekiendt. Säger och deth hoon Minnes hoon Tuå gånger waritt` | 4.17s |

Raw decoder output (before removing special tokens): `<s><s><s>bekiendt. Säger och deth hoon
Minnes hoon Tuå gånger waritt</s>`. Peak GPU memory: ~1513 MB -- smaller than Florence-2's ~3983 MB
(a dedicated encoder-decoder OCR model, not a full VLM forward pass) and larger than SATRN's ~439
MB. Software versions recorded in `Evidence.software_environment`: `torch==2.13.0+cu130`,
`transformers==4.49.0`.

**Notable, honestly reported comparison**: the fixture's ground truth (per
`tests/fixtures/htr/README.md`, also cited in `providers/florence2_htr/README.md`) is `bekiendt.
Sager och deth hon Minnes hoon Tua ganger waritt`. Swedish Lion's output differs only in a few
diacritics (`Sager`/`Säger`, `Tua`/`Tuå`, `ganger`/`gånger`) and one letter (`hon`/`hoon`) -- a
substantially closer match than Florence-2's output on the same fixture
(`Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff`, per that adapter's own README).
This is a single-fixture, single-line observation, not a claim of general accuracy superiority --
this project's reliability screening benchmark deliberately measures execution reliability and
output plausibility, never CER/WER, and no ground-truth accuracy evaluation across the sampled
corpus has been run for this adapter.
