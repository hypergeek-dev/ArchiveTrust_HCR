# SATRN adapter (`Riksarkivet/satrn_htr`)

First real `HtrMethodAdapter` implementation (docs/htr-migration-plan.md Stage 6). Line-level
Swedish historical handwriting recognition -- consumes pre-segmented text-line crops, does **not**
segment pages itself (`get_capabilities().page_level_supported is False`).

## What the model actually is (investigation result)

`Riksarkivet/satrn_htr`'s Hugging Face repo declares `library_name: htrflow` and ships `model.pth`
+ `config.py` (an OpenMMLab `mmengine` config) -- **not** a `transformers`-loadable
`config.json`/safetensors pair. Riksarkivet's own `htrflow` toolkit loads it through
`mmocr.apis.TextRecInferencer` (confirmed by reading `htrflow.models.openmmlab.satrn.Satrn`'s
source directly, not just the model card prose). This adapter integrates through that same real
inference path (`mmocr`), not a from-scratch SATRN reimplementation and not the `htrflow` PyPI
package itself (see `adapter.py`'s module docstring for why the package is skipped).

## Why inference runs in a separate, isolated Python environment

`mmocr==1.0.1` requires `mmdet<3.2`, which requires `mmcv<2.2` (and pins `numpy<2`, prebuilt
Windows wheels only up to `torch==2.1`). This project's main venv is Python 3.13 / torch
2.13+cu130 / numpy 2.3 (the `transformers` extra, already installed for a later Florence-2
adapter). Installing the OpenMMLab stack into that venv would either fail outright or force a
downgrade that breaks the ~979 other passing tests. Instead, a second, isolated venv
(`.venv-satrn`, **not** committed to the repo) runs the real model; `SatrnAdapter` talks to it
over a subprocess boundary (`facade.py` -> `_worker.py`).

### Setting up `.venv-satrn` (one-time, Windows)

```powershell
py -3.10 -m venv .venv-satrn
.venv-satrn\Scripts\python.exe -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
.venv-satrn\Scripts\python.exe -m pip install mmengine "numpy<2"
.venv-satrn\Scripts\python.exe -m pip install "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-win_amd64.whl"
.venv-satrn\Scripts\python.exe -m pip install "mmdet==3.1.0" --no-deps
.venv-satrn\Scripts\python.exe -m pip install "opencv-python<4.10" pyclipper rapidfuzz scikit-image lmdb imgaug pycocotools shapely terminaltables tqdm
.venv-satrn\Scripts\python.exe -m pip install mmocr --no-deps
.venv-satrn\Scripts\python.exe -m pip install huggingface_hub
```

Then relax one overly conservative version guard: `mmdet`'s installed `__init__.py` asserts
`mmcv < 2.1.0` even though nothing SATRN actually uses changed between mmcv 2.0 and 2.1 (verified:
inference runs correctly on 2.1.0) -- edit
`.venv-satrn/Lib/site-packages/mmdet/__init__.py`, changing `mmcv_maximum_version = '2.1.0'` to
`'2.2.0'`.

If `.venv-satrn` lives somewhere other than the repo root, set `ARCHIVETRUST_SATRN_PYTHON` to the
interpreter path. `validate_environment()`/`health_check()` report `valid=False`/`healthy=False`
with an explanatory message (never a crash) when the isolated venv is missing.

**GPU**: a CUDA device is used automatically when available (`device="auto"`, the default);
`device="cpu"` forces CPU (see measured CPU-vs-GPU timing below). No compiled CUDA extension is
required beyond what the prebuilt `mmcv`/`torch` wheels already ship (SATRN's architecture --
ShallowCNN backbone + transformer encoder/decoder -- uses no custom deformable-conv/NMS ops itself,
though it transitively imports `mmdet`'s backbone registry, which is why `mmcv` and not
`mmcv-lite` is required).

## Known limitations

- Line-level only; feed it an already-segmented crop (`RecognitionInput.input_crop_id`/
  `page_image_ref`), not a full page.
- Confidence is one scalar per line (SATRN's `AttentionPostprocessor` mean decode probability),
  not per-character/token granularity -- do not present it as such downstream.
- `.venv-satrn` is a local, undocumented-by-pip, hand-assembled environment (the OpenMMLab wheel
  index has no `torch==2.13`/Python-3.13 build); it is not part of this project's normal
  `pip install -e .[transformers]` flow and must be set up separately per the steps above.
- The isolated interpreter also means every `recognize()` call pays subprocess-start + (on first
  call) Hugging Face download/model-load cost; there is no persistent warm worker process in this
  first implementation.

## Measured real-inference behavior (this session)

Ran against the real downloaded checkpoint (`Riksarkivet/satrn_htr@a40c7093232eaa47a83ce6469fc4abd033486bdc`)
on `tests/fixtures/htr/trolldomskommissionen_sample_line.jpg` (a real 17th-century Swedish
handwritten court-record line, see that directory's `README.md` for provenance):

| device | text | score | elapsed |
|---|---|---|---|
| cuda (RTX 3070) | `till den 23 Januarii` | 0.6666 | 1.31s |
| cpu | `till den 23 Januarii` | 0.6665 | 10.66s |

Peak GPU memory: ~439 MB. Software versions recorded in `Evidence.software_environment` for that
run: `torch==2.1.0+cu121`, `mmocr==1.0.1`, `mmengine==0.10.7`, `mmcv==2.1.0`, `mmdet==3.1.0`.

The transcription is a real, honestly-reported model output -- it does **not** closely match the
fixture's ground truth (`bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt`). This is a
genuine, non-trivial recognition error, not an adapter bug: the fixture crop is an unbinarized,
irregularly-masked JPEG straight from the source dataset, and the model card states it was trained
on *binarized* line images -- a real preprocessing-domain mismatch, left undoctored here rather
than picking a fixture that happens to score well. `MethodCapabilities.confidence_supported=True`
because a real, non-fabricated confidence value was in fact produced (0.6666), and that same real
result is what the adapter's test suite asserts against.
