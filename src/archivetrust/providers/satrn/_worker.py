"""Standalone SATRN inference worker -- runs under the *isolated* `.venv-satrn` interpreter, NOT
under the main project venv (see `providers/satrn/README.md` for why: `Riksarkivet/satrn_htr` is
published for Riksarkivet's own HTRFlow toolkit, which loads it through OpenMMLab's `mmocr`
(`mmocr.apis.TextRecInferencer`) -- not a standard `transformers` `AutoModel`. `mmocr`/`mmcv`/
`mmdet`/`mmengine` are a version-triangle-pinned dependency stack (numpy<2, mmcv<2.2 for mmdet<3.2,
torch<=2.1 for prebuilt Windows wheels) that is incompatible with this project's main venv
(Python 3.13, torch 2.13+cu130, numpy 2.3) -- installing it there would either fail outright or
force a downgrade that breaks the other ~979 passing tests. This script is therefore run as a
subprocess from a separate interpreter (`.venv-satrn`, Python 3.10) by
`providers/satrn/facade.py::subprocess_satrn_facade`, communicating over stdout as a single JSON
line -- the same "isolate the incompatible runtime behind a narrow seam" discipline
`runtime/transformers_runtime.py` uses for lazy-importing torch/transformers, escalated to
process-level isolation because the conflict here is not resolvable within one interpreter.

Deliberately does NOT import anything from `archivetrust` -- it must run standalone under an
interpreter that does not have this package installed.

Communication contract (stdout is exactly one JSON object, all diagnostic output goes to stderr):
  success: {"ok": true, "text": str, "score": float | null, "elapsed_seconds": float,
            "device_used": str, "model_revision": str, "config_revision": str,
            "gpu_name": str | null, "peak_gpu_memory_mb": float | null,
            "software_environment": {...}}
  failure: {"ok": false, "category": str, "message": str}
`category` is one of "model_load_failed", "cuda_oom", "malformed_input", "empty_output",
"unknown_error" -- see module docstring of `providers/satrn/adapter.py` for how each maps to a
`FailureRecord`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time


def _hf_download_matching(repo_id: str, pattern: str, revision: str | None) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    for filename in list_repo_files(repo_id):
        if fnmatch.fnmatch(filename, pattern):
            return hf_hub_download(repo_id, filename, revision=revision)
    raise FileNotFoundError(f"No file matching {pattern!r} in {repo_id!r}")


def _load_mmlabs(model_id: str, revision: str | None) -> tuple[str, str, str | None]:
    """Downloads `model.pth` + `config.py` (+ `dictionary.txt`, patched into the config) from the
    given Hugging Face repo. Mirrors `htrflow.models.hf_utils.load_mmlabs` (Riksarkivet's own
    HTRFlow toolkit) without depending on the `htrflow` package itself -- that package additionally
    pulls in `ultralytics`/`imgaug`/pinned `nvidia-cu11-*` wheels this adapter has no use for
    (SATRN-only, no detection/segmentation models), so only the two functions actually needed are
    reimplemented here, directly against `huggingface_hub`.
    """
    from mmengine.config import Config

    weights_path = _hf_download_matching(model_id, "*.pth", revision)
    config_path = _hf_download_matching(model_id, "config.py", revision)
    try:
        dictionary_path = _hf_download_matching(model_id, "dictionary.txt", revision)
    except FileNotFoundError:
        dictionary_path = None

    if dictionary_path is not None:
        cfg = Config.fromfile(config_path)
        cfg.dictionary["dict_file"] = dictionary_path
        cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary_path
        cfg.dump(config_path)

    return weights_path, config_path, dictionary_path


def _commit_hash_from_path(path: str) -> str | None:
    import string

    _, sha = os.path.split(os.path.dirname(path))
    if sha and all(ch in string.hexdigits for ch in sha):
        return sha
    return None


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"  # "auto"


def run(*, image_path: str, device_request: str, model_id: str, revision: str | None) -> dict:
    import torch

    device = _resolve_device(device_request)

    try:
        weights_path, config_path, _dict_path = _load_mmlabs(model_id, revision)
        model_revision = _commit_hash_from_path(weights_path)
        config_revision = _commit_hash_from_path(config_path)
    except Exception as exc:  # noqa: BLE001 -- classified failure, not a swallowed exception
        return {"ok": False, "category": "model_load_failed", "message": f"{type(exc).__name__}: {exc}"}

    try:
        from mmocr.apis import TextRecInferencer

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        inferencer = TextRecInferencer(model=config_path, weights=weights_path, device=device)
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        category = "cuda_oom" if "out of memory" in message.lower() else "model_load_failed"
        return {"ok": False, "category": category, "message": message}

    if not os.path.exists(image_path):
        return {"ok": False, "category": "malformed_input", "message": f"image not found: {image_path}"}

    started = time.time()
    try:
        raw = inferencer([image_path], batch_size=1, return_datasamples=False, progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        if "out of memory" in message.lower():
            category = "cuda_oom"
        else:
            # cv2/PIL decode failures, empty/zero-size images, unreadable files, etc. all land
            # here -- an inference-time failure caused by the *input*, not the model/environment.
            category = "malformed_input"
        return {"ok": False, "category": category, "message": message}
    elapsed = time.time() - started

    predictions = raw.get("predictions", [])
    if not predictions or not predictions[0].get("text"):
        return {
            "ok": False,
            "category": "empty_output",
            "message": f"model produced no text for this input (raw={raw!r})",
        }

    prediction = predictions[0]
    gpu_name = None
    peak_gpu_memory_mb = None
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    import mmcv
    import mmdet
    import mmengine
    import mmocr

    return {
        "ok": True,
        "text": prediction["text"],
        "score": prediction.get("scores"),
        "elapsed_seconds": elapsed,
        "device_used": device,
        "model_revision": model_revision,
        "config_revision": config_revision,
        "gpu_name": gpu_name,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "software_environment": {
            "torch": torch.__version__,
            "mmocr": mmocr.__version__,
            "mmengine": mmengine.__version__,
            "mmcv": mmcv.__version__,
            "mmdet": mmdet.__version__,
            "python": sys.version.split()[0],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-id", default="Riksarkivet/satrn_htr")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    try:
        result = run(
            image_path=args.image_path,
            device_request=args.device,
            model_id=args.model_id,
            revision=args.revision,
        )
    except Exception as exc:  # noqa: BLE001 -- last-resort classification, never a bare crash
        result = {"ok": False, "category": "unknown_error", "message": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
