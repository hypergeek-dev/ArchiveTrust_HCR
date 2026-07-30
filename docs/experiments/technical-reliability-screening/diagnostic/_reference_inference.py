"""Reference SATRN inference: mmocr's own TextRecInferencer, one process, one model load.

Runs under `.venv-satrn` only. Imports nothing from `archivetrust` (that package is not installed
in the isolated interpreter). Mirrors `providers/satrn/_worker.py`'s model resolution so the same
pinned checkpoint is used, then feeds every crop to the *same* inferencer instance -- deliberately
the opposite of the adapter's fresh-process-per-crop pattern.
"""
import fnmatch, json, sys
from huggingface_hub import hf_hub_download, list_repo_files
from mmengine.config import Config

REPO, REVISION = "Riksarkivet/satrn_htr", "a40c7093232eaa47a83ce6469fc4abd033486bdc"


def grab(pattern):
    for name in list_repo_files(REPO):
        if fnmatch.fnmatch(name, pattern):
            return hf_hub_download(REPO, name, revision=REVISION)
    raise FileNotFoundError(pattern)


weights, config, dictionary = grab("*.pth"), grab("config.py"), grab("dictionary.txt")
cfg = Config.fromfile(config)
cfg.dictionary["dict_file"] = dictionary
cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary
cfg.dump(config)

from mmocr.apis import TextRecInferencer

inferencer = TextRecInferencer(model=config, weights=weights, device="cuda")
paths = json.loads(sys.argv[1])
out = []
for path in paths:
    prediction = inferencer([path], batch_size=1, return_datasamples=False, progress_bar=False)[
        "predictions"
    ][0]
    out.append({"path": path, "text": prediction["text"], "score": prediction.get("scores")})
print("REFJSON" + json.dumps(out))
