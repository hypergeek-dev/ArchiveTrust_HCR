import io
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from archivetrust.htr.benchmark import inference
from archivetrust.htr.benchmark.dryrun import LABEL, DryRunError, build_dryrun, read_dryrun, run_dryrun
from archivetrust.htr.benchmark.inference import LionBackend, LoghiBackend, RunError, read_predictions
from archivetrust.htr.benchmark.layout import LayoutError
from tests.htr.benchmark._fixtures import page_jpg, tree_digest, write


def _segmentation(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _rows(page: str, **extra) -> list[dict]:
    return [
        {"page_image": page, "line_key": "tl2", "bbox": [50.4, 219.6, 700.2, 260.1], "reading_order": 1, "segmentation_source": "seg@1", **extra},
        {"page_image": page, "line_key": "tl1", "polygon": [[50, 100], [700, 100], [700, 140], [50, 140]], "reading_order": 0,
         "segmentation_source": "seg@1", **extra},
    ]


@pytest.fixture
def pages(tmp_path):
    root = tmp_path / "pages"
    write(root, "vol A/p1.jpg", page_jpg())
    return root


def test_build_is_deterministic_hashed_and_carries_no_ground_truth(pages, tmp_path):
    seg = _segmentation(tmp_path / "seg.jsonl", _rows("vol A/p1.jpg"))
    before = tree_digest(pages)
    first = build_dryrun(pages, seg, tmp_path / "d1", dryrun_id="dry")
    second = build_dryrun(pages, seg, tmp_path / "d2", dryrun_id="dry")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert tree_digest(pages) == before
    assert first["label"] == LABEL and first["lines"] == 2 and first["findings"] == {}
    _, lines = read_dryrun(tmp_path / "d1")
    assert [line.line_key for line in lines] == ["tl1", "tl2"]
    assert lines[0].crop.bbox == (50, 100, 700, 140) and lines[1].crop.bbox == (50, 220, 700, 260)
    assert lines[0].document_id.startswith("vol_A-") and lines[0].ground_truth == "NO_GROUND_TRUTH"
    assert not any("gt" in key for key in lines[0].model_dump())
    assert (tmp_path / "d1" / lines[0].line_image_path).read_bytes() == (tmp_path / "d2" / lines[0].line_image_path).read_bytes()


def test_orientation_size_mismatch_and_empty_boxes_become_findings(pages, tmp_path):
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    Image.open(io.BytesIO(page_jpg())).save(buffer, format="JPEG", exif=exif)
    write(pages, "vol A/rot.jpg", buffer.getvalue())
    rows = (_rows("vol A/rot.jpg") + _rows("vol A/p1.jpg", page_width=1000, page_height=800)
            + [{"page_image": "missing.png", "line_key": "x", "bbox": [0, 0, 1, 1], "segmentation_source": "s"}])
    record = build_dryrun(pages, _segmentation(tmp_path / "seg.jsonl", rows), tmp_path / "d", dryrun_id="dry")
    assert record["lines"] == 0
    assert record["findings"] == {"image.exif_orientation": 1, "page.size_mismatch": 1, "page.image_missing": 1}

    empty = [{"page_image": "vol A/p1.jpg", "line_key": "e", "bbox": [900, 10, 950, 20], "segmentation_source": "s"}]
    record = build_dryrun(pages, _segmentation(tmp_path / "seg2.jsonl", empty), tmp_path / "e", dryrun_id="dry")
    assert record["findings"] == {"layout.empty_bbox": 1}


def test_build_never_overwrites_or_writes_into_the_pages(pages, tmp_path):
    seg = _segmentation(tmp_path / "seg.jsonl", _rows("vol A/p1.jpg"))
    build_dryrun(pages, seg, tmp_path / "d", dryrun_id="dry")
    with pytest.raises(DryRunError):
        build_dryrun(pages, seg, tmp_path / "d", dryrun_id="dry")
    with pytest.raises(LayoutError):
        build_dryrun(pages, seg, pages / "out", dryrun_id="dry")


class EchoLionFacade:
    def __init__(self):
        self.calls = []

    def run_inference(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "text": " line ", "elapsed_seconds": 0.1, "device_used": "cpu", "software_environment": {}}


def echo_docker(argv, **kwargs):
    specs = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    host = {t: s[: s.index(":" + t)] for t in ("/output", "/lists") for s in specs if f":{t}" in s}
    listing = Path(host["/lists"]) / argv[argv.index("--inference_list") + 1].split("/lists/")[1]
    rows = [f"{p}\t0.8\ttext {n}\n" for n, p in enumerate(listing.read_text(encoding="utf-8").splitlines())]
    (Path(host["/output"]) / "results.txt").write_text("".join(rows), encoding="utf-8")
    return subprocess.CompletedProcess(argv, 0, "", "")


def test_smoke_runs_use_primary_decoding_and_write_immutable_unscorable_predictions(pages, tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "verify_loghi_checkpoint", lambda d=None: {"checkpoint_dir": str(d), "verified": "faked-in-test"})
    ckpt = tmp_path / "ckpt"
    for name in ("model.keras", "config.json", "tokenizer.json"):
        write(ckpt, name, name)
    directory = tmp_path / "d"
    record = build_dryrun(pages, _segmentation(tmp_path / "seg.jsonl", _rows("vol A/p1.jpg")), directory, dryrun_id="dry")

    facade = EchoLionFacade()
    lion = run_dryrun(directory, LionBackend(facade=facade), limit=1)
    assert lion["label"] == LABEL and lion["lines"] == 1 and lion["model"]["decoding_profile"] == "generation_config"
    assert facade.calls[0]["generation_kwargs"]["num_beams"] == 4 and facade.calls[0]["generation_kwargs"]["no_repeat_ngram_size"] == 3

    loghi_backend = LoghiBackend(checkpoint_dir=ckpt, frozen_dir=directory, runner=echo_docker)
    loghi = run_dryrun(directory, loghi_backend)
    argv = loghi_backend.environment()["argv"]
    assert argv[argv.index("--beam_width") + 1] == "10" and "--greedy" not in argv
    assert loghi["status_counts"] == {"ok": 2}

    rows = read_predictions(directory / "predictions" / "lion.jsonl")
    assert rows[0].benchmark_id == "dryrun:dry" and rows[0].manifest_sha256 == record["manifest_sha256"]
    assert rows[0].prediction == "line" and rows[0].prediction_raw == " line "
    with pytest.raises(RunError):
        run_dryrun(directory, LionBackend(facade=facade), limit=1)
