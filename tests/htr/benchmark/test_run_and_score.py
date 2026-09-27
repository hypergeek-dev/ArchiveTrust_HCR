import json
import random
import subprocess
from pathlib import Path

import pytest

from archivetrust.evaluation.metrics import levenshtein
from archivetrust.htr.benchmark import inference
from archivetrust.htr.benchmark.build import build_candidate, freeze
from archivetrust.htr.benchmark.inference import LionBackend, LoghiBackend, RunError, read_predictions, run_model
from archivetrust.htr.benchmark.model_registry import LION, LOGHI, load_loghi_charset, loghi_checkpoint_dir, verify_loghi_checkpoint
from archivetrust.htr.benchmark.pairwise import align, tag_op
from archivetrust.htr.benchmark.report import score_run
from archivetrust.htr.benchmark.scoring import ScoringError
from archivetrust.htr.evaluation.recognition import classify_char_edits
from tests.htr.benchmark._fixtures import line_png, write

GT = {"l1": "Anno 1723", "l2": "Kongl. Maj:t", "l3": "Stockholm", "l4": "den 4 Maj"}
LION_OUT = {"l1": "Anno 1728", "l2": " Kongl. Maj:t", "l3": "", "l4": "den 4 maj"}  # l3 -> empty_output
LOGHI_OUT = {"l1": "Anno 1723", "l2": "Kongl Majt", "l4": "den 4 Maj"}  # l3 skipped by the container


@pytest.fixture
def frozen(tmp_path):
    src = tmp_path / "incoming" / "s"
    for n, key in enumerate(GT):
        write(src, f"d{n % 2}/{key}.png", line_png(GT[key], seed=n))
        write(src, f"d{n % 2}/{key}.gt.txt", GT[key])
    build_candidate(src, tmp_path / "cand", dataset_id="ds", source_id="s")
    freeze(tmp_path / "cand", tmp_path / "bench", benchmark_id="b1")
    return tmp_path / "bench"


class FakeLionFacade:
    def __init__(self):
        self.calls = []

    def run_inference(self, **kwargs):
        self.calls.append(kwargs)
        key = Path(kwargs["image_path"]).stem
        if key == "l3":
            return {"ok": False, "category": "empty_output", "message": "nothing"}
        return {"ok": True, "text": LION_OUT[key], "elapsed_seconds": 0.1, "device_used": "cpu", "software_environment": {}}


def fake_docker(argv, **kwargs):
    specs = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    host = {t: s[: s.index(":" + t)] for t in ("/model", "/output", "/lists", "/benchmark") for s in specs if f":{t}" in s}
    listing = Path(host["/lists"]) / argv[argv.index("--inference_list") + 1].split("/lists/")[1]
    out = []
    for path in listing.read_text(encoding="utf-8").splitlines():
        key = Path(path).stem
        if key in LOGHI_OUT:
            out.append(f"{path}\t0.9\t{LOGHI_OUT[key]}\n")
    (Path(host["/output"]) / "results.txt").write_text("".join(out), encoding="utf-8")
    return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def no_checkpoint_hashing(monkeypatch, tmp_path):
    ckpt = tmp_path / "ckpt"
    for name in ("model.keras", "config.json", "tokenizer.json"):
        write(ckpt, name, name)
    monkeypatch.setattr(inference, "verify_loghi_checkpoint", lambda d=None: {"checkpoint_dir": str(d), "verified": "faked-in-test"})
    return ckpt


def _run_both(frozen, report, ckpt):
    facade = FakeLionFacade()
    run_model(frozen, report, LionBackend(facade=facade))
    loghi = LoghiBackend(checkpoint_dir=ckpt, frozen_dir=frozen, runner=fake_docker)
    run_model(frozen, report, loghi)
    return facade, loghi


def test_both_models_run_on_identical_frozen_images_with_explicit_decoding(frozen, tmp_path, no_checkpoint_hashing):
    report = tmp_path / "reports" / "r1"
    facade, loghi = _run_both(frozen, report, no_checkpoint_hashing)
    assert facade.calls[0]["generation_kwargs"] == LION.profiles["generation_config"].parameters
    argv = loghi.environment()["argv"]
    assert argv[argv.index("--beam_width") + 1] == "10" and "--greedy" not in argv
    assert any(a.endswith(":/benchmark:ro") for a in argv)

    lion = {p.line_id.rsplit("/", 1)[1]: p for p in read_predictions(report / "predictions" / "lion.jsonl")}
    loghi_p = {p.line_id.rsplit("/", 1)[1]: p for p in read_predictions(report / "predictions" / "loghi.jsonl")}
    assert set(lion) == set(loghi_p) == set(GT)
    assert {k: p.image_sha256 for k, p in lion.items()} == {k: p.image_sha256 for k, p in loghi_p.items()}
    assert lion["l3"].status == "empty" and lion["l3"].prediction == ""
    assert lion["l2"].prediction_raw == " Kongl. Maj:t" and lion["l2"].prediction == "Kongl. Maj:t"
    assert loghi_p["l3"].status == "missing" and loghi_p["l3"].prediction == ""
    assert loghi_p["l1"].confidence == 0.9 and loghi_p["l1"].duration_kind == "batch_amortized"
    with pytest.raises(RunError, match="immutable"):
        run_model(frozen, report, LionBackend(facade=FakeLionFacade()))


def test_scores_count_failures_as_deletions_and_report_is_written(frozen, tmp_path, no_checkpoint_hashing):
    report = tmp_path / "reports" / "r1"
    _run_both(frozen, report, no_checkpoint_hashing)
    scores = score_run(frozen, report, loghi_charset=set("Ano 1723KngljMStckhdem.:"))
    total_chars = sum(len(t) for t in GT.values())
    lion_edits = sum(levenshtein(GT[k], LION_OUT[k].strip()) for k in GT)
    loghi_edits = sum(levenshtein(GT[k], LOGHI_OUT.get(k, "")) for k in GT)
    assert scores["corpus"]["lion"]["cer"] == pytest.approx(lion_edits / total_chars)
    assert scores["corpus"]["loghi"]["cer"] == pytest.approx(loghi_edits / total_chars)
    assert scores["corpus"]["loghi"]["char_deletions"] >= len(GT["l3"])
    assert scores["corpus"]["loghi"]["failed_or_missing_lines"] == 1 and scores["corpus"]["lion"]["empty_predictions"] == 1
    assert scores["bootstrap"]["cluster_unit"] == "line" and "warning" in scores["bootstrap"]
    assert set(scores["pairwise"]["categories"]) >= {"only_loghi_correct"}
    md = (report / "report.md").read_text(encoding="utf-8")
    assert "UNOFFICIAL" in md and "Paired difference" in md and "failures scored as empty" in md
    assert scores["loghi_out_of_vocabulary"]["characters"] == ["4", "a"]


def test_partial_or_foreign_predictions_cannot_be_scored(frozen, tmp_path, no_checkpoint_hashing):
    report = tmp_path / "reports" / "partial"
    run_model(frozen, report, LionBackend(facade=FakeLionFacade()), limit=2)
    run_model(frozen, report, LoghiBackend(checkpoint_dir=no_checkpoint_hashing, frozen_dir=frozen, runner=fake_docker))
    with pytest.raises(ScoringError, match="missing"):
        score_run(frozen, report)
    with pytest.raises(RunError, match="official"):
        run_model(frozen, tmp_path / "reports" / "x", LionBackend(facade=FakeLionFacade()), limit=2, official=True)


def test_align_agrees_with_existing_edit_counts():
    rng = random.Random(3)
    alphabet = "ab cåÅ.:1"
    for _ in range(300):
        ref = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        hyp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        ops = align(ref, hyp)
        counts = classify_char_edits(ref, hyp)
        assert [sum(1 for o in ops if o[0] == k) for k in ("match", "sub", "ins", "del")] == [
            counts.matches, counts.substitutions, counts.insertions, counts.deletions]
        assert sum(1 for o in ops if o[0] != "match") == levenshtein(ref, hyp)


@pytest.mark.parametrize("ref,hyp,tag", [
    ("a b", "ab", "spacing"), ("Maj", "maj", "case"), ("år", "ar", "diacritic"), ("a.", "a,", "punctuation"),
    ("1723", "1728", "digit"), ("ſtor", "stor", "special_character"), ("Maj:t", "Maj:k", "abbreviation_word"),
    ("hus", "hos", "letter"),
])
def test_error_tags(ref, hyp, tag):
    [op] = [o for o in align(ref, hyp) if o[0] != "match"]
    assert tag_op(op, ref) == tag


@pytest.mark.skipif(not (loghi_checkpoint_dir() / "model.keras").is_file(), reason="frozen checkpoint not on this machine")
def test_real_frozen_checkpoint_matches_pinned_identity_and_charset():
    assert verify_loghi_checkpoint()["verified"] is True
    charset = load_loghi_charset()
    assert len(charset) == 124 and {"å", "ä", "ö", " "} <= charset
    assert LOGHI.profile(None).parameters["beam_width"] == 10
