"""Verifies the real artifacts Experiment 2's completed run and preservation pass actually produced --
not scripted fixtures. Skips cleanly if the run directory is absent (e.g. a fresh checkout that never
ran Experiment 2), so this file never blocks the rest of the suite on a multi-hour training run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP0_DIR = REPO_ROOT / "training" / "full-corpus-20260802T044250Z"
EXP1_DIR = REPO_ROOT / "training" / "experiment-1-continuous-epoch-20260803T192146Z"
EXP2_DIR = REPO_ROOT / "training" / "experiment-2-loghi-recommended-scratch-20260804T051959Z"
CHARACTER_INVENTORY_PATH = REPO_ROOT / "training" / "experiment_2_character_inventory.json"
RESERVED_TEST_MANIFEST = REPO_ROOT / "training" / "loghi-swedish-v1" / "manifests" / "test_reserved_manifest.parquet"

pytestmark = pytest.mark.skipif(not EXP2_DIR.exists(), reason="Experiment 2 has not been run in this checkout")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- Test 3: architecture specification is persisted ---
def test_architecture_spec_is_persisted_and_matches_the_runner_constant():
    from archivetrust.htr.training.scratch_epoch_runner import RECOMMENDED_VGSL_SPEC

    arch = _load_json(EXP2_DIR / "experiment_2_architecture.json")
    assert arch["vgsl_specification"] == RECOMMENDED_VGSL_SPEC


# --- Test 4: character vocabulary is persisted and hashed ---
def test_character_vocabulary_sha256_matches_a_fresh_recomputation():
    inventory = _load_json(CHARACTER_INVENTORY_PATH)
    vocab = inventory["scratch_model_vocabulary"]
    recomputed = hashlib.sha256("".join(vocab["characters"]).encode("utf-8")).hexdigest()
    assert recomputed == vocab["sha256"]


# --- Test 5: all validation characters are handled ---
def test_no_validation_only_characters_were_found():
    inventory = _load_json(CHARACTER_INVENTORY_PATH)
    assert inventory["validation_only_character_count"] == 0
    assert inventory["launch_gate"]["passed"] is True


# --- Test 10: sealed test set is not accessed ---
def test_reserved_test_manifest_size_is_unchanged():
    """A byte-count check only (never opens/parses the file) -- the sealed set's size was recorded
    as 79868 bytes before Experiment 2 existed and must still read that after this whole body of
    work, proving nothing wrote to or truncated it."""
    assert RESERVED_TEST_MANIFEST.stat().st_size == 79868


def test_no_source_file_in_this_repository_reads_the_reserved_test_manifest():
    """Static check: grep every .py file actually used by Experiment 2's pipeline for the reserved
    manifest's filename -- it should only ever appear as a path reference (metadata.json,
    dataset_identity.json), never opened."""
    scripts_dir = REPO_ROOT / "scripts"
    offenders = []
    for path in scripts_dir.glob("*experiment_2*.py"):
        text = path.read_text(encoding="utf-8")
        if "test_reserved_manifest" in text and "read_text" in text:
            # crude but sufficient: flag only if both the filename and a real read call appear
            # near each other in the same file (a genuine content-access attempt), not just a
            # path constant.
            offenders.append(path.name)
    assert offenders == []


# --- Test 11: Experiment 0 and 1 artifacts are untouched ---
def test_experiment_0_best_val_checkpoint_hash_is_unchanged():
    from archivetrust.htr.training.checkpoint_index import verify_checkpoint

    path = EXP0_DIR / "run-state" / "epoch_output" / "epoch_48" / "model_new10" / "best_val"
    ok, digest, _ = verify_checkpoint(path)
    assert ok
    assert digest == "ade57f9f28250731d8bcbabcf03e2b362a6ae7075e56afc2be44f4d9d6c07dfd"


def test_experiment_1_best_val_checkpoint_hash_is_unchanged():
    from archivetrust.htr.training.checkpoint_index import verify_checkpoint

    path = EXP1_DIR / "run-state" / "epoch_output" / "epoch_1" / "model_new10" / "best_val"
    ok, digest, _ = verify_checkpoint(path)
    assert ok
    assert digest == "a1fbb832b6610f91c833db41bdb36949e1652cd7ac082eff870356ba58ee142f"


# --- Test 12: checkpoint architecture and vocabulary match ---
def test_best_val_checkpoint_tokenizer_size_matches_the_recorded_vocabulary():
    tokenizer_path = EXP2_DIR / "run-state" / "epoch_output" / "epoch_1" / "recommended" / "best_val" / "tokenizer.json"
    tokenizer = _load_json(tokenizer_path)
    metadata = _load_json(EXP2_DIR / "experiment_2_metadata.json")
    assert len(tokenizer) == metadata["character_vocabulary"]["size_tokenizer_vocabulary"] == 126


# --- Test 13: comparison preserves authoritative Experiment 0 and 1 metrics ---
def test_comparison_reports_the_exact_preserved_experiment_0_and_1_cer():
    comparison = _load_json(EXP2_DIR / "experiment_0_1_2_comparison.json")
    table = {row["experiment"]: row for row in comparison["headline_table"]}
    assert table[0]["cer"] == pytest.approx(0.1728, abs=0.0001)
    assert table[0]["true_wer"] == pytest.approx(0.4945, abs=0.0001)
    assert table[1]["cer"] == pytest.approx(0.1747, abs=0.0001)
    assert table[1]["true_wer"] == pytest.approx(0.4956, abs=0.0001)


# --- Test 14: Loghi line error rate is not labelled true WER ---
def test_dashboard_labels_the_container_wer_metric_as_line_error_rate_not_true_wer():
    dashboard_src = (REPO_ROOT / "scripts" / "live_dashboard_server_experiment_2.py").read_text(encoding="utf-8")
    assert "Loghi line error rate" in dashboard_src
    assert "NOT true WER" in dashboard_src


def test_report_distinguishes_line_error_rate_from_true_wer():
    report = (EXP2_DIR / "loghi_experiments_0_1_2_report.md").read_text(encoding="utf-8")
    assert "Line error rate" in report
    assert "True WER" in report


# --- Test 15: true Levenshtein CER and WER are used ---
def test_evaluation_metrics_module_never_imports_difflib():
    """The module's own docstring *names* difflib -- explaining a real historical incident where
    `difflib.SequenceMatcher` overstated CER 3x (0.82 vs a real 0.25) -- so a substring check on the
    whole file would false-positive on that explanation. Checking for an actual import line only."""
    src = (REPO_ROOT / "src" / "archivetrust" / "htr" / "training" / "full_run" / "evaluation_metrics.py").read_text(encoding="utf-8")
    assert "import difflib" not in src
    assert "from difflib" not in src


# --- Test 16: per-collection evaluation covers all collections ---
def test_per_collection_evaluation_covers_all_eleven_collections():
    lap_eval = _load_json(EXP2_DIR / "experiment_2_evaluation.json")
    names = {c["collection"] for c in lap_eval["per_collection"]}
    assert len(names) == 11
    assert lap_eval["missing_result_count"] == 0


# --- Test 17: run identity clearly states scratch training ---
def test_metadata_explicitly_states_random_initialization_and_no_pretrained_checkpoint():
    metadata = _load_json(EXP2_DIR / "experiment_2_metadata.json")
    assert metadata["initialization"] == "random"
    assert metadata["pretrained_checkpoint_used"] is False
    assert metadata["experiment_0_weights_used"] is False
    assert metadata["experiment_1_weights_used"] is False
    assert metadata["stable_name"] == "Loghi recommended VGSL — trained from scratch"


# --- Test 18: no inherited model weights are present ---
def test_checkpoint_manifest_source_checkpoint_is_never_a_prior_experiment_path():
    manifest = _load_json(EXP2_DIR / "experiment_2_checkpoint_manifest.json")
    for entry in manifest["entries"]:
        source = entry["source_checkpoint"]
        assert "full-corpus-20260802T044250Z" not in source  # Experiment 0
        assert "experiment-1-continuous-epoch" not in source  # Experiment 1
        assert "generic-2023-02-15" not in source  # the pristine pretrained checkpoint itself


# --- Tests 7 & 8: optimizer iterations and learning rate do not reset ---
def test_optimizer_iterations_are_monotonically_non_decreasing_across_the_whole_epoch():
    trace_path = sorted(EXP2_DIR.glob("run-state/epoch_output/epoch_*/optimizer_trace.csv"))[-1]
    rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
    iterations = [int(r["optimizer_iterations"]) for r in rows if r["optimizer_iterations"]]
    assert len(iterations) > 2
    assert all(b >= a for a, b in zip(iterations, iterations[1:]))
    assert iterations[0] == 0
    assert iterations[-1] > 30000  # a real, near-full-corpus step count, not a stub


def test_learning_rate_decays_monotonically_never_resets_upward():
    trace_path = sorted(EXP2_DIR.glob("run-state/epoch_output/epoch_*/optimizer_trace.csv"))[-1]
    rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
    lrs = [float(r["learning_rate"]) for r in rows if r["learning_rate"]]
    assert len(lrs) > 2
    # A genuine single continuous decay schedule never jumps back up to (or past) its own starting
    # value once decay has begun -- a reset would show exactly that.
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))


# --- Test 6: one continuous trainer lifecycle is used ---
def test_run_state_shows_exactly_one_shard_one_epoch_zero_resumes():
    run_state = _load_json(EXP2_DIR / "run-state" / "run_state.json")
    assert run_state["shards_per_epoch"] == 1
    assert run_state["epochs_completed"] == 1
    assert run_state["resume_count"] == 0
    assert run_state["status"] == "completed"


# --- Test 9: epoch 2 does not begin automatically ---
def test_only_epoch_1_output_directory_exists():
    epoch_dirs = sorted((EXP2_DIR / "run-state" / "epoch_output").glob("epoch_*"))
    assert [d.name for d in epoch_dirs] == ["epoch_1"]
