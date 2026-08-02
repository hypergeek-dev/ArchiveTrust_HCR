from __future__ import annotations

import json

import pytest

from archivetrust.htr.training.full_run.lap_evaluation import (
    LineResult,
    build_inference_argv,
    collection_of,
    evaluate_lap,
    join_predictions,
    parse_ground_truth_list,
    parse_results_file,
    per_collection_metrics,
    render_report,
    save_lap_evaluation,
    stratified_inspection_sample,
)

REAL_IMAGE_NAME = (
    "/lists/images/val/alvsborgs_losen_lines__alvsborgs_losen_lines_1.parquet__2721.png"
)


def test_collection_is_read_from_the_real_extracted_filename_shape():
    assert collection_of(REAL_IMAGE_NAME) == "alvsborgs_losen"
    assert collection_of(
        "/lists/images/val/goteborgs_poliskammare_fore_1900_lines__x_lines_19.parquet__4406.png"
    ) == "goteborgs_poliskammare_fore_1900"


def test_unrecognised_filename_is_labelled_not_silently_grouped():
    assert collection_of("/lists/images/val/mystery.png") == "unknown"


def test_ground_truth_with_tabs_is_not_truncated():
    """Transcriptions really do contain tabs; splitting on every tab would silently drop text and
    make the CER look better than it is."""
    parsed = parse_ground_truth_list("img.png\tSuin\ta\t5 mk\n")
    assert parsed["img.png"] == "Suin\ta\t5 mk"


def test_results_file_parses_the_real_three_column_format():
    text = f"{REAL_IMAGE_NAME}\t0.4248824715614319\ttrnds bop i 57\n"
    parsed = parse_results_file(text)
    confidence, prediction = parsed[REAL_IMAGE_NAME]
    assert confidence == pytest.approx(0.4248824715614319)
    assert prediction == "trnds bop i 57"


def test_an_empty_prediction_is_a_real_result_and_is_kept():
    """Dropping empty predictions would improve every metric by discarding the worst lines."""
    parsed = parse_results_file("img.png\t0.1\t\n")
    assert parsed["img.png"] == (0.1, "")


def test_lines_with_no_prediction_are_reported_as_missing_not_dropped():
    truth = {"a.png": "abc", "b.png": "def"}
    results = {"a.png": (0.9, "abc")}
    joined, missing = join_predictions(truth, results)
    assert [r.image_path for r in joined] == ["a.png"]
    assert missing == ["b.png"]


def test_cer_is_computed_per_line_on_true_edit_distance():
    joined, _ = join_predictions({"a.png": "kitten"}, {"a.png": (0.5, "sitting")})
    assert joined[0].cer == pytest.approx(3 / 6)


def _line(collection: str, truth: str, pred: str, conf: float = 0.5, idx: int = 0) -> LineResult:
    from archivetrust.htr.training.full_run.evaluation_metrics import character_error_rate

    path = f"/lists/images/val/{collection}_lines__{collection}_lines_1.parquet__{idx}.png"
    return LineResult(image_path=path, collection=collection, ground_truth=truth,
                      prediction=pred, confidence=conf, cer=character_error_rate(truth, pred))


def test_per_collection_metrics_put_the_worst_subgroup_first():
    lines = [
        _line("good_collection", "abcdefghij", "abcdefghij", idx=1),
        _line("alvsborgs_losen", "abcdefghij", "zzzzzzzzzz", idx=2),
    ]
    out = per_collection_metrics(lines)
    assert out[0].collection == "alvsborgs_losen"
    assert out[0].metrics.corpus_cer == pytest.approx(1.0)
    assert out[-1].collection == "good_collection"


def test_a_corpus_average_cannot_hide_a_collapsing_subgroup():
    """The point of per-collection reporting: 9 healthy collections plus 1 broken one still looks
    acceptable in aggregate, and the per-collection table is what surfaces it."""
    lines = [_line(f"c{i}", "abcdefghij", "abcdefghij", idx=i) for i in range(9)]
    lines.append(_line("broken", "abcdefghij", "zzzzzzzzzz", idx=99))
    out = per_collection_metrics(lines)
    assert out[0].collection == "broken" and out[0].metrics.corpus_cer == pytest.approx(1.0)


def test_inspection_sample_covers_every_collection():
    """The real failure this replaces: reading the head of val_list.txt sampled one collection,
    because the list is grouped by collection."""
    lines = [_line(c, "abc", "abd", idx=i)
             for c in ("alpha", "beta", "gamma") for i in range(20)]
    sample = stratified_inspection_sample(lines, per_collection=3, seed_material="h")
    assert {r.collection for r in sample} == {"alpha", "beta", "gamma"}
    assert len(sample) == 9


def test_inspection_sample_is_reproducible_for_the_same_checkpoint():
    lines = [_line("alpha", "abc", "abd", idx=i) for i in range(50)]
    a = stratified_inspection_sample(lines, per_collection=5, seed_material="checkpoint-hash")
    b = stratified_inspection_sample(lines, per_collection=5, seed_material="checkpoint-hash")
    assert [r.image_path for r in a] == [r.image_path for r in b]


def test_inspection_sample_does_not_depend_on_input_ordering():
    """Two laps must be comparable even if the join produced rows in a different order."""
    lines = [_line("alpha", "abc", "abd", idx=i) for i in range(30)]
    a = stratified_inspection_sample(lines, per_collection=4, seed_material="h")
    b = stratified_inspection_sample(list(reversed(lines)), per_collection=4, seed_material="h")
    assert [r.image_path for r in a] == [r.image_path for r in b]


def test_different_checkpoints_sample_differently():
    lines = [_line("alpha", "abc", "abd", idx=i) for i in range(50)]
    a = stratified_inspection_sample(lines, per_collection=5, seed_material="lap1")
    b = stratified_inspection_sample(lines, per_collection=5, seed_material="lap2")
    assert [r.image_path for r in a] != [r.image_path for r in b]


def test_a_collection_smaller_than_the_quota_contributes_what_it_has():
    lines = [_line("tiny", "abc", "abd", idx=0), _line("big", "abc", "abd", idx=1),
             _line("big", "abc", "abd", idx=2), _line("big", "abc", "abd", idx=3)]
    sample = stratified_inspection_sample(lines, per_collection=3, seed_material="h")
    assert len([r for r in sample if r.collection == "tiny"]) == 1
    assert len([r for r in sample if r.collection == "big"]) == 3


def _evaluation(**overrides):
    gt_lines, res_lines = [], []
    for c in ("alvsborgs_losen", "svea_hovratt"):
        for i in range(10):
            path = f"/lists/images/val/{c}_lines__{c}_lines_1.parquet__{i}.png"
            truth = "the quick brown fox"
            pred = truth if c == "svea_hovratt" else "the quick brown box"
            conf = 0.95 if c == "svea_hovratt" else 0.35
            gt_lines.append(f"{path}\t{truth}")
            res_lines.append(f"{path}\t{conf}\t{pred}")
    kwargs = dict(
        run_id="r1", checkpoint_dir="/ckpt", checkpoint_hash="deadbeef",
        epochs_completed=1, global_shards_completed=57, shards_per_epoch=57,
        ground_truth_text="\n".join(gt_lines), results_text="\n".join(res_lines),
        comparison={"pilot_9999_line_pass_cer": 0.2481},
        inspection_per_collection=2,
    )
    kwargs.update(overrides)
    return evaluate_lap(**kwargs)


def test_evaluate_lap_produces_overall_and_per_collection_evidence():
    e = _evaluation()
    assert e.scored_line_count == 20
    assert e.missing_result_count == 0
    assert e.worst_collection == "alvsborgs_losen"
    assert e.best_collection == "svea_hovratt"
    assert e.overall.corpus_cer > 0
    assert len(e.inspection_sample) == 4


def test_evaluate_lap_reports_a_short_inference_pass_rather_than_renormalising():
    """If inference returned fewer lines than the validation set, the count must surface -- otherwise
    a partially-failed pass reads as a clean, better-looking result."""
    e = _evaluation(results_text="")
    assert e.scored_line_count == 0
    assert e.missing_result_count == 20


def test_calibration_separates_the_confident_and_unconfident_lines():
    e = _evaluation()
    high = [b for b in e.confidence_buckets if b.lower == 0.8][0]
    low = [b for b in e.confidence_buckets if b.lower == 0.2][0]
    assert high.sample_count == 10 and low.sample_count == 10
    assert high.mean_cer < low.mean_cer, "higher confidence must track lower error to be calibrated"


def test_rejection_coverage_is_reported_across_thresholds():
    e = _evaluation()
    rows = {r["threshold"]: r for r in e.rejection_coverage}
    assert rows[0.0]["coverage"] == 1.0
    assert rows[0.9]["accepted_count"] == 10


def test_report_renders_every_required_section():
    text = render_report(_evaluation())
    for heading in ("## Overall", "## Per collection", "## Confidence calibration",
                    "### Rejection coverage", "## Inspection sample",
                    "## Comparison with prior reference points"):
        assert heading in text, f"missing {heading}"
    assert "alvsborgs_losen" in text


def test_report_states_when_lines_were_not_scored():
    text = render_report(_evaluation(results_text=""))
    assert "produced no result" in text


def test_saved_evaluation_round_trips(tmp_path):
    from archivetrust.htr.training.full_run.lap_evaluation import LapEvaluation

    path = tmp_path / "nested" / "lap_evaluation.json"
    e = _evaluation()
    save_lap_evaluation(path, e)
    reloaded = LapEvaluation.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert reloaded.overall.corpus_cer == pytest.approx(e.overall.corpus_cer)
    assert reloaded.worst_collection == e.worst_collection


def test_inference_argv_mounts_the_list_parent_not_the_list_itself(tmp_path):
    """The real mount trap this project already hit once: image paths inside the list are written as
    `/lists/images/...`, so `/lists` must be the list file's parent directory."""
    lst = tmp_path / "pool" / "val_list.txt"
    lst.parent.mkdir(parents=True)
    lst.write_text("x")
    argv = build_inference_argv(
        image_ref="img@sha256:abc", model_host=tmp_path / "model",
        output_host=tmp_path / "out", inference_list_host=lst, batch_size=16,
    )
    assert f"{lst.parent}:/lists:ro" in argv
    assert "--inference_list" in argv
    assert argv[argv.index("--inference_list") + 1] == "/lists/val_list.txt"
    assert "--results_file" in argv


def test_inference_argv_does_not_train():
    argv = build_inference_argv(
        image_ref="img", model_host=None, output_host=None,
        inference_list_host=__import__("pathlib").Path("p/val_list.txt"), batch_size=8,
    )
    for training_flag in ("--train_list", "--do_validate", "--epochs", "--output_checkpoints",
                          "--learning_rate", "--optimizer"):
        assert training_flag not in argv, f"{training_flag} would make this a training run"
