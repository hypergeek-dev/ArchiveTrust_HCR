"""Real-execution integration test for the Swedish Historical HTR Baseline Comparison
(`htr/experiment/baseline_execution.py`, docs/htr-migration-plan.md Stage 12).

Marked `real_model` (registered in `pyproject.toml`), same convention as
`tests/providers/satrn/test_real_inference.py` / `tests/providers/florence2_htr/
test_real_inference.py`: downloads/loads the real SATRN and Florence-2 checkpoints and runs real
forward passes (SATRN via the isolated `.venv-satrn` subprocess, Florence-2 via this venv's
`transformers`), and parses the real, hand-authored Transkribus PAGE XML fixture. Requires the same
environment as those two test files (see their module docstrings) -- skipped, not failed, when
either prerequisite is missing.
"""

from __future__ import annotations

import pytest

from archivetrust.htr.experiment.baseline_execution import (
    InputCropHashMismatchError,
    run_baseline_comparison,
)
from archivetrust.providers.florence2_htr.facade import florence2_dependencies_available
from archivetrust.providers.satrn.facade import satrn_python_available

pytestmark = pytest.mark.real_model

_satrn_ok, _satrn_message = satrn_python_available()
_florence2_ok, _florence2_message = florence2_dependencies_available()
requires_local_methods = pytest.mark.skipif(
    not (_satrn_ok and _florence2_ok),
    reason=f"satrn_available={_satrn_ok} ({_satrn_message}); florence2_available={_florence2_ok} ({_florence2_message})",
)


@requires_local_methods
def test_controlled_comparison_uses_a_byte_identical_input_crop_across_both_local_methods():
    result = run_baseline_comparison()

    # The brief's explicit hard requirement: both local recognizers received exactly the same
    # image input, verified by a matching content hash -- not merely assumed because they were
    # pointed at "the same file path" in source code.
    assert result.satrn.method_run.input_crop_id == result.shared_crop.crop_id
    assert result.florence2.method_run.input_crop_id == result.shared_crop.crop_id
    assert result.shared_crop.hash.startswith("crop_")
    # Recomputing the hash from the fixture's own bytes must match what was stored -- proves the
    # stored hash is not a fabricated/mismatched value.
    from archivetrust.htr.corpus.models import InputCrop
    from archivetrust.htr.experiment.baseline_execution import DEFAULT_LINE_FIXTURE_IMAGE

    assert InputCrop.compute_hash(DEFAULT_LINE_FIXTURE_IMAGE.read_bytes()) == result.shared_crop.hash


@requires_local_methods
def test_both_local_methods_produce_real_non_empty_output():
    result = run_baseline_comparison()

    assert result.satrn.transcript.raw_text is not None
    assert result.satrn.transcript.raw_text.strip() != ""
    assert result.florence2.transcript.parsed_text is not None
    assert result.florence2.transcript.parsed_text.strip() != ""
    # Real, not fabricated, execution telemetry.
    assert result.satrn.evidence is not None
    assert result.satrn.evidence.execution_time_ms is not None
    assert result.satrn.evidence.execution_time_ms > 0
    assert result.florence2.evidence is not None
    assert result.florence2.evidence.execution_time_ms is not None
    assert result.florence2.evidence.execution_time_ms > 0


@requires_local_methods
def test_metrics_are_computed_for_both_local_methods_against_real_ground_truth():
    result = run_baseline_comparison()

    assert result.satrn.metrics is not None
    assert 0.0 <= result.satrn.metrics.character_error_rate_normalized
    assert result.satrn.metrics.reference == result.ground_truth_text
    assert result.satrn.metrics.hypothesis == result.satrn.transcript.normalized_text

    assert result.florence2.metrics is not None
    assert 0.0 <= result.florence2.metrics.character_error_rate_normalized
    assert result.florence2.metrics.reference == result.ground_truth_text

    # Real MetricResult rows were actually produced and are queryable from the store, not just
    # held on the in-memory dataclass.
    satrn_stored_metrics = result.store.metric_results(method_run_id=result.satrn.method_run.method_run_id)
    florence2_stored_metrics = result.store.metric_results(method_run_id=result.florence2.method_run.method_run_id)
    assert len(satrn_stored_metrics) > 0
    assert len(florence2_stored_metrics) > 0


@requires_local_methods
def test_reproducibility_manifest_has_real_captured_environment_fields():
    result = run_baseline_comparison()
    manifest = result.manifest

    assert manifest.experiment_run_id == result.controlled_run.experiment_run_id
    # Real, not placeholder: python/platform are always populated by sys.version/platform.platform().
    assert "python" in manifest.software_environment
    assert "3." in manifest.software_environment["python"]
    assert "platform" in manifest.software_environment
    assert manifest.software_environment.get("torch", "").startswith(("2.", "not importable"))
    # This session's real, verified environment has torch importable with a real version string,
    # not "not importable" -- assert the strong form since we know this dev environment.
    assert manifest.software_environment["torch"] != "not importable in this process"
    assert manifest.hardware_environment.get("gpu_name") not in (None, "")
    # Adapter versions are real, non-empty strings pulled from each adapter module's own constant.
    assert manifest.software_environment["satrn_adapter_version"]
    assert manifest.software_environment["florence2_adapter_version"]
    assert manifest.software_environment["transkribus_adapter_version"]
    assert manifest.pipeline_configuration_hash is not None


@requires_local_methods
def test_transkribus_result_is_flagged_as_page_level_only_not_controlled():
    result = run_baseline_comparison()

    # The brief's explicit hard requirement: Transkribus must never be silently presented as a
    # controlled, hash-matched comparison it has no geometric correspondence to.
    assert result.transkribus.method_run.input_crop_id is None
    assert result.transkribus.method_run.experiment_run_id == result.end_to_end_run.experiment_run_id
    assert result.end_to_end_run.is_end_to_end is True
    assert result.controlled_run.is_end_to_end is False
    assert result.transkribus.method_run.experiment_run_id != result.controlled_run.experiment_run_id
    # And no fabricated CER was computed for it against unrelated ground truth.
    assert result.transkribus.metrics is None
    assert result.transkribus.transcript.parsed_text is not None
    assert result.transkribus.transcript.parsed_text.strip() != ""


@requires_local_methods
def test_full_store_graph_is_queryable_after_the_run():
    """Everything `run_baseline_comparison` registers must be readable back from the store -- not
    just held on the returned dataclass -- since that store is what a report generator or a future
    ViewModel would actually query."""
    result = run_baseline_comparison()
    store = result.store

    assert store.experiment(result.built_experiment.experiment.experiment_id) is not None
    assert len(store.method_runs(experiment_run_id=result.controlled_run.experiment_run_id)) == 2
    assert len(store.method_runs(experiment_run_id=result.end_to_end_run.experiment_run_id)) == 1
    assert store.manifests(experiment_run_id=result.controlled_run.experiment_run_id) == (result.manifest,)

    controlled_text_lines = [
        line for line in store.text_lines() if store.ground_truth_for_line(line.text_line_id) is not None
    ]
    assert len(controlled_text_lines) == 1
    assert store.ground_truth_for_line(controlled_text_lines[0].text_line_id) == result.ground_truth_text


@requires_local_methods
def test_input_crop_hash_mismatch_error_is_a_real_catchable_type():
    """Not exercised via a forced mismatch (both real adapters always read the same file in this
    module's own orchestration, so a real mismatch cannot be induced without breaking the
    function's own contract) -- this just proves the guard type exists and is importable, so a
    future refactor that could introduce a mismatch has a named exception to raise/catch."""
    assert issubclass(InputCropHashMismatchError, RuntimeError)
