from __future__ import annotations

import inspect
import importlib

from archivetrust.composition import AppContext
from archivetrust.evaluation import evaluate, ground_truth, workflow
from archivetrust.providers.registry import ProviderRegistry


def test_evaluation_modules_do_not_import_production_reconciliation_or_provider_registry():
    forbidden = (
        "archivetrust.domain.comparison",
        "archivetrust.providers.registry",
        "archivetrust.providers.ground_truth",
        "ReconciliationEngine",
        "ProviderAdapter",
    )
    sources = "\n".join(inspect.getsource(module) for module in (ground_truth, workflow, evaluate))
    for symbol in forbidden:
        assert symbol not in sources


def test_production_registry_has_no_evaluation_reference_registration_api():
    public = {name for name in dir(ProviderRegistry) if not name.startswith("_")}
    assert "register_evaluation_reference" not in public
    assert "register_ground_truth" not in public


def test_removed_ground_truth_adapter_cannot_be_imported():
    try:
        importlib.import_module("archivetrust.providers.ground_truth.adapter")
    except ModuleNotFoundError:
        return
    raise AssertionError("removed GroundTruthAdapter module remains importable")


def test_production_composition_contains_no_reference_provider(tmp_path):
    context = AppContext(deployment_root=tmp_path)
    provider_ids = {adapter.provider_id for adapter in context.provider_registry.all()}
    assert "ground_truth" not in provider_ids
    assert "evaluation_reference" not in provider_ids


def test_evaluation_and_operational_review_use_separate_durable_stores(tmp_path):
    context = AppContext(deployment_root=tmp_path)
    evaluation = context.evaluation_approval_service

    assert evaluation.assignments.path.parent.name == "evaluation"
    assert evaluation.annotations.path.parent.name == "evaluation"
    assert evaluation.annotations.path != context._current_layout.telemetry_dir / "events.jsonl"  # noqa: SLF001
    assert evaluation.annotations.path != context._current_layout.telemetry_dir / "review_interactions.jsonl"  # noqa: SLF001
