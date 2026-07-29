from __future__ import annotations

import json

from archivetrust.runtime.contracts import RuntimeCapabilities, Precision
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.model_manager import (
    IncompatibleModelError,
    scan_installed_models,
    validate_compatibility,
)
import pytest


def test_scan_returns_empty_tuple_when_models_dir_missing(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path)  # not ensured -- models_dir doesn't exist
    assert scan_installed_models(layout) == ()


def test_scan_detects_a_manifested_model(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    model_dir = layout.models_dir / "qwen2.5-vl-7b"
    model_dir.mkdir()
    (model_dir / "archivetrust_model.json").write_text(
        json.dumps({"model_id": "qwen2.5-vl-7b-instruct", "display_name": "Qwen2.5-VL 7B", "runtime_kind": "transformers", "revision": "main"})
    )
    (model_dir / "weights.bin").write_bytes(b"x" * 1024)

    installed = scan_installed_models(layout)
    assert len(installed) == 1
    assert installed[0].descriptor.model_id == "qwen2.5-vl-7b-instruct"
    assert installed[0].descriptor.runtime_kind == "transformers"
    assert installed[0].size_bytes > 1024  # includes weights.bin plus the manifest file itself
    assert installed[0].sha256 is not None


def test_scan_detects_an_unmanifested_model_as_unknown_runtime(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    (layout.models_dir / "mystery-model").mkdir()
    installed = scan_installed_models(layout)
    assert len(installed) == 1
    assert installed[0].descriptor.runtime_kind == "unknown"


def test_hash_skipped_for_large_directories(tmp_path, monkeypatch) -> None:
    import archivetrust.runtime.model_manager as mm

    monkeypatch.setattr(mm, "_HASH_SIZE_LIMIT_BYTES", 10)  # force "too large"
    layout = DeploymentLayout(root=tmp_path).ensure()
    model_dir = layout.models_dir / "big-model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"x" * 1024)
    installed = scan_installed_models(layout)
    assert installed[0].sha256 is None  # never fabricated


def test_validate_compatibility_rejects_unknown_runtime(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    (layout.models_dir / "mystery").mkdir()
    installed = scan_installed_models(layout)[0]
    caps = RuntimeCapabilities(
        supports_gpu=True, supports_cpu=True, supported_precisions=(Precision.AUTOMATIC,),
        supports_batching=False, supports_deterministic_seed=False, max_tokens_configurable=True,
    )
    with pytest.raises(IncompatibleModelError):
        validate_compatibility(installed, caps)


def test_compute_hashes_false_skips_hashing_entirely(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    model_dir = layout.models_dir / "m"
    model_dir.mkdir()
    (model_dir / "f.bin").write_bytes(b"abc")
    installed = scan_installed_models(layout, compute_hashes=False)
    assert installed[0].sha256 is None
