from __future__ import annotations

from pathlib import Path

import pytest

from archivetrust.runtime.contracts import DeviceSelection, InferenceRequest, Precision
from archivetrust.runtime.transformers_runtime import TransformersRuntime

from tests.runtime._fakes import FakeTorchAndTransformersFacade


def _runtime(facade, device=DeviceSelection.AUTOMATIC, precision=Precision.AUTOMATIC) -> TransformersRuntime:
    return TransformersRuntime(
        model_id="qwen2.5-vl-7b-instruct",
        local_path=Path("/models/qwen"),
        revision="main",
        device=device,
        precision=precision,
        facade=facade,
    )


def test_automatic_device_prefers_gpu_when_available() -> None:
    facade = FakeTorchAndTransformersFacade(cuda_available=True)
    runtime = _runtime(facade)
    runtime.warm_up()
    assert facade.loaded["device"] == "cuda"


def test_automatic_device_falls_back_to_cpu() -> None:
    facade = FakeTorchAndTransformersFacade(cuda_available=False)
    runtime = _runtime(facade)
    runtime.warm_up()
    assert facade.loaded["device"] == "cpu"


def test_gpu_only_raises_when_no_gpu_present() -> None:
    facade = FakeTorchAndTransformersFacade(cuda_available=False)
    runtime = _runtime(facade, device=DeviceSelection.GPU_ONLY)
    with pytest.raises(RuntimeError, match="GPU_ONLY"):
        runtime.warm_up()


def test_cpu_only_never_uses_gpu_even_if_available() -> None:
    facade = FakeTorchAndTransformersFacade(cuda_available=True)
    runtime = _runtime(facade, device=DeviceSelection.CPU_ONLY)
    runtime.warm_up()
    assert facade.loaded["device"] == "cpu"


def test_automatic_precision_is_fp16_on_gpu_fp32_on_cpu() -> None:
    gpu_facade = FakeTorchAndTransformersFacade(cuda_available=True)
    gpu_runtime = _runtime(gpu_facade)
    gpu_runtime.warm_up()
    assert gpu_facade.loaded["precision"] == "fp16"

    cpu_facade = FakeTorchAndTransformersFacade(cuda_available=False)
    cpu_runtime = _runtime(cpu_facade)
    cpu_runtime.warm_up()
    assert cpu_facade.loaded["precision"] == "fp32"


def test_infer_calls_warm_up_lazily_if_not_already_done() -> None:
    facade = FakeTorchAndTransformersFacade(cuda_available=False)
    runtime = _runtime(facade)
    result = runtime.infer(InferenceRequest(page_image_ref="p1.png", prompt="describe"))
    assert facade.loaded is not None  # warmed up implicitly
    assert "generated:qwen2.5-vl-7b-instruct:describe" == result.raw_text
    assert result.model_version == "qwen2.5-vl-7b-instruct@main"
    assert result.device_used == "cpu"


def test_shutdown_releases_the_model_and_is_safe_if_never_warmed_up() -> None:
    facade = FakeTorchAndTransformersFacade(cuda_available=False)
    runtime = _runtime(facade)
    runtime.shut_down()  # never warmed up -- must not raise
    assert facade.released == []

    runtime.warm_up()
    runtime.shut_down()
    assert len(facade.released) == 1


def test_capabilities_omit_int4_by_default() -> None:
    facade = FakeTorchAndTransformersFacade(cuda_available=True)
    runtime = _runtime(facade)
    assert Precision.INT4 not in runtime.capabilities().supported_precisions


def test_load_model_with_auto_class_prefers_new_name_when_model_supports_it(monkeypatch) -> None:
    from archivetrust.runtime.transformers_runtime import _load_model_with_auto_class
    import types

    class _NewClass:
        @staticmethod
        def from_pretrained(source, **kwargs):
            return "loaded-via-NEW"

    fake_transformers = types.SimpleNamespace(
        AutoModelForImageTextToText=_NewClass, AutoModelForVision2Seq="OLD", __version__="x"
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    result = _load_model_with_auto_class(source="repo", revision=None, dtype="auto", extra_kwargs={})
    assert result == "loaded-via-NEW"


def test_load_model_with_auto_class_falls_through_on_unrecognized_configuration(monkeypatch) -> None:
    """A model class that exists in the installed transformers but doesn't recognize this specific
    model's config must be skipped in favor of the next candidate, not treated as a fatal error."""
    from archivetrust.runtime.transformers_runtime import _load_model_with_auto_class
    import types

    class _WrongClass:
        @staticmethod
        def from_pretrained(source, **kwargs):
            raise ValueError("Unrecognized configuration class FooConfig for this kind of AutoModel")

    class _RightClass:
        @staticmethod
        def from_pretrained(source, **kwargs):
            return "loaded-via-vision2seq"

    fake_transformers = types.SimpleNamespace(
        AutoModelForImageTextToText=_WrongClass, AutoModelForVision2Seq=_RightClass, __version__="x"
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    result = _load_model_with_auto_class(source="repo", revision=None, dtype="auto", extra_kwargs={})
    assert result == "loaded-via-vision2seq"


def test_load_model_with_auto_class_propagates_unrelated_errors_immediately(monkeypatch) -> None:
    from archivetrust.runtime.transformers_runtime import _load_model_with_auto_class
    import types
    import pytest

    class _FailsForRealReasons:
        @staticmethod
        def from_pretrained(source, **kwargs):
            raise ValueError("401 Client Error: Unauthorized for url")

    fake_transformers = types.SimpleNamespace(AutoModelForImageTextToText=_FailsForRealReasons, __version__="x")
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    with pytest.raises(ValueError, match="Unauthorized"):
        _load_model_with_auto_class(source="repo", revision=None, dtype="auto", extra_kwargs={})


def test_load_model_with_auto_class_raises_clear_error_when_none_work(monkeypatch) -> None:
    from archivetrust.runtime.transformers_runtime import _load_model_with_auto_class
    import types
    import pytest

    fake_transformers = types.SimpleNamespace(__version__="99.0")
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    with pytest.raises(ImportError, match="99.0"):
        _load_model_with_auto_class(source="repo", revision=None, dtype="auto", extra_kwargs={})
