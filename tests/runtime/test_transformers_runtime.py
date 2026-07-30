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


# -- Isolation from the HTR methods (2026-07-30 residual cleanup) ----------------------------------
#
# `docs/htr-repository-cleanup.md`'s Stage 5 record flagged this module's Qwen-shaped assumptions and
# deferred them to "Florence-2's implementation phase", on the unverified premise that Florence-2 would
# run through this shared runtime. It does not. These three tests turn that finding from a docstring
# claim into an enforced invariant, so if a future phase *does* route an HTR method through here, it
# fails here first -- which is exactly when the inert assumptions would stop being inert.


def test_no_htr_provider_depends_on_the_shared_transformers_runtime() -> None:
    """No module under `providers/` may import `archivetrust.runtime.transformers_runtime`.

    Asserted against the real source text of every provider module rather than by importing them
    (importing `providers/satrn/adapter.py` is cheap, but a future adapter's import side effects are
    not this test's business). Docstring *mentions* of the module name are legitimate -- SATRN's and
    Florence-2's facades both cite its test-seam discipline -- so this matches import statements
    specifically.
    """
    import re
    from pathlib import Path

    providers_root = Path(__file__).resolve().parents[2] / "src" / "archivetrust" / "providers"
    import_pattern = re.compile(
        r"^\s*(?:from\s+archivetrust\.runtime[.\s]|import\s+archivetrust\.runtime\b)", re.MULTILINE
    )
    offenders = []
    for path in sorted(providers_root.rglob("*.py")):
        if import_pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(providers_root)))
    assert offenders == [], (
        "an HTR provider now imports from archivetrust.runtime: "
        f"{offenders}. If that is intended, the Qwen-specific assumptions documented in "
        "runtime/transformers_runtime.py's module docstring are no longer inert and must be "
        "addressed before this test is relaxed."
    )


def test_florence2_loads_its_own_model_rather_than_going_through_this_runtime() -> None:
    """The positive half: Florence-2 really does load independently, via the one Auto class this
    runtime deliberately does *not* offer.

    `_VISION_MODEL_CLASS_NAMES` dropped `AutoModelForCausalLM` (its docstring explains why), and that is
    precisely what Florence-2 needs -- so routing Florence-2 through this runtime could not work even
    accidentally. Asserting both halves makes the reasoning checkable rather than a claim in prose.
    """
    from pathlib import Path

    from archivetrust.runtime.transformers_runtime import _VISION_MODEL_CLASS_NAMES

    facade = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "archivetrust"
        / "providers"
        / "florence2_htr"
        / "facade.py"
    ).read_text(encoding="utf-8")
    assert "from transformers import AutoModelForCausalLM, AutoProcessor" in facade
    assert "AutoModelForCausalLM.from_pretrained(" in facade
    assert "AutoModelForCausalLM" not in _VISION_MODEL_CLASS_NAMES


def test_nothing_in_the_application_calls_infer_on_a_runtime() -> None:
    """The reachability claim: `real_torch_and_transformers_facade`'s `generate` -- where every
    genuinely Qwen-flavoured line lives -- has no caller today.

    Uses `ast` rather than text matching, so prose is not mistaken for code: comments, docstrings (this
    module's own docstring quotes the `.infer(` grep verbatim) and string literals are all invisible to
    an AST walk, and only a genuine `<expr>.infer(...)` call node counts.

    This test is expected to start failing the day a vision provider is wired up again, and its failure
    message says what to do about it rather than just going red.
    """
    import ast
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[2] / "src" / "archivetrust"
    call_sites = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "infer"
            ):
                call_sites.append(f"{path.relative_to(src_root)}:{node.lineno}")
    assert call_sites == [], (
        "something now calls InferenceRuntime.infer(): "
        f"{call_sites}. TransformersRuntime's decode path is no longer unreachable, so the "
        "chat-template/decode assumptions documented in its module docstring are now live and must "
        "be reviewed against whatever model that caller loads."
    )
