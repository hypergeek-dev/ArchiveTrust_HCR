from __future__ import annotations

import pytest

from archivetrust.presentation.first_launch_viewmodel import (
    DEFAULT_VISION_PROVIDER_ID,
    DEFAULT_VISION_RUNTIME_KIND,
    NO_VISION_PROVIDER,
    VISION_PROVIDER_CHOICES,
    FirstLaunchViewModel,
    InvalidSetupInputError,
    SetupSourceKind,
    declared_runtime_kind_for_provider,
    htr_method_readiness,
    needs_first_launch,
    runtime_kind_for_provider,
)
from archivetrust.providers.htr_adapter import (
    EnvironmentValidation,
    MethodCapabilities,
    MethodMetadata,
)
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.model_registry import ModelRegistry


def test_needs_first_launch_true_when_no_models_installed(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    assert needs_first_launch(layout)


def test_needs_first_launch_false_after_a_model_directory_exists(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    (layout.models_dir / "qwen").mkdir()
    assert not needs_first_launch(layout)


def test_complete_setup_with_hugging_face_source(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="Qwen/Qwen2.5-VL-7B-Instruct",
        runtime_kind="transformers",
    )
    assert result.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert result.runtime_kind == "transformers"
    resolved = registry.resolve("Primary Vision Provider")
    assert resolved.descriptor.source.identifier == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_complete_setup_with_local_browse(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    model_dir = layout.models_dir / "my-local-model"
    model_dir.mkdir()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.BROWSE_LOCAL, identifier=str(model_dir), runtime_kind="transformers"
    )
    assert result.model_id == "my-local-model"


def test_complete_setup_rejects_missing_local_path(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    with pytest.raises(InvalidSetupInputError):
        vm.complete_setup(
            source_kind=SetupSourceKind.BROWSE_LOCAL,
            identifier=str(layout.root / "does-not-exist"),
            runtime_kind="transformers",
        )


def test_complete_setup_rejects_empty_identifier(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    with pytest.raises(InvalidSetupInputError):
        vm.complete_setup(source_kind=SetupSourceKind.HUGGING_FACE, identifier="   ", runtime_kind="transformers")


def test_needs_setup_reflects_current_layout_state(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    assert vm.needs_setup()
    (layout.models_dir / "something").mkdir()
    assert not vm.needs_setup()


def test_the_vision_provider_catalog_offers_nothing_that_cannot_be_activated() -> None:
    """2026-07-30 residual cleanup. `test_runtime_kind_for_provider_resolves_paddleocr_vl_and_surya_to_vllm`
    stood here and asserted the runtimes of three providers whose adapters were deleted in migration
    Stage 5 -- i.e. it locked in the defect. Replaced, not deleted: the principle it was reaching for
    ("the wizard's offered choices are looked up from a catalog, never hardcoded") is preserved as the
    stronger invariant below."""
    assert VISION_PROVIDER_CHOICES == ()


def test_every_offered_vision_provider_has_an_adapter_factory() -> None:
    """The invariant that stops the original defect recurring, asserted in both directions.

    An entry in `VISION_PROVIDER_CHOICES` is activatable only if
    `composition.py::_VISION_PROVIDER_ADAPTER_FACTORIES` can build an adapter for the same
    `provider_id`. Vacuously true today (both are empty) and *not* vacuous later: adding a catalog
    entry without a factory, or removing a factory from under an entry, fails here.
    """
    from archivetrust.composition import _VISION_PROVIDER_ADAPTER_FACTORIES

    offered = {provider_id for provider_id, _l, _m, _r in VISION_PROVIDER_CHOICES}
    buildable = set(_VISION_PROVIDER_ADAPTER_FACTORIES)
    assert offered - buildable == set(), (
        "the first-launch wizard offers vision providers no adapter factory can activate: "
        f"{sorted(offered - buildable)}"
    )
    assert buildable - offered == set(), (
        "an adapter factory exists for a provider the wizard never offers, so nothing can bind it: "
        f"{sorted(buildable - offered)}"
    )


def test_no_default_names_a_deleted_provider() -> None:
    assert DEFAULT_VISION_PROVIDER_ID == NO_VISION_PROVIDER == ""
    # "vllm" would name a runtime this build cannot construct -- runtime/vllm_runtime.py is deleted.
    assert DEFAULT_VISION_RUNTIME_KIND == "transformers"


def test_runtime_kind_for_provider_falls_back_to_default_for_undeclared_provider() -> None:
    assert runtime_kind_for_provider("some-future-provider") == DEFAULT_VISION_RUNTIME_KIND


def test_declared_runtime_kind_is_none_rather_than_a_default() -> None:
    """The distinction the emptied catalog made load-bearing: a caller that *overwrites* stored state
    (`composition.py::_activate_vision_binding`'s self-heal) must be able to tell "no declaration"
    from "the default", or it rewrites every persisted binding's runtime on every workspace open."""
    assert declared_runtime_kind_for_provider("some-future-provider") is None
    assert runtime_kind_for_provider("some-future-provider") is not None


# -- HTR method readiness: the HTR-relevant replacement for the provider selection ----------------


class _FakeAdapter:
    """Implements only the three methods `htr_method_readiness` calls. Deliberately not one of the
    real adapters: the real ones' `validate_environment()` results depend on this machine, and this
    test is about the ViewModel's mapping, not about whether a GPU is present."""

    def __init__(self, *, method_id: str, valid: bool, raises: bool = False) -> None:
        self._method_id = method_id
        self._valid = valid
        self._raises = raises

    def get_metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method_id=self._method_id,
            method_name=f"{self._method_id} display name",
            vendor="Test Vendor",
            model_revision="deadbeef",
        )

    def get_capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            confidence_supported=True,
            geometry_supported=False,
            line_level_supported=True,
            page_level_supported=False,
            local_execution_supported=True,
            external_upload_required=False,
        )

    def validate_environment(self) -> EnvironmentValidation:
        if self._raises:
            raise RuntimeError("the isolated venv interpreter is missing")
        return EnvironmentValidation(
            valid=self._valid, messages=() if self._valid else ("model weights not found",)
        )


def test_htr_method_readiness_reports_each_adapters_own_validation() -> None:
    rows = htr_method_readiness(
        (
            _FakeAdapter(method_id="satrn", valid=True),
            _FakeAdapter(method_id="florence2_htr", valid=False),
        )
    )
    assert [r.method_id for r in rows] == ["satrn", "florence2_htr"]
    assert rows[0].ready is True
    assert rows[1].ready is False
    assert rows[1].messages == ("model weights not found",)
    assert rows[0].model_revision == "deadbeef"
    assert rows[0].local_execution_supported is True


def test_a_raising_adapter_is_reported_not_propagated() -> None:
    """Part 12: a first-launch screen never shows a traceback -- and one broken method must not hide
    the other."""
    rows = htr_method_readiness(
        (
            _FakeAdapter(method_id="satrn", raises=True, valid=True),
            _FakeAdapter(method_id="transkribus_swedish_lion_1", valid=True),
        )
    )
    assert rows[0].ready is False
    assert "RuntimeError" in rows[0].messages[0]
    assert "isolated venv interpreter is missing" in rows[0].messages[0]
    assert rows[1].ready is True


def test_view_model_reports_readiness_for_the_injected_adapters(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    vm = FirstLaunchViewModel(
        layout=layout,
        model_registry=ModelRegistry(layout),
        htr_method_adapters=(_FakeAdapter(method_id="satrn", valid=True),),
    )
    assert [r.method_id for r in vm.htr_method_readiness()] == ["satrn"]


def test_view_model_without_injected_adapters_reports_nothing_rather_than_failing(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    vm = FirstLaunchViewModel(layout=layout, model_registry=ModelRegistry(layout))
    assert vm.htr_method_readiness() == ()


def test_the_three_real_adapters_are_all_reportable() -> None:
    """The one test that uses the **real** adapters, asserting only what is machine-independent: all
    three are constructible, all three answer `validate_environment()` without raising, and a
    `ready=False` row always carries at least one diagnostic message (which `EnvironmentValidation`
    guarantees and this asserts end-to-end). Whether any is actually ready depends on the machine and
    is deliberately not asserted."""
    from archivetrust.composition import AppContext

    rows = htr_method_readiness(AppContext._build_htr_adapters()[0])
    assert {r.method_id for r in rows} == {"satrn", "florence2_htr", "transkribus_swedish_lion_1"}
    for row in rows:
        assert row.model_revision
        if not row.ready:
            assert row.messages, f"{row.method_id} reported not-ready with no explanation"


def test_complete_setup_without_runtime_kind_derives_it_from_provider_id(tmp_path) -> None:
    """Runtime Architecture Completion milestone, Part 11: the ingestion UI no longer exposes a
    runtime selector -- `complete_setup` must derive it from `provider_id` when omitted.

    The derived value is now `"transformers"` rather than `"vllm"` (2026-07-30 residual cleanup):
    `DEFAULT_VISION_PROVIDER_ID` is `NO_VISION_PROVIDER`, so the lookup falls back to
    `DEFAULT_VISION_RUNTIME_KIND` -- the only runtime kind with a real registered factory now that
    `runtime/vllm_runtime.py` is deleted."""
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="microsoft/Florence-2-base-ft",
        provider_id=DEFAULT_VISION_PROVIDER_ID,
    )
    assert result.runtime_kind == "transformers"
    resolved = registry.resolve("Primary Vision Provider")
    assert resolved.descriptor.runtime_kind == "transformers"


def test_complete_setup_explicit_runtime_kind_still_overrides_the_derived_default(tmp_path) -> None:
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)

    result = vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="microsoft/Florence-2-base-ft",
        provider_id=DEFAULT_VISION_PROVIDER_ID,
        runtime_kind="openai_compatible",
    )
    assert result.runtime_kind == "openai_compatible"


def test_needs_first_launch_false_after_a_binding_is_persisted_with_no_local_model(tmp_path) -> None:
    # A Hugging Face source has no local models_dir footprint -- needs_first_launch must still
    # recognize a persisted binding as "already configured" (regression: previously this only
    # scanned disk, so a non-local source would show the wizard again on every restart).
    layout = DeploymentLayout(root=tmp_path).ensure()
    registry = ModelRegistry(layout)
    vm = FirstLaunchViewModel(layout=layout, model_registry=registry)
    vm.complete_setup(
        source_kind=SetupSourceKind.HUGGING_FACE,
        identifier="Qwen/Qwen2.5-VL-7B-Instruct",
        runtime_kind="transformers",
    )
    assert not vm.needs_setup()

    # A fresh registry against the same layout, loading persisted bindings, must also see it.
    reloaded_registry = ModelRegistry(layout)
    reloaded_registry.load_bindings()
    reloaded_vm = FirstLaunchViewModel(layout=layout, model_registry=reloaded_registry)
    assert not reloaded_vm.needs_setup()
