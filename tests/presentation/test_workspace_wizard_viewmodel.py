"""Exercises `WorkspaceWizardViewModel` against a real `AppContext` (composition-tested elsewhere,
`tests/clients/test_composition.py`) rather than a hand-rolled fake, since the wizard's `create()`
drives several real collaborators (`WorkspaceStore`, `ProviderConfigurationStore`,
`AcquisitionManager`) that would otherwise need to be re-faked here.
"""

from __future__ import annotations

from archivetrust.composition import AppContext
from archivetrust.presentation.first_launch_viewmodel import (
    DEFAULT_VISION_MODEL_ID,
    DEFAULT_VISION_PROVIDER_ID,
)
from archivetrust.presentation.workspace_wizard_viewmodel import InvalidWizardStateError, WizardStep


def _context(tmp_path) -> AppContext:
    return AppContext(deployment_root=tmp_path / "app", auto_create_default_workspace=False)


def test_cannot_advance_past_name_step_without_a_name(tmp_path) -> None:
    vm = _context(tmp_path).workspace_wizard_viewmodel()
    assert vm.can_advance() is False
    try:
        vm.advance()
        assert False, "expected InvalidWizardStateError"
    except InvalidWizardStateError:
        pass


def test_full_step_sequence_reaches_review(tmp_path) -> None:
    vm = _context(tmp_path).workspace_wizard_viewmodel()
    vm.name = "Engineering Drawings"
    for _ in range(len(list(WizardStep)) - 1):
        vm.advance()
    assert vm.step == WizardStep.REVIEW


def test_create_persists_workspace_profile_and_providers(tmp_path) -> None:
    context = _context(tmp_path)
    vm = context.workspace_wizard_viewmodel()
    vm.name = "Engineering Drawings"
    vm.description = "CAD archive"
    vm.add_manual_import = True

    workspace = vm.create()

    assert workspace.name == "Engineering Drawings"
    assert context.current_workspace.id == workspace.id
    assert any(s.source_id == "manual-import" for s in context.acquisition_manager.sources())
    # This test previously also asserted `enabled_provider_ids={"docling"}` persisted
    # `provider_config_store.get("docling").enabled is True` and that
    # `"tesseract_layoutparser"` stayed disabled -- both real, eagerly-registered providers
    # deleted in docs/htr-migration-plan.md Stage 5 (EXECUTED). The underlying principle
    # (`create()` persists the operator's provider selections into `ProviderConfigurationStore`)
    # transfers to whichever future provider lands first (SATRN, Stage 6) and should get its own
    # test then, not a reimplementation against a provider_id that no longer registers to anything.


def test_create_without_name_raises(tmp_path) -> None:
    vm = _context(tmp_path).workspace_wizard_viewmodel()
    try:
        vm.create()
        assert False, "expected InvalidWizardStateError"
    except InvalidWizardStateError:
        pass


def test_new_workspace_defaults_pre_fill_no_vision_provider(tmp_path) -> None:
    """Was `test_new_workspace_defaults_inherit_paddleocr_vl` (2026-07-30 residual cleanup).

    A freshly-constructed wizard used to arrive with `paddleocr-vl` / `PaddlePaddle/PaddleOCR-VL`
    pre-filled -- a provider whose adapter was deleted in migration Stage 5, so pressing through every
    step persisted a binding nothing could ever activate. Both defaults are now empty. The principle
    the old test asserted (the wizard's defaults come from `first_launch_viewmodel`'s constants, never
    a second hardcoded copy) is unchanged and still asserted here."""
    vm = _context(tmp_path).workspace_wizard_viewmodel()
    assert vm.model_identifier == DEFAULT_VISION_MODEL_ID is None
    assert vm.model_provider_id == DEFAULT_VISION_PROVIDER_ID == ""


def test_new_workspace_binds_no_vision_provider_by_default(tmp_path) -> None:
    """The behavioural half of the change: no binding at all, rather than a binding for a deleted
    provider. `create()`'s existing `if self.model_identifier:` guard is what makes this true -- the
    same guard that has always implemented "leave blank to skip"."""
    context = _context(tmp_path)
    vm = context.workspace_wizard_viewmodel()
    vm.name = "Engineering Drawings"

    vm.create()

    assert not any(
        b.logical_name == "Primary Vision Provider" for b in context.model_registry.bindings()
    )


def test_an_operator_supplied_model_identifier_still_binds(tmp_path) -> None:
    """The default going empty must not break the path that actually works: a caller naming a real
    model still gets a binding, on the runtime `runtime_kind_for_provider` derives."""
    context = _context(tmp_path)
    vm = context.workspace_wizard_viewmodel()
    vm.name = "Bound Workspace"
    vm.model_identifier = "microsoft/Florence-2-base-ft"

    vm.create()

    binding = next(
        b for b in context.model_registry.bindings() if b.logical_name == "Primary Vision Provider"
    )
    assert binding.descriptor.model_id == "microsoft/Florence-2-base-ft"
    assert binding.descriptor.runtime_kind == "transformers"
    assert binding.provider_id == DEFAULT_VISION_PROVIDER_ID


def test_operator_can_override_default_by_clearing_the_model_field(tmp_path) -> None:
    """"Unless the operator explicitly overrides them" -- clearing the pre-filled field is the
    existing "leave blank to skip" override, still honored."""
    context = _context(tmp_path)
    vm = context.workspace_wizard_viewmodel()
    vm.name = "No Vision Provider Workspace"
    vm.model_identifier = None

    vm.create()

    assert not any(
        b.logical_name == "Primary Vision Provider" for b in context.model_registry.bindings()
    )


def test_operator_can_explicitly_select_a_provider_instead_of_the_default(tmp_path) -> None:
    """Was `test_operator_can_explicitly_select_qwen_instead_of_the_default` (2026-07-30 residual
    cleanup). The principle -- an explicitly-set `model_provider_id` reaches the persisted binding
    rather than being overwritten by the default -- is preserved with a provider id that is not a
    deleted adapter's. `"future-provider"` is deliberately unregistered: this asserts the wizard
    *records* the operator's choice, which is a different question from whether anything can activate
    it."""
    context = _context(tmp_path)
    vm = context.workspace_wizard_viewmodel()
    vm.name = "Explicit Provider Workspace"
    vm.model_provider_id = "future-provider"
    vm.model_identifier = "some-org/some-model"

    vm.create()

    binding = next(
        b for b in context.model_registry.bindings() if b.logical_name == "Primary Vision Provider"
    )
    assert binding.provider_id == "future-provider"


# `test_surya_appears_in_available_providers_once_registered` (2026-07-30 residual cleanup) was
# deleted. It asserted that `surya` appears in `available_providers()` via the Vision Provider catalog
# branch -- which is now the defect rather than the feature: `surya`'s adapter was deleted in migration
# Stage 5 and `VISION_PROVIDER_CHOICES` is deliberately empty, so nothing may appear from that branch.
# The generic principle it was written for ("adding a future provider requires no UI change --
# registration alone should make it appear") is asserted, unchanged and provider-agnostically, by
# `test_a_newly_registered_provider_appears_with_no_wizard_code_change` below, which exercises the
# provider-registry branch of the same function. Nothing is left untested by the deletion.


def test_a_newly_registered_provider_appears_with_no_wizard_code_change(tmp_path) -> None:
    """"Ensure adding a future provider requires no UI changes—registration alone should make it
    appear." Registers a provider id that exists nowhere in any wizard/composition metadata table
    and asserts it is still offered, proving the Providers step holds no hardcoded id list."""
    from archivetrust.providers.base import (
        DeploymentLocality,
        DeterministicProviderAdapter,
        Introspectability,
        ProviderRegistration,
        Reproducibility,
    )

    class _FutureAdapter(DeterministicProviderAdapter):
        registration = ProviderRegistration(
            provider_id="future-provider",
            reproducibility=Reproducibility.DETERMINISTIC,
            deployment_locality=DeploymentLocality.IN_PROCESS,
            introspectability=Introspectability.OPEN,
        )

        def observe(self, *, document_ref, invocation_id, source):  # pragma: no cover - unused
            raise AssertionError("not invoked by this presentation-only test")

    context = _context(tmp_path)
    context.create_workspace("Seed Workspace")  # populates provider_registry (empty until opened)
    context.provider_registry.register(_FutureAdapter())
    context.provider_availability["future-provider"] = True
    vm = context.workspace_wizard_viewmodel()

    choices = {c.provider_id: c for c in vm.available_providers()}

    assert "future-provider" in choices
    assert choices["future-provider"].available is True
    assert choices["future-provider"].label  # falls back to a sensible label, never blank


# `test_unavailable_provider_is_shown_but_create_never_enables_it` and
# `test_create_activates_the_bound_vision_provider_immediately` (docs/htr-migration-plan.md
# Stage 5 -- EXECUTED) were deleted: both depended on `docling`/`paddleocr-vl` being real,
# eagerly-registered/activatable providers, which no OCR-era provider is anymore. Their
# principles -- "unavailable providers are shown but unselectable" and "`create()` must actually
# activate what it binds, not just persist the binding" -- transfer to whichever future provider
# lands first (SATRN, Stage 6) and belong in that phase's own activation tests.
