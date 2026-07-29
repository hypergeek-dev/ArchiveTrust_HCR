from __future__ import annotations

from archivetrust.runtime.contracts import DeviceSelection, Precision
from archivetrust.runtime.provider_config import (
    DETERMINISTIC_LOCAL_CAPABILITIES,
    PROBABILISTIC_VISION_CAPABILITIES,
    ExecutionMode,
    ProcessingMode,
    ProviderConfiguration,
    ProviderConfigurationStore,
)
from archivetrust.runtime.provider_profiles import ProviderProfileName, default_profiles


def test_default_profiles_cover_the_four_named_presets() -> None:
    names = {p.name for p in default_profiles()}
    assert names == {
        ProviderProfileName.ARCHIVE,
        ProviderProfileName.FAST_REVIEW,
        ProviderProfileName.MAXIMUM_QUALITY,
        ProviderProfileName.RESEARCH,
    }


def test_research_profile_is_deterministic_with_a_seed() -> None:
    research = next(p for p in default_profiles() if p.name == ProviderProfileName.RESEARCH)
    assert research.deterministic
    assert research.seed is not None


def test_execution_mode_maps_to_device_selection() -> None:
    assert ExecutionMode.GPU_ONLY.to_device_selection() == DeviceSelection.GPU_ONLY
    assert ExecutionMode.AUTOMATIC.to_device_selection() == DeviceSelection.AUTOMATIC


def test_processing_mode_maps_to_a_profile() -> None:
    assert ProcessingMode.FAST.to_profile_name() == ProviderProfileName.FAST_REVIEW
    assert ProcessingMode.MAXIMUM_QUALITY.to_profile_name() == ProviderProfileName.MAXIMUM_QUALITY


def test_configuration_matching_a_preset_keeps_its_profile_label() -> None:
    archive_preset = next(p for p in default_profiles() if p.name == ProviderProfileName.ARCHIVE)
    config = ProviderConfiguration(
        provider_id="qwen2.5-vl",
        profile=ProviderProfileName.ARCHIVE,
        device=archive_preset.device,
        precision=archive_preset.precision,
        batch_size=archive_preset.batch_size,
        parallel_documents=archive_preset.parallel_documents,
        deterministic=archive_preset.deterministic,
    )
    assert config.with_effective_profile_marker().profile == ProviderProfileName.ARCHIVE


def test_hand_edited_configuration_reports_custom() -> None:
    config = ProviderConfiguration(
        provider_id="qwen2.5-vl",
        profile=ProviderProfileName.FAST_REVIEW,
        precision=Precision.FP32,  # diverges from the Fast Review preset's INT8
    )
    assert config.with_effective_profile_marker().profile == ProviderProfileName.CUSTOM


def test_store_returns_default_configuration_for_unconfigured_provider() -> None:
    store = ProviderConfigurationStore()
    config = store.get("docling")
    assert config.provider_id == "docling"
    assert config.enabled  # sensible default, never a crash for an unconfigured provider


def test_store_primary_vision_provider_is_first_enabled() -> None:
    store = ProviderConfigurationStore()
    store.set(ProviderConfiguration(provider_id="docling", enabled=False))
    store.set(ProviderConfiguration(provider_id="qwen2.5-vl", enabled=True))
    assert store.primary_vision_provider() == "qwen2.5-vl"


def test_store_primary_vision_provider_none_when_nothing_enabled() -> None:
    store = ProviderConfigurationStore()
    store.set(ProviderConfiguration(provider_id="docling", enabled=False))
    assert store.primary_vision_provider() is None


def test_capability_presets_differ_for_probabilistic_vs_deterministic() -> None:
    assert PROBABILISTIC_VISION_CAPABILITIES.has_prompt_management
    assert not DETERMINISTIC_LOCAL_CAPABILITIES.has_prompt_management
    assert not DETERMINISTIC_LOCAL_CAPABILITIES.has_device_selection
    assert DETERMINISTIC_LOCAL_CAPABILITIES.has_grounding  # Tesseract/Docling still emit boxes
