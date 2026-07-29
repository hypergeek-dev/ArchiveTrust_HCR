"""Provider Profiles (Part 7): reusable presets an operator selects instead of tuning dozens of
individual options. A profile encapsulates device, precision, batch size, prompt template,
parallelism, and runtime settings as one named choice.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from archivetrust.runtime.contracts import DeviceSelection, Precision


class ProviderProfileName(str, Enum):
    ARCHIVE = "archive"
    """Large, mixed-quality historical batches: favors throughput and CPU availability over peak
    per-page quality."""
    FAST_REVIEW = "fast_review"
    """Low latency for interactive review workflows: small batches, low token ceilings."""
    MAXIMUM_QUALITY = "maximum_quality"
    """Highest achievable fidelity: full precision, larger token budget, no batching shortcuts."""
    RESEARCH = "research"
    """Deterministic, seeded runs for reproducible experimentation (Comparison-Engine-adjacent
    research, ROADMAP.md S13) -- always requests a fixed seed where the runtime supports one."""
    CUSTOM = "custom"
    """An operator-authored profile; not one of the four presets above."""


class ProviderProfile(BaseModel):
    """One reusable configuration preset (Part 7)."""

    model_config = ConfigDict(frozen=True)

    name: ProviderProfileName
    label: str
    device: DeviceSelection
    precision: Precision
    batch_size: int
    parallel_documents: int
    prompt_template_name: str
    prompt_template_version: int
    deterministic: bool
    seed: int | None = None


def default_profiles(prompt_template_name: str = "default") -> tuple[ProviderProfile, ...]:
    """The four named presets (Part 7). `CUSTOM` has no preset instance here -- it is what an
    operator gets by starting from one of these and changing a value in Advanced Provider Settings
    (Part 9), at which point the effective profile is reported as CUSTOM rather than silently still
    claiming to be e.g. FAST_REVIEW.
    """
    return (
        ProviderProfile(
            name=ProviderProfileName.ARCHIVE,
            label="Archive",
            device=DeviceSelection.AUTOMATIC,
            precision=Precision.FP16,
            batch_size=8,
            parallel_documents=4,
            prompt_template_name=prompt_template_name,
            prompt_template_version=1,
            deterministic=False,
        ),
        ProviderProfile(
            name=ProviderProfileName.FAST_REVIEW,
            label="Fast Review",
            device=DeviceSelection.AUTOMATIC,
            precision=Precision.INT8,
            batch_size=1,
            parallel_documents=1,
            prompt_template_name=prompt_template_name,
            prompt_template_version=1,
            deterministic=False,
        ),
        ProviderProfile(
            name=ProviderProfileName.MAXIMUM_QUALITY,
            label="Maximum Quality",
            device=DeviceSelection.GPU_ONLY,
            precision=Precision.FP32,
            batch_size=1,
            parallel_documents=1,
            prompt_template_name=prompt_template_name,
            prompt_template_version=1,
            deterministic=False,
        ),
        ProviderProfile(
            name=ProviderProfileName.RESEARCH,
            label="Research",
            device=DeviceSelection.AUTOMATIC,
            precision=Precision.FP32,
            batch_size=1,
            parallel_documents=1,
            prompt_template_name=prompt_template_name,
            prompt_template_version=1,
            deterministic=True,
            seed=1337,
        ),
    )
