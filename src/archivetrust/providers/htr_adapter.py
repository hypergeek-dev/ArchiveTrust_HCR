"""`HtrMethodAdapter` (docs/htr-domain-design.md; docs/htr-migration-plan.md Stage 2).

Added alongside the existing `ProviderAdapter` (`providers/base.py`), which is not modified --
both interfaces exist temporarily per the migration plan, until `providers/base.py` is deleted in
a later stage once every method has an `HtrMethodAdapter` implementation.

Every capability is explicit (Constitution Article 6, Full Exposure -- applied here as "full
exposure of what a method *cannot* do", not just what it can): `MethodCapabilities` declares every
flag as a required boolean with no default, so a `False` capability can never be an accidental
omission indistinguishable from "not yet declared".
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class MethodMetadata(BaseModel):
    """Identity of one HTR method (docs/htr-domain-design.md's `Method`/`ModelVersion`)."""

    model_config = ConfigDict(frozen=True)

    method_id: str
    method_name: str
    vendor: str
    model_revision: str
    """Pinned model/checkpoint version -- the same value `Evidence.model_revision` and
    `MethodRun.model_version_id` refer to for this method's runs."""


class MethodCapabilities(BaseModel):
    """What this method can and cannot produce -- every flag required, none defaulted, so an
    unsupported capability is always an explicit `False`, never a silent omission.
    """

    model_config = ConfigDict(frozen=True)

    confidence_supported: bool
    geometry_supported: bool
    line_level_supported: bool
    page_level_supported: bool
    local_execution_supported: bool
    external_upload_required: bool
    image_color_normalization_required: bool = False
    """Whether this method's input images must pass through the versioned RGB-normalization stage
    (`htr/preprocessing/`, docs/methods/transkribus-swedish-lion-1.md) before being handed to it.

    Added 2026-07-30 with that stage. **The one defaulted flag on this model**, and the exception is
    deliberate rather than a convenience: every other flag describes what a method *can produce* and
    must be an explicit `True`/`False` per the class docstring, whereas this one describes a
    *pipeline obligation* that only applies to a method whose input ArchiveTrust prepares and hands
    to an external service. Defaulting it to `False` means "this method's inputs are not prepared by
    that stage", which is the correct and true statement for every local recognizer -- including any
    future one -- and avoids forcing an unrelated adapter to answer a question about a stage it has
    nothing to do with.

    `True` for Transkribus Swedish Lion I page-level workflows: what is uploaded *is* the
    experimental input, so its colour representation is an experimental variable and must be a
    recorded fact. `False` for SATRN and Florence-2, whose controlled line crops are deliberately
    **not** routed through the stage -- doing so would change the bytes they read relative to the
    committed baseline and invalidate its recorded results. Florence-2's own inline
    `convert("RGB")` is its model's separate input requirement, not this stage (see
    `providers/florence2_htr/facade.py` and the method doc §8)."""


class EnvironmentValidation(BaseModel):
    """Result of checking whether this method's runtime prerequisites (model weights, GPU
    drivers, external service reachability, ...) are satisfied, before `recognize()` is ever
    called."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    messages: tuple[str, ...] = ()
    """Human-readable diagnostic messages -- always populated when `valid=False` (never a bare
    `False` with no explanation), optional when `valid=True`."""


class HealthCheckResult(BaseModel):
    """Result of a lightweight runtime health probe (distinct from `EnvironmentValidation`, which
    checks prerequisites before any run; this checks liveness during operation)."""

    model_config = ConfigDict(frozen=True)

    healthy: bool
    message: str | None = None


class RecognitionInput(BaseModel):
    """What `recognize()` consumes -- an `InputCrop` reference (never embedded image bytes, per
    the graph-reference discipline already enforced elsewhere in this codebase) plus optional
    method-specific configuration. `page_image_ref`/`region_ref` are alternatives to
    `input_crop_id` for methods that are only `page_level_supported` (no line-level input),
    matching `MethodCapabilities.page_level_supported`."""

    model_config = ConfigDict(frozen=True)

    input_crop_id: str | None = None
    page_image_ref: str | None = None
    region_id: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)


class RecognitionResult(BaseModel):
    """The common envelope over one `recognize()` call. `raw_response` preserves whatever
    provider-specific structure the underlying method actually returned -- never forced into a
    common schema (Constitution Article 6: full exposure; Article 20: provider independence
    applies downstream of this envelope, not to what this envelope is permitted to carry
    internally)."""

    model_config = ConfigDict(frozen=True)

    text: str | None
    confidence: float | None = None
    """Only populated when `MethodCapabilities.confidence_supported` is True for the producing
    method; `None` otherwise -- never fabricated."""
    raw_response: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float | None = None
    model_revision: str | None = None


@runtime_checkable
class HtrMethodAdapter(Protocol):
    """Every HTR recognition method -- local or external-upload, deterministic or probabilistic --
    is architecturally identical: an implementer of this Protocol. Mirrors `ProviderAdapter`'s
    role (Constitution Article 3) at the HTR-method layer.
    """

    def get_metadata(self) -> MethodMetadata: ...

    def get_capabilities(self) -> MethodCapabilities: ...

    def validate_environment(self) -> EnvironmentValidation: ...

    def recognize(self, input: RecognitionInput) -> RecognitionResult: ...

    def health_check(self) -> HealthCheckResult: ...
