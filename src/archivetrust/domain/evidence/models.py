"""The Canonical Evidence Model (ROADMAP.md S5.2, Constitution Article 5).

Evidence is the immutable, content-addressed record of one raw provider capture. It is never
mutated after creation -- not even when later shown to be erroneous, and not even when the
captured output was malformed and no Observation could validly be derived from it (Constitution
Article 5). Observations reference Evidence by id; they never embed or copy it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archivetrust.domain.shared.ids import content_address


class Precision(str, Enum):
    """The geometric precision class of a BoundingBox.

    Required per MILESTONE1_DOMAIN_MODEL.md S4.2: a pixel-accurate box (Docling, Tesseract) and a
    coarse quadrant estimate (Qwen2.5-VL) must never be treated as equivalent-precision evidence
    by geometric comparison logic.
    """

    PIXEL_ACCURATE = "pixel_accurate"
    COARSE_ESTIMATE = "coarse_estimate"


class BoundingBox(BaseModel):
    """A geometric location claim, authored only on Evidence (S4.2 -- 'both, but only one is
    authored'). Observations and Canonical Observations never author their own BoundingBox; they
    expose location by reference to their contributing Evidence's BoundingBox(es).
    """

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float
    precision: Precision

    @model_validator(mode="after")
    def _validate_extent(self) -> "BoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("BoundingBox extent must be non-negative (x1>=x0, y1>=y0)")
        return self


class ProcessingStage(str, Enum):
    """Which stage of a provider's pipeline produced this Evidence.

    An open, string-backed enum by design: Full Exposure (Constitution Article 6) requires
    surfacing whatever stages a provider actually has, and new providers will have stages this
    ontology cannot anticipate. RAW is the only stage every provider is guaranteed to have.
    """

    RAW = "raw"
    LAYOUT_ANALYSIS = "layout_analysis"
    OCR = "ocr"
    STRUCTURE_ANALYSIS = "structure_analysis"
    VLM_INFERENCE = "vlm_inference"
    POST_PROCESSING = "post_processing"


class Evidence(BaseModel):
    """One immutable, content-addressed raw provider capture.

    Content-addressed: identical raw output from the same provider/version/stage/prompt always
    produces the same `evidence_id`, so it is never duplicated in storage (ROADMAP.md S5.2).
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    provider: str
    provider_version: str
    raw_output: str
    processing_stage: ProcessingStage
    page: int | None = None
    region: str | None = None
    bounding_box: BoundingBox | None = None
    provider_confidence: float | None = None
    prompt: str | None = None
    supporting_metadata: dict[str, Any] = Field(default_factory=dict)
    model_revision: str | None = None
    """Pinned HTR model/checkpoint version, e.g. `MethodMetadata.model_revision`
    (docs/htr-domain-design.md §2, §4). Optional, defaults to `None` so every existing Evidence
    construction call site is unaffected (migration Stage 3). Included in the content-addressed
    `evidence_id` when set (see `compute_id`), so it must be stored on the instance -- otherwise
    re-validating an already-constructed Evidence's id (below) could never reproduce it."""
    pipeline_configuration_hash: str | None = None
    """Hash of the pipeline configuration that produced this Evidence (docs/htr-domain-design.md
    §2: "so line-level HTR results content-address correctly per the brief's 'hash of each shared
    input crop' requirement"). Same storage rationale as `model_revision` above -- included in the
    id hash when set, and therefore stored so the id stays self-verifiable."""
    execution_device: str | None = None
    """e.g. `"cuda:0"`, `"cpu"` -- which device produced this Evidence, when known."""
    execution_time_ms: float | None = None
    gpu_memory_mb: float | None = None
    software_environment: dict[str, Any] | None = None
    hardware_environment: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_id(self) -> "Evidence":
        expected = Evidence.compute_id(
            provider=self.provider,
            provider_version=self.provider_version,
            raw_output=self.raw_output,
            processing_stage=self.processing_stage,
            prompt=self.prompt,
            model_revision=self.model_revision,
            pipeline_configuration_hash=self.pipeline_configuration_hash,
        )
        if self.evidence_id != expected:
            raise ValueError(
                f"evidence_id {self.evidence_id!r} does not match its content address "
                f"{expected!r} -- construct Evidence via Evidence.create(), never by hand-setting "
                "evidence_id"
            )
        return self

    @staticmethod
    def compute_id(
        *,
        provider: str,
        provider_version: str,
        raw_output: str,
        processing_stage: ProcessingStage,
        prompt: str | None,
        model_revision: str | None = None,
        pipeline_configuration_hash: str | None = None,
    ) -> str:
        """Backward-compatible by construction (migration Stage 3, docs/htr-domain-design.md §2):
        `model_revision`/`pipeline_configuration_hash` are appended to the hashed parts only when
        supplied, so any existing caller that never passes them gets byte-identical output to
        before this extension landed."""
        parts = [
            provider,
            provider_version,
            processing_stage.value,
            prompt or "",
            raw_output,
        ]
        if model_revision is not None:
            parts.append(model_revision)
        if pipeline_configuration_hash is not None:
            parts.append(pipeline_configuration_hash)
        return content_address(*parts)

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        provider_version: str,
        raw_output: str,
        processing_stage: ProcessingStage,
        page: int | None = None,
        region: str | None = None,
        bounding_box: BoundingBox | None = None,
        provider_confidence: float | None = None,
        prompt: str | None = None,
        supporting_metadata: dict[str, Any] | None = None,
        model_revision: str | None = None,
        pipeline_configuration_hash: str | None = None,
        execution_device: str | None = None,
        execution_time_ms: float | None = None,
        gpu_memory_mb: float | None = None,
        software_environment: dict[str, Any] | None = None,
        hardware_environment: dict[str, Any] | None = None,
    ) -> "Evidence":
        """The only supported construction path -- computes the content-addressed id. Every
        Stage-3 parameter is optional and defaults to `None`/unset, so existing callers that pass
        none of them get identical `evidence_id` output to before this extension (see
        `compute_id`'s docstring)."""
        evidence_id = cls.compute_id(
            provider=provider,
            provider_version=provider_version,
            raw_output=raw_output,
            processing_stage=processing_stage,
            prompt=prompt,
            model_revision=model_revision,
            pipeline_configuration_hash=pipeline_configuration_hash,
        )
        return cls(
            evidence_id=evidence_id,
            provider=provider,
            provider_version=provider_version,
            raw_output=raw_output,
            processing_stage=processing_stage,
            page=page,
            region=region,
            bounding_box=bounding_box,
            provider_confidence=provider_confidence,
            prompt=prompt,
            supporting_metadata=supporting_metadata or {},
            model_revision=model_revision,
            pipeline_configuration_hash=pipeline_configuration_hash,
            execution_device=execution_device,
            execution_time_ms=execution_time_ms,
            gpu_memory_mb=gpu_memory_mb,
            software_environment=software_environment,
            hardware_environment=hardware_environment,
        )
