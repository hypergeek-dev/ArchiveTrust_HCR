"""ProviderAdapter contracts (ROADMAP.md S5.7, S5.8; MILESTONE0_REVIEW.md required change #7;
Constitution Article 3, Article 21).

Every provider -- an OCR engine, a layout parser, a vision-language model, present or future -- is
architecturally identical: an observation producer behind this one interface (Article 3). The
three registration axes below (Reproducibility, DeploymentLocality, Introspectability) are
independently-set per Article 21: none may be inferred from another. A concrete counter-example
this guards against (S5.5's clarification): Google Document AI is Deterministic yet
out-of-process; Donut is Deterministic yet black-box. This module must never special-case a
provider by name.

**docs/htr-migration-plan.md Stage 5 -- kept, deliberately, not deleted (a documented deviation
from the plan's literal text).** The plan describes this interface as "fully superseded" by
`providers/htr_adapter.py::HtrMethodAdapter` and slates it for deletion in the same stage as the
five OCR-era provider packages. Investigation before executing that step found it is not actually
superseded yet: `application/pipeline.py`, `acquisition/processing.py`,
`presentation/provider_manager_viewmodel.py`, and the composition root (`composition.py`) all still
orchestrate document-level provider invocation through `ProviderAdapter.observe()` /
`ProviderRunResult` / `ProviderRegistry`, and nothing in this codebase has migrated that
orchestration layer to `HtrMethodAdapter` (which operates one level down -- per-crop/per-line
`recognize()`, not per-document `observe()`; see `htr_adapter.py`'s own docstring). Deleting this
module now would not finish a completed migration, it would break the application pipeline with no
replacement in place -- that migration is a separate, larger piece of work than "delete the
replaced OCR provider architecture" (Stage 5's actual scope), and belongs with whichever future
stage first threads an `HtrMethodAdapter`-based method through document-level orchestration (SATRN,
Stage 6, is the natural candidate to prove out that shape). What Stage 5 *did* remove from this
module: `DeploymentProfile.STRUCTURED_PIPELINE` (PaddleOCR-VL Structured's now-deleted two-stage
pipeline shape) -- `SINGLE_PASS` is the only remaining profile.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.evidence.models import Evidence, ProcessingStage
from archivetrust.domain.ontology.base import Observation


class Reproducibility(str, Enum):
    """Does identical input + pinned version reproduce identical output? (S5.5) Never conflated
    with deployment locality or introspectability (Article 21)."""

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"


class DeploymentLocality(str, Enum):
    """Where the provider actually runs -- independent of whether it is deterministic."""

    IN_PROCESS = "in_process"
    OUT_OF_PROCESS = "out_of_process"


class Introspectability(str, Enum):
    """Can internals be inspected/exposed upstream, or is the provider black-box by design --
    independent of both other axes (S5.5's Donut example: deterministic yet black-box)."""

    OPEN = "open"
    BLACK_BOX = "black_box"


class DeploymentProfile(str, Enum):
    """Which pipeline architecture a provider is deployed as -- independent of `provider_id`. A
    single vendor system could in principle expose more than one deployment shape without that
    being a different provider in any sense that matters to the rest of this codebase; declared
    per-registration (see `ProviderRegistration.deployment_profile`) so `ProviderRegistry` can hold
    more than one adapter for the same `provider_id`, one per profile, rather than a provider
    needing a second, synthetic `provider_id` to express "I have two shapes." Defaults to
    `SINGLE_PASS` so every existing provider's registration remains valid unchanged.

    `STRUCTURED_PIPELINE` (PaddleOCR-VL Structured's detect-then-crop-then-recognize two-stage
    shape) was deleted in docs/htr-migration-plan.md Stage 5 (EXECUTED) along with that provider --
    `SINGLE_PASS` is the only remaining value. The independent-segmentation-stage concept it modeled
    too vendor-specifically is superseded by `docs/htr-domain-design.md` §7's generic
    `SegmentationAdapter`.
    """

    SINGLE_PASS = "single_pass"
    """One call (or one call per page) produces the final Evidence/Observations directly."""


class ProviderInputKind(str, Enum):
    """What shape of `source` a provider's `observe()` expects (Multi-Provider Activation
    milestone) -- the missing piece the Vision Provider Activation investigation identified:
    `ProviderAdapter.observe`'s own docstring always allowed adapter-specific input shapes, but
    nothing composed a per-adapter source before this. Declared per-adapter-class, never inferred
    from provider name (the same non-special-casing discipline `capability_matrix.py` follows).
    """

    DOCUMENT = "document"
    """The provider reads the whole Archive Object itself (e.g. Docling, which does its own
    multi-page/multi-format handling) -- the default, preserving every adapter's existing
    behavior unless it opts into `PAGE_IMAGE`."""
    PAGE_IMAGE = "page_image"
    """The provider must be invoked once per rendered page image (e.g. Tesseract, Qwen2.5-VL) --
    it cannot read a multi-page document directly."""


class ProviderRegistration(BaseModel):
    """The three independent axes, plus identity, recorded once per adapter -- never derived from
    one another and never inferred from provider name elsewhere in the codebase.
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    reproducibility: Reproducibility
    deployment_locality: DeploymentLocality
    introspectability: Introspectability
    deployment_profile: DeploymentProfile = DeploymentProfile.SINGLE_PASS
    """Which internal pipeline shape this registration represents (Phase 31). Defaulted so every
    pre-existing `ProviderRegistration` in this codebase is valid unchanged -- old code that never
    heard of `DeploymentProfile` implicitly means `SINGLE_PASS`, which is exactly what it already
    does today."""


class ProviderAttempt(BaseModel):
    """Mirrors `ProviderObservationAttempted` telemety (domain/telemetry/events.py) -- recorded by
    every adapter invocation regardless of outcome, so a caller wiring this into telemetry never
    has to reconstruct "was this even attempted" after the fact (Constitution Article 18).
    """

    model_config = ConfigDict(frozen=True)

    provider_id: str
    provider_version: str
    invocation_id: str
    target_region: str | None = None


class RejectedRawOutput(BaseModel):
    """Mirrors `EvidenceRejected` telemetry -- a provider response that failed schema/contract
    validation. The raw output is preserved even though no Evidence could validly be created from
    it (Constitution Article 5).
    """

    model_config = ConfigDict(frozen=True)

    processing_stage: ProcessingStage
    raw_output: str
    rejection_reason: str


class ProviderRunResult(BaseModel):
    """What one adapter invocation produces. Deliberately does not itself emit telemetry --
    telemetry emission is the pipeline/application layer's responsibility (S5.8), so this type
    stays a plain data carrier a pipeline can translate into `ProviderObservationAttempted`,
    `EvidenceCreated`, `ObservationCreated`, and `EvidenceRejected` events without the adapter
    needing to depend on `domain.telemetry` at all.
    """

    model_config = ConfigDict(frozen=True)

    attempt: ProviderAttempt
    evidence: tuple[Evidence, ...] = ()
    observations: tuple[Observation, ...] = ()
    rejections: tuple[RejectedRawOutput, ...] = ()
    failure_reason: str | None = None
    """Why the underlying provider could not complete, when it couldn't (Operational Hardening
    milestone). Distinguishes "ran and genuinely observed nothing" (empty `observations`,
    `failure_reason=None`) from "failed to run at all" — previously both surfaced as an empty
    result and the recorded cause (e.g. a real client's `last_error`) died with the client object,
    unreachable by telemetry (Article 18: silence must be distinguishable from failure)."""


class ProviderAdapter(ABC):
    """Base contract every provider adapter implements. Concrete subclasses set `registration`
    and implement `observe`; nothing else may construct Evidence or Observations on a provider's
    behalf (that would violate Article 20 -- providers are plugins behind this interface, not a
    set of special cases known to the rest of the system).
    """

    registration: ClassVar[ProviderRegistration]
    input_kind: ClassVar[ProviderInputKind] = ProviderInputKind.DOCUMENT

    @property
    def provider_id(self) -> str:
        return self.registration.provider_id

    @abstractmethod
    def observe(self, *, document_ref: str, invocation_id: str, source: Any) -> ProviderRunResult:
        """Runs this provider over `source` (provider-native input -- a file path, an image
        reference, an already-loaded native document; the shape is intentionally adapter-specific,
        since standardizing it further would leak provider vocabulary upstream) and returns
        whatever Evidence/Observations/rejections resulted. Must never raise for a malformed
        provider response -- that is a `RejectedRawOutput`, not an exception (Constitution
        Article 18: failure must be a recorded fact, not a silently swallowed or crashing one).
        """


class DeterministicProviderAdapter(ProviderAdapter):
    """Identical input + pinned provider_version must reproduce identical Evidence/Observation
    *content* (ids aside -- new Observation ids are expected on every invocation, S2.5). Full
    Exposure Principle (Article 6) applies: this adapter must surface the richest observation set
    the underlying provider is capable of producing, never silently discard something it computed.
    """


class ProbabilisticProviderAdapter(ProviderAdapter):
    """No determinism guarantee. Every Evidence record this adapter produces must carry `prompt`
    and model/version provenance (S5.5) -- enforced by convention here (concrete subclasses must
    populate `Evidence.prompt`), since the domain layer's `Evidence.prompt` field is optional for
    deterministic providers that have no such concept.
    """
