"""The application composition root / dependency injection (First Goal: dependency injection).

Framework-independent on purpose — **no PySide6 import here.** Constructs the Learning Platform
services and every Center's ViewModel and wires them together; the Qt widgets receive an
`AppContext` and bind to what it exposes. Keeping construction here, and Qt-free, means the entire
wiring is testable headless (`tests/test_composition.py`) and the Qt layer stays a thin
binding over already-verified objects.

`build_demo_context` seeds a small, realistic municipal archive (see `demo_data`) so the operator
lands on a populated operational picture rather than an empty prototype. It is clearly demo data —
produced by exercising the real engines and emitting the exact telemetry a live pipeline emits, not
a fabrication that bypasses the architecture.

**Provider Runtime Abstraction wiring:** `AppContext` constructs a real `ProviderRegistry`, a
`ProviderConfigurationStore`, per-provider `ProviderHealthTracker`s, and a `ModelRegistry` bound to
a `DeploymentLayout`. None of this changes what the Review/Processing/Quality/Evolution/Evidence-
Explorer Centers do; it only makes the Provider Manager, Settings, and First-Launch surfaces real
rather than placeholders.

**Workspace & Acquisition Management wiring (ROADMAP.md §5.13, Revision 5):** `AppContext` no
longer holds one global telemetry sink/provider config/model registry. It holds a `WorkspaceStore`
and, once a Workspace is opened (`open_workspace`), a full set of per-Workspace state — its own
`WorkspaceLayout`-backed `ModelRegistry`/`ProviderConfigurationStore`/`ProviderHealthTracker`s, its
own file-backed telemetry sink, its own `AcquisitionManager` and `WorkspaceProcessingService`.
`self.telemetry`/`self.interaction_sink` are now properties over the current Workspace's state, so
every existing Center ViewModel factory below is unchanged — only what it is fed changes (ROADMAP.md
§5.13, decision 4 in `IMPLEMENTATION_STATUS.md`).

**docs/htr-migration-plan.md Stage 5 (EXECUTED):** relocated here from
`clients/desktop/composition.py` (a "desktop"-namespaced module was never an appropriate home for
the app's non-UI composition root -- `admin/qualification.py` and `worker/service.py` always
imported it despite doing no UI work). All eager real-adapter registration for the five deleted
OCR-era providers (Docling, Tesseract+LayoutParser, Qwen2.5-VL, PaddleOCR-VL, PaddleOCR-VL
Structured, Surya) and the `vllm`/`paddle_layout` runtime factories those adapters needed is
removed: `_build_provider_registry` now registers no adapters (there are none left to register --
the first real one arrives with the SATRN `HtrMethodAdapter` implementation, Stage 6), and the
`vllm`/`paddle_layout` runtime kinds are always reported unavailable rather than probed, since the
modules that could ever back them (`runtime/vllm_runtime.py`, `runtime/pp_doclayout_runtime.py`)
are deleted, not merely uninstalled. `runtime_kind="transformers"` is untouched (Florence-2, a
later phase, needs it).
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Callable

from archivetrust.acquisition.folder_watch import FolderWatchConfig, FolderWatchSource
from archivetrust.acquisition.events import AcquisitionTelemetrySink, FileAcquisitionTelemetrySink
from archivetrust.acquisition.manager import AcquisitionManager
from archivetrust.acquisition.manual_import import ManualImportSource
from archivetrust.acquisition.processing import WorkspaceProcessingService
from archivetrust.admin.identity import (
    AdminAction,
    AdminAuditLog,
    AuthenticatedSession,
    LocalAccount,
    Permission,
    PermissionDenied,
    require_permission,
)
from archivetrust.application.progress import (
    FileProcessingProgressSink,
    InMemoryProcessingProgressSink,
)
from archivetrust.clients.desktop.demo_data import DemoDocument, seed_realistic_archive
from archivetrust.core import HeadlessCoreService
from archivetrust.domain.comparison.capability_matrix_data import production_capability_matrix
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink, InMemoryTelemetrySink
from archivetrust.evaluation.ground_truth import FileGroundTruthStore
from archivetrust.evaluation.workflow import (
    EvaluationApprovalService,
    FileEvaluationAssignmentStore,
)
from archivetrust.learning.review.sink import (
    FileReviewInteractionSink,
    InMemoryReviewInteractionSink,
    ReviewInteractionSink,
)
from archivetrust.learning.source import TelemetrySource
from archivetrust.presentation.acquisition_viewmodel import AcquisitionManagerViewModel
from archivetrust.presentation.evidence_explorer_viewmodel import EvidenceExplorerViewModel
from archivetrust.presentation.evolution_viewmodel import EvolutionCenterViewModel
from archivetrust.presentation.first_launch_viewmodel import FirstLaunchViewModel
from archivetrust.presentation.operational_stats import TelemetryAggregate
from archivetrust.presentation.operations_viewmodel import ProcessingCenterViewModel
from archivetrust.presentation.read_model import ReadModel
from archivetrust.presentation.provider_manager_viewmodel import (
    ProviderManagerViewModel,
    capabilities_for_provider,
)
from archivetrust.presentation.quality_viewmodel import QualityCenterViewModel
from archivetrust.presentation.review_viewmodel import ReviewViewModel
from archivetrust.presentation.settings_viewmodel import (
    AdvancedProviderSettingsViewModel,
    GeneralSettingsViewModel,
)
from archivetrust.presentation.shell_viewmodel import ShellViewModel
from archivetrust.presentation.workspace_viewmodel import WorkspaceManagerViewModel
from archivetrust.presentation.workspace_wizard_viewmodel import (
    WizardProviderChoice,
    WorkspaceWizardViewModel,
)
from archivetrust.providers.registry import ProviderRegistry
from archivetrust.review.service import ReviewService
from archivetrust.review.sampling.coordinator import AdaptiveReviewCoordinator
from archivetrust.review.sampling.log import (
    FileSamplingLogSink,
    InMemorySamplingLogSink,
    SamplingLogSink,
)
from archivetrust.providers.decoding.base import DecoderTelemetrySink, InMemoryDecoderTelemetrySink
from archivetrust.runtime.contracts import DeviceSelection, InferenceRequest, InferenceResult, InferenceRuntime, Precision, RuntimeCapabilities
from archivetrust.runtime.deployment_layout import DeploymentLayout
from archivetrust.runtime.gpu_resource_manager import GPUResourceManager, real_gpu_info_provider
from archivetrust.runtime.model_registry import LogicalModelBinding, ModelRegistry
from archivetrust.runtime.models import ModelDescriptor, ModelSource, ModelSourceKind
from archivetrust.runtime.provider_config import ProviderConfiguration, ProviderConfigurationStore
from archivetrust.runtime.provider_health import ProviderHealthTracker
from archivetrust.runtime.runtime_manager import RuntimeManager
from archivetrust.runtime.runtime_telemetry import (
    FileRuntimeTelemetrySink,
    RuntimeTelemetryEvent,
    RuntimeTelemetryKind,
    RuntimeTelemetrySink,
)
from archivetrust.workspace.models import Workspace
from archivetrust.workspace.store import WorkspaceStore
from archivetrust.security.input_policy import InputSecurityPolicy


class _UnavailableRuntime:
    """A placeholder `InferenceRuntime` registered for the "transformers" runtime kind when the
    real `transformers`/`torch` extra is not installed. Its `capabilities()`/`warm_up()` are enough
    for the Provider Manager and Model Registry to reason about the binding; `infer()` raises a
    clear, operator-facing error rather than an import traceback if actually invoked (Part 12: never
    expose a Python traceback).
    """

    runtime_kind = "transformers"

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_gpu=True, supports_cpu=True,
            supported_precisions=(Precision.AUTOMATIC, Precision.FP32, Precision.FP16, Precision.BF16, Precision.INT8),
            supports_batching=True, supports_deterministic_seed=True, max_tokens_configurable=True,
        )

    def warm_up(self) -> None:
        pass

    def infer(self, request: InferenceRequest) -> InferenceResult:
        raise RuntimeError(
            "The 'transformers' runtime extra is not installed. Install it with "
            "`pip install -e \".[transformers]\"` to run real vision-language inference."
        )

    def shut_down(self) -> None:
        pass


class _LazyFacade:
    """Defers building the real facade until its first attribute access (release WS5): a
    `TransformersRuntime` is constructed at workspace open whenever a binding names it, but
    importing torch/transformers costs ~10 s on this machine and belongs to the first actual
    inference (`warm_up`/`infer`), not to opening a workspace. Import errors still surface at
    that first real use, exactly where the non-lazy facade raised them."""

    def __init__(self, build) -> None:
        self._build = build
        self._real = None

    def __getattr__(self, name):
        if self._real is None:
            self._real = self._build()
        return getattr(self._real, name)


def _transformers_runtime_factory(resolved):
    """Constructs a real `TransformersRuntime` bound to a `ResolvedModel`. Imported lazily so this
    module stays importable (and the rest of the desktop app testable) whether or not the
    `transformers` extra is installed -- selection between this and `_UnavailableRuntime` happens
    once, at `AppContext` construction, per which import actually succeeds.
    """
    from archivetrust.runtime.transformers_runtime import TransformersRuntime

    def _build_real_facade():
        from archivetrust.runtime.transformers_runtime import real_torch_and_transformers_facade

        return real_torch_and_transformers_facade()

    return TransformersRuntime(
        model_id=resolved.descriptor.model_id,
        local_path=resolved.local_path,
        revision=resolved.descriptor.source.revision,
        device=resolved.device,
        precision=resolved.precision,
        facade=_LazyFacade(_build_real_facade),
    )


def _transformers_available() -> bool:
    """Probed via `find_spec`, never a real import (release WS5): actually importing torch +
    transformers costs ~10 s and was the dominant share of every workspace open on this machine —
    an availability *probe* must not pay a model framework's import bill. The real import still
    happens lazily inside `_transformers_runtime_factory` on first use, where an import failure
    surfaces as that runtime's own clear error."""
    import importlib.util

    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def _register_transformers_runtime_factory(model_registry: ModelRegistry) -> None:
    """Registers the real `TransformersRuntime` factory if `torch`/`transformers` are importable,
    else `_UnavailableRuntime` -- checked once at startup, never assumed. This is what makes
    `pip install -e ".[transformers]"` actually take effect: previously this binding was hardcoded
    to the placeholder regardless of what was installed.
    """
    if not _transformers_available():
        model_registry.register_runtime_factory("transformers", lambda resolved: _UnavailableRuntime())
    else:
        model_registry.register_runtime_factory("transformers", _transformers_runtime_factory)


def _vllm_runtime_factory_with_budget(
    resolved, *, gpu_memory_utilization: float, telemetry: RuntimeTelemetrySink | None = None
):
    """`RuntimeManager` requires a `vllm_runtime_factory` (its `VLLMRuntimeFactory` protocol
    parameter); `runtime/vllm_runtime.py` itself is deleted (docs/htr-migration-plan.md Stage 5 --
    EXECUTED, only consumers were the deleted PaddleOCR-VL/Surya adapters), so this always reports
    unavailable rather than probing for an extra that could never back a real runtime again. A real
    `vllm`-backed factory returns if/when a future HTR method needs vLLM-served inference."""
    return _UnavailableRuntime()  # `infer()` raises a clear, operator-facing error if ever called


def _vllm_discovery_probe(resolved) -> bool:
    """`RuntimeManager`'s discovery probe seam -- always `False`, mirroring
    `_vllm_runtime_factory_with_budget` above (`runtime/vllm_runtime.py` is deleted, not merely
    uninstalled, so there is nothing to discover)."""
    return False


_RUNTIME_AVAILABILITY_PROBES: dict[str, Callable[[], bool]] = {
    "transformers": _transformers_available,
    "vllm": lambda: False,
    "paddle_layout": lambda: False,
}
"""`runtime_kind -> "can this process actually construct the real runtime right now"`. `vllm`/
`paddle_layout` are always `False` (docs/htr-migration-plan.md Stage 5 -- EXECUTED): the modules
that could ever back them are deleted, not merely uninstalled extras."""

_RUNTIME_UNAVAILABLE_REASONS: dict[str, str] = {
    "transformers": "The 'transformers' extra is not installed. Install it with "
    "`pip install -e \".[transformers]\"`.",
    "vllm": "vLLM-served runtimes are not available in this build (the OCR-era providers that used "
    "them were removed; docs/htr-migration-plan.md Stage 5).",
    "paddle_layout": "The Structured Pipeline layout-detection runtime is not available in this "
    "build (its only consumer, PaddleOCR-VL Structured, was removed; docs/htr-migration-plan.md "
    "Stage 5).",
}


class AppContext:
    """Constructed once at startup; the single source every View reads its dependencies from.

    Holds no global telemetry/provider/model state itself (ROADMAP.md §5.13, Revision 5) — that
    state belongs to whichever Workspace is currently open (`open_workspace`). `self.telemetry`/
    `self.interaction_sink` are properties over the current Workspace's own state, so every
    existing Center ViewModel factory below reads exactly as it did before this revision; only what
    it is fed changes.
    """

    def __init__(
        self,
        *,
        reviewer_ref: str = "reviewer",
        deployment_root: Path | None = None,
        auto_create_default_workspace: bool = True,
        persistent_telemetry: bool = True,
        use_process_worker: bool = False,
        authenticated_session: AuthenticatedSession | None = None,
        require_authentication: bool = False,
    ) -> None:
        if require_authentication and authenticated_session is None:
            raise PermissionDenied("production desktop startup requires an authenticated local account")
        self.authenticated_session = authenticated_session
        self.account: LocalAccount | None = (
            authenticated_session.require_active() if authenticated_session is not None else None
        )
        self.reviewer_ref = self.account.reviewer_ref if self.account is not None else reviewer_ref
        self.persistent_telemetry = persistent_telemetry
        self.use_process_worker = use_process_worker
        """When True (the default — Operational Hardening milestone, Priority 1), each Workspace's
        Trust Engine and Acquisition telemetry streams are `FileTelemetrySink`/
        `FileAcquisitionTelemetrySink` under the Workspace's own `telemetry/` directory, so every
        event survives restart and replay works across sessions. `False` keeps the old in-memory
        sinks — used by `build_demo_context` (whose seeded demo events must not accrete into a
        durable stream on every launch) and by tests that want a throwaway context."""
        self.shell = ShellViewModel()

        # -- app-level installation layout (Provider Runtime Abstraction milestone) -------------
        self.app_layout = DeploymentLayout(
            root=deployment_root or (Path.cwd() / "archivetrust_data")
        ).ensure()
        self.workspace_store = WorkspaceStore(app_root=self.app_layout.root / "workspaces")
        self.admin_audit_log = AdminAuditLog(self.app_layout.root / "identity" / "admin-audit.jsonl")

        self.current_workspace: Workspace | None = None
        self._telemetry: TelemetrySink | TelemetrySource | None = None
        self.interaction_sink: ReviewInteractionSink | None = None
        self.review_service: ReviewService | None = None
        self.evaluation_approval_service: EvaluationApprovalService | None = None
        self.sampling_log: SamplingLogSink | None = None
        self.adaptive_review_coordinator: AdaptiveReviewCoordinator | None = None
        self.model_registry: ModelRegistry | None = None
        # -- GPU-aware runtime management (Production Runtime Completion milestone) -------------
        # Process-wide, constructed once regardless of how many Workspaces are opened in this
        # session -- GPU memory and Docker containers are physical-machine resources, not
        # Workspace-scoped ones, so reservations and warm runtimes must survive a workspace switch
        # (`rebind`, called from `open_workspace` below) rather than being torn down and rebuilt.
        self.gpu_resource_manager = GPUResourceManager(gpu_info=real_gpu_info_provider())
        self.runtime_telemetry = FileRuntimeTelemetrySink(
            self.app_layout.telemetry_dir / "runtime_events.jsonl"
        )
        """Durable unconditionally -- deliberately *not* gated by `persistent_telemetry` (Production
        Investigation, 2026-07-13: found `--demo`'s `persistent_telemetry=False` silently discarding
        real `RUNTIME_START_FAILED`/`WARMUP_TIME` events from a real vLLM startup failure, because
        this line originally reused that flag). `persistent_telemetry=False` exists so `--demo`'s
        *seeded document data* doesn't accrete into domain telemetry on every launch (see
        `build_demo_context`) -- it says nothing about whether the GPU/Docker activity a demo
        Workspace's real, runtime-backed provider generates should be durable. That activity is real
        regardless of demo mode (same container, same GPU reservation, same possible startup
        failure), so it's recorded the same way `self.telemetry`/`self.acquisition_telemetry` are by
        default, matching container starts/stops, GPU allocation, and runtime construction failures
        (`RuntimeTelemetryKind.RUNTIME_START_FAILED`) that previously lived only in
        `InMemoryRuntimeTelemetrySink` and vanished on process exit, leaving a crash mid-run with no
        runtime-level trace. Process-wide (`app_layout`, not a per-Workspace `layout`), matching the
        GPU/Docker resources it describes."""
        self.decoder_telemetry: DecoderTelemetrySink = InMemoryDecoderTelemetrySink()
        """Shared across every provider's Importer (Production Observability Completion milestone)
        -- process-wide, like `runtime_telemetry`, so `ArtifactDecoded`/`ArtifactDecodeFailed`
        events from every provider land in one queryable stream regardless of which Workspace is
        open when they're recorded."""
        self.runtime_manager: RuntimeManager | None = None
        self.provider_registry: ProviderRegistry | None = None
        self.provider_config_store: ProviderConfigurationStore | None = None
        self.provider_health_trackers: dict[str, ProviderHealthTracker] = {}
        self.provider_availability: dict[str, bool] = {}
        """`provider_id -> can this provider's real client actually execute right now`
        (Operational Completion milestone) -- library/binary/model-binding present or not. Feeds
        the Provider Manager's "Unavailable" status and `processing_service()`'s adapter filter."""
        self.running_provider_ids: set[str] = set()
        """Providers with an invocation currently in flight, mutated by the background queue
        worker (`queue_worker.py`) around each `WorkspaceProcessingService.process()` call. Plain
        in-memory UI state, not telemetry (Article 16: process-level state doesn't belong in
        domain telemetry) -- granularity is per-document, not per-provider, since `run_pipeline`
        invokes every enabled adapter for one document inside a single atomic call."""
        self.queue_failure_count: int = 0
        """Documents the queue worker could not process at all this session (e.g. no provider was
        both enabled and available, so `run_pipeline` never even started) -- distinct from
        `EvidenceRejected` telemetry (a provider rejecting one page mid-run). Plain in-memory UI
        state incremented by `QueueWorker`, not a new telemetry event (Article 16) -- feeds
        `ProcessingCenterViewModel.documents_failed()` so the "Failed" tile is never dishonestly
        `0` while every document in a run actually failed."""
        self.acquisition_telemetry: AcquisitionTelemetrySink | None = None
        self.acquisition_manager: AcquisitionManager | None = None
        self.processing_progress: InMemoryProcessingProgressSink | None = None
        """Per-Workspace processing-run progress stream (Production Hardening Review, 2026-07-13):
        run started (with which providers), document handed to the pipeline, document finished,
        run finished -- durable under `telemetry/processing.jsonl` so a process that dies
        mid-document leaves a trace of exactly what was in flight, which the buffered-per-document
        domain stream structurally cannot provide."""
        self.processing_stats: TelemetryAggregate | None = None
        self.read_model: ReadModel | None = None
        """Shared incremental aggregate over the current Workspace's domain telemetry stream --
        one instance per opened Workspace, handed to every `processing_viewmodel()` so a
        Processing Center refresh costs O(events since last refresh), not a full rescan of the
        entire history per refresh (the 2026-07-13 incident's primary root cause)."""

        existing = self.workspace_store.list()
        if existing:
            self.open_workspace(existing[0].id)
        elif auto_create_default_workspace:
            # A bare `AppContext()` remains immediately usable (`.telemetry`, etc.) exactly as it
            # was before Workspace & Acquisition Management -- callers that want the real Workspace
            # Wizard experience call `create_workspace`/`workspace_wizard_viewmodel` themselves;
            # this default only exists so nothing constructed against the old single-global-context
            # shape breaks (ROADMAP.md §5.13, decision 4). `build_demo_context` opts out
            # (`auto_create_default_workspace=False`) so it can create its own, realistically named
            # demo Workspace instead of this generic placeholder.
            self.create_workspace("Default Workspace")

    # -- Workspace lifecycle -------------------------------------------------------------------

    @property
    def telemetry(self) -> TelemetrySink | TelemetrySource:
        if self._telemetry is None:
            raise RuntimeError("No Workspace is open — call open_workspace() or create one first.")
        return self._telemetry

    def open_workspace(self, workspace_id: str) -> Workspace:
        """Rebuilds every per-Workspace object: layout, provider/model configuration, telemetry,
        Acquisition Manager, and the Workspace Processing Service. Existing Center ViewModels need
        no change — only the state their factory methods below read from changes.
        """
        workspace = self.workspace_store.open(workspace_id)
        layout = self.workspace_store.layout_for(workspace_id)

        # The Workspace's own durable streams (Operational Hardening milestone, Priority 1):
        # events.jsonl is the Trust Engine's knowledge record, acquisition.jsonl the acquisition
        # history — both under this Workspace's telemetry/ directory, so switching Workspaces
        # switches histories and a restart reopens exactly the state that was recorded.
        if self.persistent_telemetry:
            self._telemetry = FileTelemetrySink(layout.telemetry_dir / "events.jsonl")
        else:
            self._telemetry = InMemoryTelemetrySink()
        self.interaction_sink = (
            FileReviewInteractionSink(layout.telemetry_dir / "review_interactions.jsonl")
            if self.persistent_telemetry
            else InMemoryReviewInteractionSink()
        )
        self.review_service = ReviewService(
            telemetry_source=self._telemetry,  # type: ignore[arg-type]
            telemetry_sink=self._telemetry,  # type: ignore[arg-type]
            interaction_sink=self.interaction_sink,
        )
        # Evaluation approval is a separate durable data plane. It is never registered as a
        # provider and never shares the operational review telemetry stream.
        evaluation_root = layout.root / "evaluation"
        self.evaluation_approval_service = EvaluationApprovalService(
            assignments=FileEvaluationAssignmentStore(evaluation_root / "assignments.jsonl"),
            annotations=FileGroundTruthStore(evaluation_root / "annotations.jsonl"),
        )

        self.model_registry = ModelRegistry(layout.deployment_layout)
        _register_transformers_runtime_factory(self.model_registry)
        # "vllm"/"paddle_layout" runtime kinds are never registered here anymore (Stage 5 --
        # EXECUTED): the modules that could ever back them (`runtime/vllm_runtime.py`,
        # `runtime/pp_doclayout_runtime.py`) are deleted, and no remaining provider names either
        # runtime kind in its binding. `RuntimeManager.get_or_create` would raise a clear error if
        # a stale persisted binding ever did (never silently construct something wrong).
        self.model_registry.load_bindings()

        if self.runtime_manager is None:
            self.runtime_manager = RuntimeManager(
                model_registry=self.model_registry,
                gpu_resource_manager=self.gpu_resource_manager,
                vllm_runtime_factory=functools.partial(
                    _vllm_runtime_factory_with_budget, telemetry=self.runtime_telemetry
                ),
                vllm_discovery_probe=_vllm_discovery_probe,
                telemetry=self.runtime_telemetry,
            )
        else:
            self.runtime_manager.rebind(self.model_registry)
        self.provider_registry, self.provider_availability = _build_provider_registry(
            self.decoder_telemetry, self.runtime_telemetry
        )
        self.provider_config_store = ProviderConfigurationStore()
        self.provider_health_trackers = {
            adapter.provider_id: ProviderHealthTracker(adapter.provider_id)
            for adapter in self.provider_registry.all()
        }
        self.running_provider_ids = set()
        self.queue_failure_count = 0

        if self.persistent_telemetry:
            self.acquisition_telemetry = FileAcquisitionTelemetrySink(
                layout.telemetry_dir / "acquisition.jsonl"
            )
            self.processing_progress = FileProcessingProgressSink(
                layout.telemetry_dir / "processing.jsonl"
            )
            self.sampling_log = FileSamplingLogSink(layout.telemetry_dir / "sampling.jsonl")
        else:
            self.acquisition_telemetry = AcquisitionTelemetrySink()
            self.processing_progress = InMemoryProcessingProgressSink()
            self.sampling_log = InMemorySamplingLogSink()
        self.adaptive_review_coordinator = AdaptiveReviewCoordinator(
            review_service=self.review_service,
            telemetry_source=self._telemetry,  # type: ignore[arg-type]
            sampling_log=self.sampling_log,
            telemetry_sink=self._telemetry,  # type: ignore[arg-type]  # F4: packet dispatch/closure
            reviewer_ref=self.reviewer_ref,
        )
        self.processing_stats = TelemetryAggregate()
        self.read_model = ReadModel(
            telemetry_source=self._telemetry,  # type: ignore[arg-type]
            review_interactions=self.interaction_sink,
            core=self.processing_stats,
        )
        self.acquisition_manager = AcquisitionManager(
            workspace.id,
            layout,
            self.acquisition_telemetry,
            self.processing_progress,
            security_policy=(
                InputSecurityPolicy()
                if self.use_process_worker or self.authenticated_session is not None
                else None
            ),
        )

        self.current_workspace = workspace
        self._current_layout = layout
        self.activate_configured_vision_provider()
        return workspace

    def create_workspace(self, name: str, description: str = "") -> Workspace:
        if self.account is not None:
            require_permission(self.account, Permission.WORKSPACE_CREATE)
        workspace = self.workspace_store.create(name, description=description)
        self.open_workspace(workspace.id)
        if self.account is not None:
            self.admin_audit_log.record(
                actor=self.account,
                action=AdminAction.WORKSPACE_CREATED,
                target_ref=workspace.id,
                target_label=workspace.name,
                resulting_state_ref=workspace.id,
                session_id=self.authenticated_session.session_id if self.authenticated_session else None,
            )
        return workspace

    def require_permission(self, permission: Permission) -> LocalAccount:
        if self.authenticated_session is None:
            raise PermissionDenied("this production action requires an authenticated session")
        account = self.authenticated_session.require_active()
        require_permission(account, permission)
        return account

    def audit_action(
        self,
        action: AdminAction,
        *,
        target_ref: str,
        target_label: str | None = None,
        prior_state_ref: str | None = None,
        resulting_state_ref: str | None = None,
        reason: str | None = None,
        success: bool = True,
    ) -> None:
        if self.account is None:
            return
        self.admin_audit_log.record(
            actor=self.account,
            action=action,
            target_ref=target_ref,
            target_label=target_label,
            prior_state_ref=prior_state_ref,
            resulting_state_ref=resulting_state_ref,
            reason=reason,
            success=success,
            session_id=self.authenticated_session.session_id if self.authenticated_session else None,
        )

    def enabled_provider_ids(self) -> set[str]:
        """Providers this Workspace would actually invoke right now: enabled *and* available (a
        real client, not the "Unavailable" placeholder) — the single filter both
        `processing_service()` and the background queue worker use, so "which providers will run"
        is computed in exactly one place.
        """
        return {
            adapter.provider_id
            for adapter in self.provider_registry.all()
            if self.provider_config_store.get(adapter.provider_id).enabled
            and self.provider_availability.get(adapter.provider_id, False)
        }

    def processing_service(self) -> WorkspaceProcessingService:
        """A `WorkspaceProcessingService` for the current Workspace, using whichever providers are
        currently enabled and available (ROADMAP.md §5.13.2, decision 6). Constructed fresh per
        call rather than cached, since which providers are enabled can change between calls.
        """
        enabled_ids = self.enabled_provider_ids()
        enabled_adapters = tuple(a for a in self.provider_registry.all() if a.provider_id in enabled_ids)
        return WorkspaceProcessingService(
            adapters=enabled_adapters,
            reconciliation_policy=ReconciliationPolicy(policy_version=1),
            capability_matrix=production_capability_matrix(),
            confidence_policy=ConfidencePolicy(confidence_policy_version=1),
            telemetry_sink=self.telemetry,  # type: ignore[arg-type]
            layout=self._current_layout,
            workspace_id=self.current_workspace.id if self.current_workspace else "",
            security_policy=(
                InputSecurityPolicy()
                if self.use_process_worker or self.authenticated_session is not None
                else None
            ),
        )

    def provider_health_probes(self) -> dict[str, Callable[[], bool]]:
        """`provider_id -> cheap reachability probe` for every currently registered provider,
        reusing the exact probe function `activate_configured_vision_provider` already calls for it
        (`_RUNTIME_AVAILABILITY_PROBES[runtime_kind]`) -- never new probing logic, and never a
        provider-identity branch (Provider Health Guard, Phase 22, Constitution Article 20).

        Only providers actually activated through a `ModelRegistry` binding (`_activate_vision_
        binding`) are runtime-backed at all; a provider registered without one (docs/htr-migration-
        plan.md Stage 5's now-empty `_build_provider_registry` register nothing today, but a future
        in-process, non-runtime-backed provider -- or the `ProviderAdapter` conformance fakes tests
        register directly -- would) has no external dependency to probe and is trivially healthy,
        mirroring how the deleted `docling`/`tesseract_layoutparser` deterministic providers (no
        runtime binding at all) were always reported healthy pre-Stage-5.
        """
        from archivetrust.presentation.first_launch_viewmodel import runtime_kind_for_provider

        probes: dict[str, Callable[[], bool]] = {}
        if self.provider_registry is None:
            return probes
        bound_provider_ids = {b.provider_id for b in self.model_registry.bindings()} if self.model_registry else set()
        for adapter in self.provider_registry.all():
            provider_id = adapter.provider_id
            if provider_id not in bound_provider_ids:
                probes[provider_id] = lambda: True
                continue
            runtime_kind = runtime_kind_for_provider(provider_id)
            probe = _RUNTIME_AVAILABILITY_PROBES.get(runtime_kind)
            if probe is not None:
                probes[provider_id] = probe
        return probes

    def provider_health_guard(self, required_provider_ids: frozenset[str]):
        """A `ProviderHealthGuard` (Phase 22) for exactly the providers named, wiring
        `provider_health_probes()` plus `attempt_docker_recovery` for whichever of those providers
        are backed by a Docker/`vllm` runtime -- the only recovery this guard attempts
        automatically, per its own conservative-by-design policy.
        """
        from archivetrust.acquisition.provider_health_guard import (
            ProviderHealthGuard,
            RecoveryAttempt,
            attempt_docker_recovery,
        )
        from archivetrust.presentation.first_launch_viewmodel import runtime_kind_for_provider

        probes = self.provider_health_probes()
        recovery: dict[str, Callable[[], RecoveryAttempt]] = {
            provider_id: (lambda provider_id=provider_id: attempt_docker_recovery(provider_id))
            for provider_id in required_provider_ids
            if runtime_kind_for_provider(provider_id) == "vllm"
        }
        return ProviderHealthGuard(required_provider_ids=required_provider_ids, probes=probes, recovery=recovery)

    # -- ViewModel factories (one per Center) -------------------------------------------------

    def resolve_archive_path(self, archive_object_ref: str) -> Path | None:
        """The absolute path to an Archive Object's immutable original file (Priority 14) — how
        the Review Center opens the real document instead of drawing a placeholder rectangle.
        `None` if the Archive Object isn't found in this Workspace's acquisition history (e.g. a
        stale reference, or acquisition telemetry that predates `FileAcquisitionTelemetrySink`).
        """
        archive_object = self.acquisition_manager.archive_object_by_ref(archive_object_ref)
        if archive_object is None:
            return None
        return self._current_layout.archive_dir / archive_object.storage_path

    def review_viewmodel(self, *, document_ref: str, archive_object_ref: str) -> ReviewViewModel:
        if self.authenticated_session is not None:
            self.require_permission(Permission.REVIEW)
        return ReviewViewModel(
            service=self.review_service,
            reviewer_ref=self.reviewer_ref,
            document_ref=document_ref,
            archive_object_ref=archive_object_ref,
        )

    def processing_viewmodel(self) -> ProcessingCenterViewModel:
        return ProcessingCenterViewModel(
            self.telemetry,  # type: ignore[arg-type]
            workspace=self.current_workspace,
            acquisition_manager=self.acquisition_manager,
            queue_failure_count=self.queue_failure_count,
            enabled_provider_ids=frozenset(self.enabled_provider_ids()),
            configured_provider_ids=frozenset(a.provider_id for a in self.provider_registry.all()),
            aggregate=self.processing_stats,
            processing_progress=self.processing_progress,
            gpu_resource_manager=self.gpu_resource_manager,
            current_state=self.read_model.current_state,
        )

    def core_service(self) -> HeadlessCoreService:
        """In-process C1 service boundary over the current Workspace."""
        return HeadlessCoreService(
            workspace=self.current_workspace,
            telemetry_source=self.telemetry,  # type: ignore[arg-type]
            read_model=self.read_model,
            processing_viewmodel_factory=self.processing_viewmodel,
            acquisition_manager=self.acquisition_manager,
            processing_progress=self.processing_progress,
        )

    def quality_viewmodel(self) -> QualityCenterViewModel:
        return QualityCenterViewModel(
            self.telemetry,  # type: ignore[arg-type]
            self.interaction_sink,
            read_model=self.read_model,
        )

    def evolution_viewmodel(self) -> EvolutionCenterViewModel:
        return EvolutionCenterViewModel(self.telemetry)  # type: ignore[arg-type]

    def evidence_explorer_viewmodel(self) -> EvidenceExplorerViewModel:
        return EvidenceExplorerViewModel(
            self.telemetry, current_state=self.read_model.current_state  # type: ignore[arg-type]
        )

    # -- HTR research interface (docs/htr-migration-plan.md Stage 11) --------------------------

    @property
    def htr_research_store(self):
        """The process-wide `HtrResearchStore` backing the research surfaces.

        Process-wide rather than per-Workspace, and lazily constructed: it holds the HTR corpus,
        experiment, canonical and external-import entities, which have no telemetry persistence yet
        (see `htr/research_store.py`'s module docstring for why that is deliberate rather than an
        oversight). It is therefore empty on a fresh launch -- the research pages render an honest
        "nothing registered yet" state rather than seeded placeholder data.
        """
        from archivetrust.htr.research_store import HtrResearchStore

        if getattr(self, "_htr_research_store", None) is None:
            self._htr_research_store = HtrResearchStore()
        return self._htr_research_store

    @property
    def blind_review_store(self):
        """The process-wide `BlindReviewStore` backing the Review Center. In-memory by that class's
        own design; constructed lazily for the same reason as `htr_research_store`."""
        from archivetrust.review.blind_review.store import BlindReviewStore

        if getattr(self, "_blind_review_store", None) is None:
            self._blind_review_store = BlindReviewStore()
        return self._blind_review_store

    def htr_method_adapters(self) -> tuple:
        """The three real `HtrMethodAdapter`s, constructed but never invoked here.

        Each constructor is inert (no model load, no network, no subprocess) -- only `recognize()`
        and `validate_environment()` reach outward, and the Methods page probes those only on
        explicit request. An adapter whose optional dependencies are missing is skipped with its
        import error surfaced by `htr_method_adapter_errors`, never silently dropped.
        """
        adapters, _errors = self._build_htr_adapters()
        return adapters

    def htr_method_adapter_errors(self) -> tuple[str, ...]:
        """Import/construction failures from `htr_method_adapters()`, so a missing method is a
        reported absence rather than a shorter list nobody notices."""
        _adapters, errors = self._build_htr_adapters()
        return errors

    @staticmethod
    def _build_htr_adapters() -> tuple[tuple, tuple[str, ...]]:
        adapters: list = []
        errors: list[str] = []
        for module_name, class_name in (
            ("archivetrust.providers.satrn.adapter", "SatrnAdapter"),
            ("archivetrust.providers.florence2_htr.adapter", "Florence2Adapter"),
            ("archivetrust.providers.transkribus.adapter", "TranskribusAdapter"),
        ):
            try:
                import importlib

                module = importlib.import_module(module_name)
                adapters.append(getattr(module, class_name)())
            except Exception as exc:  # noqa: BLE001 - a missing method is reported, not fatal
                errors.append(f"{class_name}: {exc}")
        return tuple(adapters), tuple(errors)

    def research_dashboard_viewmodel(self):
        from archivetrust.presentation.htr_dashboard_viewmodel import ResearchDashboardViewModel

        return ResearchDashboardViewModel(
            self.htr_research_store,
            review_center=self.review_center_viewmodel(),
            review_target_refs=tuple(
                sorted({a.target_ref for a in self.blind_review_store.all_assignments()})
            ),
        )

    def method_overview_viewmodel(self):
        from archivetrust.presentation.htr_methods_viewmodel import MethodOverviewViewModel

        return MethodOverviewViewModel(
            self.htr_method_adapters(),
            method_runs=self.htr_research_store.method_runs(),
        )

    def dataset_explorer_viewmodel(self):
        from archivetrust.presentation.htr_dataset_viewmodel import DatasetExplorerViewModel

        archive_objects: dict[str, object] = {}
        if self.acquisition_manager is not None:
            for archive_object in self.acquisition_manager.pending:
                archive_objects[archive_object.id] = archive_object
        return DatasetExplorerViewModel(self.htr_research_store, archive_objects=archive_objects)

    def experiment_builder_viewmodel(self):
        from archivetrust.htr.evaluation import definitions
        from archivetrust.presentation.htr_experiment_builder_viewmodel import (
            ExperimentBuilderViewModel,
        )

        metric_ids = tuple(
            sorted(
                value.metric_definition_id
                for name, value in vars(definitions).items()
                if not name.startswith("_") and hasattr(value, "metric_definition_id")
            )
        )
        return ExperimentBuilderViewModel(
            available_method_ids=tuple(
                adapter.get_metadata().method_id for adapter in self.htr_method_adapters()
            ),
            available_metric_definition_ids=metric_ids,
        )

    def htr_comparison_viewmodel(self):
        from archivetrust.presentation.htr_comparison_viewmodel import ComparisonViewModel

        return ComparisonViewModel(self.htr_research_store)

    def htr_evidence_chain_viewmodel(self):
        from archivetrust.presentation.htr_evidence_viewmodel import HtrEvidenceChainViewModel

        return HtrEvidenceChainViewModel(self.htr_research_store)

    def review_center_viewmodel(self):
        from archivetrust.presentation.htr_review_center_viewmodel import ReviewCenterViewModel

        return ReviewCenterViewModel(self.blind_review_store)

    def local_worker_client(self):
        from archivetrust.worker.client import LocalWorkerClient

        if self.current_workspace is None:
            raise RuntimeError("No Workspace is open")
        if self.authenticated_session is not None:
            self.require_permission(Permission.PROCESSING_CONTROL)
        return LocalWorkerClient(
            deployment_root=self.app_layout.root,
            workspace_id=self.current_workspace.id,
            session=self.authenticated_session,
        )

    def provider_manager_viewmodel(self) -> ProviderManagerViewModel:
        return ProviderManagerViewModel(
            provider_registry=self.provider_registry,
            configuration_store=self.provider_config_store,
            health_trackers=self.provider_health_trackers,
            model_registry=self.model_registry,
            availability=self.provider_availability,
            running_provider_ids=self.running_provider_ids,
        )

    def general_settings_viewmodel(self) -> GeneralSettingsViewModel:
        return GeneralSettingsViewModel(
            provider_registry=self.provider_registry, configuration_store=self.provider_config_store
        )

    def advanced_provider_settings_viewmodel(self, provider_id: str) -> AdvancedProviderSettingsViewModel:
        adapter = self.provider_registry.get(provider_id)
        return AdvancedProviderSettingsViewModel(
            provider_id=provider_id,
            configuration_store=self.provider_config_store,
            capabilities=capabilities_for_provider(adapter),
        )

    def first_launch_viewmodel(self) -> FirstLaunchViewModel:
        return FirstLaunchViewModel(layout=self._current_layout.deployment_layout, model_registry=self.model_registry)

    def workspace_manager_viewmodel(self) -> WorkspaceManagerViewModel:
        return WorkspaceManagerViewModel(
            store=self.workspace_store,
            context=self,
            audit_log=self.admin_audit_log,
            actor=self.account,
            session_id=self.authenticated_session.session_id if self.authenticated_session else None,
        )

    def workspace_wizard_viewmodel(self) -> WorkspaceWizardViewModel:
        return WorkspaceWizardViewModel(store=self.workspace_store, context=self)

    def available_wizard_providers(self) -> tuple[WizardProviderChoice, ...]:
        """Every provider the Workspace Wizard's Providers step may offer — enumerated, never
        hand-listed. Two sources:

        - **Eagerly-registered providers** (`self.provider_registry.all()`): whatever this
          composition root's `_build_provider_registry` (or, in tests, a caller directly
          registering a fake/future adapter) actually registered. Empty in production as of
          docs/htr-migration-plan.md Stage 5 (EXECUTED) -- the five OCR-era providers this used to
          eagerly register are deleted, and no `HtrMethodAdapter`-era provider has landed yet
          (Stage 6) -- but the enumeration itself stays generic, never assuming "empty today" means
          "never populate this branch again."
        - **The Vision Provider catalog** (`VISION_PROVIDER_CHOICES`) for any provider not yet
          registered. Availability is checked via `_RUNTIME_AVAILABILITY_PROBES`, the same side-
          effect-free check `activate_configured_vision_provider` uses — never a runtime
          construction, so listing an inactive provider here starts no container and reserves no
          GPU memory.

        A provider already registered is taken from the first source only, so it is never listed
        twice.
        """
        from archivetrust.presentation.first_launch_viewmodel import VISION_PROVIDER_CHOICES

        vision_labels = {pid: label for pid, label, _default_model, _runtime_kind in VISION_PROVIDER_CHOICES}
        choices: list[WizardProviderChoice] = []
        seen: set[str] = set()

        if self.provider_registry is not None:
            for adapter in self.provider_registry.all():
                provider_id = adapter.provider_id
                available = self.provider_availability.get(provider_id, False)
                choices.append(
                    WizardProviderChoice(
                        provider_id=provider_id,
                        label=vision_labels.get(provider_id, provider_id),
                        description="",
                        group="Vision-Language Models",
                        available=available,
                        unavailable_reason=None if available else f"'{provider_id}' is not available on this machine.",
                        default_enabled=False,
                    )
                )
                seen.add(provider_id)

        for provider_id, label, _default_model, runtime_kind in VISION_PROVIDER_CHOICES:
            if provider_id in seen:
                continue
            probe = _RUNTIME_AVAILABILITY_PROBES.get(runtime_kind)
            available = probe() if probe is not None else False
            choices.append(
                WizardProviderChoice(
                    provider_id=provider_id,
                    label=label,
                    description="",
                    group="Vision-Language Models",
                    available=available,
                    unavailable_reason=None if available else _RUNTIME_UNAVAILABLE_REASONS.get(
                        runtime_kind, f"The '{runtime_kind}' runtime is not available on this machine."
                    ),
                    default_enabled=False,
                )
            )
        return tuple(choices)

    def acquisition_manager_viewmodel(self) -> AcquisitionManagerViewModel:
        return AcquisitionManagerViewModel(
            manager=self.acquisition_manager,
            layout=self._current_layout,
            workspace_id=self.current_workspace.id if self.current_workspace else "",
        )

    def queue_worker(self):
        """A fresh `QueueWorker` (Operational Completion milestone) for the current Workspace's
        pending queue. Imported lazily so `composition.py` stays importable without PySide6."""
        from archivetrust.clients.desktop.queue_worker import QueueWorker

        return QueueWorker(self)

    def activate_configured_vision_provider(self) -> None:
        """Registers whichever VLM adapter each vision-provider binding names (Multi-Provider
        Activation milestone: `LogicalModelBinding.provider_id`, defaulting to Qwen2.5-VL for
        bindings persisted before that field existed), backed by the runtime bridge, for every
        binding that exists and isn't already registered. Idempotent and safe to call repeatedly —
        e.g. once at `AppContext` construction (covers a restart after First Launch already ran)
        and again right after the wizard completes (covers the first run, when construction
        happened before the operator had configured anything).

        Iterates both `PRIMARY_VISION_PROVIDER` and `SECONDARY_VISION_PROVIDER` (P4-P10
        Prequalification, 2026-07-20) rather than only the primary one -- the wizard only ever creates a
        Primary binding (Part 11: one choice, never expanded), but an admin/qualification caller
        may also have created a Secondary one via `bind_additional_vision_provider` to run two
        vision-capable providers (e.g. `paddleocr-vl` and `surya`) concurrently, which a single
        binding slot cannot express.
        """
        from archivetrust.presentation.first_launch_viewmodel import (
            PRIMARY_VISION_PROVIDER,
            SECONDARY_VISION_PROVIDER,
        )

        for logical_name in (PRIMARY_VISION_PROVIDER, SECONDARY_VISION_PROVIDER):
            self._activate_vision_binding(logical_name)

    def bind_additional_vision_provider(self, provider_id: str, *, logical_name: str | None = None) -> None:
        """Admin/automation-only entry point (P4-P10 Prequalification, 2026-07-20) that binds a
        second vision-capable provider under `SECONDARY_VISION_PROVIDER` and activates it. Never
        called by the First Launch wizard or any operator-facing view -- the wizard's one-choice
        design (Part 11) is unchanged; this exists for the qualification runner, which must run
        more than one vision-capable provider at once to satisfy a `required_providers` list naming
        more than one.
        """
        from archivetrust.presentation.first_launch_viewmodel import (
            SECONDARY_VISION_PROVIDER,
            default_model_id_for_provider,
            runtime_kind_for_provider,
        )

        resolved_logical_name = logical_name or SECONDARY_VISION_PROVIDER
        model_id = default_model_id_for_provider(provider_id)
        if model_id is None:
            raise ValueError(f"no default model id known for vision provider {provider_id!r}")
        descriptor = ModelDescriptor(
            model_id=model_id,
            display_name=model_id,
            runtime_kind=runtime_kind_for_provider(provider_id),
            source=ModelSource(kind=ModelSourceKind.HUGGING_FACE, identifier=model_id),
        )
        self.model_registry.bind(
            LogicalModelBinding(logical_name=resolved_logical_name, descriptor=descriptor, provider_id=provider_id)
        )
        # Unlike the wizard's ephemeral `--demo` bindings (see the self-heal comment in
        # `_activate_vision_binding`, which persists only if a bindings file already exists), this
        # is always an explicit admin/automation action -- it must persist unconditionally so the
        # separately-spawned `archivetrust-worker` process (which loads bindings from disk, not
        # from this in-memory `AppContext`) sees the same provider the qualification runner just
        # activated. Without this, `_ensure_vision_providers_bound` would silently only take effect
        # for whichever process called it, never for the spawned worker that does the real work.
        self.model_registry.save_bindings()
        self._activate_vision_binding(resolved_logical_name)

    def _activate_vision_binding(self, logical_name: str) -> None:
        """The per-binding registration logic `activate_configured_vision_provider` used to run
        inline for `PRIMARY_VISION_PROVIDER` only -- extracted, unchanged, so it can run once per
        vision-provider logical name instead."""
        from archivetrust.presentation.first_launch_viewmodel import (
            default_model_id_for_provider,
            runtime_kind_for_provider,
        )

        binding = next(
            (b for b in self.model_registry.bindings() if b.logical_name == logical_name),
            None,
        )
        if binding is None:
            return
        provider_id = binding.provider_id

        # `runtime_kind` and `model_id` are both properties of the *provider*, never a persisted
        # operator choice (Part 2/11) -- but both are still stored inside the persisted
        # `ModelDescriptor`. A binding saved before a provider's declared runtime changed (e.g.
        # PaddleOCR-VL moving from "transformers" to "vllm", Runtime Architecture Completion
        # milestone -- Root-cause finding, Production Investigation: PaddleOCR-VL invoking
        # Transformers instead of vLLM) or one whose `model_id`/`source` was left pointing at a
        # different provider's model than its own `provider_id` names (Root-cause finding,
        # Production Investigation: Settings UI showing Qwen / stale config, 2026-07-12 -- a
        # `provider_id="paddleocr-vl"` binding found with `model_id="Qwen/Qwen2.5-VL-7B-Instruct"`
        # still attached, a self-inconsistent pairing no operator would deliberately choose, never
        # a legitimate legacy binding to preserve) would otherwise keep silently running the wrong
        # model/runtime forever; `load_bindings()` loads whatever was persisted verbatim and nothing
        # ever re-derives it. Self-heal on every `open_workspace()` instead of trusting the file.
        # A binding whose `model_id` already matches its `provider_id` (including every genuine
        # legacy Qwen binding, where both correctly name Qwen) is left untouched -- this only
        # corrects a `provider_id`/`model_id` pairing that could not have been a deliberate choice.
        correct_runtime_kind = runtime_kind_for_provider(provider_id)
        correct_model_id = default_model_id_for_provider(provider_id)
        needs_runtime_fix = binding.descriptor.runtime_kind != correct_runtime_kind
        needs_model_fix = (
            correct_model_id is not None and binding.descriptor.model_id != correct_model_id
        )
        if needs_runtime_fix or needs_model_fix:
            descriptor_updates: dict[str, object] = {}
            if needs_runtime_fix:
                descriptor_updates["runtime_kind"] = correct_runtime_kind
            if needs_model_fix:
                descriptor_updates.update(
                    {
                        "model_id": correct_model_id,
                        "display_name": correct_model_id,
                        "source": ModelSource(
                            kind=ModelSourceKind.HUGGING_FACE, identifier=correct_model_id
                        ),
                    }
                )
            binding = binding.model_copy(
                update={"descriptor": binding.descriptor.model_copy(update=descriptor_updates)}
            )
            self.model_registry.bind(binding)
            # Persist the healed binding only when a persisted binding exists to heal (Provider
            # Execution Verification, 2026-07-13): this used to save unconditionally, so healing
            # an *in-memory-only* binding wrote it to disk -- which is exactly how `--demo`'s
            # deliberately-unpersisted Qwen binding (short model id, "corrected" by the model-fix
            # branch above) ended up durably bound to a real production Workspace, silently
            # switching its vision provider to Qwen for every later non-demo session.
            if self.model_registry.bindings_path().exists():
                self.model_registry.save_bindings()

        # `DeploymentProfile.STRUCTURED_PIPELINE` (Phase 31, PaddleOCR-VL's detect-then-crop-then-
        # recognize profile) was removed with the rest of the OCR-era provider architecture
        # (docs/htr-migration-plan.md Stage 5 -- EXECUTED); every registration is `SINGLE_PASS` now.
        deployment_profile = self.provider_config_store.get(provider_id).deployment_profile

        try:
            self.provider_registry.get(provider_id, deployment_profile)
            return  # already registered
        except Exception:  # noqa: BLE001 -- UnknownProviderError, the expected "not yet" case
            pass

        runtime = self.runtime_manager.get_or_create(logical_name)
        health = self.provider_health_trackers.setdefault(provider_id, ProviderHealthTracker(provider_id))

        factory = _VISION_PROVIDER_ADAPTER_FACTORIES.get(provider_id)
        if factory is None:
            return  # a binding naming a provider this composition root has no factory for --
            # never guessed at; Provider Manager will simply not show it until a factory is added
        adapter = factory(runtime, health, self.decoder_telemetry)

        self.provider_registry.register(adapter)
        self.provider_config_store.set(
            self.provider_config_store.get(provider_id).model_copy(update={"enabled": True})
        )
        # "available" here means the runtime that backs *this specific binding's* runtime_kind is
        # real (not one of the `_Unavailable*Runtime` placeholders) -- looked up by runtime_kind
        # (`_RUNTIME_AVAILABILITY_PROBES`), never assumed to be "transformers" regardless of what
        # the binding actually names (Runtime Architecture Completion milestone bug fix: a
        # PaddleOCR-VL/Surya binding on `runtime_kind="vllm"` was previously reported available or
        # unavailable based on whether `transformers` was installed, an unrelated fact).
        probe = _RUNTIME_AVAILABILITY_PROBES.get(binding.descriptor.runtime_kind)
        self.provider_availability[provider_id] = probe() if probe is not None else False


_VISION_PROVIDER_ADAPTER_FACTORIES: dict[
    str, Callable[[InferenceRuntime, ProviderHealthTracker, DecoderTelemetrySink], ProviderAdapter]
] = {}
"""Dispatch table, not a special-case chain -- every runtime-backed VLM adapter's construction has
the identical shape (a `RuntimeBackedLocalInferenceRunner` wrapped in that provider's own backend,
wrapped in that provider's own adapter, wrapped in an `Importer` carrying the shared
`decoder_telemetry` sink); this table is the one place that shape is instantiated per `provider_id`,
so `activate_configured_vision_provider` below never branches on which provider was configured.

Empty as of docs/htr-migration-plan.md Stage 5 (EXECUTED): the three vision-language adapters that
used to populate this table (Qwen2.5-VL, PaddleOCR-VL, Surya) are deleted. A future HTR method run
through this same `InferenceRuntime`-backed shape (if any -- SATRN/Florence-2/Transkribus each have
their own adapter shape per docs/htr-domain-design.md) would add one entry here."""


def _build_provider_registry(
    decoder_telemetry: DecoderTelemetrySink,
    runtime_telemetry: RuntimeTelemetrySink | None = None,
) -> tuple[ProviderRegistry, dict[str, bool]]:
    """Returns an empty `ProviderRegistry` (docs/htr-migration-plan.md Stage 5 -- EXECUTED): the
    five OCR-era providers this used to eagerly register (Docling, Tesseract+LayoutParser,
    Qwen2.5-VL, PaddleOCR-VL, Surya) are deleted. The first real registration returns with the
    SATRN `HtrMethodAdapter` implementation (Stage 6) -- this composition root's job until then is
    only to keep the rest of the Provider Manager/Settings/Wizard machinery honestly showing "no
    providers configured" rather than crashing or fabricating one.
    """
    del decoder_telemetry, runtime_telemetry  # unused until a real adapter is registered again
    return ProviderRegistry(), {}


def build_demo_context(deployment_root: Path | None = None) -> tuple[AppContext, list[DemoDocument]]:
    """An `AppContext` pre-seeded with a small realistic municipal archive, inside its own demo
    Workspace ("Lund kommun – Historiska kyrkoarkiv", ROADMAP.md §5.13.1). Returns the context and
    the seeded documents (the first is the one the Review Center opens — it has real uncertainties).

    docs/htr-migration-plan.md Stage 5 (EXECUTED): this used to also bind a demo "Primary Vision
    Provider" (Qwen2.5-VL) in memory and activate it, so the Provider Manager showed a real,
    runtime-backed probabilistic provider alongside the (also since-deleted) deterministic ones.
    Qwen2.5-VL is deleted along with the rest of the OCR-era provider architecture and nothing
    replaces it yet (SATRN/Florence-2/Transkribus are future phases), so demo mode no longer binds
    or activates any vision provider -- the Provider Manager honestly shows none configured, rather
    than a binding that silently activates nothing. Inference was never actually invoked for the
    seeded documents in the first place (their telemetry is emitted directly by `demo_data.py`,
    exercising the real Comparison/Confidence/Alignment engines without a real provider client) --
    but the demo Acquisition Sources below are real `ManualImportSource`/`FolderWatchSource`
    instances exercised through their genuine discover -> hash -> register path, not fabricated
    statistics.
    """
    import tempfile

    context = AppContext(
        reviewer_ref="reviewer.anna",
        deployment_root=deployment_root,
        auto_create_default_workspace=False,
        # Demo telemetry is seeded fresh on every launch; a durable stream would accrete the same
        # seeded events run after run. In-memory keeps --demo side-effect-free (see AppContext).
        persistent_telemetry=False,
    )
    if context.current_workspace is None:
        context.create_workspace(
            "Lund kommun – Historiska kyrkoarkiv",
            description="Kommunfullmäktige, byggnadsnämnden och kommunstyrelsens protokoll- och beslutsarkiv.",
        )
    documents = seed_realistic_archive(context.telemetry, context.interaction_sink)  # type: ignore[arg-type]
    _seed_demo_acquisition_sources(context)
    return context, documents


def _seed_demo_acquisition_sources(context: AppContext) -> None:
    """Adds one real `ManualImportSource` and one real `FolderWatchSource` to the demo Workspace's
    Acquisition Manager, and exercises each through its genuine discover/import path against a
    couple of small, real (`.html`, not a fabricated PDF) scanned-record stand-ins — so the
    Acquisition Manager UI shows real, non-fabricated health statistics. (Stage 5 -- EXECUTED: with
    no OCR-era provider left registered, pressing Process Queue against them in demo mode no longer
    has a real recognition path to run; the Acquisition-side discover/hash/register behavior this
    helper exercises is unaffected.)
    """
    import tempfile
    from pathlib import Path as _Path

    manager = context.acquisition_manager

    manual = ManualImportSource(source_id="manual-import")
    manager.add_source(manual)
    demo_manual_dir = _Path(tempfile.mkdtemp(prefix="archivetrust_demo_manual_"))
    manual_file = demo_manual_dir / "kf_protokoll_2019_03_14_skannad.html"
    manual_file.write_text(
        "<html><body><h1>Kommunfullmäktige — Sammanträdesprotokoll 2019-03-14</h1>"
        "<p>Ärendet föredrogs av ekonomichefen.</p></body></html>",
        encoding="utf-8",
    )
    manager.import_files(manual, (manual_file,))

    demo_watch_dir = _Path(tempfile.mkdtemp(prefix="archivetrust_demo_watch_"))
    folder_watch = FolderWatchSource(
        FolderWatchConfig(path=demo_watch_dir, stability_check_interval_seconds=0.05),
        source_id="folder-watch-inkorg",
    )
    manager.add_source(folder_watch)
    (demo_watch_dir / "bn_beslut_47_skannad.html").write_text(
        "<html><body><h1>Byggnadsnämnden — Beslut § 47</h1>"
        "<p>Nämnden beviljar bygglov för nybyggnad av flerbostadshus.</p></body></html>",
        encoding="utf-8",
    )
    import time as _time

    _time.sleep(0.2)
    manager.run_scan_cycle()
