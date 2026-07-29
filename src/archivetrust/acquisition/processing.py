"""`WorkspaceProcessingService` (ROADMAP.md §5.13.2, decision 6) — drains a Workspace's pending,
registered-but-unprocessed Archive Objects and calls the **unchanged**
`application.pipeline.run_pipeline` for each, using whichever provider adapters are enabled for the
Workspace. This is the one seam through which Acquisition hands off to the Trust Engine — it never
constructs Evidence/Observations itself, and it stops being Acquisition's concern the moment
`run_pipeline` is called (Article 25).

**Per-adapter source preparation (Multi-Provider Activation milestone).** Every adapter used to
receive the same whole-document path — correct for `ProviderInputKind.DOCUMENT` adapters (Docling,
which does its own multi-page/multi-format handling) but wrong for `ProviderInputKind.PAGE_IMAGE`
adapters (Tesseract+LayoutParser, Qwen2.5-VL), whose own contracts have always specified a
rendered-page-image input (`ProviderAdapter.observe`'s docstring; `QwenVLAdapter`'s page-image
convention). This service now renders one invocation per page for those adapters, via the shared,
Qt-free `PageRenderCache` — pages become reproducible derived artifacts under the Workspace's own
`derived/` directory, never a second source of truth, always reconstructible from the immutable
Archive Object. `run_pipeline` itself is unchanged.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from archivetrust.acquisition.archive_object import ArchiveObject
from archivetrust.application.pipeline import AdapterInvocation, PipelineRunResult, run_pipeline
from archivetrust.domain.comparison.capability_matrix import CapabilityMatrix
from archivetrust.domain.comparison.policy import ReconciliationPolicy
from archivetrust.domain.confidence.policy import ConfidencePolicy
from archivetrust.domain.shared.ids import content_address, new_id
from archivetrust.domain.shared.versioning import CURRENT_ONTOLOGY_VERSION
from archivetrust.domain.telemetry.events import (
    ProvenanceContextEstablished,
    ProviderFailureCategory,
    ProviderInvocationOutcome,
    ProviderObservationAttempted,
    TelemetryEvent,
    stamp_recorded_at,
)
from archivetrust.domain.telemetry.sink import TelemetrySink
from archivetrust.infrastructure.rendering import PageRenderCache, PdfRenderError, pdf_page_count
from archivetrust.infrastructure.rendering.pdf_renderer import DEFAULT_SCALE
from archivetrust.providers.base import ProviderAdapter, ProviderInputKind
from archivetrust.workspace.layout import WorkspaceLayout
from archivetrust.security.input_policy import InputSecurityPolicy

_PDF_MIME_TYPE = "application/pdf"
_REPO_ROOT = Path(__file__).resolve().parents[3]


class WorkspaceProcessingService:
    """`adapters` is the set of provider adapters enabled for this Workspace's Processing Profile
    (`ProviderRegistry.all()` filtered by `ProviderConfigurationStore`, composed one level up in
    `composition.py`) — this service has no opinion about which providers exist, only how to
    invoke whichever it is given, exactly as `run_pipeline` already requires.

    `layout` resolves an `ArchiveObject.storage_path` (relative, as registration recorded it) to
    the real absolute path under `archive/` a real provider client can actually open (Operational
    Completion milestone — the path handed to `AdapterInvocation` used to be the bare relative
    `storage_path`, unresolvable, which never surfaced while every real client was a `client=None`
    placeholder that ignored `source` entirely).
    """

    def __init__(
        self,
        *,
        adapters: tuple[ProviderAdapter, ...],
        reconciliation_policy: ReconciliationPolicy,
        capability_matrix: CapabilityMatrix,
        confidence_policy: ConfidencePolicy,
        telemetry_sink: TelemetrySink,
        layout: WorkspaceLayout,
        workspace_id: str,
        security_policy: InputSecurityPolicy | None = None,
    ) -> None:
        self._adapters = adapters
        self._reconciliation_policy = reconciliation_policy
        self._capability_matrix = capability_matrix
        self._confidence_policy = confidence_policy
        self._telemetry_sink = telemetry_sink
        self._layout = layout
        self._workspace_id = workspace_id
        self._security_policy = security_policy
        """The owning Workspace's id (ROADMAP.md §5.13.1) — carried only as an opaque foreign key
        for `ProvenanceContextEstablished.workspace_identifier`
        (`ARCHITECTURE_TELEMETRY_STANDARD.md` S8.3), the same reference-without-dependency pattern
        `DatasetCandidateCreated.archive_object_ref` already uses; this service has no other use
        for it and depends on no Workspace/Acquisition concept because of it (Article 25)."""
        self._render_cache = PageRenderCache(layout.derived_dir)
        self._warm_providers: set[str] = set()
        """Which `provider_id`s this service instance has already invoked (Observability
        milestone, 2026-07-13) — lives as long as this service does (in practice, the app
        process's lifetime), so `run_pipeline`'s `cold_start` reflects a real "first execution
        since process startup" fact across every document this service processes, not just
        within one document."""

    def _prepare_invocations(
        self, archive_object: ArchiveObject
    ) -> tuple[tuple[AdapterInvocation, ...], tuple[TelemetryEvent, ...]]:
        """Builds one `AdapterInvocation` per adapter honoring each adapter's own `input_kind`
        (Multi-Provider Activation milestone): the document path for `DOCUMENT` adapters; one
        rendered-page invocation per page for `PAGE_IMAGE` adapters, when the Archive Object is a
        PDF. A non-PDF Archive Object falls back to handing `PAGE_IMAGE` adapters the raw file
        directly — correct as-is for an already-image Archive Object, and the same honest
        "invoked, found nothing, client recorded why" behavior as before this milestone for any
        other format, never a silent skip.

        Returns the invocations plus any *render-failure* telemetry events — attributed here as
        the `PAGE_IMAGE` adapter's own `ProviderObservationAttempted(outcome=FAILED)`, with a
        `failure_reason` naming the real cause ("page rendering failed: …"), never a generic
        provider failure that hides where the pipeline actually stopped (Article 18).
        """
        document_path = self._layout.archive_dir / archive_object.storage_path
        is_pdf = archive_object.mime_type == _PDF_MIME_TYPE

        invocations: list[AdapterInvocation] = []
        failure_events: list[TelemetryEvent] = []
        page_count: int | None = None
        page_count_error: str | None = None

        for adapter in self._adapters:
            if adapter.input_kind == ProviderInputKind.DOCUMENT or not is_pdf:
                invocations.append(
                    AdapterInvocation(
                        adapter=adapter, source=str(document_path), invocation_id=new_id("invocation")
                    )
                )
                continue

            if page_count is None and page_count_error is None:
                try:
                    page_count = pdf_page_count(document_path)
                    if self._security_policy is not None and page_count > self._security_policy.max_pdf_pages:
                        raise PdfRenderError(
                            f"PDF has {page_count} pages; policy limit is {self._security_policy.max_pdf_pages}"
                        )
                except PdfRenderError as exc:
                    page_count_error = str(exc)

            if page_count_error is not None:
                failure_events.append(self._render_failure_event(archive_object, adapter, page_count_error))
                continue

            for page_no in range(1, (page_count or 0) + 1):
                try:
                    artifact = self._render_cache.get_or_render(
                        content_hash=archive_object.content_hash,
                        pdf_path=document_path,
                        page_no=page_no,
                        scale=DEFAULT_SCALE,
                    )
                    if self._security_policy is not None and max(
                        artifact.page_width_pt, artifact.page_height_pt
                    ) > self._security_policy.max_page_dimension_points:
                        raise PdfRenderError(
                            f"page dimension exceeds {self._security_policy.max_page_dimension_points:g} points"
                        )
                except PdfRenderError as exc:
                    failure_events.append(
                        self._render_failure_event(archive_object, adapter, str(exc), page_no=page_no)
                    )
                    continue
                invocations.append(
                    AdapterInvocation(
                        adapter=adapter, source=str(artifact.path), invocation_id=new_id("invocation"),
                        pages_processed=1,
                    )
                )

        return tuple(invocations), tuple(failure_events)

    @staticmethod
    def _render_failure_event(
        archive_object: ArchiveObject, adapter: ProviderAdapter, reason: str, *, page_no: int | None = None
    ) -> ProviderObservationAttempted:
        target_region = f"page-{page_no}" if page_no is not None else None
        provider_version = getattr(adapter, "_provider_version", "unknown")
        return ProviderObservationAttempted(
            event_id=new_id("event"),
            document_ref=archive_object.id,
            provider_id=adapter.provider_id,
            provider_version=provider_version,
            invocation_id=new_id("invocation"),
            target_region=target_region,
            outcome=ProviderInvocationOutcome.FAILED,
            observation_count=0,
            failure_reason=f"page rendering failed: {reason}",
            # No adapter was ever invoked here (rendering failed first) -- duration/retry/cold-start
            # genuinely do not apply, unlike an invocation that ran and then failed.
            duration_ms=None,
            retry_count=0,
            cold_start=None,
            timeout=False,
            failure_category=ProviderFailureCategory.EXCEPTION,
            pages_processed=1 if page_no is not None else None,
        )

    def _current_git_commit(self) -> str | None:
        """Best-effort; `None` when genuinely unresolvable (e.g. a packaged install with no `.git`
        directory) — never guessed (Article 18's silence-vs-failure discipline, applied to this
        fact; `ARCHITECTURE_TELEMETRY_STANDARD.md` S8.3)."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _configuration_snapshot(self) -> tuple[str, str]:
        """Returns `(configuration_hash, configuration_snapshot_reference)`
        (`ARCHITECTURE_TELEMETRY_STANDARD.md` S8.3): the enabled-adapter roster and invocation
        order — the one part of "effective configuration" not already covered by
        `reconciliation_policy_version`/`capability_matrix_version`/`confidence_policy_version`,
        so no separate `pipeline_definition_hash` field is needed (S8.3's rejected-field list).
        Snapshot content is inline (no external blob store exists elsewhere in this codebase,
        matching `Evidence.raw_output`'s own inline-storage precedent)."""
        locked_hash: str | None = None
        lock_file = self._layout.config_dir / "configuration.lock.json"
        if lock_file.exists():
            try:
                locked_hash = json.loads(lock_file.read_text(encoding="utf-8")).get("configuration_hash")
            except (OSError, json.JSONDecodeError):
                locked_hash = None
        snapshot = json.dumps(
            {
                "configuration_lock_hash": locked_hash,
                "enabled_adapters": [
                {
                    "provider_id": adapter.provider_id,
                    "provider_version": getattr(adapter, "_provider_version", "unknown"),
                }
                for adapter in self._adapters
                ],
            },
            sort_keys=True,
        )
        return content_address(snapshot, prefix="configuration"), snapshot

    def _establish_provenance_context(self, archive_object: ArchiveObject) -> None:
        """Computes this run's `ProvenanceContextEstablished` event and appends it only if no
        event with the same `context_id` already exists for this `document_ref`
        (`ARCHITECTURE_TELEMETRY_STANDARD.md` S8.4's deduplication rule) — prepended before any
        other event for this run (S8.5). Never reached by replay (S8.7): this method, and this
        method alone, is the one call site that creates this event kind."""
        configuration_hash, configuration_snapshot = self._configuration_snapshot()
        context = ProvenanceContextEstablished.create(
            document_ref=archive_object.id,
            ontology_version=CURRENT_ONTOLOGY_VERSION,
            git_commit=self._current_git_commit(),
            configuration_hash=configuration_hash,
            configuration_snapshot_reference=configuration_snapshot,
            machine_identifier=platform.node() or None,
            workspace_identifier=self._workspace_id,
            reconciliation_policy_version=self._reconciliation_policy.policy_version,
            capability_matrix_version=self._capability_matrix.matrix_version,
            confidence_policy_version=self._confidence_policy.confidence_policy_version,
        )
        existing_context_ids = {
            event.context_id
            for event in self._telemetry_sink.events_for_document(archive_object.id)
            if isinstance(event, ProvenanceContextEstablished)
        }
        if context.context_id not in existing_context_ids:
            self._telemetry_sink.append(stamp_recorded_at(context))

    def process(self, archive_object: ArchiveObject) -> PipelineRunResult:
        self._establish_provenance_context(archive_object)
        invocations, render_failure_events = self._prepare_invocations(archive_object)
        if not invocations and render_failure_events:
            # Every adapter was a PAGE_IMAGE adapter and every page failed to render -- a real,
            # recordable outcome, not an exception: the render-failure attempts above are this
            # run's complete honest record (Article 18), same shape as run_pipeline's own "no
            # provider produced any Observation" case. Distinct from `self._adapters` being empty
            # to begin with (no provider enabled/available at all) -- that case has no attempts to
            # report and must still raise, exactly as before this milestone, so it keeps surfacing
            # as a per-document processing failure (`AppContext.queue_failure_count`'s documented
            # purpose), not a silent no-op.
            result = PipelineRunResult(
                reconciled_graph=None, canonical_document=None, events=render_failure_events
            )
        else:
            result = run_pipeline(
                document_ref=archive_object.id,
                archive_object_ref=archive_object.id,
                invocations=invocations,
                reconciliation_policy=self._reconciliation_policy,
                capability_matrix=self._capability_matrix,
                confidence_policy=self._confidence_policy,
                warm_providers=self._warm_providers,
            )
            if render_failure_events:
                result = result.model_copy(update={"events": render_failure_events + result.events})
        for event in result.events:
            self._telemetry_sink.append(stamp_recorded_at(event))
        return result

    def process_pending(self, pending: list[ArchiveObject]) -> tuple[PipelineRunResult, ...]:
        """Processes and removes every currently-pending Archive Object, in order. A processing
        failure for one Archive Object (e.g. no adapters configured) propagates rather than being
        silently swallowed — never a fake success (Constitution Article 18's discipline)."""
        results = []
        while pending:
            archive_object = pending.pop(0)
            results.append(self.process(archive_object))
        return tuple(results)
