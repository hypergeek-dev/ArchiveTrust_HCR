"""Evidence-chain ViewModel (Stage 11, brief's "User interface" -> evidence view).

Walks `docs/htr-domain-design.md` §4's traceability chain and returns it as an ordered sequence of
labeled links a View renders as breadcrumbs:

    CanonicalResult -> reviewed -> normalized -> parsed -> raw -> MethodRun -> ModelVersion
        -> InputCrop -> segmentation result (TextLine -> Region) -> Page -> Document
        -> Collection -> Dataset(+Version) -> ResearchProject

**Distinct from `evidence_explorer_viewmodel.py`.** That one reconstructs the *operational*
Evidence->Observation->Comparison->Canonical chain by replaying telemetry for a document. This one
walks the *HTR research* chain over `htr/research_store.py`'s corpus/experiment entities, which
telemetry replay does not currently carry (see that store's module docstring). They answer
different questions about different entities; neither replaces the other.

**A broken link is shown, not hidden.** §4's chain is "a stored id reference, never an inline
copy", so a hop can genuinely dangle -- an id whose target this store has never seen. Such a link
is emitted with `resolved=False` and the id it could not resolve, and the walk stops there. It is
never silently truncated into a shorter-looking but apparently-complete chain.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.presentation.display_names import (
    canonicalization_strategy_label,
    document_label,
    method_label,
    result_stage_label,
    short_ref,
)


class ChainLink(BaseModel):
    """One hop of the traceability chain."""

    model_config = ConfigDict(frozen=True)

    kind: str
    """The entity kind this hop names -- `"canonical_result"`, `"result_stage"`, `"method_run"`,
    `"model_version"`, `"input_crop"`, `"text_line"`, `"region"`, `"page"`, `"document"`,
    `"collection"`, `"dataset_version"`, `"dataset"`, `"project"`."""
    entity_id: str
    label: str
    detail: str = ""
    resolved: bool = True
    """`False` when `entity_id` was referenced by the previous hop but is not registered in this
    store -- a real dangling reference, surfaced rather than dropped."""


class EvidenceChain(BaseModel):
    """The full walk for one starting point."""

    model_config = ConfigDict(frozen=True)

    origin_kind: str
    origin_id: str
    links: tuple[ChainLink, ...]
    complete: bool
    """`True` only when every hop resolved all the way to a `ResearchProject`. `False` when the
    walk stopped early -- either at a dangling reference or because the corpus genuinely does not
    reach that far up yet."""
    broken_at: str | None = None
    """The id of the first unresolvable hop, when `complete` is `False` because of one."""


class HtrEvidenceChainViewModel:
    """Builds `EvidenceChain`s over an `HtrResearchStore`."""

    def __init__(self, store: HtrResearchStore) -> None:
        self._store = store

    def chain_for_method_run(self, method_run_id: str) -> EvidenceChain:
        """The chain from one `MethodRun` upward to its `ResearchProject`, including the result
        stages it produced. Does not include a `CanonicalResult` hop -- canonical selection is
        page-scoped and may not have chosen this run; use `chain_for_canonical_span` for that."""
        links: list[ChainLink] = []
        run = self._store.method_run(method_run_id)
        if run is None:
            return EvidenceChain(
                origin_kind="method_run",
                origin_id=method_run_id,
                links=(
                    ChainLink(
                        kind="method_run",
                        entity_id=method_run_id,
                        label=short_ref(method_run_id),
                        detail="Not registered in this store",
                        resolved=False,
                    ),
                ),
                complete=False,
                broken_at=method_run_id,
            )

        links.extend(self._stage_links(method_run_id))
        links.append(
            ChainLink(
                kind="method_run",
                entity_id=run.method_run_id,
                label=method_label(run.method_id),
                detail=(
                    f"{run.outcome}; started {run.started_at}"
                    + (f", completed {run.completed_at}" if run.completed_at else "")
                ),
            )
        )
        links.append(
            ChainLink(
                kind="model_version",
                entity_id=run.model_version_id or "",
                label=run.model_version_id or "Model version not recorded",
                detail=f"Evidence {short_ref(run.evidence_id, prefix=18)}",
                resolved=run.model_version_id is not None,
            )
        )

        upward, complete, broken = self._corpus_links_from_crop(run.input_crop_id)
        links.extend(upward)
        return EvidenceChain(
            origin_kind="method_run",
            origin_id=method_run_id,
            links=tuple(links),
            complete=complete,
            broken_at=broken,
        )

    def chain_for_canonical_span(self, canonical_result_id: str, text_line_id: str) -> EvidenceChain:
        """The full §4 chain, starting at the `CanonicalResult` span that selected one line."""
        links: list[ChainLink] = []
        canonical = next(
            (
                result
                for result in self._store.canonical_results()
                if result.canonical_result_id == canonical_result_id
            ),
            None,
        )
        if canonical is None:
            return EvidenceChain(
                origin_kind="canonical_result",
                origin_id=canonical_result_id,
                links=(
                    ChainLink(
                        kind="canonical_result",
                        entity_id=canonical_result_id,
                        label=short_ref(canonical_result_id),
                        detail="Not registered in this store",
                        resolved=False,
                    ),
                ),
                complete=False,
                broken_at=canonical_result_id,
            )

        span = next((s for s in canonical.spans if s.text_line_id == text_line_id), None)
        links.append(
            ChainLink(
                kind="canonical_result",
                entity_id=canonical.canonical_result_id,
                label=canonicalization_strategy_label(canonical.strategy),
                detail=(
                    f"strategy version {canonical.strategy_version}; "
                    f"{len(canonical.spans)} span(s); created {canonical.created_at}"
                ),
            )
        )
        if span is None:
            return EvidenceChain(
                origin_kind="canonical_result",
                origin_id=canonical_result_id,
                links=tuple(links),
                complete=False,
                broken_at=text_line_id,
            )

        downstream = self.chain_for_method_run(span.source_method_run_id)
        links.extend(downstream.links)
        return EvidenceChain(
            origin_kind="canonical_result",
            origin_id=canonical_result_id,
            links=tuple(links),
            complete=downstream.complete,
            broken_at=downstream.broken_at,
        )

    # -- Internals -----------------------------------------------------------------------------

    def _stage_links(self, method_run_id: str) -> list[ChainLink]:
        """The reviewed -> normalized -> parsed -> raw hops. Only stages that actually exist are
        emitted; a missing stage is not a dangling reference, it is simply a stage this run never
        produced, so it is omitted rather than shown as broken."""
        transcript = self._store.transcript(method_run_id)
        if transcript is None:
            return []
        links: list[ChainLink] = []
        for stage, text in (
            ("reviewed", transcript.reviewed_text),
            ("normalized", transcript.normalized_text),
            ("parsed", transcript.parsed_text),
            ("raw", transcript.raw_text),
        ):
            if text is None:
                continue
            attribution = transcript.reviewer_ref if stage == "reviewed" else None
            links.append(
                ChainLink(
                    kind="result_stage",
                    entity_id=f"{method_run_id}:{stage}",
                    label=result_stage_label(stage),
                    detail=text if attribution is None else f"{text} - {attribution}",
                )
            )
        return links

    def _corpus_links_from_crop(
        self, input_crop_id: str | None
    ) -> tuple[list[ChainLink], bool, str | None]:
        links: list[ChainLink] = []
        if input_crop_id is None:
            # An end-to-end run legitimately has no shared crop (`MethodRun.input_crop_id`'s own
            # docstring). That is a documented mode, not a broken chain -- but the walk genuinely
            # cannot continue upward from here, so `complete` is False with no `broken_at`.
            links.append(
                ChainLink(
                    kind="input_crop",
                    entity_id="",
                    label="No shared input crop",
                    detail="End-to-end run: this method used its own segmentation",
                    resolved=False,
                )
            )
            return links, False, None

        crop = self._store.input_crop(input_crop_id)
        if crop is None:
            links.append(self._broken("input_crop", input_crop_id))
            return links, False, input_crop_id
        links.append(
            ChainLink(
                kind="input_crop",
                entity_id=crop.crop_id,
                label=short_ref(crop.hash, prefix=18),
                detail=f"{crop.byte_size} bytes at {crop.storage_path}",
            )
        )

        line = self._store.text_line(crop.text_line_id)
        if line is None:
            links.append(self._broken("text_line", crop.text_line_id))
            return links, False, crop.text_line_id
        links.append(
            ChainLink(
                kind="text_line",
                entity_id=line.text_line_id,
                label=f"Line {line.reading_order_index}",
                detail="Segmentation result",
            )
        )

        region = self._store.region(line.region_id)
        if region is None:
            links.append(self._broken("region", line.region_id))
            return links, False, line.region_id
        links.append(
            ChainLink(
                kind="region",
                entity_id=region.region_id,
                label=region.region_type or "Region",
                detail="Segmentation result",
            )
        )

        page = self._store.page(region.page_id)
        if page is None:
            links.append(self._broken("page", region.page_id))
            return links, False, region.page_id
        links.append(
            ChainLink(
                kind="page",
                entity_id=page.page_id,
                label=f"Page {page.page_number}",
                detail=page.archive_object_ref,
            )
        )

        links.append(
            ChainLink(
                kind="document",
                entity_id=page.archive_object_ref,
                label=document_label(page.archive_object_ref),
                detail="Archive object",
            )
        )

        collection = next(
            (
                c
                for c in self._store.collections()
                if page.archive_object_ref in c.archive_object_refs
            ),
            None,
        )
        if collection is None:
            links.append(self._broken("collection", page.archive_object_ref))
            return links, False, page.archive_object_ref
        links.append(
            ChainLink(
                kind="collection",
                entity_id=collection.collection_id,
                label=collection.name,
                detail=f"{len(collection.archive_object_refs)} document(s)",
            )
        )

        dataset = self._store.dataset(collection.dataset_id)
        if dataset is None:
            links.append(self._broken("dataset", collection.dataset_id))
            return links, False, collection.dataset_id

        for version in self._store.dataset_versions(dataset_id=dataset.dataset_id):
            if collection.collection_id in version.collection_ids:
                links.append(
                    ChainLink(
                        kind="dataset_version",
                        entity_id=version.dataset_version_id,
                        label=f"{dataset.name} v{version.version}",
                        detail=f"Snapshotted {version.created_at}",
                    )
                )
        links.append(
            ChainLink(
                kind="dataset",
                entity_id=dataset.dataset_id,
                label=dataset.name,
                detail=dataset.description or "No description recorded",
            )
        )

        project = self._store.project(dataset.project_id)
        if project is None:
            links.append(self._broken("project", dataset.project_id))
            return links, False, dataset.project_id
        links.append(
            ChainLink(
                kind="project",
                entity_id=project.project_id,
                label=project.name,
                detail=project.description or "No description recorded",
            )
        )
        return links, True, None

    @staticmethod
    def _broken(kind: str, entity_id: str) -> ChainLink:
        return ChainLink(
            kind=kind,
            entity_id=entity_id,
            label=short_ref(entity_id),
            detail="Referenced but not registered in this store",
            resolved=False,
        )
