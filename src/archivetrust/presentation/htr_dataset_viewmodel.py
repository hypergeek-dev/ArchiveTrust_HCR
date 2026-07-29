"""Dataset explorer ViewModel (Stage 11, brief's "User interface" -> dataset explorer).

A read-only navigation over `htr/corpus/models.py`'s entity chain, as held by
`htr/research_store.py`:

    ResearchProject -> Dataset -> DatasetVersion -> Collection -> Document(ArchiveObject)
        -> Page -> Region -> TextLine -> InputCrop

Framework-independent and strictly read-only: this ViewModel has no mutating method at all, so a
Qt tree bound to it structurally cannot edit the corpus. Every node carries the id it projects, so
a View can hand that id straight to the evidence-chain ViewModel without re-deriving anything.

**"Document" is `ArchiveObject`, by reference.** `htr/corpus/models.py`'s own module docstring
explains there is deliberately no `Document` class -- `Page.archive_object_ref` is the document
identity. The document level of this tree is therefore synthesized by grouping pages on that ref,
not read from a `Document` table that does not exist.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.presentation.display_names import document_label, method_label, short_ref


class CorpusNode(BaseModel):
    """One node in the read-only corpus tree. `children` is populated lazily by the ViewModel's
    per-level methods rather than eagerly for the whole corpus -- a dataset version can span
    thousands of lines, and a tree View expands one level at a time."""

    model_config = ConfigDict(frozen=True)

    kind: str
    """`"project" | "dataset" | "dataset_version" | "collection" | "document" | "page" | "region"
    | "text_line" | "input_crop"`."""
    node_id: str
    label: str
    detail: str = ""
    child_count: int | None = None
    """`None` when the child count is not knowable without loading children; an integer otherwise.
    Never `0` as a stand-in for "unknown"."""


class LineDetail(BaseModel):
    """Everything the explorer shows for one selected `TextLine` -- including whether it has ground
    truth, which methods produced output for it, and its review status."""

    model_config = ConfigDict(frozen=True)

    text_line_id: str
    region_id: str
    page_id: str
    reading_order_index: int
    crop_ids: tuple[str, ...]
    crop_hashes: tuple[str, ...]
    """Content addresses of this line's crops. In a controlled comparison every method run for this
    line consumes the same hash (`docs/htr-domain-design.md` §7); more than one distinct hash here
    means the runs were NOT byte-identical-input comparable."""
    shared_crop_verified: bool | None = None
    """`True` when every crop for this line shares one hash, `False` when they differ, `None` when
    the line has no crop at all (not yet segmented) -- three distinguishable states."""
    ground_truth: str | None = None
    method_outputs: tuple[str, ...] = ()
    """Labels of the methods that produced a run for this line's crops."""
    review_status: str = "Not reviewed"


class DatasetExplorerViewModel:
    """Read-only projection of the corpus held in an `HtrResearchStore`."""

    def __init__(
        self,
        store: HtrResearchStore,
        *,
        archive_objects: dict[str, object] | None = None,
        review_status_by_line: dict[str, str] | None = None,
    ) -> None:
        self._store = store
        self._archive_objects = archive_objects or {}
        """`archive_object_ref -> ArchiveObject`, used only for human-readable document labels.
        Absent refs fall back to a short id -- a missing acquisition record never blanks a row."""
        self._review_status_by_line = review_status_by_line or {}

    # -- Tree levels ---------------------------------------------------------------------------

    def projects(self) -> tuple[CorpusNode, ...]:
        return tuple(
            CorpusNode(
                kind="project",
                node_id=project.project_id,
                label=project.name,
                detail=project.description or "No description recorded",
                child_count=len(self._store.datasets(project_id=project.project_id)),
            )
            for project in self._store.projects()
        )

    def datasets(self, project_id: str) -> tuple[CorpusNode, ...]:
        return tuple(
            CorpusNode(
                kind="dataset",
                node_id=dataset.dataset_id,
                label=dataset.name,
                detail=dataset.description or "No description recorded",
                child_count=len(self._store.dataset_versions(dataset_id=dataset.dataset_id)),
            )
            for dataset in self._store.datasets(project_id=project_id)
        )

    def dataset_versions(self, dataset_id: str) -> tuple[CorpusNode, ...]:
        return tuple(
            CorpusNode(
                kind="dataset_version",
                node_id=version.dataset_version_id,
                label=f"Version {version.version}",
                detail=(
                    f"{len(version.collection_ids)} collection(s), snapshotted {version.created_at}"
                    + (f", supersedes {short_ref(version.supersedes)}" if version.supersedes else "")
                ),
                child_count=len(version.collection_ids),
            )
            for version in self._store.dataset_versions(dataset_id=dataset_id)
        )

    def collections(self, dataset_version_id: str) -> tuple[CorpusNode, ...]:
        version = self._store.dataset_version(dataset_version_id)
        if version is None:
            return ()
        nodes: list[CorpusNode] = []
        for collection_id in version.collection_ids:
            collection = self._store.collection(collection_id)
            if collection is None:
                # A snapshot naming a collection this store has never seen is a real, reportable
                # gap -- shown as such, not silently dropped from the tree.
                nodes.append(
                    CorpusNode(
                        kind="collection",
                        node_id=collection_id,
                        label=short_ref(collection_id),
                        detail="Referenced by this dataset version but not registered in this store",
                    )
                )
                continue
            nodes.append(
                CorpusNode(
                    kind="collection",
                    node_id=collection.collection_id,
                    label=collection.name,
                    detail=f"{len(collection.archive_object_refs)} document(s)",
                    child_count=len(collection.archive_object_refs),
                )
            )
        return tuple(nodes)

    def documents(self, collection_id: str) -> tuple[CorpusNode, ...]:
        collection = self._store.collection(collection_id)
        if collection is None:
            return ()
        nodes: list[CorpusNode] = []
        for ref in collection.archive_object_refs:
            pages = self._store.pages(archive_object_ref=ref)
            nodes.append(
                CorpusNode(
                    kind="document",
                    node_id=ref,
                    label=document_label(ref, archive_object=self._archive_objects.get(ref)),
                    detail=f"{len(pages)} page(s) registered",
                    child_count=len(pages),
                )
            )
        return tuple(nodes)

    def pages(self, archive_object_ref: str) -> tuple[CorpusNode, ...]:
        nodes: list[CorpusNode] = []
        for page in self._store.pages(archive_object_ref=archive_object_ref):
            regions = self._store.regions(page_id=page.page_id)
            dimensions = (
                f"{page.width}x{page.height}"
                if page.width is not None and page.height is not None
                else "dimensions not recorded"
            )
            nodes.append(
                CorpusNode(
                    kind="page",
                    node_id=page.page_id,
                    label=f"Page {page.page_number}",
                    detail=f"{dimensions}; {len(regions)} region(s)",
                    child_count=len(regions),
                )
            )
        return tuple(nodes)

    def regions(self, page_id: str) -> tuple[CorpusNode, ...]:
        nodes: list[CorpusNode] = []
        for region in self._store.regions(page_id=page_id):
            lines = self._store.text_lines(region_id=region.region_id)
            nodes.append(
                CorpusNode(
                    kind="region",
                    node_id=region.region_id,
                    label=region.region_type or "Region",
                    detail=f"{len(lines)} line(s)",
                    child_count=len(lines),
                )
            )
        return tuple(nodes)

    def text_lines(self, region_id: str) -> tuple[CorpusNode, ...]:
        nodes: list[CorpusNode] = []
        for line in self._store.text_lines(region_id=region_id):
            crops = self._store.crops_for_line(line.text_line_id)
            ground_truth = self._store.ground_truth_for_line(line.text_line_id)
            nodes.append(
                CorpusNode(
                    kind="text_line",
                    node_id=line.text_line_id,
                    label=f"Line {line.reading_order_index}",
                    detail=(
                        f"{len(crops)} crop(s); "
                        + ("ground truth present" if ground_truth else "no ground truth")
                    ),
                    child_count=len(crops),
                )
            )
        return tuple(nodes)

    def input_crops(self, text_line_id: str) -> tuple[CorpusNode, ...]:
        return tuple(
            CorpusNode(
                kind="input_crop",
                node_id=crop.crop_id,
                label=short_ref(crop.hash, prefix=18),
                detail=(
                    f"{crop.byte_size} bytes"
                    + (
                        f", {crop.width}x{crop.height}"
                        if crop.width is not None and crop.height is not None
                        else ""
                    )
                ),
                child_count=0,
            )
            for crop in self._store.crops_for_line(text_line_id)
        )

    # -- Selection detail ----------------------------------------------------------------------

    def line_detail(self, text_line_id: str) -> LineDetail | None:
        line = self._store.text_line(text_line_id)
        if line is None:
            return None
        region = self._store.region(line.region_id)
        crops = self._store.crops_for_line(text_line_id)
        hashes = tuple(sorted({crop.hash for crop in crops}))

        method_labels: list[str] = []
        for crop in crops:
            for run in self._store.method_runs_for_crop(crop.crop_id):
                label = method_label(run.method_id)
                if label not in method_labels:
                    method_labels.append(label)

        return LineDetail(
            text_line_id=line.text_line_id,
            region_id=line.region_id,
            page_id=region.page_id if region is not None else "",
            reading_order_index=line.reading_order_index,
            crop_ids=tuple(crop.crop_id for crop in crops),
            crop_hashes=hashes,
            shared_crop_verified=None if not crops else len(hashes) == 1,
            ground_truth=self._store.ground_truth_for_line(text_line_id),
            method_outputs=tuple(method_labels),
            review_status=self._review_status_by_line.get(text_line_id, "Not reviewed"),
        )
