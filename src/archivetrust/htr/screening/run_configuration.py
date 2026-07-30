"""The resolved, content-hashed configuration one reliability run executes under.

This module answers a single question precisely: *what is this run, exactly?* -- so that a run
stopped today and resumed next week can prove it is still the same run, and so that a run resumed
under a changed corpus, a changed sample, a changed model revision or changed heuristics **refuses**
rather than quietly producing a report that mixes two configurations.

Nothing here performs inference, reads a model, or touches `dataset-rgb/` other than to hash the
page bytes it is about to feed to the pipeline.

## What is fixed, and where it comes from

Every field below is read from an already-committed, already-reviewed artifact rather than invented
here:

* dataset identity, corpus digest, and the 60 sampled `page_id`s --
  `docs/experiments/technical-reliability-screening/dataset-manifest.json` (Stage 1's real
  inventory, seed 20260730, non-binding stratified proposal, adopted as the default configuration);
* per-page paths, content hashes and dimensions -- that stage's `dataset-inventory.csv`;
* preprocessing version and configuration hash -- `htr/preprocessing/models.py`;
* reliability-heuristics version and threshold hash -- `htr/screening/reliability_signals.py`;
* segmentation configuration -- the real `Florence2LineDetectorAdapter`'s own configuration object;
* model revisions -- each adapter's own `get_metadata()`.

## `configuration_hash` and `experiment_version_id`

`configuration_hash` is a deterministic sha256 over every resolved field *except* the derived
identifiers, in canonical JSON. `experiment_version_id` is derived from it, so two runs configured
identically share an experiment version and two runs configured differently can never appear to.
Neither is a random id: a random one would make "is this the same configuration?" unanswerable,
which is the only question this module exists to answer.

## Profiles

`full` is the real benchmark and is the default. `fixture` exists solely so the stop/resume
machinery can be demonstrated against real models on two pages in minutes instead of hours; it is
selected by the `ARCHIVETRUST_RELIABILITY_PROFILE` environment variable, is named in the resolved
configuration, and produces a different `configuration_hash` -- so fixture evidence can never be
mistaken for, or resumed into, the real run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.preprocessing.models import (
    RGB_NORMALIZATION_VERSION,
    RgbNormalizationConfig,
)
from archivetrust.htr.screening.reliability_signals import (
    RELIABILITY_HEURISTICS_VERSION,
    ReliabilityThresholds,
)

RELIABILITY_RUN_SCHEMA_VERSION = "1.0.0"
"""Version of the *run harness's* own contract: the task-identity tuple, the completion-marker
convention, and the shape of the resolved configuration. Bumped when any of those change meaning,
because a change to any of them changes what "already completed" means for previously recorded
work."""

SAMPLING_VERSION = "stratified-proposal/seed-20260730/v1"
"""Names the sampling *rule*, not the drawn sample. The page ids are carried explicitly in the
configuration as well; this string is what a resume checks first, so that a re-drawn sample under a
new rule is refused with a rule-level explanation rather than a diff of sixty ids."""

PROFILE_ENV_VAR = "ARCHIVETRUST_RELIABILITY_PROFILE"

SCREENING_DIR_NAME = "technical-reliability-screening"

DEFAULT_CROPS_PER_PAGE = 10
FIXTURE_CROPS_PER_PAGE = 2
FIXTURE_PAGE_COUNT = 2

SATRN_METHOD_ID = "satrn"
FLORENCE2_METHOD_ID = "florence2_htr"
SEGMENTATION_METHOD_ID = "segmentation"
"""The segmentation stage is a task like any other in this harness -- it has an identity, it
completes or it does not, and it is skipped when already complete. It is given a `method_id` so it
travels through the same task-identity tuple as recognition rather than needing a parallel one."""

RECOGNITION_METHOD_IDS: tuple[str, ...] = (SATRN_METHOD_ID, FLORENCE2_METHOD_ID)


class ConfigurationResolutionError(RuntimeError):
    """The configuration could not be resolved from the committed Stage 1 evidence -- a missing
    manifest, a sampled page absent from the inventory, a page file absent from `dataset-rgb/`.

    Raised rather than defaulted, because every one of those means the run would silently execute
    over something other than the approved sample.
    """


class PageSelection(BaseModel):
    """One sampled page, as Stage 1's real inventory recorded it. `content_hash` is the inventory's
    own page content address and is what the run's dataset check is made against -- never a
    re-derived one, so a mismatch means the bytes changed, not that two hashing schemes disagree."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    document_id: str
    page_number: int
    relative_path: str
    content_hash: str
    automatic_category: str
    width: int
    height: int


class ResolvedConfiguration(BaseModel):
    """Everything a run is pinned to, hashed, and comparable across processes."""

    model_config = ConfigDict(frozen=True)

    schema_version: str
    profile: str

    dataset_id: str
    dataset_version_id: str
    dataset_root_relative_path: str
    corpus_digest_sha256: str
    corpus_page_count: int

    sampling_version: str
    sampling_seed: int
    pages: tuple[PageSelection, ...]

    crops_per_page: int
    crop_selection_rule: str

    method_ids: tuple[str, ...]
    model_revisions: Mapping[str, str]

    segmentation_adapter_name: str
    segmentation_configuration_hash: str

    preprocessing_version: str
    preprocessing_configuration_hash: str

    reliability_heuristics_version: str
    reliability_thresholds_hash: str

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_recognition_tasks(self) -> int:
        """Upper bound only. A page whose segmentation yields fewer than `crops_per_page` lines
        contributes fewer real tasks, and the run reports the real count -- this number is what the
        status display shows as the denominator before segmentation has told it otherwise."""
        return self.page_count * self.crops_per_page

    def identity_payload(self) -> dict:
        """The canonical dict `configuration_hash` is computed over: every pinned field, and no
        derived identifier (including the hash itself), so the hash is a function of the
        configuration rather than of a previous hash."""
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "dataset_id": self.dataset_id,
            "dataset_version_id": self.dataset_version_id,
            "dataset_root_relative_path": self.dataset_root_relative_path,
            "corpus_digest_sha256": self.corpus_digest_sha256,
            "corpus_page_count": self.corpus_page_count,
            "sampling_version": self.sampling_version,
            "sampling_seed": self.sampling_seed,
            "page_ids": [page.page_id for page in self.pages],
            "page_content_hashes": [page.content_hash for page in self.pages],
            "crops_per_page": self.crops_per_page,
            "crop_selection_rule": self.crop_selection_rule,
            "method_ids": list(self.method_ids),
            "model_revisions": dict(sorted(self.model_revisions.items())),
            "segmentation_adapter_name": self.segmentation_adapter_name,
            "segmentation_configuration_hash": self.segmentation_configuration_hash,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_configuration_hash": self.preprocessing_configuration_hash,
            "reliability_heuristics_version": self.reliability_heuristics_version,
            "reliability_thresholds_hash": self.reliability_thresholds_hash,
        }

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return "reliability_config_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def experiment_version_id(self) -> str:
        """Derived from `configuration_hash`, never minted randomly -- see the module docstring."""
        return "experiment_version_" + self.configuration_hash.split("_", 2)[-1][:48]

    @property
    def dataset_fingerprint(self) -> str:
        """A single value summarizing the *content* of the sampled pages, used by the resume
        dataset-mismatch check. Distinct from `corpus_digest_sha256` (the whole corpus) because a
        run is only entitled to refuse over the pages it actually reads."""
        payload = "\n".join(f"{p.page_id}:{p.content_hash}" for p in self.pages)
        return "dataset_fingerprint_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def comparable_fields(self) -> dict[str, str]:
        """The named values a resume compares, one line per human-readable reason a resume can
        refuse. Ordered so the coarsest (the corpus) is reported before the finest (a threshold)."""
        return {
            "harness schema version": self.schema_version,
            "profile": self.profile,
            "dataset version": self.dataset_version_id,
            "corpus digest": self.corpus_digest_sha256,
            "sampled-page content fingerprint": self.dataset_fingerprint,
            "sampling version": self.sampling_version,
            "crops per page": str(self.crops_per_page),
            "methods": ",".join(self.method_ids),
            **{
                f"model revision [{method}]": revision
                for method, revision in sorted(self.model_revisions.items())
            },
            "segmentation adapter": self.segmentation_adapter_name,
            "segmentation configuration": self.segmentation_configuration_hash,
            "preprocessing version": self.preprocessing_version,
            "preprocessing configuration": self.preprocessing_configuration_hash,
            "reliability heuristics version": self.reliability_heuristics_version,
            "reliability thresholds": self.reliability_thresholds_hash,
            "configuration hash": self.configuration_hash,
        }


def screening_directory(repo_root: Path) -> Path:
    return repo_root / "docs" / "experiments" / SCREENING_DIR_NAME


def active_profile(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(PROFILE_ENV_VAR, "full").strip().lower() or "full"


def _load_manifest(repo_root: Path) -> dict:
    path = screening_directory(repo_root) / "dataset-manifest.json"
    if not path.exists():
        raise ConfigurationResolutionError(
            f"Stage 1's dataset manifest is missing at {path}. The run configuration is read from "
            "the committed inventory, never guessed."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inventory(repo_root: Path) -> dict[str, dict]:
    path = screening_directory(repo_root) / "dataset-inventory.csv"
    if not path.exists():
        raise ConfigurationResolutionError(
            f"Stage 1's dataset inventory is missing at {path}."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["page_id"]: row for row in csv.DictReader(handle)}


def _segmentation_configuration_hash(configuration) -> str:
    payload = json.dumps(
        configuration.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return "segmentation_config_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_configuration(
    repo_root: Path,
    *,
    profile: str | None = None,
    model_revisions: Mapping[str, str] | None = None,
    segmentation_adapter_name: str | None = None,
    segmentation_configuration_hash: str | None = None,
    thresholds: ReliabilityThresholds | None = None,
    verify_page_files: bool = True,
) -> ResolvedConfiguration:
    """Resolves the run configuration from committed Stage 1 evidence plus the live adapters.

    `model_revisions`, `segmentation_adapter_name` and `segmentation_configuration_hash` are
    injectable so a test can resolve a configuration without constructing a real GPU adapter. When
    omitted they are read from the real adapters -- which is what the interactive script does, and
    is why a silently swapped checkpoint is caught by the resume check rather than averaged into the
    results.
    """
    profile = (profile or active_profile()).strip().lower()
    if profile not in {"full", "fixture"}:
        raise ConfigurationResolutionError(
            f"unknown profile {profile!r}: expected 'full' (the real benchmark) or 'fixture' "
            f"(the small real-model demonstration). Set {PROFILE_ENV_VAR} to one of those."
        )

    manifest = _load_manifest(repo_root)
    inventory = _load_inventory(repo_root)
    dataset = manifest["dataset"]
    proposal = manifest["proposed_sample"]

    sampled_ids: list[str] = []
    for stratum in sorted(proposal["selected_page_ids_by_stratum"]):
        sampled_ids.extend(proposal["selected_page_ids_by_stratum"][stratum])

    missing = [page_id for page_id in sampled_ids if page_id not in inventory]
    if missing:
        raise ConfigurationResolutionError(
            f"{len(missing)} sampled page id(s) are not present in dataset-inventory.csv "
            f"(first: {missing[0]}). The manifest and the inventory disagree; refusing to run over "
            "a sample that cannot be resolved to real files."
        )

    pages = [
        PageSelection(
            page_id=page_id,
            document_id=inventory[page_id]["document_id"],
            page_number=int(inventory[page_id]["page_number"]),
            relative_path=inventory[page_id]["relative_path"],
            content_hash=inventory[page_id]["content_hash"],
            automatic_category=inventory[page_id]["automatic_category"],
            width=int(inventory[page_id]["width"]),
            height=int(inventory[page_id]["height"]),
        )
        for page_id in sampled_ids
    ]

    crops_per_page = DEFAULT_CROPS_PER_PAGE
    if profile == "fixture":
        # The two physically smallest sampled pages, so the demonstration exercises the real
        # segmentation + real recognition path at the lowest wall-clock cost available inside the
        # approved sample. Deterministic (sorted by pixel count then page id), never random.
        pages.sort(key=lambda p: (p.width * p.height, p.page_id))
        pages = pages[:FIXTURE_PAGE_COUNT]
        pages.sort(key=lambda p: p.page_id)
        crops_per_page = FIXTURE_CROPS_PER_PAGE

    if verify_page_files:
        absent = [p for p in pages if not (repo_root / p.relative_path).is_file()]
        if absent:
            raise ConfigurationResolutionError(
                f"{len(absent)} sampled page file(s) are missing from the working tree (first: "
                f"{absent[0].relative_path}). `dataset-rgb/` is not committed; restore it before "
                "running."
            )

    if model_revisions is None or segmentation_adapter_name is None:
        from archivetrust.htr.segmentation import (  # noqa: PLC0415
            Florence2LineDetectorAdapter,
        )
        from archivetrust.providers.florence2_htr.adapter import (  # noqa: PLC0415
            Florence2Adapter,
        )
        from archivetrust.providers.satrn.adapter import SatrnAdapter  # noqa: PLC0415

        if model_revisions is None:
            model_revisions = {
                SATRN_METHOD_ID: SatrnAdapter().get_metadata().model_revision,
                FLORENCE2_METHOD_ID: Florence2Adapter().get_metadata().model_revision,
            }
        if segmentation_adapter_name is None:
            detector = Florence2LineDetectorAdapter()
            segmentation_adapter_name = detector.name
            if segmentation_configuration_hash is None:
                segmentation_configuration_hash = _segmentation_configuration_hash(
                    detector.configuration
                )

    if segmentation_configuration_hash is None:
        from archivetrust.htr.segmentation import (  # noqa: PLC0415
            Florence2LineDetectorAdapter,
        )

        segmentation_configuration_hash = _segmentation_configuration_hash(
            Florence2LineDetectorAdapter().configuration
        )

    thresholds = thresholds or ReliabilityThresholds()
    return ResolvedConfiguration(
        schema_version=RELIABILITY_RUN_SCHEMA_VERSION,
        profile=profile,
        dataset_id=dataset["dataset_id"],
        dataset_version_id=dataset["dataset_version"],
        dataset_root_relative_path=dataset["root_relative_path"],
        corpus_digest_sha256=dataset["corpus_digest_sha256"],
        corpus_page_count=int(manifest["counts"]["total_files"]),
        sampling_version=SAMPLING_VERSION,
        sampling_seed=int(proposal["seed"]),
        pages=tuple(pages),
        crops_per_page=crops_per_page,
        crop_selection_rule="evenly-spaced-over-detected-lines/v1",
        method_ids=RECOGNITION_METHOD_IDS,
        model_revisions=dict(model_revisions or {}),
        segmentation_adapter_name=segmentation_adapter_name or "unknown",
        segmentation_configuration_hash=segmentation_configuration_hash,
        preprocessing_version=RGB_NORMALIZATION_VERSION,
        preprocessing_configuration_hash=RgbNormalizationConfig().configuration_hash,
        reliability_heuristics_version=RELIABILITY_HEURISTICS_VERSION,
        reliability_thresholds_hash=thresholds.configuration_hash,
    )


def evenly_spaced_indices(total: int, wanted: int) -> tuple[int, ...]:
    """The `crop_selection_rule`: `wanted` indices spread evenly across `total`, deterministically.

    Evenly spaced rather than the first N (which the smoke test used) because the first lines of a
    page are systematically headers, dates and short marginal marks -- exactly the degenerate-input
    regime the SATRN diagnostic already characterized. Taking the first ten would have measured that
    regime again instead of the page.

    Returns every index when `total <= wanted`, and `()` for an empty page -- never a fabricated
    index into lines that were not detected.
    """
    if total <= 0 or wanted <= 0:
        return ()
    if total <= wanted:
        return tuple(range(total))
    step = total / wanted
    seen: list[int] = []
    for position in range(wanted):
        index = min(total - 1, int(position * step + step / 2))
        if index not in seen:
            seen.append(index)
    return tuple(seen)
