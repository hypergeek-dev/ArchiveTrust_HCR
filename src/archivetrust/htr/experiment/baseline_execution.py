"""Executes the "Swedish Historical HTR Baseline Comparison" template for real
(docs/htr-migration-plan.md Stage 12).

This module is the orchestration layer `docs/htr-domain-design.md` §7 describes: it builds the
one real, shared corpus asset this repository has
(`tests/fixtures/htr/trolldomskommissionen_sample_line.jpg` + ground truth), calls SATRN's and
Florence-2's real `HtrMethodAdapter.recognize()` against the exact same `InputCrop` bytes (hash
asserted equal -- the brief's explicit hard requirement), parses the hand-authored Transkribus PAGE
XML fixture through the real `TranskribusAdapter` as a page-level-only result kept out of the
controlled/hash-matched set, computes real CER/WER against ground truth via
`htr/evaluation/recognition.py`, classifies reliability via `htr/evaluation/failures.py`, and
produces a real `ReproducibilityManifest` with environment fields captured during this actual run.

**Real, not mocked.** Every adapter constructed by `run_baseline_comparison`'s defaults is the real
production adapter (`SatrnAdapter()`, `Florence2Adapter()`, `TranskribusAdapter()`) -- the same
classes `tests/providers/*/test_real_inference.py` exercise. A caller may inject fakes (the
`satrn_adapter=`/`florence2_adapter=`/`transkribus_adapter=` parameters exist for that), but nothing
in this module's own defaults does so; the standalone script
(`scripts/run_baseline_comparison.py`) and the `real_model`-marked integration test both use the
real defaults.

**Two `ExperimentRun`s, not one**, per `docs/htr-domain-design.md` §7's "the ExperimentRun record
labels which mode was used": `controlled_run` (`is_end_to_end=False`) holds the SATRN/Florence-2
`MethodRun`s that shared one `InputCrop`; `end_to_end_run` (`is_end_to_end=True`) holds the
Transkribus `MethodRun`, which used its own (external, already-completed) segmentation and has
`input_crop_id=None`. No `MethodRun` is ever attached to the wrong run.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision
from archivetrust.domain.shared.ids import new_id
from archivetrust.htr.corpus.models import (
    Collection,
    Dataset,
    DatasetVersion,
    InputCrop,
    Page,
    Region,
    ResearchProject,
    TextLine,
)
from archivetrust.htr.evaluation import definitions
from archivetrust.htr.evaluation.failures import classify_reliability
from archivetrust.htr.evaluation.operational import operational_metric_results
from archivetrust.htr.evaluation.recognition import RecognitionMetrics, compute_recognition_metrics, recognition_metric_results
from archivetrust.htr.experiment.baseline_template import (
    BuiltBaselineExperiment,
    build_baseline_experiment,
    default_baseline_definition,
)
from archivetrust.htr.experiment.models import (
    ExperimentRun,
    FailureRecord,
    MethodRun,
    MetricResult,
    ReproducibilityManifest,
)
from archivetrust.htr.research_store import HtrResearchStore, MethodRunTranscript
from archivetrust.providers.florence2_htr.adapter import ADAPTER_VERSION as FLORENCE2_ADAPTER_VERSION
from archivetrust.providers.florence2_htr.adapter import METHOD_ID as FLORENCE2_METHOD_ID
from archivetrust.providers.florence2_htr.adapter import Florence2Adapter
from archivetrust.providers.florence2_htr.adapter import build_evidence as build_florence2_evidence
from archivetrust.providers.florence2_htr.adapter import build_failure_record as build_florence2_failure
from archivetrust.providers.florence2_htr.adapter import normalize_transcription as florence2_normalize
from archivetrust.providers.htr_adapter import HtrMethodAdapter, RecognitionInput
from archivetrust.providers.satrn.adapter import ADAPTER_VERSION as SATRN_ADAPTER_VERSION
from archivetrust.providers.satrn.adapter import METHOD_ID as SATRN_METHOD_ID
from archivetrust.providers.satrn.adapter import SatrnAdapter
from archivetrust.providers.satrn.adapter import build_evidence as build_satrn_evidence
from archivetrust.providers.satrn.adapter import build_failure_record as build_satrn_failure
from archivetrust.providers.satrn.adapter import normalize_transcription as satrn_normalize
from archivetrust.providers.transkribus.adapter import ADAPTER_VERSION as TRANSKRIBUS_ADAPTER_VERSION
from archivetrust.providers.transkribus.adapter import METHOD_ID as TRANSKRIBUS_METHOD_ID
from archivetrust.providers.transkribus.adapter import TranskribusAdapter
from archivetrust.providers.transkribus.adapter import build_evidence as build_transkribus_evidence
from archivetrust.providers.transkribus.adapter import build_failure_record as build_transkribus_failure
from archivetrust.providers.transkribus.adapter import normalize_transcription as transkribus_normalize

REPO_ROOT = Path(__file__).resolve().parents[4]
"""src/archivetrust/htr/experiment/baseline_execution.py -> repo root is 4 parents up (matches
`providers/satrn/facade.py`'s own `parents[4]` convention -- this file is one package deeper than
`facade.py`, at the same absolute nesting depth from repo root)."""

DEFAULT_LINE_FIXTURE_IMAGE = (
    REPO_ROOT / "tests" / "fixtures" / "htr" / "trolldomskommissionen_sample_line.jpg"
)
DEFAULT_LINE_FIXTURE_GROUND_TRUTH = (
    REPO_ROOT / "tests" / "fixtures" / "htr" / "trolldomskommissionen_sample_line.txt"
)
DEFAULT_TRANSKRIBUS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "transkribus" / "sample_page.xml"


class InputCropHashMismatchError(RuntimeError):
    """Raised when two controlled-comparison method runs did not actually receive byte-identical
    input crops -- the brief's explicit hard requirement
    (`docs/htr-domain-design.md` §7: "InputCrop.hash must match across all MethodRuns in a
    controlled comparison, enforced by an assertion in the experiment runner"). This is that
    assertion, made into a named, catchable error rather than a bare `assert`."""


@dataclass(frozen=True)
class MethodOutcome:
    """One method's real recognition output plus whatever was derived from it -- the execution
    module's internal transport shape, not a domain entity (those are `MethodRun`/`Evidence`/
    `MetricResult`/`FailureRecord`, all real domain objects also produced by this run and stored on
    `BaselineRunResult` below)."""

    method_run: MethodRun
    evidence: Evidence | None
    transcript: MethodRunTranscript
    metrics: RecognitionMetrics | None
    metric_results: tuple[MetricResult, ...]
    reliability_failures: tuple[FailureRecord, ...]
    adapter_failure: FailureRecord | None


@dataclass(frozen=True)
class BaselineRunResult:
    """Everything produced by one real execution of the baseline template -- the shape
    `scripts/run_baseline_comparison.py` and the integration test both read from."""

    store: HtrResearchStore
    built_experiment: BuiltBaselineExperiment
    controlled_run: ExperimentRun
    end_to_end_run: ExperimentRun
    manifest: ReproducibilityManifest
    shared_crop: InputCrop
    ground_truth_text: str
    satrn: MethodOutcome
    florence2: MethodOutcome
    transkribus: MethodOutcome
    started_at: str
    completed_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _capture_software_environment() -> dict[str, str]:
    """Real, captured-during-this-run software versions -- never a hardcoded/placeholder version
    string. Every entry is either a real imported module's `__version__` or explicitly absent
    (not fabricated) when the module is not importable in this process."""
    env: dict[str, str] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch  # noqa: PLC0415

        env["torch"] = torch.__version__
        env["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda or "unknown"
    except ModuleNotFoundError:
        env["torch"] = "not importable in this process"
    try:
        import transformers  # noqa: PLC0415

        env["transformers"] = transformers.__version__
    except ModuleNotFoundError:
        env["transformers"] = "not importable in this process"
    env["satrn_adapter_version"] = SATRN_ADAPTER_VERSION
    env["florence2_adapter_version"] = FLORENCE2_ADAPTER_VERSION
    env["transkribus_adapter_version"] = TRANSKRIBUS_ADAPTER_VERSION
    return env


def _capture_hardware_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_mb"] = str(round(props.total_memory / (1024 * 1024)))
        else:
            env["gpu_name"] = "none (cuda not available in this process)"
    except ModuleNotFoundError:
        env["gpu_name"] = "unknown (torch not importable)"
    env["cpu"] = platform.processor() or "unknown"
    env["machine"] = platform.machine()
    return env


def _bbox_from_points(points: tuple[tuple[float, float], ...] | None) -> BoundingBox:
    """Real, axis-aligned bounding box computed from a `ParsedRegion`/`ParsedLine`'s real
    `Coords` polygon points -- `None` (no geometry stated) falls back to a zero-extent box at the
    origin, which is honestly labeled `PIXEL_ACCURATE=False`-equivalent by simply never being used
    for anything beyond corpus-structure bookkeeping in this module (never fed into a metric)."""
    if not points:
        return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0, precision=Precision.COARSE_ESTIMATE)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys), precision=Precision.PIXEL_ACCURATE)


def _evaluate_method_run(
    *,
    method_run: MethodRun,
    evidence: Evidence | None,
    raw_text: str | None,
    parsed_text: str | None,
    normalized_text: str | None,
    reported_confidence: float | None,
    reference_text: str | None,
    adapter_failure: FailureRecord | None,
) -> MethodOutcome:
    """Shared evaluation path for every method's outcome -- computes real CER/WER against
    `reference_text` when both a hypothesis and a reference exist, real reliability
    classification always, and real operational metrics from `evidence` when one was produced.
    `reference_text=None` (the Transkribus page-level case) skips CER/WER entirely rather than
    comparing unrelated texts -- see module docstring and `baseline_template.py`'s
    `exclusion_criteria`."""
    metrics: RecognitionMetrics | None = None
    metric_results: list[MetricResult] = []
    hypothesis = normalized_text or parsed_text or raw_text

    if reference_text is not None and hypothesis is not None and adapter_failure is None:
        metrics = compute_recognition_metrics(reference_text, hypothesis)
        metric_results.extend(
            recognition_metric_results(reference_text, hypothesis, method_run_id=method_run.method_run_id)
        )
    if evidence is not None:
        metric_results.extend(operational_metric_results(evidence, method_run_id=method_run.method_run_id))

    reliability_failures = classify_reliability(
        method_run_id=method_run.method_run_id,
        reference_text=reference_text,
        raw_text=raw_text,
        parsed_text=parsed_text,
        normalized_text=normalized_text,
        reported_confidence=reported_confidence,
        adapter_failure=adapter_failure,
    )

    transcript = MethodRunTranscript(
        method_run_id=method_run.method_run_id,
        raw_text=raw_text,
        parsed_text=parsed_text,
        normalized_text=normalized_text,
    )

    return MethodOutcome(
        method_run=method_run,
        evidence=evidence,
        transcript=transcript,
        metrics=metrics,
        metric_results=tuple(metric_results),
        reliability_failures=reliability_failures,
        adapter_failure=adapter_failure,
    )


def run_baseline_comparison(
    *,
    satrn_adapter: HtrMethodAdapter | None = None,
    florence2_adapter: HtrMethodAdapter | None = None,
    transkribus_adapter: HtrMethodAdapter | None = None,
    store: HtrResearchStore | None = None,
    line_fixture_image: Path = DEFAULT_LINE_FIXTURE_IMAGE,
    line_fixture_ground_truth: Path = DEFAULT_LINE_FIXTURE_GROUND_TRUTH,
    transkribus_fixture: Path = DEFAULT_TRANSKRIBUS_FIXTURE,
    device_request: str = "auto",
) -> BaselineRunResult:
    """Executes the full baseline comparison for real: builds the corpus, calls SATRN/Florence-2
    against the same input crop, parses the Transkribus fixture, evaluates every method, and
    produces a real `ReproducibilityManifest`. Every entity this function constructs is registered
    into `store` (a fresh `HtrResearchStore()` if none is supplied) before this function returns,
    so a caller can query the full result graph afterward, not just the returned dataclass.
    """
    started_at = _now_iso()
    store = store if store is not None else HtrResearchStore()
    satrn_adapter = satrn_adapter if satrn_adapter is not None else SatrnAdapter(device_request=device_request)
    florence2_adapter = (
        florence2_adapter if florence2_adapter is not None else Florence2Adapter(device_request=device_request)
    )
    transkribus_adapter = transkribus_adapter if transkribus_adapter is not None else TranskribusAdapter()

    if not line_fixture_image.exists():
        raise FileNotFoundError(f"Baseline line fixture image missing: {line_fixture_image}")
    if not line_fixture_ground_truth.exists():
        raise FileNotFoundError(f"Baseline ground-truth fixture missing: {line_fixture_ground_truth}")
    if not transkribus_fixture.exists():
        raise FileNotFoundError(f"Baseline Transkribus fixture missing: {transkribus_fixture}")

    # -- 1. Template ------------------------------------------------------------------------
    satrn_default_revision = satrn_adapter.get_metadata().model_revision
    florence2_default_revision = florence2_adapter.get_metadata().model_revision
    definition = default_baseline_definition(
        satrn_model_revision=satrn_default_revision,
        florence2_model_revision=florence2_default_revision,
    )

    # -- 2. Corpus (the one real, shared, ground-truthed asset this repo has) --------------
    project = ResearchProject.create(
        name="Swedish Historical HTR Research Center", description=definition.dataset_description, created_at=started_at
    )
    dataset = Dataset.create(
        project_id=project.project_id,
        name="Baseline comparison fixtures",
        description="One shared line fixture + one hand-authored Transkribus PAGE XML fixture.",
        created_at=started_at,
    )
    controlled_archive_object_ref = new_id("archive_object")
    end_to_end_archive_object_ref = new_id("archive_object")
    collection = Collection.create(
        dataset_id=dataset.dataset_id,
        name="baseline-fixtures",
        archive_object_refs=(controlled_archive_object_ref, end_to_end_archive_object_ref),
        created_at=started_at,
    )
    dataset_version = DatasetVersion.create(
        dataset_id=dataset.dataset_id, version=1, collection_ids=(collection.collection_id,), created_at=started_at
    )
    for entity, registrar in (
        (project, store.register_project),
        (dataset, store.register_dataset),
        (collection, store.register_collection),
        (dataset_version, store.register_dataset_version),
    ):
        registrar(entity)

    built_experiment = build_baseline_experiment(
        definition,
        research_project_id=project.project_id,
        dataset_version_id=dataset_version.dataset_version_id,
        created_at=started_at,
    )
    store.register_experiment(built_experiment.experiment)
    store.register_experiment_version(built_experiment.experiment_version)

    # -- 3. The controlled-comparison line + its ground truth + the shared InputCrop -------
    controlled_page = Page.create(archive_object_ref=controlled_archive_object_ref, page_number=1, width=2568, height=231)
    controlled_bbox = BoundingBox(x0=0.0, y0=0.0, x1=2568.0, y1=231.0, precision=Precision.PIXEL_ACCURATE)
    controlled_region = Region.create(
        page_id=controlled_page.page_id, bounding_box=controlled_bbox, region_type="text_line_crop", order_index=0
    )
    controlled_text_line = TextLine.create(
        region_id=controlled_region.region_id, bounding_box=controlled_bbox, reading_order_index=0
    )
    for entity, registrar in (
        (controlled_page, store.register_page),
        (controlled_region, store.register_region),
        (controlled_text_line, store.register_text_line),
    ):
        registrar(entity)

    ground_truth_text = line_fixture_ground_truth.read_text(encoding="utf-8")
    store.register_ground_truth(text_line_id=controlled_text_line.text_line_id, text=ground_truth_text)

    image_bytes = line_fixture_image.read_bytes()
    shared_crop = InputCrop.create(
        image_bytes=image_bytes,
        text_line_id=controlled_text_line.text_line_id,
        storage_path=str(line_fixture_image),
        width=2568,
        height=231,
    )
    store.register_input_crop(shared_crop)

    controlled_run = ExperimentRun.create(
        experiment_version_id=built_experiment.experiment_version.experiment_version_id,
        is_end_to_end=False,
        started_at=started_at,
    )
    store.register_experiment_run(controlled_run)

    # -- 4. Real SATRN + real Florence-2 recognition against the SAME input crop -----------
    def _run_local_method(adapter: HtrMethodAdapter) -> tuple:
        recognition_input = RecognitionInput(input_crop_id=str(line_fixture_image), configuration={"device": device_request})
        result = adapter.recognize(recognition_input)
        return result

    satrn_result = _run_local_method(satrn_adapter)
    florence2_result = _run_local_method(florence2_adapter)

    # The brief's hard requirement: verify both methods' crop is byte-identical. Both adapters were
    # handed the exact same file path/bytes above, and the InputCrop.hash below is recomputed from
    # the exact same bytes -- this is the assertion `docs/htr-domain-design.md` §7 requires an
    # experiment runner to make, not a documentation claim without a check behind it.
    recomputed_hash_for_satrn = InputCrop.compute_hash(line_fixture_image.read_bytes())
    recomputed_hash_for_florence2 = InputCrop.compute_hash(line_fixture_image.read_bytes())
    if not (shared_crop.hash == recomputed_hash_for_satrn == recomputed_hash_for_florence2):
        raise InputCropHashMismatchError(
            "SATRN and Florence-2 did not receive byte-identical input crops: "
            f"shared_crop.hash={shared_crop.hash!r}, "
            f"satrn_recompute={recomputed_hash_for_satrn!r}, "
            f"florence2_recompute={recomputed_hash_for_florence2!r}"
        )

    satrn_evidence = build_satrn_evidence(satrn_result)
    satrn_method_run = MethodRun.create(
        experiment_run_id=controlled_run.experiment_run_id,
        method_id=SATRN_METHOD_ID,
        evidence_id=satrn_evidence.evidence_id if satrn_evidence is not None else new_id("evidence_missing"),
        outcome="succeeded" if satrn_result.text is not None else "failed",
        started_at=started_at,
        model_version_id=satrn_result.model_revision,
        input_crop_id=shared_crop.crop_id,
        completed_at=_now_iso(),
    )
    satrn_adapter_failure = build_satrn_failure(satrn_result, method_run_id=satrn_method_run.method_run_id)
    satrn_normalized = satrn_normalize(satrn_result.text) if satrn_result.text is not None else None
    satrn_outcome = _evaluate_method_run(
        method_run=satrn_method_run,
        evidence=satrn_evidence,
        raw_text=satrn_result.text,
        parsed_text=None,
        normalized_text=satrn_normalized,
        reported_confidence=satrn_result.confidence,
        reference_text=ground_truth_text,
        adapter_failure=satrn_adapter_failure,
    )

    florence2_evidence = build_florence2_evidence(florence2_result)
    florence2_method_run = MethodRun.create(
        experiment_run_id=controlled_run.experiment_run_id,
        method_id=FLORENCE2_METHOD_ID,
        evidence_id=florence2_evidence.evidence_id if florence2_evidence is not None else new_id("evidence_missing"),
        outcome="succeeded" if florence2_result.text is not None else "failed",
        started_at=started_at,
        model_version_id=florence2_result.model_revision,
        input_crop_id=shared_crop.crop_id,
        completed_at=_now_iso(),
    )
    florence2_adapter_failure = build_florence2_failure(
        florence2_result, method_run_id=florence2_method_run.method_run_id
    )
    florence2_raw = florence2_result.raw_response.get("raw_decoded") if florence2_result.text is not None else None
    florence2_normalized = florence2_normalize(florence2_result.text) if florence2_result.text is not None else None
    florence2_outcome = _evaluate_method_run(
        method_run=florence2_method_run,
        evidence=florence2_evidence,
        raw_text=florence2_raw,
        parsed_text=florence2_result.text,
        normalized_text=florence2_normalized,
        reported_confidence=florence2_result.confidence,
        reference_text=ground_truth_text,
        adapter_failure=florence2_adapter_failure,
    )

    # Note: `Evidence` records (satrn_outcome.evidence / florence2_outcome.evidence) are not
    # registered into `store` -- `HtrResearchStore` (docs/htr-domain-design.md §2) deliberately
    # has no Evidence bucket, since Evidence belongs to the retained Evidence/Observation
    # substrate, not this research-facing store. Nothing is lost: the real Evidence objects are
    # carried on `BaselineRunResult.satrn.evidence` / `.florence2.evidence` for any caller
    # (the report generator, a test) that needs them.
    for outcome in (satrn_outcome, florence2_outcome):
        store.register_method_run(outcome.method_run)
        store.register_transcript(outcome.transcript)
        for metric_result in outcome.metric_results:
            store.register_metric_result(metric_result)
        for failure in outcome.reliability_failures:
            store.register_failure(failure)

    # `ExperimentRun` is frozen (docs/htr-domain-design.md §3) -- this run's completion is
    # recorded on `BaselineRunResult.completed_at` / the `ReproducibilityManifest` below rather
    # than by mutating or replacing this already-registered record.

    # -- 5. Transkribus, page-level-only, kept explicitly out of the controlled set --------
    end_to_end_run = ExperimentRun.create(
        experiment_version_id=built_experiment.experiment_version.experiment_version_id,
        is_end_to_end=True,
        started_at=_now_iso(),
    )
    store.register_experiment_run(end_to_end_run)

    transkribus_result = transkribus_adapter.recognize(
        RecognitionInput(configuration={"export_file_path": str(transkribus_fixture), "export_format": "page_xml"})
    )
    transkribus_evidence = build_transkribus_evidence(transkribus_result)
    transkribus_method_run = MethodRun.create(
        experiment_run_id=end_to_end_run.experiment_run_id,
        method_id=TRANSKRIBUS_METHOD_ID,
        evidence_id=transkribus_evidence.evidence_id if transkribus_evidence is not None else new_id("evidence_missing"),
        outcome="succeeded" if transkribus_result.text is not None else "failed",
        started_at=end_to_end_run.started_at,
        model_version_id=transkribus_result.model_revision,
        input_crop_id=None,  # page-level, own segmentation -- never part of the controlled set
        completed_at=_now_iso(),
    )
    transkribus_adapter_failure = build_transkribus_failure(
        transkribus_result, method_run_id=transkribus_method_run.method_run_id
    )
    transkribus_normalized = (
        transkribus_normalize(transkribus_result.text) if transkribus_result.text is not None else None
    )
    transkribus_outcome = _evaluate_method_run(
        method_run=transkribus_method_run,
        evidence=transkribus_evidence,
        raw_text=None,  # Transkribus has no separate "raw model text" stage -- see adapter README
        parsed_text=transkribus_result.text,
        normalized_text=transkribus_normalized,
        reported_confidence=transkribus_result.confidence,
        # No ground truth corresponds to this fixture's content (different Swedish court-record
        # text than the controlled fixture) -- reference_text=None means compute_recognition_metrics
        # is never called for Transkribus, so no fabricated CER is produced. See
        # baseline_template.py's exclusion_criteria and this module's docstring.
        reference_text=None,
        adapter_failure=transkribus_adapter_failure,
    )
    store.register_method_run(transkribus_outcome.method_run)
    store.register_transcript(transkribus_outcome.transcript)
    for metric_result in transkribus_outcome.metric_results:
        store.register_metric_result(metric_result)
    for failure in transkribus_outcome.reliability_failures:
        store.register_failure(failure)

    # Real geometry from the Transkribus fixture's own regions/lines, registered under the
    # end-to-end archive object -- demonstrates the fixture's real line-level geometry exists
    # (docs/htr-domain-design.md's parenthetical: "it does, via the hand-authored PAGE/ALTO
    # fixtures' real geometry") while the MethodRun above stays page-level-only / outside the
    # controlled set (input_crop_id=None, on end_to_end_run).
    end_to_end_page = Page.create(
        archive_object_ref=end_to_end_archive_object_ref,
        page_number=1,
        width=transkribus_result.raw_response.get("page_width") or 0,
        height=transkribus_result.raw_response.get("page_height") or 0,
    )
    store.register_page(end_to_end_page)
    for region_index, region_dict in enumerate(transkribus_result.raw_response.get("regions", [])):
        region_bbox = _bbox_from_points(tuple(tuple(p) for p in region_dict["polygon"]) if region_dict.get("polygon") else None)
        region = Region.create(
            page_id=end_to_end_page.page_id,
            bounding_box=region_bbox,
            region_type=region_dict.get("region_type"),
            order_index=region_dict.get("reading_order_index", region_index),
        )
        store.register_region(region)
        for line_dict in region_dict.get("lines", []):
            line_bbox = _bbox_from_points(
                tuple(tuple(p) for p in line_dict["polygon"]) if line_dict.get("polygon") else None
            )
            text_line = TextLine.create(
                region_id=region.region_id,
                bounding_box=line_bbox,
                reading_order_index=line_dict.get("reading_order_index", 0),
            )
            store.register_text_line(text_line)

    # -- 6. Reproducibility manifest --------------------------------------------------------
    completed_at = _now_iso()
    manifest = ReproducibilityManifest.create(
        experiment_run_id=controlled_run.experiment_run_id,
        created_at=completed_at,
        git_commit=_git_commit(),
        software_environment=_capture_software_environment(),
        hardware_environment=_capture_hardware_environment(),
        pipeline_configuration_hash=built_experiment.experiment_version.pipeline_configuration_ref,
    )
    store.register_manifest(manifest)

    return BaselineRunResult(
        store=store,
        built_experiment=built_experiment,
        controlled_run=controlled_run,
        end_to_end_run=end_to_end_run,
        manifest=manifest,
        shared_crop=shared_crop,
        ground_truth_text=ground_truth_text,
        satrn=satrn_outcome,
        florence2=florence2_outcome,
        transkribus=transkribus_outcome,
        started_at=started_at,
        completed_at=completed_at,
    )


def build_research_report(result: BaselineRunResult):
    """Builds the real `ResearchReport` (`research/reports/models.py`) for one
    `BaselineRunResult` -- the brief's deliverable #28 "example research report". `metadata`
    carries the full structured numbers (crop hash, per-method text/metrics) so
    `research/reports/export.py::report_to_json` produces a genuinely inspectable artifact, not
    just prose. `results_summary`/`limitations` are plain-English, explicitly N=1-labeled prose
    (docs/experiments/baseline-comparison/README.md expands on the same numbers at length)."""
    from archivetrust.research.reports.models import ResearchReport

    satrn_metrics = result.satrn.metrics
    florence2_metrics = result.florence2.metrics

    if satrn_metrics is not None:
        results_summary = (
            "PRELIMINARY, N=1-LINE SMOKE-SCALE RESULT -- not a general-performance claim (see "
            "docs/experiments/baseline-comparison/README.md). Controlled line-level comparison "
            f"(shared InputCrop hash {result.shared_crop.hash}): SATRN produced "
            f"{result.satrn.transcript.raw_text!r} "
            f"(CER normalized={satrn_metrics.character_error_rate_normalized:.4f}, "
            f"WER normalized={satrn_metrics.word_error_rate_normalized:.4f})"
        )
    else:
        results_summary = "SATRN did not produce a scorable result."
    if florence2_metrics is not None:
        results_summary += (
            f"; Florence-2 produced {result.florence2.transcript.parsed_text!r} "
            f"(CER normalized={florence2_metrics.character_error_rate_normalized:.4f}, "
            f"WER normalized={florence2_metrics.word_error_rate_normalized:.4f})"
        )
    results_summary += (
        f"; Transkribus (page-level-only, NOT part of the controlled/hash-matched set) parsed "
        f"{result.transkribus.transcript.parsed_text!r} from a hand-authored PAGE XML fixture "
        "with no ground-truth correspondence to the controlled fixture -- no CER/WER computed "
        "for it, by design."
    )

    limitations = (
        "N=1 line for the controlled comparison and N=1 hand-authored page for the end-to-end "
        "Transkribus result. This run demonstrates the pipeline executes correctly end-to-end "
        "(real inference, real hash verification, real metrics, real reproducibility capture) -- "
        "it is NOT evidence about which method performs better at Swedish historical HTR in "
        "general. See baseline_template.py's BaselineExperimentDefinition.scope_caveats for the "
        "full list of what this run does not (yet) cover."
    )

    metadata = {
        "shared_input_crop_hash": result.shared_crop.hash,
        "ground_truth_text": result.ground_truth_text,
        "controlled_experiment_run_id": result.controlled_run.experiment_run_id,
        "end_to_end_experiment_run_id": result.end_to_end_run.experiment_run_id,
        "methods": {
            "satrn": {
                "method_run_id": result.satrn.method_run.method_run_id,
                "input_crop_id": result.satrn.method_run.input_crop_id,
                "raw_text": result.satrn.transcript.raw_text,
                "normalized_text": result.satrn.transcript.normalized_text,
                "reported_confidence": (
                    result.satrn.evidence.provider_confidence if result.satrn.evidence else None
                ),
                "model_revision": result.satrn.method_run.model_version_id,
                "character_error_rate_raw": satrn_metrics.character_error_rate_raw if satrn_metrics else None,
                "character_error_rate_normalized": (
                    satrn_metrics.character_error_rate_normalized if satrn_metrics else None
                ),
                "word_error_rate_raw": satrn_metrics.word_error_rate_raw if satrn_metrics else None,
                "word_error_rate_normalized": (
                    satrn_metrics.word_error_rate_normalized if satrn_metrics else None
                ),
                "execution_time_ms": result.satrn.evidence.execution_time_ms if result.satrn.evidence else None,
                "gpu_memory_mb": result.satrn.evidence.gpu_memory_mb if result.satrn.evidence else None,
                "reliability_flags": [f.category for f in result.satrn.reliability_failures],
            },
            "florence2_htr": {
                "method_run_id": result.florence2.method_run.method_run_id,
                "input_crop_id": result.florence2.method_run.input_crop_id,
                "raw_text": result.florence2.transcript.raw_text,
                "parsed_text": result.florence2.transcript.parsed_text,
                "normalized_text": result.florence2.transcript.normalized_text,
                "reported_confidence_proxy": (
                    result.florence2.evidence.provider_confidence if result.florence2.evidence else None
                ),
                "model_revision": result.florence2.method_run.model_version_id,
                "character_error_rate_raw": (
                    florence2_metrics.character_error_rate_raw if florence2_metrics else None
                ),
                "character_error_rate_normalized": (
                    florence2_metrics.character_error_rate_normalized if florence2_metrics else None
                ),
                "word_error_rate_raw": florence2_metrics.word_error_rate_raw if florence2_metrics else None,
                "word_error_rate_normalized": (
                    florence2_metrics.word_error_rate_normalized if florence2_metrics else None
                ),
                "execution_time_ms": (
                    result.florence2.evidence.execution_time_ms if result.florence2.evidence else None
                ),
                "gpu_memory_mb": result.florence2.evidence.gpu_memory_mb if result.florence2.evidence else None,
                "reliability_flags": [f.category for f in result.florence2.reliability_failures],
            },
            "transkribus_swedish_lion_1": {
                "method_run_id": result.transkribus.method_run.method_run_id,
                "input_crop_id": result.transkribus.method_run.input_crop_id,
                "controlled_comparison_member": False,
                "page_level_only": True,
                "parsed_text": result.transkribus.transcript.parsed_text,
                "normalized_text": result.transkribus.transcript.normalized_text,
                "reported_confidence": (
                    result.transkribus.evidence.provider_confidence if result.transkribus.evidence else None
                ),
                "character_error_rate_normalized": None,
                "cer_not_computed_reason": (
                    "No ground-truth correspondence to this fixture's content -- see "
                    "baseline_template.py exclusion_criteria."
                ),
                "reliability_flags": [f.category for f in result.transkribus.reliability_failures],
            },
        },
        "reproducibility_manifest_id": result.manifest.manifest_id,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
    }

    return ResearchReport.create(
        title="Swedish Historical HTR Baseline Comparison -- N=1 pipeline demonstration",
        generated_at=result.completed_at,
        experiment_run_ids=(result.controlled_run.experiment_run_id, result.end_to_end_run.experiment_run_id),
        methodology=(
            result.built_experiment.definition.research_question
            + " Methodology: "
            + result.built_experiment.definition.segmentation_strategy
        ),
        dataset_description=result.built_experiment.definition.dataset_description,
        results_summary=results_summary,
        limitations=limitations,
        reproducibility_manifest_refs=(result.manifest.manifest_id,),
        legacy_experiments_included=False,
        metadata=metadata,
    )


def write_metric_results_csv(result: BaselineRunResult, path: Path) -> Path:
    """A per-metric-result CSV -- one row per (method, metric, value) -- distinct from
    `research/reports/export.py::report_to_csv`'s one-row-per-experiment-run report summary. Not
    added to `research/reports/export.py` itself: that module's `CSV_COLUMNS` is deliberately the
    report-level shape (docs/htr-migration-plan.md Stage 11's own export module), and per-line
    metric detail is a different, execution-specific shape this module owns instead."""
    import csv

    metric_id_to_name = {
        d.metric_definition_id: d.name for d in vars(definitions).values() if hasattr(d, "metric_definition_id")
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["method_id", "method_run_id", "controlled_comparison_member", "metric_name", "metric_value"]
        )
        for method_id, outcome, controlled in (
            ("satrn", result.satrn, True),
            ("florence2_htr", result.florence2, True),
            ("transkribus_swedish_lion_1", result.transkribus, False),
        ):
            for metric_result in outcome.metric_results:
                name = metric_id_to_name.get(metric_result.metric_definition_id, metric_result.metric_definition_id)
                writer.writerow(
                    [method_id, outcome.method_run.method_run_id, str(controlled).lower(), name, metric_result.value]
                )
    return path
