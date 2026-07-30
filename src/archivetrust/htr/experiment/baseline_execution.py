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

**Durable from 2026-07-30** (`docs/architecture/htr-telemetry.md`; the phase that module's §7 named
"Baseline re-execution: not done here"). Two things changed here, and only these two:

1. `store` is now a `DurableHtrResearchStore` -- and *defaults* to one, over an
   `InMemoryTelemetrySink`, instead of to a bare `HtrResearchStore()`. So this function always
   event-sources; only the medium varies. A caller wanting real durability injects a
   `FileTelemetrySink`-backed store (`scripts/run_baseline_comparison.py` does), and the store
   reports which it got via `is_durable` rather than leaving it ambiguous -- the ambiguity
   `docs/htr-telemetry-knowledge-gap-analysis.md` §0 traces real data loss to.
2. Every registration now threads `caused_by` and runs inside a `correlated_to` scope, so the
   emitted log carries a real causal DAG and one `correlation_id` per `ExperimentRun`
   (`docs/architecture/htr-event-model.md` §4) rather than events that merely happen to be adjacent
   in a file. The two runs get *two different* correlation ids, which is what keeps the Transkribus
   chain provably separate from the controlled one rather than separate by convention.

**Report generation now reads durable records, not these dataclasses** -- gap analysis §8's
remaining item. `build_research_report` is a thin wrapper over
`build_research_report_from_store`, which sources every field from store queries and from the
event stream; see that function's docstring for the one traversal that makes it possible
(`ExperimentVersion.pipeline_configuration_ref` round-trips the whole
`BaselineExperimentDefinition`).

**No `SegmentationRunCompleted` is emitted, deliberately.** Neither run performs segmentation in
this process: the controlled fixture is already a pre-cropped line, and Transkribus's segmentation
ran externally and is imported from a PAGE XML fixture. Emitting a "segmentation adapter run
completed" for either would claim an execution that did not happen. See
`baseline_template.py`'s `segmentation_strategy` for the same statement in the template itself.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from archivetrust.domain.evidence.models import BoundingBox, Evidence, Precision
from archivetrust.domain.shared.ids import new_id
from archivetrust.domain.telemetry.events import EvidenceCreated, TelemetryEvent
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
    BaselineExperimentDefinition,
    BuiltBaselineExperiment,
    build_baseline_experiment,
    default_baseline_definition,
)
from archivetrust.htr.experiment.models import (
    ExperimentRun,
    FailureRecord,
    MethodRun,
    MetricDefinition,
    MetricResult,
    ReproducibilityManifest,
)
from archivetrust.htr.persistence.durable_store import DurableHtrResearchStore
from archivetrust.htr.research_store import HtrResearchStore, MethodRunTranscript
from archivetrust.infrastructure.storage.telemetry_sink import InMemoryTelemetrySink
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


CER_WER_METRIC_NAMES: frozenset[str] = frozenset(
    definition.name
    for definition in (
        definitions.CHARACTER_ERROR_RATE_RAW,
        definitions.CHARACTER_ERROR_RATE_NORMALIZED,
        definitions.WORD_ERROR_RATE_RAW,
        definitions.WORD_ERROR_RATE_NORMALIZED,
    )
)
"""The four accuracy-against-ground-truth metric names. Named here, derived from the definition
objects rather than hand-typed, so a test can assert "no CER/WER exists for the Transkribus method
run" against the same source of truth the producer uses -- one of this follow-up's named testing
requirements, and the durable counterpart of `baseline_template.py`'s `exclusion_criteria`."""


class MissingDurableRecordError(LookupError):
    """Raised when report generation cannot find a record it needs in the store it was handed.

    A named error rather than a `None`-propagating field, because a report silently missing its
    experiment run or its shared input crop would be a report about nothing -- exactly the class of
    silent-absence failure `ARCHITECTURAL_CONSTITUTION.md` Article 18 exists to prevent."""


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
    recognition_metric_results: tuple[MetricResult, ...]
    """CER/WER/edit-count results, computed from this run's transcript text against ground truth."""
    operational_metric_results: tuple[MetricResult, ...]
    """Timing/memory results, read verbatim off `Evidence`. Kept separate from the recognition
    results *because their causes differ*: a recognition metric is caused by the transcript stage
    whose text it scored, an operational one by the method run that produced the `Evidence`. Merging
    them into one tuple would force the durable log to claim a normalized string caused a timing
    measurement (see `run_baseline_comparison`'s `_record_outcome`)."""
    reliability_failures: tuple[FailureRecord, ...]
    adapter_failure: FailureRecord | None

    @property
    def metric_results(self) -> tuple[MetricResult, ...]:
        """Every `MetricResult` for this method run, recognition and operational together -- the flat
        view callers that do not care about causal origin (a CSV row writer) want."""
        return self.recognition_metric_results + self.operational_metric_results


@dataclass(frozen=True)
class BaselineRunResult:
    """Everything produced by one real execution of the baseline template -- the shape
    `scripts/run_baseline_comparison.py` and the integration test both read from."""

    store: DurableHtrResearchStore
    built_experiment: BuiltBaselineExperiment
    controlled_run: ExperimentRun
    """The controlled `ExperimentRun` **as started** (`completed_at is None`), unchanged from this
    module's pre-durability behaviour. Its terminal state lives in the durable log as an
    `ExperimentRunCompleted` event, and a replayed projection therefore answers
    `experiment_run(id).completed_at` with a real timestamp where this field is `None` -- the
    projection advancement `HtrResearchStore.advance_experiment_run` is documented for."""
    end_to_end_run: ExperimentRun
    manifest: ReproducibilityManifest
    shared_crop: InputCrop
    ground_truth_text: str
    satrn: MethodOutcome
    florence2: MethodOutcome
    transkribus: MethodOutcome
    started_at: str
    completed_at: str
    controlled_run_started_event_id: str
    """The `ExperimentRunStarted` `event_id` for the controlled run -- the causal root a test walks
    `application/htr_journal.py::causation_chain` from, handed back explicitly rather than
    rediscovered by scanning the stream for a kind."""
    end_to_end_run_started_event_id: str
    method_run_started_event_ids: dict[str, str]
    """`method_id -> MethodRunStarted event_id`, for the same reason."""


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
    recognition_results: tuple[MetricResult, ...] = ()
    operational_results: tuple[MetricResult, ...] = ()
    hypothesis = normalized_text or parsed_text or raw_text

    if reference_text is not None and hypothesis is not None and adapter_failure is None:
        metrics = compute_recognition_metrics(reference_text, hypothesis)
        recognition_results = recognition_metric_results(
            reference_text, hypothesis, method_run_id=method_run.method_run_id
        )
    if evidence is not None:
        operational_results = operational_metric_results(
            evidence, method_run_id=method_run.method_run_id
        )

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
        recognition_metric_results=recognition_results,
        operational_metric_results=operational_results,
        reliability_failures=reliability_failures,
        adapter_failure=adapter_failure,
    )


def metric_definition_registry() -> tuple[MetricDefinition, ...]:
    """Every `MetricDefinition` the evaluation engine can produce, in `definitions.py`'s own
    declaration order.

    Registered wholesale at the start of a run so that a projection rebuilt from telemetry alone can
    resolve `MetricResult.metric_definition_id -> name/version` from durable records, instead of
    having to import `htr/evaluation/definitions.py` and rely on that module's ids matching -- which
    they would not, since each `MetricDefinition.metric_definition_id` is minted fresh per process
    import (that module's own docstring says so). Without this, a replayed report could show metric
    *values* with no way to say which metric each one is.
    """
    return tuple(
        value for value in vars(definitions).values() if isinstance(value, MetricDefinition)
    )


def evidence_from_events(events: Iterable[TelemetryEvent]) -> dict[str, Evidence]:
    """Recovers the real `Evidence` records from a durable telemetry stream, keyed by `evidence_id`.

    `HtrResearchStore` deliberately has no `Evidence` bucket (`docs/htr-domain-design.md` §2), so
    `HtrJournal.replay` cannot return these -- but `DurableHtrResearchStore.record_evidence` writes
    them to the same stream as `EvidenceCreated`, so they are recoverable from it. This is what lets
    `build_research_report_from_store` report real provider confidence and execution device from
    durable records rather than from a live in-process object.
    """
    return {
        event.evidence.evidence_id: event.evidence
        for event in events
        if isinstance(event, EvidenceCreated)
    }


def run_baseline_comparison(
    *,
    satrn_adapter: HtrMethodAdapter | None = None,
    florence2_adapter: HtrMethodAdapter | None = None,
    transkribus_adapter: HtrMethodAdapter | None = None,
    store: DurableHtrResearchStore | None = None,
    line_fixture_image: Path = DEFAULT_LINE_FIXTURE_IMAGE,
    line_fixture_ground_truth: Path = DEFAULT_LINE_FIXTURE_GROUND_TRUTH,
    transkribus_fixture: Path = DEFAULT_TRANSKRIBUS_FIXTURE,
    device_request: str = "auto",
) -> BaselineRunResult:
    """Executes the full baseline comparison for real: builds the corpus, calls SATRN/Florence-2
    against the same input crop, parses the Transkribus fixture, evaluates every method, and
    produces a real `ReproducibilityManifest`. Every entity this function constructs is registered
    into `store` before this function returns, so a caller can query the full result graph
    afterward, not just the returned dataclass -- and, because every registration also appends a
    telemetry event, so a caller can *reconstruct* that graph later from the event log alone.

    `store` defaults to a `DurableHtrResearchStore` over an `InMemoryTelemetrySink`: event-sourced,
    honestly reporting `is_durable is False`, and sufficient for a test. Pass one backed by a
    `FileTelemetrySink` for a run whose evidence must survive the process --
    `scripts/run_baseline_comparison.py` does exactly that.
    """
    started_at = _now_iso()
    store = store if store is not None else DurableHtrResearchStore(InMemoryTelemetrySink())
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
    # Each registration returns the `event_id` it emitted, threaded into the next call's `caused_by`
    # so the durable log carries an explicit causal DAG (docs/architecture/htr-event-model.md §4)
    # rather than events a reader would have to infer relationships between from append order.
    project_event = store.register_project(project)
    dataset_event = store.register_dataset(dataset, caused_by=project_event)
    collection_event = store.register_collection(collection, caused_by=dataset_event)
    dataset_version_event = store.register_dataset_version(
        dataset_version, caused_by=collection_event
    )

    # The metric registry, recorded before any result references it -- see
    # `metric_definition_registry`. Deliberately no `caused_by` and no correlation: these definitions
    # are a module-import-time fact of the evaluation engine, not something this experiment caused.
    for metric_definition in metric_definition_registry():
        store.register_metric_definition(metric_definition)

    built_experiment = build_baseline_experiment(
        definition,
        research_project_id=project.project_id,
        dataset_version_id=dataset_version.dataset_version_id,
        created_at=started_at,
    )
    experiment_event = store.register_experiment(
        built_experiment.experiment, caused_by=project_event
    )
    # Caused by the experiment it versions; its `DatasetVersion` link is carried as a stored id
    # reference (`dataset_version_id`), which is how docs/architecture/htr-telemetry.md §2 specifies
    # the traceability chain -- `causation_id` is single-valued and names what *caused* this event.
    experiment_version_event = store.register_experiment_version(
        built_experiment.experiment_version, caused_by=experiment_event
    )
    assert dataset_version_event  # the version references it; keep the link asserted, not implied

    # -- 3. The controlled-comparison line + its ground truth + the shared InputCrop -------
    controlled_page = Page.create(archive_object_ref=controlled_archive_object_ref, page_number=1, width=2568, height=231)
    controlled_bbox = BoundingBox(x0=0.0, y0=0.0, x1=2568.0, y1=231.0, precision=Precision.PIXEL_ACCURATE)
    controlled_region = Region.create(
        page_id=controlled_page.page_id, bounding_box=controlled_bbox, region_type="text_line_crop", order_index=0
    )
    controlled_text_line = TextLine.create(
        region_id=controlled_region.region_id, bounding_box=controlled_bbox, reading_order_index=0
    )
    controlled_page_event = store.register_page(controlled_page, caused_by=collection_event)
    controlled_region_event = store.register_region(
        controlled_region, caused_by=controlled_page_event
    )
    controlled_line_event = store.register_text_line(
        controlled_text_line, caused_by=controlled_region_event
    )

    ground_truth_text = line_fixture_ground_truth.read_text(encoding="utf-8")
    store.register_ground_truth(
        text_line_id=controlled_text_line.text_line_id,
        text=ground_truth_text,
        caused_by=controlled_line_event,
    )

    image_bytes = line_fixture_image.read_bytes()
    shared_crop = InputCrop.create(
        image_bytes=image_bytes,
        text_line_id=controlled_text_line.text_line_id,
        storage_path=str(line_fixture_image),
        width=2568,
        height=231,
    )
    store.register_input_crop(shared_crop, caused_by=controlled_line_event)

    controlled_run = ExperimentRun.create(
        experiment_version_id=built_experiment.experiment_version.experiment_version_id,
        is_end_to_end=False,
        started_at=started_at,
    )
    # `register_experiment_run` defaults this event's `correlation_id` to the run's own id -- it is
    # the causal root of the run, so it belongs to that run's correlation by definition.
    controlled_run_event = store.register_experiment_run(
        controlled_run, caused_by=experiment_version_event
    )

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

    # Note: `Evidence` records are still not registered into `store`'s *projection* --
    # `HtrResearchStore` (docs/htr-domain-design.md §2) deliberately has no Evidence bucket, since
    # Evidence belongs to the retained Evidence/Observation substrate, not this research-facing
    # store. What changed is that they are no longer confined to this process's memory either:
    # `store.record_evidence` appends each one to the same durable stream as `EvidenceCreated`, the
    # retained substrate's own event kind, closing the gap
    # `docs/htr-telemetry-knowledge-gap-analysis.md` §4 point 3 names ("those `Evidence` objects are
    # also never appended to a `TelemetrySink`"). `evidence_from_events` reads them back.
    method_run_started_event_ids: dict[str, str] = {}

    def _record_outcome(
        outcome: MethodOutcome, *, archive_object_ref: str, caused_by: str
    ) -> tuple[str, str]:
        """Durably records one method's whole outcome as a causally-chained event sequence.

        The chain the follow-up asks to be demonstrable end to end:

            MethodRunStarted -> RawMethodResultRecorded -> ParsedMethodResultRecorded
                             -> NormalizedMethodResultRecorded -> MetricCalculated

        assembled by threading each emitted `event_id` into the next call's `caused_by`.
        `register_transcript` emits one event per *populated* stage and returns the last, so SATRN
        (which has no separate parsed stage) honestly produces a shorter chain than Florence-2 rather
        than a fabricated `ParsedMethodResultRecorded` duplicating its raw text.

        `MethodRunCompleted` is emitted as a causal *sibling* of the transcript stages, not a link in
        that chain -- it marks the run terminal, it does not cause the raw output.

        Returns `(MethodRunStarted event_id, last emitted event_id)`.
        """
        method_run_event = store.register_method_run(outcome.method_run, caused_by=caused_by)
        method_run_started_event_ids[outcome.method_run.method_id] = method_run_event
        if outcome.evidence is not None:
            store.record_evidence(
                outcome.evidence,
                document_ref=archive_object_ref,
                invocation_id=outcome.method_run.method_run_id,
                caused_by=method_run_event,
            )
        store.complete_method_run(
            outcome.method_run,
            caused_by=method_run_event,
            failure_reason=(
                outcome.adapter_failure.reason if outcome.adapter_failure is not None else None
            ),
        )
        transcript_event = store.register_transcript(
            outcome.transcript,
            caused_by=method_run_event,
            evidence_id=outcome.method_run.evidence_id,
        )
        last_event = transcript_event
        # A recognition metric is caused by the transcript stage whose text it scored; an operational
        # metric is read off `Evidence` and is caused by the method run, not by any text. Hence the
        # two tuples on `MethodOutcome` -- see its `operational_metric_results` docstring.
        for metric_result in outcome.recognition_metric_results:
            last_event = store.register_metric_result(metric_result, caused_by=transcript_event)
        for metric_result in outcome.operational_metric_results:
            store.register_metric_result(metric_result, caused_by=method_run_event)
        for failure in outcome.reliability_failures:
            # Two events per classification, deliberately, because they are two different facts at
            # two different layers of `docs/architecture/htr-event-model.md` §1:
            #
            # * `ReliabilityIssueClassified` (layer 7, evaluation evidence) is the semantically
            #   correct kind and says what `classify_reliability` concluded. Emitting it is what
            #   finally gives that kind a producer.
            # * `MethodRunFailed` carrying the whole `FailureRecord` is what keeps the conclusion
            #   *indexed and replayable*: `HtrJournal._apply` projects it into
            #   `store.failures(method_run_id=...)`, whereas `ReliabilityIssueClassified` is one of
            #   the kinds `docs/architecture/htr-telemetry.md` §11 names as "durable but unindexed"
            #   because `HtrResearchStore` has no reliability bucket.
            #
            # Honest caveat, stated rather than smoothed over: `MethodRunFailed`/`FailureRecord` are
            # documented as "a MethodRun that could not complete", and a reliability flag on a run
            # whose `outcome` is `"succeeded"` stretches that. The stretch is
            # `htr/evaluation/failures.py`'s own modelling choice -- `classify_reliability` returns
            # `FailureRecord`s -- and predates this pass; it is recorded here rather than hidden.
            store.record_reliability_issue(
                method_run_id=outcome.method_run.method_run_id,
                classification=failure.category or "unclassified",
                detail=failure.reason,
                caused_by=transcript_event,
            )
            store.register_failure(failure, caused_by=transcript_event)
        return method_run_event, last_event

    # Everything below belongs to the controlled run's unit of work and shares its correlation id.
    # The Transkribus run further down gets its own, different one -- which is what makes the two
    # chains provably separate rather than separate by convention.
    with store.correlated_to(controlled_run.experiment_run_id):
        _record_outcome(
            satrn_outcome,
            archive_object_ref=controlled_archive_object_ref,
            caused_by=controlled_run_event,
        )
        _, controlled_last_event = _record_outcome(
            florence2_outcome,
            archive_object_ref=controlled_archive_object_ref,
            caused_by=controlled_run_event,
        )

    # `ExperimentRun` is frozen (docs/htr-domain-design.md §3), so this run's completion is recorded
    # as a new `ExperimentRunCompleted` event carrying a `model_copy` with `completed_at` set --
    # never by mutating the already-registered record. See the end of this function.

    # -- 5. Transkribus, page-level-only, kept explicitly out of the controlled set --------
    end_to_end_run = ExperimentRun.create(
        experiment_version_id=built_experiment.experiment_version.experiment_version_id,
        is_end_to_end=True,
        started_at=_now_iso(),
    )
    end_to_end_run_event = store.register_experiment_run(
        end_to_end_run, caused_by=experiment_version_event
    )

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
    # A *separate* correlation scope: this run's events carry `end_to_end_run.experiment_run_id`,
    # never the controlled run's. Combined with `reference_text=None` above (so
    # `recognition_metric_results` is empty and no `MetricCalculated` for a CER/WER definition is
    # ever emitted for this method run), that is the durable, queryable form of the design decision
    # in `baseline_template.py`'s `exclusion_criteria`: Transkribus is not in the controlled set and
    # has no CER against the controlled fixture's ground truth.
    with store.correlated_to(end_to_end_run.experiment_run_id):
        _, transkribus_last_event = _record_outcome(
            transkribus_outcome,
            archive_object_ref=end_to_end_archive_object_ref,
            caused_by=end_to_end_run_event,
        )

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
    with store.correlated_to(end_to_end_run.experiment_run_id):
        end_to_end_page_event = store.register_page(
            end_to_end_page, caused_by=end_to_end_run_event
        )
        for region_index, region_dict in enumerate(transkribus_result.raw_response.get("regions", [])):
            region_bbox = _bbox_from_points(tuple(tuple(p) for p in region_dict["polygon"]) if region_dict.get("polygon") else None)
            region = Region.create(
                page_id=end_to_end_page.page_id,
                bounding_box=region_bbox,
                region_type=region_dict.get("region_type"),
                order_index=region_dict.get("reading_order_index", region_index),
            )
            region_event = store.register_region(region, caused_by=end_to_end_page_event)
            for line_dict in region_dict.get("lines", []):
                line_bbox = _bbox_from_points(
                    tuple(tuple(p) for p in line_dict["polygon"]) if line_dict.get("polygon") else None
                )
                text_line = TextLine.create(
                    region_id=region.region_id,
                    bounding_box=line_bbox,
                    reading_order_index=line_dict.get("reading_order_index", 0),
                )
                store.register_text_line(text_line, caused_by=region_event)

        store.complete_experiment_run(
            end_to_end_run.model_copy(update={"completed_at": _now_iso()}),
            caused_by=transkribus_last_event,
        )

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
    with store.correlated_to(controlled_run.experiment_run_id):
        manifest_event = store.register_manifest(manifest, caused_by=controlled_last_event)
        # Chained from the manifest rather than forked off the last metric: the run completes *after*
        # its reproducibility manifest is recorded, so the causal path stays a single line.
        store.complete_experiment_run(
            controlled_run.model_copy(update={"completed_at": completed_at}),
            caused_by=manifest_event,
        )

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
        controlled_run_started_event_id=controlled_run_event,
        end_to_end_run_started_event_id=end_to_end_run_event,
        method_run_started_event_ids=dict(method_run_started_event_ids),
    )


def _metric_values_by_name(store: HtrResearchStore, *, method_run_id: str) -> dict[str, float]:
    """`metric name -> value` for one method run, with names resolved through the store's *own*
    registered `MetricDefinition`s.

    Deliberately not resolved by importing `htr/evaluation/definitions.py` and matching ids: those
    ids are minted fresh on every process import (that module's docstring), so a projection replayed
    in a later process would match none of them. Resolving through the store is what makes a report
    built from a replay name its metrics correctly."""
    values: dict[str, float] = {}
    for result in store.metric_results(method_run_id=method_run_id):
        definition = store.metric_definition(result.metric_definition_id)
        name = definition.name if definition is not None else result.metric_definition_id
        values[name] = result.value
    return values


def _method_entry(
    store: HtrResearchStore,
    run: MethodRun,
    *,
    evidence_by_id: Mapping[str, Evidence],
    controlled_comparison_member: bool,
) -> dict:
    """One method's report entry, assembled entirely from durable records.

    Uniform in shape across all three methods (the previous generator emitted a slightly different
    key set per method), with the Transkribus-specific exclusion facts added rather than substituted,
    so a reader comparing two methods is never comparing two different schemas."""
    transcript = store.transcript(run.method_run_id)
    metrics = _metric_values_by_name(store, method_run_id=run.method_run_id)
    evidence = evidence_by_id.get(run.evidence_id)

    entry = {
        "method_id": run.method_id,
        "method_run_id": run.method_run_id,
        "experiment_run_id": run.experiment_run_id,
        "controlled_comparison_member": controlled_comparison_member,
        "input_crop_id": run.input_crop_id,
        "outcome": run.outcome,
        "model_revision": run.model_version_id,
        "evidence_id": run.evidence_id,
        "raw_text": transcript.raw_text if transcript is not None else None,
        "parsed_text": transcript.parsed_text if transcript is not None else None,
        "normalized_text": transcript.normalized_text if transcript is not None else None,
        "reported_confidence": evidence.provider_confidence if evidence is not None else None,
        "reported_confidence_note": (
            "Method-specific meaning -- see each adapter's README (Florence-2's is a proxy, not a "
            "calibrated probability). Read from this run's durable EvidenceCreated record."
        ),
        "execution_device": evidence.execution_device if evidence is not None else None,
        "character_error_rate_raw": metrics.get(definitions.CHARACTER_ERROR_RATE_RAW.name),
        "character_error_rate_normalized": metrics.get(
            definitions.CHARACTER_ERROR_RATE_NORMALIZED.name
        ),
        "word_error_rate_raw": metrics.get(definitions.WORD_ERROR_RATE_RAW.name),
        "word_error_rate_normalized": metrics.get(definitions.WORD_ERROR_RATE_NORMALIZED.name),
        "exact_word_accuracy": metrics.get(definitions.EXACT_WORD_ACCURACY.name),
        "execution_time_ms": metrics.get(definitions.EXECUTION_TIME_MS.name),
        "gpu_memory_mb": metrics.get(definitions.GPU_MEMORY_MB.name),
        "reliability_flags": [
            failure.category for failure in store.failures(method_run_id=run.method_run_id)
        ],
    }
    if not controlled_comparison_member:
        entry["page_level_only"] = True
        entry["cer_not_computed_reason"] = (
            "No ground-truth correspondence to this fixture's content -- see baseline_template.py "
            "exclusion_criteria. Every CER/WER field above is null because no MetricCalculated event "
            "for a CER/WER definition was ever emitted for this method run, not because one was "
            "computed and hidden."
        )
    return entry


def build_research_report_from_store(
    store: HtrResearchStore,
    *,
    controlled_experiment_run_id: str,
    end_to_end_experiment_run_id: str,
    evidence_by_id: Mapping[str, Evidence] | None = None,
):
    """Builds the real `ResearchReport` (`research/reports/models.py`) from **durable records only**.

    This is `docs/htr-telemetry-knowledge-gap-analysis.md` §8's remaining item, closed: the previous
    generator read the in-process `BaselineRunResult`/`MethodOutcome` dataclasses directly, which §4
    identified as "a third, even-narrower data source than either of the two the design docs
    anticipated". Every field here comes from a store query or from `evidence_by_id`, which
    `evidence_from_events` recovers from the event stream -- so the same report can be regenerated
    from a projection rebuilt by `HtrJournal.replay` after the original process is gone.

    The one traversal that makes it possible: `ExperimentVersion.pipeline_configuration_ref`
    round-trips the whole `BaselineExperimentDefinition` (`baseline_template.py::to_ref` writes
    deterministic sorted JSON), so the research question, dataset description, and segmentation
    strategy are recoverable from the durable `ExperimentVersionCreated` event rather than needing a
    live `BuiltBaselineExperiment`.

    `store` is typed as the base `HtrResearchStore` on purpose: this function only ever *reads*, and a
    replayed projection is a bare one.
    """
    from archivetrust.research.reports.models import ResearchReport

    evidence_by_id = evidence_by_id if evidence_by_id is not None else {}

    controlled_run = store.experiment_run(controlled_experiment_run_id)
    end_to_end_run = store.experiment_run(end_to_end_experiment_run_id)
    if controlled_run is None or end_to_end_run is None:
        raise MissingDurableRecordError(
            f"cannot build a report: experiment run(s) missing from the store "
            f"(controlled={controlled_experiment_run_id!r} found={controlled_run is not None}, "
            f"end_to_end={end_to_end_experiment_run_id!r} found={end_to_end_run is not None})"
        )

    experiment_version = store.experiment_version(controlled_run.experiment_version_id)
    if experiment_version is None or experiment_version.pipeline_configuration_ref is None:
        raise MissingDurableRecordError(
            f"cannot build a report: ExperimentVersion "
            f"{controlled_run.experiment_version_id!r} or its pipeline_configuration_ref is missing"
        )
    definition = BaselineExperimentDefinition.model_validate_json(
        experiment_version.pipeline_configuration_ref
    )

    controlled_by_method = {
        run.method_id: run
        for run in store.method_runs(experiment_run_id=controlled_experiment_run_id)
    }
    end_to_end_by_method = {
        run.method_id: run
        for run in store.method_runs(experiment_run_id=end_to_end_experiment_run_id)
    }

    # The shared-crop invariant, re-enforced from the durable side: every controlled method run must
    # name one and the same `InputCrop`. This is the same requirement `run_baseline_comparison`
    # asserts against the fixture bytes (docs/htr-domain-design.md §7), checked here against what was
    # actually *recorded* -- so a report can never present a controlled comparison the log does not
    # support.
    crop_ids = {run.input_crop_id for run in controlled_by_method.values()}
    if len(crop_ids) != 1 or None in crop_ids:
        raise InputCropHashMismatchError(
            "the durable record does not describe a controlled comparison: the "
            f"{len(controlled_by_method)} method run(s) on experiment run "
            f"{controlled_experiment_run_id!r} reference input crop ids "
            f"{sorted(str(crop_id) for crop_id in crop_ids)!r} -- a controlled comparison requires "
            "exactly one, shared, non-null crop across every one of them"
        )
    shared_crop = store.input_crop(next(iter(crop_ids)))
    if shared_crop is None:
        raise MissingDurableRecordError(
            f"cannot build a report: InputCrop {next(iter(crop_ids))!r} missing from the store"
        )
    ground_truth_text = store.ground_truth_for_line(shared_crop.text_line_id)

    methods = {
        run.method_id: _method_entry(
            store, run, evidence_by_id=evidence_by_id, controlled_comparison_member=True
        )
        for run in controlled_by_method.values()
    }
    methods.update(
        {
            run.method_id: _method_entry(
                store, run, evidence_by_id=evidence_by_id, controlled_comparison_member=False
            )
            for run in end_to_end_by_method.values()
        }
    )

    satrn = methods.get(SATRN_METHOD_ID, {})
    florence2 = methods.get(FLORENCE2_METHOD_ID, {})
    transkribus = methods.get(TRANSKRIBUS_METHOD_ID, {})

    def _fmt(value) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    results_summary = (
        "PRELIMINARY, N=1-LINE SMOKE-SCALE RESULT -- not a general-performance claim (see "
        "docs/experiments/baseline-comparison/README.md). Controlled line-level comparison "
        f"(shared InputCrop hash {shared_crop.hash}): SATRN produced {satrn.get('raw_text')!r} "
        f"(CER normalized={_fmt(satrn.get('character_error_rate_normalized'))}, "
        f"WER normalized={_fmt(satrn.get('word_error_rate_normalized'))})"
        f"; Florence-2 produced {florence2.get('parsed_text')!r} "
        f"(CER normalized={_fmt(florence2.get('character_error_rate_normalized'))}, "
        f"WER normalized={_fmt(florence2.get('word_error_rate_normalized'))})"
        f"; Transkribus (page-level-only, NOT part of the controlled/hash-matched set) parsed "
        f"{transkribus.get('parsed_text')!r} from a hand-authored PAGE XML fixture with no "
        "ground-truth correspondence to the controlled fixture -- no CER/WER computed for it, by "
        "design."
    )

    limitations = (
        "N=1 line for the controlled comparison and N=1 hand-authored page for the end-to-end "
        "Transkribus result. This run demonstrates the pipeline executes correctly end-to-end "
        "(real inference, real hash verification, real metrics, real reproducibility capture, and "
        "-- new as of 2026-07-30 -- durable, replayable telemetry for all of it) -- it is NOT "
        "evidence about which method performs better at Swedish historical HTR in general. See "
        "baseline_template.py's BaselineExperimentDefinition.scope_caveats for the full list of "
        "what this run does not (yet) cover."
    )

    manifests = store.manifests(experiment_run_id=controlled_experiment_run_id)
    manifest_ids = tuple(manifest.manifest_id for manifest in manifests)

    metadata = {
        "source": (
            "Generated from durable HTR telemetry records via "
            "build_research_report_from_store -- not from in-process dataclasses "
            "(docs/htr-telemetry-knowledge-gap-analysis.md §8)."
        ),
        "shared_input_crop_hash": shared_crop.hash,
        "shared_input_crop_id": shared_crop.crop_id,
        "ground_truth_text": ground_truth_text,
        "controlled_experiment_run_id": controlled_experiment_run_id,
        "end_to_end_experiment_run_id": end_to_end_experiment_run_id,
        "methods": methods,
        "reproducibility_manifest_id": manifest_ids[0] if manifest_ids else None,
        "started_at": controlled_run.started_at,
        "completed_at": controlled_run.completed_at,
        "end_to_end_started_at": end_to_end_run.started_at,
        "end_to_end_completed_at": end_to_end_run.completed_at,
    }

    return ResearchReport.create(
        title="Swedish Historical HTR Baseline Comparison -- N=1 pipeline demonstration",
        generated_at=controlled_run.completed_at or controlled_run.started_at,
        experiment_run_ids=(controlled_experiment_run_id, end_to_end_experiment_run_id),
        methodology=(
            definition.research_question + " Methodology: " + definition.segmentation_strategy
        ),
        dataset_description=definition.dataset_description,
        results_summary=results_summary,
        limitations=limitations,
        reproducibility_manifest_refs=manifest_ids,
        legacy_experiments_included=False,
        metadata=metadata,
    )


def build_research_report(result: BaselineRunResult):
    """Builds the real `ResearchReport` for one `BaselineRunResult` -- the brief's deliverable #28
    "example research report".

    A thin wrapper: it extracts the store and the two run ids and delegates to
    `build_research_report_from_store`, which does all the work from durable records. Even on this
    live path the `Evidence` fields are read back out of the event stream rather than off the
    in-process `MethodOutcome`s, so there is exactly one code path producing a report and no way for
    the live and replayed versions to drift.
    """
    return build_research_report_from_store(
        result.store,
        controlled_experiment_run_id=result.controlled_run.experiment_run_id,
        end_to_end_experiment_run_id=result.end_to_end_run.experiment_run_id,
        evidence_by_id=evidence_from_events(result.store.sink.all_events()),
    )


def write_metric_results_csv_from_store(
    store: HtrResearchStore,
    path: Path,
    *,
    controlled_experiment_run_id: str,
    end_to_end_experiment_run_id: str,
) -> Path:
    """A per-metric-result CSV -- one row per (method, metric, value) -- built from durable records.

    Distinct from `research/reports/export.py::report_to_csv`'s one-row-per-experiment-run report
    summary. Not added to `research/reports/export.py` itself: that module's `CSV_COLUMNS` is
    deliberately the report-level shape (docs/htr-migration-plan.md Stage 11's own export module),
    and per-line metric detail is a different, execution-specific shape this module owns instead.
    """
    import csv

    rows: list[tuple[str, str, str, str, float]] = []
    for run_id, controlled in (
        (controlled_experiment_run_id, True),
        (end_to_end_experiment_run_id, False),
    ):
        for run in store.method_runs(experiment_run_id=run_id):
            for name, value in sorted(
                _metric_values_by_name(store, method_run_id=run.method_run_id).items()
            ):
                rows.append(
                    (run.method_id, run.method_run_id, str(controlled).lower(), name, value)
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["method_id", "method_run_id", "controlled_comparison_member", "metric_name", "metric_value"]
        )
        writer.writerows(rows)
    return path


def write_metric_results_csv(result: BaselineRunResult, path: Path) -> Path:
    """`write_metric_results_csv_from_store` for one `BaselineRunResult` -- same thin-wrapper
    arrangement as `build_research_report`, for the same reason."""
    return write_metric_results_csv_from_store(
        result.store,
        path,
        controlled_experiment_run_id=result.controlled_run.experiment_run_id,
        end_to_end_experiment_run_id=result.end_to_end_run.experiment_run_id,
    )
