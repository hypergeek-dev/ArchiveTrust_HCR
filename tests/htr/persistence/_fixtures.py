"""Synthetic HTR research entities for the durable-persistence tests.

Synthetic on purpose: this pass is about the persistence layer, and none of these tests needs -- or
should need -- real SATRN/Florence-2/Transkribus inference to prove that an event written to disk
replays into the same entity. Re-running the baseline through this path is a separate,
later task (`docs/htr-telemetry-knowledge-gap-analysis.md` §0/§9).
"""

from __future__ import annotations

from dataclasses import dataclass

from archivetrust.domain.evidence.models import BoundingBox, Precision
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
from archivetrust.htr.experiment.models import (
    Experiment,
    ExperimentRun,
    ExperimentVersion,
    MethodRun,
    MetricDefinition,
    MetricResult,
    ReproducibilityManifest,
)
from archivetrust.htr.persistence import DurableHtrResearchStore
from archivetrust.htr.research_store import MethodRunTranscript

AT = "2026-07-30T09:00:00+00:00"
ARCHIVE_OBJECT_REF = "archive_object_synthetic_1"


@dataclass(frozen=True)
class RegisteredCorpus:
    """Everything one `register_small_corpus` call created, plus the event ids it emitted, so a test
    can assert on both the entities and the causal edges between the events announcing them."""

    project: ResearchProject
    dataset: Dataset
    collection: Collection
    dataset_version: DatasetVersion
    page: Page
    region: Region
    text_line: TextLine
    crop: InputCrop
    experiment: Experiment
    experiment_version: ExperimentVersion
    experiment_run: ExperimentRun
    completed_experiment_run: ExperimentRun
    method_run: MethodRun
    transcript: MethodRunTranscript
    metric_definition: MetricDefinition
    metric_result: MetricResult
    manifest: ReproducibilityManifest

    experiment_run_started_event_id: str
    method_run_started_event_id: str
    transcript_last_event_id: str
    metric_event_id: str


def register_small_corpus(store: DurableHtrResearchStore) -> RegisteredCorpus:
    """Registers one project / dataset / collection / dataset version / page / region / line / crop /
    experiment / experiment version / experiment run / method run (with all three machine transcript
    stages) / metric definition / metric result / reproducibility manifest.

    Every call passes `caused_by` explicitly, so the resulting event log carries a real causal DAG
    rather than events that merely happen to be adjacent in the file.
    """
    project = ResearchProject.create(name="Swedish court records", created_at=AT)
    project_event = store.register_project(project)

    dataset = Dataset.create(project_id=project.project_id, name="Göta hovrätt", created_at=AT)
    dataset_event = store.register_dataset(dataset, caused_by=project_event)

    collection = Collection.create(
        dataset_id=dataset.dataset_id,
        name="1841 batch",
        archive_object_refs=(ARCHIVE_OBJECT_REF,),
        created_at=AT,
    )
    collection_event = store.register_collection(collection, caused_by=dataset_event)

    dataset_version = DatasetVersion.create(
        dataset_id=dataset.dataset_id,
        version=1,
        collection_ids=(collection.collection_id,),
        created_at=AT,
    )
    dataset_version_event = store.register_dataset_version(
        dataset_version, caused_by=collection_event
    )

    page = Page.create(archive_object_ref=ARCHIVE_OBJECT_REF, page_number=1, width=2480, height=3508)
    page_event = store.register_page(page, caused_by=collection_event)

    region = Region.create(
        page_id=page.page_id,
        bounding_box=BoundingBox(x0=100, y0=200, x1=1900, y1=600, precision=Precision.PIXEL_ACCURATE),
        region_type="text_block",
        order_index=0,
    )
    region_event = store.register_region(region, caused_by=page_event)

    text_line = TextLine.create(
        region_id=region.region_id,
        bounding_box=BoundingBox(x0=100, y0=200, x1=1900, y1=260, precision=Precision.PIXEL_ACCURATE),
        reading_order_index=0,
    )
    line_event = store.register_text_line(text_line, caused_by=region_event)

    crop = InputCrop.create(
        image_bytes=b"synthetic-crop-bytes",
        text_line_id=text_line.text_line_id,
        storage_path="crops/synthetic.png",
        width=1800,
        height=60,
    )
    store.register_input_crop(crop, caused_by=line_event)

    experiment = Experiment.create(
        name="SATRN vs Florence-2",
        research_project_id=project.project_id,
        created_at=AT,
    )
    experiment_event = store.register_experiment(experiment, caused_by=project_event)

    experiment_version = ExperimentVersion.create(
        experiment_id=experiment.experiment_id,
        version=1,
        dataset_version_id=dataset_version.dataset_version_id,
        method_ids=("satrn", "florence2"),
        created_at=AT,
    )
    version_event = store.register_experiment_version(
        experiment_version, caused_by=experiment_event
    )
    assert dataset_version_event  # the version references it; keep the link asserted, not implied

    experiment_run = ExperimentRun.create(
        experiment_version_id=experiment_version.experiment_version_id,
        is_end_to_end=False,
        started_at=AT,
    )
    run_event = store.register_experiment_run(experiment_run, caused_by=version_event)

    # Everything below belongs to this run's unit of work and shares its correlation id.
    with store.correlated_to(experiment_run.experiment_run_id):
        method_run = MethodRun.create(
            experiment_run_id=experiment_run.experiment_run_id,
            method_id="satrn",
            evidence_id="evidence_synthetic_1",
            outcome="succeeded",
            started_at=AT,
            model_version_id="satrn-swedish-v1",
            input_crop_id=crop.crop_id,
            completed_at="2026-07-30T09:00:05+00:00",
        )
        method_run_event = store.register_method_run(method_run, caused_by=run_event)
        store.complete_method_run(method_run, caused_by=method_run_event)

        transcript = MethodRunTranscript(
            method_run_id=method_run.method_run_id,
            raw_text="Anno 1841 den 3 Martii",
            parsed_text="Anno 1841 den 3 Martii",
            normalized_text="anno 1841 den 3 martii",
        )
        transcript_event = store.register_transcript(
            transcript, caused_by=method_run_event, evidence_id="evidence_synthetic_1"
        )

        store.register_ground_truth(
            text_line_id=text_line.text_line_id, text="Anno 1841 den 3 Martii"
        )

        metric_definition = MetricDefinition.create(
            name="character_error_rate", version=1, higher_is_better=False
        )
        definition_event = store.register_metric_definition(metric_definition)

        metric_result = MetricResult.create(
            metric_definition_id=metric_definition.metric_definition_id,
            method_run_id=method_run.method_run_id,
            value=0.0417,
        )
        # Caused by the transcript stage whose text was scored, not by the definition.
        metric_event = store.register_metric_result(metric_result, caused_by=transcript_event)
        assert definition_event

        manifest = ReproducibilityManifest.create(
            experiment_run_id=experiment_run.experiment_run_id,
            created_at=AT,
            git_commit="0951e91",
            software_environment={"python": "3.11"},
            hardware_environment={"device": "cpu"},
            pipeline_configuration_hash="cfg_synthetic",
        )
        manifest_event = store.register_manifest(manifest, caused_by=metric_event)

        completed_run = experiment_run.model_copy(
            update={"completed_at": "2026-07-30T09:01:00+00:00"}
        )
        # Chained from the manifest, not forked off the metric: the run completes after its
        # reproducibility manifest is recorded, so the causal path stays a single line rather than
        # two same-length branches whose ordering would be arbitrary.
        store.complete_experiment_run(completed_run, caused_by=manifest_event)

    return RegisteredCorpus(
        project=project,
        dataset=dataset,
        collection=collection,
        dataset_version=dataset_version,
        page=page,
        region=region,
        text_line=text_line,
        crop=crop,
        experiment=experiment,
        experiment_version=experiment_version,
        experiment_run=experiment_run,
        completed_experiment_run=completed_run,
        method_run=method_run,
        transcript=transcript,
        metric_definition=metric_definition,
        metric_result=metric_result,
        manifest=manifest,
        experiment_run_started_event_id=run_event,
        method_run_started_event_id=method_run_event,
        transcript_last_event_id=transcript_event,
        metric_event_id=metric_event,
    )
