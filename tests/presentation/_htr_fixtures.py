"""Shared fixture corpus for the Stage 11 HTR research-interface ViewModel tests.

Builds a small but *real* corpus out of the actual domain entities -- no mocks, no stub models --
so every assertion in the HTR ViewModel tests is made against the same Pydantic types the
application constructs at runtime.

Shape of the fixture (deliberately small enough to reason about, rich enough to exercise every
branch the ViewModels have):

    project "Swedish Historical HTR"
      dataset "Trolldomskommissionen" -> version 1 -> collection "Court records 1670s"
        document archive_object_court_1
          page 1
            region "text_block"
              line 0  -- crop A, shared by all three method runs (controlled comparison)
              line 1  -- crop B, one succeeded run + one FAILED run

    method runs on line 0: satrn (succeeded), florence2_htr (succeeded),
                           transkribus_swedish_lion_1 (succeeded)
    method runs on line 1: satrn (succeeded), florence2_htr (failed, with a FailureRecord)

    a CanonicalResult over page 1 selecting SATRN for line 0
    a Transkribus ExternalImport wrapping the transkribus method run
    ground truth on both lines
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.canonical.result import (
    CanonicalResult,
    CanonicalResultSpan,
    CanonicalizationStrategy,
)
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
    FailureRecord,
    MethodRun,
    ReproducibilityManifest,
)
from archivetrust.htr.research_store import HtrResearchStore, MethodRunTranscript
from archivetrust.providers.htr_adapter import (
    EnvironmentValidation,
    HealthCheckResult,
    MethodCapabilities,
    MethodMetadata,
    RecognitionInput,
    RecognitionResult,
)
from archivetrust.providers.transkribus.external_import import (
    ExternalImport,
    ExternalImportFormat,
)

ARCHIVE_OBJECT_REF = "archive_object_court_1"
GROUND_TRUTH_LINE_0 = "till den 23 Januarii"
GROUND_TRUTH_LINE_1 = "waritt i Stockholm"


class FixtureCorpus(BaseModel):
    """Ids of the entities the fixture registered, so a test names what it asserts on instead of
    re-deriving ids by searching the store."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    store: HtrResearchStore
    project_id: str
    dataset_id: str
    dataset_version_id: str
    collection_id: str
    page_id: str
    region_id: str
    line_0_id: str
    line_1_id: str
    crop_a_id: str
    crop_b_id: str
    crop_a_hash: str
    experiment_id: str
    experiment_version_id: str
    experiment_run_id: str
    satrn_run_line_0: str
    florence_run_line_0: str
    transkribus_run_line_0: str
    satrn_run_line_1: str
    florence_run_line_1_failed: str
    canonical_result_id: str
    external_import_id: str


def build_fixture_corpus() -> FixtureCorpus:
    """Constructs and registers the fixture described in this module's docstring."""
    store = HtrResearchStore()
    box = BoundingBox(x0=10, y0=20, x1=400, y1=60, precision=Precision.PIXEL_ACCURATE)

    project = ResearchProject.create(
        name="Swedish Historical HTR",
        description="Baseline comparison of SATRN, Florence-2 and Transkribus.",
        created_at="2026-07-01T00:00:00Z",
    )
    store.register_project(project)

    dataset = Dataset.create(
        project_id=project.project_id,
        name="Trolldomskommissionen",
        description="17th-century Swedish court records.",
        created_at="2026-07-02T00:00:00Z",
    )
    store.register_dataset(dataset)

    collection = Collection.create(
        dataset_id=dataset.dataset_id,
        name="Court records 1670s",
        archive_object_refs=(ARCHIVE_OBJECT_REF,),
        created_at="2026-07-02T01:00:00Z",
    )
    store.register_collection(collection)

    version = DatasetVersion.create(
        dataset_id=dataset.dataset_id,
        version=1,
        collection_ids=(collection.collection_id,),
        created_at="2026-07-02T02:00:00Z",
    )
    store.register_dataset_version(version)

    page = Page.create(
        archive_object_ref=ARCHIVE_OBJECT_REF, page_number=1, width=2480, height=3508
    )
    store.register_page(page)

    region = Region.create(
        page_id=page.page_id, bounding_box=box, region_type="text_block", order_index=0
    )
    store.register_region(region)

    line_0 = TextLine.create(region_id=region.region_id, bounding_box=box, reading_order_index=0)
    line_1 = TextLine.create(region_id=region.region_id, bounding_box=box, reading_order_index=1)
    store.register_text_line(line_0)
    store.register_text_line(line_1)

    crop_a = InputCrop.create(
        image_bytes=b"line-zero-image-bytes",
        text_line_id=line_0.text_line_id,
        storage_path="crops/a.png",
        width=390,
        height=40,
    )
    crop_b = InputCrop.create(
        image_bytes=b"line-one-image-bytes",
        text_line_id=line_1.text_line_id,
        storage_path="crops/b.png",
    )
    store.register_input_crop(crop_a)
    store.register_input_crop(crop_b)

    store.register_ground_truth(text_line_id=line_0.text_line_id, text=GROUND_TRUTH_LINE_0)
    store.register_ground_truth(text_line_id=line_1.text_line_id, text=GROUND_TRUTH_LINE_1)

    experiment = Experiment.create(
        name="SATRN vs Florence-2 vs Transkribus",
        research_project_id=project.project_id,
        created_at="2026-07-03T00:00:00Z",
    )
    store.register_experiment(experiment)

    experiment_version = ExperimentVersion.create(
        experiment_id=experiment.experiment_id,
        version=1,
        dataset_version_id=version.dataset_version_id,
        method_ids=("satrn", "florence2_htr", "transkribus_swedish_lion_1"),
        segmentation_configuration_ref="segmentation_baseline_v1",
        created_at="2026-07-03T00:10:00Z",
    )
    store.register_experiment_version(experiment_version)

    experiment_run = ExperimentRun.create(
        experiment_version_id=experiment_version.experiment_version_id,
        is_end_to_end=False,
        started_at="2026-07-03T01:00:00Z",
        completed_at="2026-07-03T01:30:00Z",
    )
    store.register_experiment_run(experiment_run)
    store.register_manifest(
        ReproducibilityManifest.create(
            experiment_run_id=experiment_run.experiment_run_id,
            created_at="2026-07-03T01:30:01Z",
            git_commit="0951e91",
            software_environment={"torch": "2.1.0"},
            hardware_environment={"gpu_name": "RTX 3070"},
        )
    )

    def _run(
        *,
        method_id: str,
        crop_id: str,
        outcome: str,
        model_version_id: str | None,
        started_at: str,
    ) -> MethodRun:
        run = MethodRun.create(
            experiment_run_id=experiment_run.experiment_run_id,
            method_id=method_id,
            evidence_id=f"evidence_{method_id}_{crop_id[-6:]}",
            outcome=outcome,
            started_at=started_at,
            model_version_id=model_version_id,
            input_crop_id=crop_id,
            completed_at=started_at,
        )
        store.register_method_run(run)
        return run

    satrn_0 = _run(
        method_id="satrn",
        crop_id=crop_a.crop_id,
        outcome="succeeded",
        model_version_id="a40c7093232eaa47a83ce6469fc4abd033486bdc",
        started_at="2026-07-03T01:01:00Z",
    )
    # SATRN's own README records this exact string as its real measured output on a real
    # 17th-century Swedish line -- so the fixture's "correct" transcription is not invented.
    store.register_transcript(
        MethodRunTranscript(
            method_run_id=satrn_0.method_run_id,
            raw_text="till den 23 Januarii ",
            parsed_text="till den 23 Januarii ",
            normalized_text="till den 23 Januarii",
            reviewed_text="till den 23 Januarii",
            reviewer_ref="reviewer_a",
        )
    )

    florence_0 = _run(
        method_id="florence2_htr",
        crop_id=crop_a.crop_id,
        outcome="succeeded",
        model_version_id="nazounoryuu/florence_base__mixed__line_bbox__ocr@main",
        started_at="2026-07-03T01:02:00Z",
    )
    store.register_transcript(
        MethodRunTranscript(
            method_run_id=florence_0.method_run_id,
            raw_text="</s><s>till den 23 Januari</s>",
            parsed_text="till den 23 Januari",
            normalized_text="till den 23 Januari",
        )
    )

    transkribus_0 = _run(
        method_id="transkribus_swedish_lion_1",
        crop_id=crop_a.crop_id,
        outcome="succeeded",
        model_version_id="swedish_lion_1",
        started_at="2026-07-03T01:03:00Z",
    )
    store.register_transcript(
        MethodRunTranscript(
            method_run_id=transkribus_0.method_run_id,
            raw_text="till den 23 Januarii",
            parsed_text="till den 23 Januarii",
            normalized_text="till den 23 Januarii",
        )
    )

    satrn_1 = _run(
        method_id="satrn",
        crop_id=crop_b.crop_id,
        outcome="succeeded",
        model_version_id="a40c7093232eaa47a83ce6469fc4abd033486bdc",
        started_at="2026-07-03T01:04:00Z",
    )
    store.register_transcript(
        MethodRunTranscript(
            method_run_id=satrn_1.method_run_id,
            raw_text="waritt i Stockholm",
            parsed_text="waritt i Stockholm",
            normalized_text="waritt i Stockholm",
        )
    )

    florence_1 = _run(
        method_id="florence2_htr",
        crop_id=crop_b.crop_id,
        outcome="failed",
        model_version_id=None,
        started_at="2026-07-03T01:05:00Z",
    )
    store.register_failure(
        FailureRecord.create(
            method_run_id=florence_1.method_run_id,
            reason="CUDA out of memory while loading the fine-tuned checkpoint",
            category="cuda_oom",
        )
    )

    canonical = CanonicalResult.create(
        page_id=page.page_id,
        strategy=CanonicalizationStrategy.HUMAN_APPROVED_SELECTION,
        strategy_version=1,
        spans=(
            CanonicalResultSpan(
                text_line_id=line_0.text_line_id,
                source_method_run_id=satrn_0.method_run_id,
                text="till den 23 Januarii",
            ),
        ),
        created_at="2026-07-03T02:00:00Z",
    )
    store.register_canonical_result(canonical)

    external = ExternalImport.create(
        method_run_id=transkribus_0.method_run_id,
        source_file_path="exports/transkribus_court_1.xml",
        export_format=ExternalImportFormat.PAGE_XML,
        imported_by="researcher_1",
        imported_at="2026-07-03T00:50:00Z",
        transkribus_document_id="12345",
        vendor_reported_accuracy=0.94,
    )
    store.register_external_import(external)

    return FixtureCorpus(
        store=store,
        project_id=project.project_id,
        dataset_id=dataset.dataset_id,
        dataset_version_id=version.dataset_version_id,
        collection_id=collection.collection_id,
        page_id=page.page_id,
        region_id=region.region_id,
        line_0_id=line_0.text_line_id,
        line_1_id=line_1.text_line_id,
        crop_a_id=crop_a.crop_id,
        crop_b_id=crop_b.crop_id,
        crop_a_hash=crop_a.hash,
        experiment_id=experiment.experiment_id,
        experiment_version_id=experiment_version.experiment_version_id,
        experiment_run_id=experiment_run.experiment_run_id,
        satrn_run_line_0=satrn_0.method_run_id,
        florence_run_line_0=florence_0.method_run_id,
        transkribus_run_line_0=transkribus_0.method_run_id,
        satrn_run_line_1=satrn_1.method_run_id,
        florence_run_line_1_failed=florence_1.method_run_id,
        canonical_result_id=canonical.canonical_result_id,
        external_import_id=external.external_import_id,
    )


class StubAdapter:
    """A real `HtrMethodAdapter` implementation (structurally -- the Protocol is
    `runtime_checkable`), used where a test needs a method whose environment probe is deterministic.
    Not a mock: every method returns a genuine, validated model instance.
    """

    def __init__(
        self,
        *,
        method_id: str,
        method_name: str,
        vendor: str,
        model_revision: str,
        local: bool,
        external_upload: bool,
        confidence: bool = True,
        valid: bool = True,
        healthy: bool = True,
    ) -> None:
        self._metadata = MethodMetadata(
            method_id=method_id,
            method_name=method_name,
            vendor=vendor,
            model_revision=model_revision,
        )
        self._capabilities = MethodCapabilities(
            confidence_supported=confidence,
            geometry_supported=False,
            line_level_supported=True,
            page_level_supported=False,
            local_execution_supported=local,
            external_upload_required=external_upload,
        )
        self._valid = valid
        self._healthy = healthy

    def get_metadata(self) -> MethodMetadata:
        return self._metadata

    def get_capabilities(self) -> MethodCapabilities:
        return self._capabilities

    def validate_environment(self) -> EnvironmentValidation:
        return EnvironmentValidation(
            valid=self._valid,
            messages=() if self._valid else ("model weights not found",),
        )

    def recognize(self, input: RecognitionInput) -> RecognitionResult:
        return RecognitionResult(text=None)

    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=self._healthy, message="stub probe")
