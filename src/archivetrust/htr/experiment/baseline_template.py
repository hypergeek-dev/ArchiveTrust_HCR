""""Swedish Historical HTR Baseline Comparison" experiment template
(docs/htr-migration-plan.md Stage 12; `docs/htr-domain-design.md` §1).

This module builds the **predefined experiment template** the brief asks for: a reusable
definition comparing SATRN, Florence-2, and Transkribus Swedish Lion I, supporting both a
controlled line-level comparison (identical line crops fed to every local recognizer, hash-
verified) and an end-to-end page-level comparison (each method's own pipeline). It does **not**
execute anything -- no adapter is imported here, no inference runs, no GPU is touched. Execution
lives in `htr/experiment/baseline_execution.py`, which imports this module's output and actually
runs it. This split mirrors `presentation/htr_experiment_builder_viewmodel.py`'s own
"build() returns constructed entities; registering/running them is the caller's decision" rule --
a template must be constructible and testable without a GPU.

**What is a real `ExperimentVersion` field vs. a template-only field.** `ExperimentVersion`
(`htr/experiment/models.py`) was frozen in an earlier stage with exactly two free-form
configuration hooks: `segmentation_configuration_ref` and `pipeline_configuration_ref`. The
brief's "Experiment definition" asks for many more fields than that record carries (research
question, hypothesis, inclusion/exclusion criteria, sampling strategy, acceptance criteria, ...).
Rather than smuggling a domain-model change into this stage, this module follows exactly the
precedent `ExperimentBuilderViewModel.PipelineConfiguration`/`SELECTION_NOT_MODELED` already set:
every field the brief asks for that has no dedicated `ExperimentVersion` column is captured on
`BaselineExperimentDefinition` below and serialized, deterministically, into
`ExperimentVersion.pipeline_configuration_ref` via `BaselineExperimentDefinition.to_ref()` -- nothing
is silently dropped, and nothing is invented as a new frozen-model field this late.

**Honest scope.** The brief's "dataset" field is filled honestly, not fabricated: this repository
has exactly one real, shared, ground-truthed HTR asset
(`tests/fixtures/htr/trolldomskommissionen_sample_line.jpg` + its ground-truth `.txt`) and one
hand-authored (not genuine-vendor-output) Transkribus PAGE XML fixture
(`tests/fixtures/transkribus/sample_page.xml`, see that directory's own README for provenance).
`dataset_description` says this plainly -- it does not claim a larger corpus exists.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from archivetrust.htr.experiment.models import Experiment, ExperimentVersion
from archivetrust.providers.florence2_htr.adapter import METHOD_ID as FLORENCE2_METHOD_ID
from archivetrust.providers.satrn.adapter import METHOD_ID as SATRN_METHOD_ID
from archivetrust.providers.transkribus.adapter import METHOD_ID as TRANSKRIBUS_METHOD_ID

BASELINE_TEMPLATE_TITLE = "Swedish Historical HTR Baseline Comparison"
BASELINE_TEMPLATE_VERSION = 1
"""Bumped whenever this template's *definition* (not a single run's captured values) changes
meaning -- mirrors `htr/evaluation/definitions.py::EVALUATION_ENGINE_VERSION`'s own convention."""


class BaselineExperimentDefinition(BaseModel):
    """Every "Experiment definition" field the brief names, for the predefined baseline template.
    Real values are used wherever this repository actually has them (real model revisions read
    from each adapter's own `get_metadata()`, real hardware/software requirements read from this
    project's own README files); genuinely unavailable values are honest placeholders/TBDs, never
    fabricated numbers (see `scope_caveats`)."""

    model_config = ConfigDict(frozen=True)

    title: str = BASELINE_TEMPLATE_TITLE
    research_question: str
    hypothesis: str
    dataset_description: str
    dataset_version_note: str
    inclusion_criteria: str
    exclusion_criteria: str
    sampling_strategy: str
    transcription_convention: str
    ground_truth_requirements: str
    method_ids: tuple[str, ...]
    method_versions: dict[str, str]
    """`method_id -> model_revision`, read from each adapter's real `get_metadata()` at template-
    build time -- not hand-typed version strings that could drift from the adapter."""
    segmentation_strategy: str
    controlled_variables: tuple[str, ...]
    independent_variables: tuple[str, ...]
    dependent_variables: tuple[str, ...]
    metric_names: tuple[str, ...]
    failure_policy: str
    acceptance_criteria: str
    hardware_environment_requirements: dict[str, str]
    software_environment_requirements: dict[str, str]
    random_seed: int
    reporting_strategy: str
    scope_caveats: tuple[str, ...] = ()
    """Explicit, honest "what this template does NOT have yet" notes -- e.g. "dataset is one
    document, not a corpus" -- so a reader of the definition sees the scope-down decisions instead
    of inferring absence from silence."""

    def to_ref(self) -> str:
        """A deterministic, sorted JSON string -- what actually lands in
        `ExperimentVersion.pipeline_configuration_ref` (see module docstring). Two definitions
        built with identical field values always produce an identical ref."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


class BuiltBaselineExperiment(BaseModel):
    """What `build_baseline_experiment()` returns -- the real `Experiment` + `ExperimentVersion`
    plus the full definition that produced `pipeline_configuration_ref`, so a caller (the
    execution module, or a test) can inspect either the compact real entities or the rich
    definition without re-deriving one from the other."""

    model_config = ConfigDict(frozen=True)

    experiment: Experiment
    experiment_version: ExperimentVersion
    definition: BaselineExperimentDefinition


def default_baseline_definition(
    *,
    satrn_model_revision: str,
    florence2_model_revision: str,
) -> BaselineExperimentDefinition:
    """The predefined baseline template's definition. `satrn_model_revision`/
    `florence2_model_revision` are passed in by the caller (read from each adapter's real
    `MethodMetadata.model_revision`, e.g. `SatrnAdapter().get_metadata().model_revision`) rather
    than hand-typed here, so this function never silently drifts from what the adapters actually
    report. Transkribus's method-version field is filled with its own adapter's honest
    "unpinned -- export declares no fixed checkpoint hash" string (see
    `providers/transkribus/adapter.py::DEFAULT_METADATA_MODEL_REVISION`) rather than invented.
    """
    from archivetrust.providers.transkribus.adapter import DEFAULT_METADATA_MODEL_REVISION

    return BaselineExperimentDefinition(
        research_question=(
            "How do SATRN (Riksarkivet), Florence-2 (hoanghapham/vlm-htr fine-tuned OCR "
            "checkpoint), and Transkribus Swedish Lion I compare on Swedish historical "
            "handwriting recognition, under (a) a controlled line-level comparison where the "
            "local recognizers read byte-identical input crops, and (b) an end-to-end, "
            "page-level comparison where each method's own pipeline (including Transkribus's "
            "own segmentation, run outside this application) is used as-is?"
        ),
        hypothesis=(
            "No directional hypothesis is asserted about which method performs better -- the "
            "brief's own instruction is not to make broad general-performance claims from a "
            "small sample. The falsifiable prediction this template's first run is designed to "
            "test is procedural, not competitive: that the controlled-comparison pipeline can "
            "feed two independent local HTR adapters the exact same input-crop bytes (verified "
            "by a matching content hash) and produce two real, non-empty, independently computed "
            "recognition results plus honestly-computed CER/WER against ground truth."
        ),
        dataset_description=(
            "Exactly one real, shared, ground-truthed HTR asset exists in this repository at "
            "template-authoring time: tests/fixtures/htr/trolldomskommissionen_sample_line.jpg "
            "(a single line from Riksarkivet's trolldomskommissionen_lines dataset, 17th-century "
            "Swedish court-record handwriting) plus its ground-truth transcription .txt file. "
            "A separate, hand-authored (not genuine Transkribus vendor output -- see "
            "tests/fixtures/transkribus/README.md) PAGE XML fixture, sample_page.xml, supplies "
            "the page-level, non-corresponding Transkribus input. This is NOT a multi-document "
            "research corpus; see scope_caveats."
        ),
        dataset_version_note=(
            "DatasetVersion.version=1, collection_ids referencing one Collection containing one "
            "ArchiveObject reference for the shared line fixture. No second DatasetVersion has "
            "ever been created (would require a membership change), consistent with "
            "docs/htr-domain-design.md §3's 'immutable snapshot' rule."
        ),
        inclusion_criteria=(
            "Any TextLine with (a) a real, unmodified source image crop and (b) a real, "
            "human-transcribed ground-truth string is eligible for the controlled comparison. "
            "The Transkribus page-level fixture is eligible for the end-to-end comparison only "
            "(see exclusion_criteria) because it has no known geometric correspondence to the "
            "controlled fixture's ground truth."
        ),
        exclusion_criteria=(
            "A method run is excluded from the controlled-comparison hash-matched set if its "
            "input crop's content hash does not match every other controlled-run's input crop "
            "hash for the same TextLine (enforced by an assertion at execution time, "
            "docs/htr-domain-design.md §7). Transkribus's page-level result is unconditionally "
            "excluded from the controlled set and from any line-level CER computed against the "
            "shared fixture's ground truth, because its fixture's transcribed content is a "
            "different, unrelated Swedish court-record phrase with no established line-to-line "
            "correspondence to that ground truth -- computing a CER between them would compare "
            "two unrelated texts, not measure recognition accuracy."
        ),
        sampling_strategy=(
            "No sampling is performed -- every eligible input (the one real line, the one "
            "hand-authored page) is used. Sampling strategy is TBD for a future run against a "
            "larger corpus; ExperimentBuilderViewModel.PipelineConfiguration already models "
            "sample_size/sample_seed for when one exists."
        ),
        transcription_convention=(
            "TBD -- no versioned TranscriptionConvention record has been created for this "
            "template's ground truth. The reference text used is exactly the upstream "
            "Riksarkivet dataset's published transcription "
            "(tests/fixtures/htr/README.md), including its own special-character convention for "
            "uncertain characters, taken as-is rather than re-transcribed under a "
            "project-defined convention."
        ),
        ground_truth_requirements=(
            "One human-produced reference transcription per TextLine, taken verbatim from the "
            "upstream published dataset for the controlled fixture. No blind dual-annotation + "
            "adjudication was performed for this ground truth (docs/htr-domain-design.md's "
            "ReviewAssignment/AgreementResult/Adjudication workflow exists but was not exercised "
            "here) -- it is a single external annotation, not this project's own adjudicated "
            "ground truth. The Transkribus page-level fixture has no ground truth at all (its "
            "TextEquiv content is itself a hand-authored stand-in for a recognition result, not "
            "a reference transcription)."
        ),
        method_ids=(SATRN_METHOD_ID, FLORENCE2_METHOD_ID, TRANSKRIBUS_METHOD_ID),
        method_versions={
            SATRN_METHOD_ID: satrn_model_revision,
            FLORENCE2_METHOD_ID: florence2_model_revision,
            TRANSKRIBUS_METHOD_ID: DEFAULT_METADATA_MODEL_REVISION,
        },
        segmentation_strategy=(
            "Controlled comparison: no segmentation stage runs at all -- the shared fixture is "
            "already a single, pre-cropped line image, so its InputCrop is built directly from "
            "the fixture's own bytes rather than produced by a htr/segmentation/"
            "SegmentationAdapter. End-to-end comparison: Transkribus's own (external, "
            "already-completed, not re-run by this application) segmentation is used as-is -- "
            "its PAGE XML fixture's TextRegion/TextLine/Coords geometry is exactly what "
            "Transkribus itself produced (or, for this hand-authored fixture, what stands in "
            "for it)."
        ),
        controlled_variables=(
            "input_crop_hash (identical bytes across SATRN and Florence-2, asserted equal)",
            "device_request='auto' (both adapters resolve the same CUDA GPU when available)",
        ),
        independent_variables=("method_id (satrn | florence2_htr | transkribus_swedish_lion_1)",),
        dependent_variables=(
            "character_error_rate_raw",
            "character_error_rate_normalized",
            "word_error_rate_raw",
            "word_error_rate_normalized",
            "exact_word_accuracy",
            "reported confidence (method-specific meaning -- see each adapter's README)",
            "execution_time_ms / gpu_memory_mb (controlled-comparison methods only)",
        ),
        metric_names=(
            "character_error_rate_raw",
            "character_error_rate_normalized",
            "word_error_rate_raw",
            "word_error_rate_normalized",
            "exact_word_accuracy",
            "execution_time_ms",
            "gpu_memory_mb",
        ),
        failure_policy="preserve_and_continue",
        acceptance_criteria=(
            "This run is accepted as a valid smoke-scale pipeline demonstration (NOT as a "
            "general-performance claim) if: (1) both SATRN and Florence-2 produce a real, "
            "non-empty RecognitionResult.text from the same input-crop hash; (2) the input-crop "
            "hash used by both is verified identical by assertion, not merely assumed; (3) "
            "CER/WER is computed for both against the real ground truth; (4) the Transkribus "
            "page-level result is present and explicitly labeled as excluded from the "
            "controlled/hash-matched set; (5) a ReproducibilityManifest with real (non-"
            "placeholder) environment fields is produced."
        ),
        hardware_environment_requirements={
            "gpu": "CUDA-capable GPU recommended (both SATRN and Florence-2 fall back to CPU; "
            "see each adapter's README for measured CPU-vs-GPU timing)",
            "satrn_runtime": ".venv-satrn: an isolated Python 3.10 environment with torch==2.1.0, "
            "mmengine, mmcv==2.1.0, mmdet==3.1.0, mmocr (see providers/satrn/README.md)",
            "florence2_runtime": "this project's main venv with the 'transformers' extra "
            "(torch, transformers==4.49.0, timm, einops)",
        },
        software_environment_requirements={
            "python": ">=3.11 (main venv); SATRN's isolated .venv-satrn is Python 3.10",
            "transformers": "==4.49.0 (Florence-2; verified working, >=5.0 verified broken -- "
            "see providers/florence2_htr/README.md)",
            "mmocr": "==1.0.1 (SATRN, isolated venv only)",
        },
        random_seed=0,
        reporting_strategy=(
            "JSON (round-trippable) + CSV (lossy, spreadsheet-friendly) via "
            "research/reports/export.py, written under "
            "docs/experiments/baseline-comparison/. Findings are labeled preliminary/N=1 "
            "throughout -- see docs/experiments/baseline-comparison/README.md's explicit caveat, "
            "per the brief's 'do not make broad general-performance claims from a small sample' "
            "instruction."
        ),
        scope_caveats=(
            "The 'dataset' is one line image, not a corpus of documents -- Dataset/"
            "DatasetVersion/Collection/Document/Page/Region/TextLine records are built for "
            "structural correctness, not because a large corpus exists.",
            "No TranscriptionConvention record or blind dual-annotation ground-truth workflow "
            "was exercised for this template's single ground-truth string.",
            "The Transkribus fixture is hand-authored (not genuine Transkribus vendor output) "
            "per tests/fixtures/transkribus/README.md -- its result is real adapter output "
            "(really parsed from a real file), but the file itself is a test fixture, not a "
            "live Transkribus export.",
            "No sampling strategy, no CanonicalResult (cross-method selection), and no review/ "
            "adjudication workflow were exercised in this template's first run.",
        ),
    )


def build_baseline_experiment(
    definition: BaselineExperimentDefinition,
    *,
    research_project_id: str,
    dataset_version_id: str,
    created_at: str,
    version: int = BASELINE_TEMPLATE_VERSION,
) -> BuiltBaselineExperiment:
    """Constructs the real `Experiment` + `ExperimentVersion` for the baseline template. Pure --
    no adapter import, no I/O, no randomness beyond what `Experiment.create`/`ExperimentVersion.
    create` already use for id generation (`domain/shared/ids.py::new_id`, not seeded by
    `definition.random_seed` -- that seed governs sampling/inference determinism, not entity ids).
    """
    experiment = Experiment.create(
        name=definition.title,
        research_project_id=research_project_id,
        description=definition.research_question,
        created_at=created_at,
    )
    experiment_version = ExperimentVersion.create(
        experiment_id=experiment.experiment_id,
        version=version,
        dataset_version_id=dataset_version_id,
        method_ids=definition.method_ids,
        segmentation_configuration_ref=definition.segmentation_strategy,
        pipeline_configuration_ref=definition.to_ref(),
        created_at=created_at,
    )
    return BuiltBaselineExperiment(
        experiment=experiment,
        experiment_version=experiment_version,
        definition=definition,
    )
