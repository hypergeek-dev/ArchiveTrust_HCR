"""The ground-truth annotation store: append-only, provenance-first, supersession-only edits.

Mirrors the storage discipline every other durable stream in this codebase follows (JSONL, one
record per line, the file is the single source of truth, corrections happen by writing a new
record that `supersedes` an old one — never by rewriting history).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator
from archivetrust.domain.evidence.models import BoundingBox

GROUND_TRUTH_ANNOTATION_SCHEMA = "archivetrust.evaluation_reference_annotation.v2"


class AdjudicationStatus(str, Enum):
    UNADJUDICATED = "unadjudicated"
    """A single annotator's reading, not independently checked — the honest default."""
    ADJUDICATED = "adjudicated"
    """Confirmed by a second reviewer or an adjudication pass."""
    DISPUTED = "disputed"
    """Two readings disagree and no adjudication has resolved them; excluded from evaluation."""


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified_evaluation_reference"
    EXCLUDED = "excluded"


class EvaluationReferenceStatus(str, Enum):
    DRAFTED = "drafted"
    ASSIGNED = "assigned"
    INDEPENDENTLY_ANNOTATED = "independently_annotated"
    AWAITING_SECOND_REVIEW = "awaiting_second_review"
    DISAGREEMENT = "disagreement"
    AWAITING_ADJUDICATION = "awaiting_adjudication"
    ADJUDICATED = "adjudicated"
    VERIFIED = "verified_evaluation_reference"
    EXCLUDED = "excluded"
    SUPERSEDED = "superseded"


class ReviewerIndependenceState(str, Enum):
    SINGLE_REVIEW = "single_review"
    INDEPENDENT = "independent"
    ADJUDICATOR = "adjudicator"


class BlindingState(str, Enum):
    BLINDED = "blinded"
    UNBLINDED = "unblinded"
    LEGACY_UNKNOWN = "legacy_unknown"


class AnnotatorKind(str, Enum):
    HUMAN = "human"
    AI_ASSISTED = "ai_assisted"
    AI = "ai"


class GroundTruthGranularity(str, Enum):
    """The segment level a `GroundTruthAnnotation` was made at (docs/htr-domain-design.md §1:
    "GroundTruthItem (page | region | line | word | entity | custom segment)"). Distinct from the
    pre-existing free-form `scope` field, which names *what part of the payload* was annotated
    (e.g. `"field"`); `granularity` names the *segmentation level*, needed once line-level HTR
    ground truth exists alongside the pre-existing page/paragraph-level annotations. Optional and
    defaulted so every existing `GroundTruthAnnotation` construction call site keeps working
    unmodified."""

    PAGE = "page"
    REGION = "region"
    LINE = "line"
    WORD = "word"
    ENTITY = "entity"
    CUSTOM_SEGMENT = "custom_segment"


class TranscriptionConvention(BaseModel):
    """A versioned transcription convention (docs/htr-domain-design.md §1, §3): the rules an
    annotator follows when transcribing historical handwriting (e.g. abbreviation expansion,
    superscript handling, damaged-text markup). Versioned and immutable once referenced: "once any
    GroundTruthItem references a (convention_id, version) pair, that pair is frozen -- a change
    creates a new version, and existing ground truth keeps pointing at the old one" (§3). This
    type carries no enforcement of that freeze itself (it has no way to know what references it);
    enforcement is the same store-level responsibility `FileGroundTruthStore` already exercises
    for `supersedes` chains.
    """

    model_config = ConfigDict(frozen=True)

    convention_id: str
    version: int
    name: str
    rules: str
    """The convention text itself (free-form, e.g. Markdown) -- rules a human annotator follows."""
    created_at: str
    supersedes_version: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> "TranscriptionConvention":
        if self.version < 1:
            raise ValueError("TranscriptionConvention.version must be >= 1")
        return self


class GroundTruthAnnotation(BaseModel):
    """One curator-authored fact about one immutable Archive Object, at a named scope.

    `field` names *what* was annotated (e.g. `page1_heading`, `page1_first_paragraph`) — the
    population every metric computed from these annotations describes. `content_hash` ties the
    annotation to the exact bytes that were read; a hash mismatch means the annotation describes
    a different document and must never be scored.
    """

    schema_id: str = Field(default=GROUND_TRUTH_ANNOTATION_SCHEMA, alias="schema")
    annotation_id: str
    archive_object_ref: str
    content_hash: str
    page: int
    field: str
    observation_type: str
    """Which ontology type this field is evaluated against (e.g. "heading", "paragraph")."""
    text: str | None
    """The human reading. `None` is legal only with `illegible=True`."""
    uncertain: bool = False
    """The annotator could read it but is not certain — kept, reported separately."""
    illegible: bool = False
    """The annotator could not read it; scored as "not evaluable", never as an empty string."""
    annotator: str
    method: str
    """How the reading was produced (e.g. "visual transcription from rendered page image")."""
    source: str
    """Where the reading came from (campaign name / provenance of the annotator's access)."""
    created_at: str
    revision: int = 1
    supersedes: str | None = None
    """`annotation_id` of the record this one corrects. Append-only: the superseded record stays
    in the file; readers resolve the chain."""
    adjudication_status: AdjudicationStatus = AdjudicationStatus.UNADJUDICATED
    notes: str | None = None
    region_geometry: BoundingBox | None = None
    semantic_slot_id: str | None = None
    task_type: str = "transcription"
    scope: str = "field"
    proposed_value: str | None = None
    submitted_value: str | None = None
    assignment_id: str | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    reference_status: EvaluationReferenceStatus = EvaluationReferenceStatus.DRAFTED
    reviewer_independence_state: ReviewerIndependenceState = (
        ReviewerIndependenceState.SINGLE_REVIEW
    )
    blinding_state: BlindingState = BlindingState.LEGACY_UNKNOWN
    campaign_id: str | None = None
    sampling_stratum_id: str | None = None
    exclusion_reason: str | None = None
    annotator_kind: AnnotatorKind = AnnotatorKind.HUMAN
    adjudicates: tuple[str, ...] = ()
    legal_basis: str | None = None
    sampling_basis: str | None = None
    granularity: "GroundTruthGranularity | None" = None
    """Segment level this annotation was made at (docs/htr-domain-design.md §1). Optional and
    defaulted so every existing GroundTruthAnnotation construction call site keeps working
    unmodified (migration Stage 1) -- distinct from the pre-existing `scope` field, which names
    *what part of the payload* was annotated, not the segmentation level."""
    convention_id: str | None = None
    convention_version: int | None = None
    """Ties this annotation to a frozen `(convention_id, convention_version)` pair
    (docs/htr-domain-design.md §3) -- both `None` for annotations made before
    `TranscriptionConvention` existed, never backfilled."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class GroundTruthValidationError(ValueError):
    """Raised when an annotation cannot be accepted (identity mismatch, broken supersession)."""


class FileGroundTruthStore:
    """Append-only JSONL store of `GroundTruthAnnotation` records for one campaign."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    @property
    def path(self) -> Path:
        return self._path

    def all_records(self) -> tuple[GroundTruthAnnotation, ...]:
        records = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(GroundTruthAnnotation.model_validate_json(line))
        return tuple(records)

    def append(
        self,
        annotation: GroundTruthAnnotation,
        *,
        expected_content_hash: str | None = None,
    ) -> None:
        """Validates then appends. `expected_content_hash` is the acquisition record's hash for
        the annotation's Archive Object, when the caller can supply it — a mismatch is refused,
        never silently stored."""
        if annotation.illegible and annotation.text is not None:
            raise GroundTruthValidationError("illegible=True requires text=None")
        if (
            not annotation.illegible
            and annotation.text is None
            and annotation.exclusion_reason is None
        ):
            raise GroundTruthValidationError(
                "text=None is only legal for illegible or explicitly excluded annotations"
            )
        if expected_content_hash is not None and annotation.content_hash != expected_content_hash:
            raise GroundTruthValidationError(
                f"content hash mismatch for {annotation.archive_object_ref}: annotation says "
                f"{annotation.content_hash[:12]}…, the immutable archive object is "
                f"{expected_content_hash[:12]}… — this annotation describes a different document"
            )
        if annotation.verification_status is VerificationStatus.VERIFIED and (
            not annotation.legal_basis or not annotation.sampling_basis
        ):
            raise GroundTruthValidationError(
                "verified evaluation references require legal_basis and sampling_basis"
            )
        existing_ids = {record.annotation_id for record in self.all_records()}
        if annotation.annotation_id in existing_ids:
            raise GroundTruthValidationError(f"duplicate annotation_id {annotation.annotation_id}")
        if annotation.supersedes is not None and annotation.supersedes not in existing_ids:
            raise GroundTruthValidationError(
                f"supersedes unknown annotation {annotation.supersedes}"
            )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(annotation.model_dump_json(by_alias=True))
            handle.write("\n")

    def latest(self) -> tuple[GroundTruthAnnotation, ...]:
        """Every annotation that has not been superseded, in file order — the evaluable set
        (callers additionally filter `illegible` and `DISPUTED` as their metrics require)."""
        records = self.all_records()
        superseded = {r.supersedes for r in records if r.supersedes is not None}
        return tuple(r for r in records if r.annotation_id not in superseded)

    def iter_for_document(self, archive_object_ref: str) -> Iterator[GroundTruthAnnotation]:
        return (r for r in self.latest() if r.archive_object_ref == archive_object_ref)

    def verified_references(
        self, *, include_ai: bool = False
    ) -> tuple[GroundTruthAnnotation, ...]:
        """Return evaluation-eligible references.

        Human-independent references are the default population. AI and AI-assisted records
        remain queryable, but a caller must opt in explicitly so they cannot be silently mixed
        into human evaluation results.
        """

        return tuple(
            record
            for record in self.latest()
            if record.verification_status is VerificationStatus.VERIFIED
            and record.adjudication_status is not AdjudicationStatus.DISPUTED
            and not record.illegible
            and record.exclusion_reason is None
            and (include_ai or record.annotator_kind is AnnotatorKind.HUMAN)
        )

    def export_with_provenance(self, output_path: Path | str) -> Path:
        """Writes the resolved (latest) dataset plus a provenance block naming every superseded
        revision, with a file-digest sidecar for the export itself."""
        from archivetrust.infrastructure.storage.integrity import write_file_digest_manifest

        output = Path(output_path)
        records = self.all_records()
        superseded = {r.supersedes for r in records if r.supersedes is not None}
        payload = {
            "schema": "archivetrust.evaluation_reference_dataset.v2",
            "source_store": self._path.name,
            "total_records": len(records),
            "superseded_records": sorted(r for r in superseded if r),
            "annotations": [
                json.loads(r.model_dump_json(by_alias=True))
                for r in records
                if r.annotation_id not in superseded
            ],
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_file_digest_manifest(output)
        return output
