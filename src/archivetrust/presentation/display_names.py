"""Researcher-facing names for internal ArchiveTrust identifiers.

**HTR vocabulary (Stage 11 terminology sweep).** "HTR" is the main technical term throughout this
application: the three things a researcher compares are **methods** (SATRN, Florence-2, Transkribus
Swedish Lion I), not "OCR providers". `method_label` is the primary entry point; `provider_label`
remains as an alias because `providers/base.py`'s `ProviderAdapter`/`ProviderRegistry` were
deliberately *not* deleted (see that module's docstring and
`docs/htr-repository-cleanup.md`'s providers table) and the legacy provider-management surfaces in
`clients/desktop/pages/` still administer them under that name -- renaming the display helper alone
would not make those surfaces stop being about `ProviderAdapter`s.

**Legacy OCR methods are labeled, not erased.** `docs/htr-domain-design.md` §8 requires any
pre-transformation `MethodRun` from a deleted OCR provider to be "clearly labeled `legacy: true` in
any report or UI surface". Stored telemetry still names `docling`/`tesseract_layoutparser`/
`qwen2.5-vl`/`paddleocr-vl`/`surya`, and the Evidence page still replays it, so those ids keep a
label here -- one that says plainly that they are retired OCR methods rather than presenting them
as peers of the three real HTR methods.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


_METHOD_LABELS = {
    # Keys are the real `METHOD_ID` constants each adapter module declares -- not invented ids.
    "satrn": "SATRN (Riksarkivet)",
    "florence2_htr": "Florence-2 (vlm-htr line OCR)",
    "transkribus_swedish_lion_1": "Transkribus Swedish Lion I",
}

LEGACY_METHOD_SUFFIX = " (retired OCR method)"
"""Appended to every deleted-provider label. `docs/htr-domain-design.md` §8: legacy runs stay
readable and are never blended into HTR baseline statistics, so their label must say what they are
at a glance rather than looking like a fourth supported HTR method."""

_LEGACY_METHOD_LABELS = {
    "docling": "Docling",
    "tesseract": "Tesseract",
    "tesseract_layoutparser": "Tesseract + LayoutParser",
    "qwen2.5-vl": "Qwen2.5-VL",
    "qwen": "Qwen2.5-VL",
    "paddleocr-vl": "PaddleOCR-VL",
    "paddleocr_vl_structured": "PaddleOCR-VL Structured",
    "surya": "Surya",
}

_PROFILE_LABELS = {
    "archive": "Balanced archive processing",
    "fast_review": "Fast review",
    "maximum_quality": "Maximum quality",
    "research": "Research and replay",
}

_SOURCE_KIND_LABELS = {
    "manual_import": "Manual import",
    "manual-import": "Manual import",
    "folder_watch": "Watched folder",
    "folder-watch": "Watched folder",
    "network_share": "Network share",
    "email_inbox": "Email inbox",
    "rest_api": "REST API",
    "sharepoint": "SharePoint",
    "zip_import": "ZIP import",
    "cloud_storage": "Cloud storage",
}

_BASIS_CODE_LABELS = {
    "single_source": "Only one method supports this reading",
    "uncorroborated_single_source": "Only one method supports this reading",
    "contested": "Methods disagree",
    "corroborated": "Multiple methods agree",
    "missing": "A method did not provide this field",
}

_RUNTIME_LABELS = {
    "local_transformers": "Local Transformers",
    "transformers": "Transformers",
    "fake": "Test runtime",
    "cpu": "CPU",
    "cuda": "CUDA",
    "mps": "Apple GPU",
}

_RESULT_STAGE_LABELS = {
    "raw": "Raw output",
    "parsed": "Parsed",
    "normalized": "Normalized",
    "reviewed": "Human-reviewed",
    "canonical": "Canonical",
}
"""The five stages the comparison surface must keep distinct (`docs/htr-domain-design.md` §1, §4).
Deliberately five separate labels: collapsing them into one "transcription" string is exactly what
Stage 11 forbids."""

_CANONICALIZATION_STRATEGY_LABELS = {
    "best_single_method": "Best single method",
    "metric_selected_method": "Metric-selected method",
    "segment_level_selection": "Segment-level selection",
    "human_approved_selection": "Human-approved selection",
    "consensus_based_selection": "Consensus-based selection",
}

_BENCHMARK_STATUS_LABELS = {
    "agreed": "Reviewers agreed",
    "minor_disagreement": "Minor disagreement",
    "material_disagreement": "Material disagreement",
    "requires_adjudication": "Requires adjudication",
    "excluded_from_benchmark": "Excluded from benchmark",
}

_METHOD_RUN_OUTCOME_LABELS = {
    "succeeded": "Succeeded",
    "failed": "Failed",
    "no_output": "Ran, produced no text",
}

_OBSERVATION_TYPE_LABELS = {
    "successful_recognition_behavior": "Recognition worked as hoped",
    "recurring_recognition_failure": "Recognition failure that recurred",
    "segmentation_problem": "Segmentation problem",
    "handwriting_feature": "Handwriting feature",
    "document_layout_feature": "Document layout feature",
    "model_limitation": "Model limitation",
    "confidence_anomaly": "Confidence did not match accuracy",
    "performance_bottleneck": "Performance bottleneck",
    "environment_issue": "Environment issue",
    "reviewer_observation": "Noted by a reviewer",
    "possible_hypothesis": "Possible hypothesis",
    "unexpected_method_disagreement": "Methods disagreed unexpectedly",
    "experiment_validity_boundary": "What this comparison cannot measure",
    "reproducibility_anomaly": "Did not reproduce across sessions",
}
"""All fourteen `htr/knowledge/models.py::ObservationType` members.

Two are worded to prevent a specific misreading rather than to be short.
`experiment_validity_boundary` reads "What this comparison cannot measure" because that member exists
precisely so a statement about an *experiment's* limits can never be filed as one about the *method*
that ran inside it -- "Experiment validity boundary" is jargon a reader could gloss as a judgement on
the method. `confidence_anomaly` reads "Confidence did not match accuracy" because the bare word
"confidence" in this application otherwise means a model's own scalar, and this observation type is
about the *disagreement* between that scalar and a measurement.
"""

_FINDING_STATUS_LABELS = {
    "Draft": "Draft",
    "Candidate": "Candidate (not yet reviewed)",
    "Under review": "Under review",
    "Provisionally supported": "Provisionally supported (not reproduced)",
    "Supported": "Supported (reproduced in another run)",
    "Disputed": "Disputed",
    "Superseded": "Superseded",
    "Rejected": "Rejected",
}
"""All eight `FindingStatus` members.

`Candidate`, `Provisionally supported` and `Supported` carry a parenthetical because the difference
between them is the whole point of the lifecycle and is invisible in the bare words. A reader who sees
"Supported" next to "Provisionally supported" with no explanation has no way to know that the gap
between them is reproduction in an experiment run outside the finding's own scope -- which is the rule
`docs/knowledge-lifecycle.md`'s rule 3 makes unskippable, and the reason no finding in this repository
is `Supported`.
"""

_OBSERVATION_REVIEW_STATUS_LABELS = {
    "Unreviewed": "Nobody has checked this yet",
    "Under review": "Under review",
    "Accepted": "Confirmed against the records",
    "Rejected": "Rejected",
}
"""`ObservationReviewStatus`. "Accepted" reads "Confirmed against the records" because accepting an
observation means only "yes, this is really what the records show" -- never "yes, this generalizes",
which is a finding's business. The four-value enum has no `Supported`/`Disputed` for that reason and
the labels must not import the connotation back."""

_KNOWLEDGE_CONFIDENCE_LABELS = {
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
}
"""`ObservationConfidence` and `FindingConfidence`. One table for both: they are different fields
with different subjects, but the same three ordinal words, and two identical tables would drift.
Neither is a recognition confidence -- the *field names* carry that distinction
(`observation_confidence`, `confidence_level`), which is why these labels are bare ordinals rather
than trying to explain the subject in the value."""

_EVIDENCE_REFERENCE_KIND_LABELS = {
    "telemetry_event": "Telemetry event",
    "experiment_run": "Experiment run",
    "method_run": "Method run",
    "metric_result": "Metric result",
    "reliability_classification": "Reliability classification",
    "evidence_record": "Evidence record",
    "input_crop": "Input crop",
    "text_line": "Text line",
    "ground_truth_text": "Ground truth",
    "research_observation": "Research observation",
    "research_finding": "Research finding",
    "reproducibility_manifest": "Reproducibility manifest",
    "external_document": "Repository file (not a telemetry record)",
}
"""`EvidenceReferenceKind`. `external_document` says "not a telemetry record" in the label because it
is the one member whose target is not in an event stream, and a UI that rendered it identically to the
others would imply an evidence trail it does not have."""

_RESEARCH_QUESTION_STATUS_LABELS = {
    "Open": "Open (nothing drafted yet)",
    "Being investigated": "Being investigated",
    "Answered": "Answered by a finding",
}
"""`ResearchQuestionStatus`. "Answered by a finding" states the rule the model enforces: a question is
answered by a reviewed claim, never by a run completing."""

_HYPOTHESIS_RELATIONSHIP_LABELS = {
    "supports": "Supports the stated hypothesis",
    "contradicts": "Contradicts the stated hypothesis",
    "refines": "Refines the stated hypothesis",
    "untested": "Hypothesis untested by this finding",
    "no_hypothesis_asserted": "The experiment asserted no hypothesis",
}
"""`HypothesisRelationship`. `no_hypothesis_asserted` is spelled out rather than shortened because it
is the value all five real baseline findings carry: the baseline experiment's own `hypothesis` field
declines to predict which method performs better, and a label reading "None" would look like missing
data rather than a recorded decision."""

_CAPABILITY_LABELS = {
    "confidence_supported": "Reports confidence",
    "geometry_supported": "Reports geometry",
    "line_level_supported": "Line-level input",
    "page_level_supported": "Page-level input",
    "local_execution_supported": "Runs locally",
    "external_upload_required": "Requires external upload",
}


def method_label(method_id: Any) -> str:
    """Researcher-facing name for one HTR method id. A retired OCR provider id gets its historical
    name plus `LEGACY_METHOD_SUFFIX`, never a bare name that would read as a supported method."""
    value = _identifier_value(method_id)
    if value in _METHOD_LABELS:
        return _METHOD_LABELS[value]
    if value in _LEGACY_METHOD_LABELS:
        return f"{_LEGACY_METHOD_LABELS[value]}{LEGACY_METHOD_SUFFIX}"
    return _humanize(value)


def is_legacy_method(method_id: Any) -> bool:
    """Whether this id names one of the deleted OCR providers (`docs/htr-domain-design.md` §8).
    Callers aggregating HTR baseline statistics must exclude these; callers rendering history must
    still show them."""
    return _identifier_value(method_id) in _LEGACY_METHOD_LABELS


def provider_label(provider_id: Any) -> str:
    """Alias of `method_label` for the surfaces that genuinely still administer `ProviderAdapter`
    registrations (`clients/desktop/pages/provider_manager.py`, `pages/settings.py`) -- see this
    module's docstring for why that interface still exists."""
    return method_label(provider_id)


def profile_label(profile: Any) -> str:
    value = _identifier_value(profile)
    return _PROFILE_LABELS.get(value, _humanize(value))


def source_kind_label(kind: Any) -> str:
    value = _identifier_value(kind)
    return _SOURCE_KIND_LABELS.get(value, _humanize(value))


def basis_code_label(code: Any) -> str:
    value = _identifier_value(code)
    return _BASIS_CODE_LABELS.get(value, _humanize(value))


def runtime_label(runtime_kind: Any) -> str:
    value = _identifier_value(runtime_kind)
    if not value:
        return "-"
    return _RUNTIME_LABELS.get(value, _humanize(value))


def result_stage_label(stage: Any) -> str:
    value = _identifier_value(stage)
    return _RESULT_STAGE_LABELS.get(value, _humanize(value))


def canonicalization_strategy_label(strategy: Any) -> str:
    value = _identifier_value(strategy)
    return _CANONICALIZATION_STRATEGY_LABELS.get(value, _humanize(value))


def benchmark_status_label(status: Any) -> str:
    value = _identifier_value(status)
    return _BENCHMARK_STATUS_LABELS.get(value, _humanize(value))


def method_run_outcome_label(outcome: Any) -> str:
    value = _identifier_value(outcome)
    return _METHOD_RUN_OUTCOME_LABELS.get(value, _humanize(value))


def capability_label(capability_field: Any) -> str:
    value = _identifier_value(capability_field)
    return _CAPABILITY_LABELS.get(value, _humanize(value))


def observation_type_label(observation_type: Any) -> str:
    """Researcher-facing name for one of the fourteen `ObservationType` members."""
    value = _identifier_value(observation_type)
    return _OBSERVATION_TYPE_LABELS.get(value, _humanize(value))


def finding_status_label(status: Any) -> str:
    """Researcher-facing name for one of the eight `FindingStatus` members."""
    value = _identifier_value(status)
    return _FINDING_STATUS_LABELS.get(value, _humanize(value))


def observation_review_status_label(status: Any) -> str:
    """Researcher-facing name for one of the four `ObservationReviewStatus` members."""
    value = _identifier_value(status)
    return _OBSERVATION_REVIEW_STATUS_LABELS.get(value, _humanize(value))


def knowledge_confidence_label(confidence: Any) -> str:
    """Researcher-facing name for an `ObservationConfidence` or `FindingConfidence`.

    Never for a recognition confidence: those are floats and are rendered as numbers, not labels.
    """
    value = _identifier_value(confidence)
    return _KNOWLEDGE_CONFIDENCE_LABELS.get(value, _humanize(value))


def evidence_reference_kind_label(kind: Any) -> str:
    """Researcher-facing name for one of the thirteen `EvidenceReferenceKind` members."""
    value = _identifier_value(kind)
    return _EVIDENCE_REFERENCE_KIND_LABELS.get(value, _humanize(value))


def research_question_status_label(status: Any) -> str:
    """Researcher-facing name for one of the three `ResearchQuestionStatus` members."""
    value = _identifier_value(status)
    return _RESEARCH_QUESTION_STATUS_LABELS.get(value, _humanize(value))


def hypothesis_relationship_label(relationship: Any) -> str:
    """Researcher-facing name for one of the five `HypothesisRelationship` members."""
    value = _identifier_value(relationship)
    return _HYPOTHESIS_RELATIONSHIP_LABELS.get(value, _humanize(value))


def short_ref(ref: Any, *, prefix: int = 10) -> str:
    value = _identifier_value(ref)
    if len(value) <= prefix:
        return value
    return f"{value[:prefix]}..."


def document_label(document_ref: Any, *, archive_object: Any | None = None) -> str:
    original_filename = getattr(archive_object, "original_filename", None)
    if isinstance(original_filename, str) and original_filename.strip():
        return original_filename.strip()
    return short_ref(document_ref)


def _identifier_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _humanize(value: str) -> str:
    value = value.strip()
    if not value:
        return "-"
    parts = value.replace("-", " ").replace("_", " ").split()
    acronyms = {"api", "cer", "cpu", "cuda", "gpu", "htr", "iou", "ocr", "pdf", "vl", "wer", "xml", "zip"}
    return " ".join(part.upper() if part.lower() in acronyms else part.capitalize() for part in parts)
