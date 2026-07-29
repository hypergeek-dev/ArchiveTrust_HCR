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
