"""Per-stage result types for one Loghi pipeline run (Laypa -> Loghi Tooling -> Loghi HTR).

**Why this exists.** The brief is explicit: "Do not treat only the final text as evidence." A single
`RecognitionResult` per page would collapse three independently-failing stages into one pass/fail,
losing exactly the information a "which stage broke" failure taxonomy needs. `LoghiPipelineResult`
bundles all three `LoghiStageResult`s plus the line crops and final PAGE XML, and `adapter.py`'s
`build_evidence`/`build_failure_record` read from it -- never from a single collapsed string.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LoghiStageResult(BaseModel):
    """One pipeline stage's outcome -- Laypa (layout analysis), Loghi Tooling (line extraction /
    reading order), or Loghi HTR (recognition)."""

    model_config = ConfigDict(frozen=True)

    stage_name: str
    """`"laypa" | "loghi_tooling" | "loghi_htr"`. Kept as a plain string (not an enum), same
    rationale as `KNOWN_FAILURE_CATEGORIES` elsewhere in this codebase: a future pipeline revision
    that adds a stage is still carried verbatim."""
    started_at: str
    completed_at: str | None = None
    ok: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    exit_code: int | None = None
    duration_ms: float | None = None


class LoghiPipelineResult(BaseModel):
    """The full, multi-stage record of one Loghi run over one page. `adapter.py` maps this onto the
    common `RecognitionResult` envelope for `HtrMethodAdapter` conformance, but this richer object --
    not the envelope -- is what gets persisted into `Evidence.raw_output`/`raw_response` and the
    telemetry stage events, so no stage's detail is lost in translation."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    stages: tuple[LoghiStageResult, ...]
    line_crop_refs: tuple[str, ...] = ()
    """References (paths/ids), never embedded bytes, to whatever line crops Laypa/Tooling produced
    internally -- present so "only the final text is evidence" never happens, per the brief."""
    reading_order_output: str | None = None
    final_page_xml: str | None = None
    """The untouched PAGE XML text Loghi HTR/Tooling produced, if the pipeline reached that stage --
    hashed separately by `page_xml.parse_loghi_page_xml` when parsed, never re-encoded here."""
    total_duration_ms: float | None = None
    suspicious_output: bool = False
    """True when the pipeline completed successfully but produced output an automated check flags as
    worth human review (e.g. near-empty text on a page with detected regions). Per the brief: "A
    successful run with poor-looking text is not an execution failure" -- this is a flag on a
    successful `LoghiPipelineResult`, never a `FailureRecord` category."""
    suspicious_output_reason: str | None = None
