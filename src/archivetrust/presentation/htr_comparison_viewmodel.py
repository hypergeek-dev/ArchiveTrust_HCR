"""Comparison ViewModel (Stage 11, brief's "User interface" -> comparison view).

Side-by-side, per page/region/line: source crop reference, ground truth, each method's output, and
the real CER/WER/confidence/time/failure/diff/reviewer-decision/canonical-selection facts for it.

**The five stages never collapse.** `docs/htr-domain-design.md` §1 models
`RawResult`/`ParsedResult`/`NormalizedResult` as distinct entities under `MethodRun`, a human
`ReviewSubmission`/`Adjudication` as a separate lineage, and `CanonicalResult` as a page-scoped
selection across methods. `MethodComparisonCell.stages` therefore exposes five separately-labeled
`ResultStage` entries -- `raw`, `parsed`, `normalized`, `reviewed`, `canonical` -- each with its own
`present` flag. A stage that does not exist is `present=False` with `text=None`; it is never
back-filled from an adjacent stage, because "the parser did not run" and "the parser ran and
changed nothing" are different facts and a researcher comparing methods needs to tell them apart.

**Metric and diff primitives are reused, not reimplemented.** CER/WER and the character/word
edit-operation classification come from `htr/evaluation/recognition.py::compute_recognition_metrics`
(which itself reuses `evaluation/metrics.py`). This module computes no distance of its own.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from archivetrust.domain.canonical.result import CanonicalResult
from archivetrust.htr.evaluation.recognition import (
    RecognitionMetrics,
    compute_recognition_metrics,
)
from archivetrust.htr.research_store import HtrResearchStore
from archivetrust.presentation.display_names import (
    canonicalization_strategy_label,
    method_label,
    method_run_outcome_label,
    result_stage_label,
)

STAGE_ORDER = ("raw", "parsed", "normalized", "reviewed", "canonical")
"""The five stages, in pipeline order. Exposed as a module constant so a View renders them in a
stable order without hardcoding the vocabulary itself."""


class ResultStage(BaseModel):
    """One of the five stages of one method's result for one line."""

    model_config = ConfigDict(frozen=True)

    stage: str
    label: str
    present: bool
    """`False` means this stage genuinely does not exist for this run -- not that it is empty."""
    text: str | None = None
    attribution: str | None = None
    """Who/what produced this stage: the method for raw/parsed/normalized, the reviewer for
    reviewed, the canonicalization strategy for canonical. `None` when `present` is `False`."""


class EditOpSummary(BaseModel):
    """Character- and word-level edit-operation counts against the ground truth, from
    `htr/evaluation/recognition.py`'s classification (never recomputed here)."""

    model_config = ConfigDict(frozen=True)

    char_matches: int
    char_substitutions: int
    char_insertions: int
    char_deletions: int
    word_matches: int
    word_substitutions: int
    word_insertions: int
    word_deletions: int


class MethodComparisonCell(BaseModel):
    """One method's column for one line."""

    model_config = ConfigDict(frozen=True)

    method_id: str
    method_display_name: str
    method_run_id: str
    outcome: str
    outcome_label: str
    failed: bool
    failure_reasons: tuple[str, ...] = ()
    stages: tuple[ResultStage, ...] = ()
    confidence: float | None = None
    """Only when the producing method's `MethodCapabilities.confidence_supported` is true and the
    run actually reported one -- never fabricated (`providers/htr_adapter.py`'s discipline)."""
    processing_time_ms: float | None = None
    character_error_rate_normalized: float | None = None
    character_error_rate_raw: float | None = None
    word_error_rate_normalized: float | None = None
    word_error_rate_raw: float | None = None
    exact_match_normalized: bool | None = None
    edit_ops: EditOpSummary | None = None
    """`None` when there is no ground truth for this line -- metrics against nothing are not
    computed and not shown as zero."""
    selected_as_canonical: bool = False
    """Whether the page's current `CanonicalResult` selected THIS method run's text for this line."""
    is_legacy_method: bool = False


class LineComparison(BaseModel):
    """One line's full side-by-side row."""

    model_config = ConfigDict(frozen=True)

    text_line_id: str
    region_id: str
    page_id: str
    reading_order_index: int
    source_crop_ids: tuple[str, ...]
    source_crop_hashes: tuple[str, ...]
    controlled_comparison: bool | None
    """`True` when every method in this row consumed one byte-identical crop hash
    (`docs/htr-domain-design.md` §7). `False` when hashes differ -- the row is then an end-to-end
    comparison and must not be read as a pure recognizer comparison. `None` when no crop exists."""
    ground_truth: str | None
    reviewer_decision: str | None = None
    canonical_text: str | None = None
    canonical_strategy_label: str | None = None
    cells: tuple[MethodComparisonCell, ...] = ()


class ComparisonViewModel:
    """Builds `LineComparison` rows from an `HtrResearchStore`."""

    def __init__(
        self,
        store: HtrResearchStore,
        *,
        run_confidences: dict[str, float] | None = None,
        run_times_ms: dict[str, float] | None = None,
        reviewer_decisions: dict[str, str] | None = None,
        legacy_method_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._store = store
        self._run_confidences = run_confidences or {}
        """`method_run_id -> confidence`, from the run's Evidence. Passed in rather than read here
        so this ViewModel never reaches into the telemetry store."""
        self._run_times_ms = run_times_ms or {}
        self._reviewer_decisions = reviewer_decisions or {}
        """`text_line_id -> reviewer decision summary`."""
        self._legacy_method_ids = legacy_method_ids

    def page_comparison(self, page_id: str) -> tuple[LineComparison, ...]:
        rows: list[LineComparison] = []
        for region in self._store.regions(page_id=page_id):
            rows.extend(self.region_comparison(region.region_id))
        return tuple(rows)

    def region_comparison(self, region_id: str) -> tuple[LineComparison, ...]:
        return tuple(
            row
            for line in self._store.text_lines(region_id=region_id)
            if (row := self.line_comparison(line.text_line_id)) is not None
        )

    def line_comparison(self, text_line_id: str) -> LineComparison | None:
        line = self._store.text_line(text_line_id)
        if line is None:
            return None
        region = self._store.region(line.region_id)
        page_id = region.page_id if region is not None else ""

        crops = self._store.crops_for_line(text_line_id)
        crop_hashes = tuple(sorted({crop.hash for crop in crops}))
        ground_truth = self._store.ground_truth_for_line(text_line_id)

        canonical = self._store.latest_canonical_result(page_id) if page_id else None
        canonical_span = None
        if canonical is not None:
            canonical_span = next(
                (span for span in canonical.spans if span.text_line_id == text_line_id), None
            )

        cells = tuple(
            self._cell(
                run_id=run.method_run_id,
                ground_truth=ground_truth,
                canonical=canonical,
                canonical_selected_run_id=(
                    canonical_span.source_method_run_id if canonical_span is not None else None
                ),
            )
            for crop in crops
            for run in self._store.method_runs_for_crop(crop.crop_id)
        )

        return LineComparison(
            text_line_id=text_line_id,
            region_id=line.region_id,
            page_id=page_id,
            reading_order_index=line.reading_order_index,
            source_crop_ids=tuple(crop.crop_id for crop in crops),
            source_crop_hashes=crop_hashes,
            controlled_comparison=None if not crops else len(crop_hashes) == 1,
            ground_truth=ground_truth,
            reviewer_decision=self._reviewer_decisions.get(text_line_id),
            canonical_text=canonical_span.text if canonical_span is not None else None,
            canonical_strategy_label=(
                canonicalization_strategy_label(canonical.strategy) if canonical is not None else None
            ),
            cells=tuple(sorted(cells, key=lambda cell: cell.method_display_name)),
        )

    def _cell(
        self,
        *,
        run_id: str,
        ground_truth: str | None,
        canonical: CanonicalResult | None,
        canonical_selected_run_id: str | None,
    ) -> MethodComparisonCell:
        run = self._store.method_run(run_id)
        assert run is not None  # every id here came from the store's own index
        transcript = self._store.transcript(run_id)
        failures = self._store.failures(method_run_id=run_id)

        selected = canonical_selected_run_id == run_id
        canonical_text = None
        if selected and canonical is not None:
            canonical_text = next(
                (
                    span.text
                    for span in canonical.spans
                    if span.source_method_run_id == run_id
                ),
                None,
            )

        stages = self._stages(
            transcript=transcript,
            method_id=run.method_id,
            canonical=canonical,
            canonical_text=canonical_text,
            selected=selected,
        )

        # Metrics are computed against the stage a researcher actually compares -- the normalized
        # text -- falling back through parsed to raw only when the later stages genuinely do not
        # exist, so a method that never got a normalization step is still measurable.
        hypothesis = None
        for candidate in (
            transcript.normalized_text if transcript is not None else None,
            transcript.parsed_text if transcript is not None else None,
            transcript.raw_text if transcript is not None else None,
        ):
            if candidate is not None:
                hypothesis = candidate
                break

        metrics: RecognitionMetrics | None = None
        if ground_truth is not None and hypothesis is not None:
            metrics = compute_recognition_metrics(ground_truth, hypothesis)

        return MethodComparisonCell(
            method_id=run.method_id,
            method_display_name=method_label(run.method_id),
            method_run_id=run_id,
            outcome=run.outcome,
            outcome_label=method_run_outcome_label(run.outcome),
            failed=run.outcome == "failed",
            failure_reasons=tuple(failure.reason for failure in failures),
            stages=stages,
            confidence=self._run_confidences.get(run_id),
            processing_time_ms=self._run_times_ms.get(run_id),
            character_error_rate_normalized=(
                metrics.character_error_rate_normalized if metrics else None
            ),
            character_error_rate_raw=metrics.character_error_rate_raw if metrics else None,
            word_error_rate_normalized=metrics.word_error_rate_normalized if metrics else None,
            word_error_rate_raw=metrics.word_error_rate_raw if metrics else None,
            exact_match_normalized=metrics.exact_match_normalized if metrics else None,
            edit_ops=(
                EditOpSummary(
                    char_matches=metrics.char_edits_normalized.matches,
                    char_substitutions=metrics.char_edits_normalized.substitutions,
                    char_insertions=metrics.char_edits_normalized.insertions,
                    char_deletions=metrics.char_edits_normalized.deletions,
                    word_matches=metrics.word_edits_normalized.matches,
                    word_substitutions=metrics.word_edits_normalized.substitutions,
                    word_insertions=metrics.word_edits_normalized.insertions,
                    word_deletions=metrics.word_edits_normalized.deletions,
                )
                if metrics
                else None
            ),
            selected_as_canonical=selected,
            is_legacy_method=run.method_id in self._legacy_method_ids,
        )

    @staticmethod
    def _stages(
        *,
        transcript,
        method_id: str,
        canonical: CanonicalResult | None,
        canonical_text: str | None,
        selected: bool,
    ) -> tuple[ResultStage, ...]:
        """Builds all five stages, every one explicitly present-or-not. Never derives one stage's
        text from another."""
        texts: dict[str, str | None] = {
            "raw": transcript.raw_text if transcript is not None else None,
            "parsed": transcript.parsed_text if transcript is not None else None,
            "normalized": transcript.normalized_text if transcript is not None else None,
            "reviewed": transcript.reviewed_text if transcript is not None else None,
            "canonical": canonical_text if selected else None,
        }
        attributions: dict[str, str | None] = {
            "raw": method_label(method_id),
            "parsed": method_label(method_id),
            "normalized": method_label(method_id),
            "reviewed": (transcript.reviewer_ref if transcript is not None else None),
            "canonical": (
                canonicalization_strategy_label(canonical.strategy)
                if canonical is not None and selected
                else None
            ),
        }
        return tuple(
            ResultStage(
                stage=stage,
                label=result_stage_label(stage),
                present=texts[stage] is not None,
                text=texts[stage],
                attribution=attributions[stage] if texts[stage] is not None else None,
            )
            for stage in STAGE_ORDER
        )
