"""Ground-truth evaluation over recorded telemetry (release WS7).

For each evaluable annotation, every provider's observations and the canonical result are scored
against the human reading. Matching is **best-candidate by normalized similarity** among the
document's observations of the annotated type — a deliberate, documented choice: it needs no
region ground truth, but it is optimistic (a system is never penalized for *which* candidate it
would have chosen, only for its best available one). Region/IoU detection metrics and table
structure metrics are therefore *not* produced by this evaluator; producing them requires region
ground truth this campaign does not collect. Nothing here invokes a provider.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from archivetrust.application.current_state import CurrentStateService
from archivetrust.application.journal import Journal
from archivetrust.domain.current_state import CurrentDocumentState
from archivetrust.evaluation.ground_truth import (
    AdjudicationStatus,
    FileGroundTruthStore,
    GroundTruthAnnotation,
)
from archivetrust.evaluation.metrics import METRICS_VERSION, TextComparison, compare_text

EVALUATION_SCHEMA = "archivetrust.ground_truth_evaluation.v2"


class FieldScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    annotation_id: str
    archive_object_ref: str
    field: str
    system: str
    """Provider id, "canonical" (latest, including human corrections), or
    "canonical_pre_correction" (the initial reconciled value)."""
    matched: bool
    """False when the system produced no observation of the annotated type for this document —
    a miss, reported as such, never silently dropped and never scored as an empty string."""
    best_candidate: str | None
    comparison: TextComparison | None
    canonical_confidence: float | None = None


class SystemSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    fields_evaluated: int
    misses: int
    exact_matches: int
    mean_cer_matched: float | None
    mean_wer_matched: float | None
    mean_cer_all: float | None
    """Misses scored as CER 1.0 — the honest all-in number; both are reported."""


class ConfidenceBin(BaseModel):
    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    count: int
    exact_match_rate: float | None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = EVALUATION_SCHEMA
    metrics_version: int = METRICS_VERSION
    annotations_total: int
    annotations_evaluable: int
    annotations_unverified: int
    annotations_excluded: int
    annotations_ai_excluded: int
    annotations_illegible: int
    annotations_disputed: int
    annotations_uncertain_included: int
    documents: int
    field_scores: tuple[FieldScore, ...]
    summaries: tuple[SystemSummary, ...]
    canonical_confidence_bins: tuple[ConfidenceBin, ...]
    central_result: dict
    scope_statement: str
    known_biases: tuple[str, ...]


_KNOWN_BIASES = (
    "Best-candidate matching: each system is scored on its closest observation of the annotated "
    "type, not on the candidate it would surface first — optimistic for all systems equally.",
    "No page filtering: candidates from any page of the document compete; for multi-page "
    "documents a lucky match from another page could flatter a system.",
    "Single annotator per record unless adjudication_status says otherwise; annotator identity "
    "and method are recorded per annotation.",
    "Annotated fields (main heading and first body paragraph of page 1) are not a random sample "
    "of all content in the documents; results describe those fields only.",
)


def evaluate(
    *,
    store: FileGroundTruthStore,
    telemetry_source,
    seed: int = 20260716,
) -> EvaluationResult:
    annotations = store.latest()
    verified = store.verified_references()
    disputed = [a for a in annotations if a.adjudication_status is AdjudicationStatus.DISPUTED]
    illegible = [a for a in annotations if a.illegible]
    evaluable = list(verified)

    journal = Journal()
    current_state = CurrentStateService(telemetry_source)
    states: dict[str, tuple[object, CurrentDocumentState]] = {}
    field_scores: list[FieldScore] = []
    for annotation in evaluable:
        ref = annotation.archive_object_ref
        if ref not in states:
            states[ref] = (
                journal.replay(telemetry_source.events_for_document(ref)),
                current_state.document(ref),
            )
        replayed, current = states[ref]
        if current.evaluation_eligible:
            field_scores.extend(_score_annotation(annotation, replayed, current))

    summaries = _summaries(field_scores)
    bins = _confidence_bins(field_scores)
    central = _central_result(summaries, field_scores, seed=seed)

    return EvaluationResult(
        annotations_total=len(annotations),
        annotations_evaluable=len(evaluable),
        annotations_unverified=sum(
            1 for annotation in annotations if annotation.verification_status.value == "unverified"
        ),
        annotations_excluded=sum(
            1 for annotation in annotations if annotation.verification_status.value == "excluded"
        ),
        annotations_ai_excluded=sum(
            1
            for annotation in annotations
            if annotation.verification_status.value == "verified_evaluation_reference"
            and annotation.annotator_kind.value != "human"
        ),
        annotations_illegible=len(illegible),
        annotations_disputed=len(disputed),
        annotations_uncertain_included=sum(1 for a in evaluable if a.uncertain),
        documents=len({a.archive_object_ref for a in evaluable}),
        field_scores=tuple(field_scores),
        summaries=tuple(summaries),
        canonical_confidence_bins=tuple(bins),
        central_result=central,
        scope_statement=(
            "Metrics describe verified, human-independent evaluation references only, for the "
            "annotated fields/documents, matched by best candidate of the annotated observation "
            "type. They are not corpus-wide accuracy claims."
        ),
        known_biases=_KNOWN_BIASES,
    )


def _text_of(payload) -> str | None:
    text = getattr(payload, "text", None)
    if isinstance(text, str):
        return text
    value = getattr(payload, "value", None)
    return value if isinstance(value, str) else None


def _score_annotation(
    annotation: GroundTruthAnnotation, state, current: CurrentDocumentState
) -> Iterable[FieldScore]:
    reference = annotation.text or ""

    # Per provider: best-matching observation of the annotated type.
    by_provider: dict[str, list[str]] = {}
    for observation in state.all_observations():
        if observation.observation_type.value != annotation.observation_type:
            continue
        text = _text_of(observation.payload)
        if text:
            by_provider.setdefault(observation.provider_id, []).append(text)

    for provider_id in sorted(by_provider):
        yield _best_score(annotation, provider_id, by_provider[provider_id], reference)

    # Providers that ran but produced no candidate of this type are misses; providers that never
    # ran on this document are simply absent (never fabricated).
    invoked = {o.provider_id for o in state.all_observations()}
    for provider_id in sorted(invoked - set(by_provider)):
        yield FieldScore(
            annotation_id=annotation.annotation_id,
            archive_object_ref=annotation.archive_object_ref,
            field=annotation.field,
            system=provider_id,
            matched=False,
            best_candidate=None,
            comparison=None,
        )

    # Canonical: initial reconciliation and latest (includes human corrections).
    initial: list[tuple[str, float | None]] = []
    latest: list[tuple[str, float | None]] = []
    for slot in current.slots:
        history = slot.history
        for target, canonical in ((initial, history[0]), (latest, slot.current)):
            if canonical.observation_type.value != annotation.observation_type:
                continue
            text = _text_of(canonical.payload)
            if text:
                confidence = (
                    canonical.canonical_confidence.value
                    if canonical.canonical_confidence is not None
                    else None
                )
                target.append((text, confidence))

    for system, candidates in (("canonical_pre_correction", initial), ("canonical", latest)):
        if candidates:
            best = max(
                candidates, key=lambda item: compare_text(reference, item[0]).normalized_similarity
            )
            yield FieldScore(
                annotation_id=annotation.annotation_id,
                archive_object_ref=annotation.archive_object_ref,
                field=annotation.field,
                system=system,
                matched=True,
                best_candidate=best[0],
                comparison=compare_text(reference, best[0]),
                canonical_confidence=best[1],
            )
        else:
            yield FieldScore(
                annotation_id=annotation.annotation_id,
                archive_object_ref=annotation.archive_object_ref,
                field=annotation.field,
                system=system,
                matched=False,
                best_candidate=None,
                comparison=None,
            )


def _best_score(
    annotation: GroundTruthAnnotation, system: str, candidates: list[str], reference: str
) -> FieldScore:
    best = max(candidates, key=lambda c: compare_text(reference, c).normalized_similarity)
    return FieldScore(
        annotation_id=annotation.annotation_id,
        archive_object_ref=annotation.archive_object_ref,
        field=annotation.field,
        system=system,
        matched=True,
        best_candidate=best,
        comparison=compare_text(reference, best),
    )


def _summaries(scores: list[FieldScore]) -> list[SystemSummary]:
    systems = sorted({s.system for s in scores})
    summaries = []
    for system in systems:
        rows = [s for s in scores if s.system == system]
        matched = [s for s in rows if s.matched and s.comparison is not None]
        cer_matched = [s.comparison.character_error_rate for s in matched]
        wer_matched = [s.comparison.word_error_rate for s in matched]
        cer_all = cer_matched + [1.0] * (len(rows) - len(matched))
        summaries.append(
            SystemSummary(
                system=system,
                fields_evaluated=len(rows),
                misses=len(rows) - len(matched),
                exact_matches=sum(1 for s in matched if s.comparison.exact_match),
                mean_cer_matched=(sum(cer_matched) / len(cer_matched)) if cer_matched else None,
                mean_wer_matched=(sum(wer_matched) / len(wer_matched)) if wer_matched else None,
                mean_cer_all=(sum(cer_all) / len(cer_all)) if cer_all else None,
            )
        )
    return summaries


def _confidence_bins(scores: list[FieldScore]) -> list[ConfidenceBin]:
    """Descriptive only (the release forbids fitting calibration from this sample): canonical
    exact-match rate per confidence bin, for the canonical (latest) system."""
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    rows = [
        s
        for s in scores
        if s.system == "canonical" and s.matched and s.canonical_confidence is not None
    ]
    bins = []
    for lower, upper in zip(edges, edges[1:]):
        in_bin = [s for s in rows if lower <= s.canonical_confidence < upper]
        bins.append(
            ConfidenceBin(
                lower=lower,
                upper=min(upper, 1.0),
                count=len(in_bin),
                exact_match_rate=(
                    sum(1 for s in in_bin if s.comparison.exact_match) / len(in_bin)
                    if in_bin
                    else None
                ),
            )
        )
    return bins


def _central_result(
    summaries: list[SystemSummary], scores: list[FieldScore], *, seed: int
) -> dict:
    """Did the canonical result outperform the best individual provider, on mean all-in CER
    (misses = 1.0), with a paired bootstrap interval over annotated fields?"""
    providers = [
        s
        for s in summaries
        if s.system not in ("canonical", "canonical_pre_correction") and s.fields_evaluated > 0
    ]
    canonical = next((s for s in summaries if s.system == "canonical"), None)
    if canonical is None or not providers:
        return {"verdict": "insufficient_data", "detail": "no canonical or no provider scores"}

    best_provider = min(providers, key=lambda s: s.mean_cer_all)
    per_field: dict[str, dict[str, float]] = {}
    for score in scores:
        cer = score.comparison.character_error_rate if score.comparison is not None else 1.0
        per_field.setdefault(score.annotation_id, {})[score.system] = cer
    paired = [
        (fields["canonical"], fields[best_provider.system])
        for fields in per_field.values()
        if "canonical" in fields and best_provider.system in fields
    ]
    diffs = [c - p for c, p in paired]
    rng = random.Random(seed)
    resampled_means = []
    if diffs:
        for _ in range(2000):
            sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            resampled_means.append(sum(sample) / len(sample))
        resampled_means.sort()
        low = resampled_means[int(0.025 * len(resampled_means))]
        high = resampled_means[int(0.975 * len(resampled_means))]
    else:
        low = high = 0.0

    mean_diff = (sum(diffs) / len(diffs)) if diffs else 0.0
    if not diffs or len(diffs) < 5:
        verdict = "sample_insufficient"
    elif high < 0:
        verdict = "canonical_better"
    elif low > 0:
        verdict = "canonical_worse"
    else:
        verdict = "equivalent_within_uncertainty"

    return {
        "question": "Did the canonical result outperform the best individual provider?",
        "metric": "mean all-in character error rate (misses scored 1.0), paired per field",
        "best_provider": best_provider.system,
        "best_provider_mean_cer_all": best_provider.mean_cer_all,
        "canonical_mean_cer_all": canonical.mean_cer_all,
        "mean_paired_cer_difference_canonical_minus_provider": mean_diff,
        "bootstrap_95ci": [low, high],
        "paired_fields": len(diffs),
        "verdict": verdict,
    }


def write_evaluation(result: EvaluationResult, output_path: Path | str) -> Path:
    output = Path(output_path)
    output.write_text(
        json.dumps(json.loads(result.model_dump_json()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
