"""The Loghi feasibility smoke test (docs/methods/loghi.md §"Feasibility gate").

Runs a small number of pages (5 Swedish + 5 Dutch by default, per the brief) through the real
`LoghiAdapter` and records every stage's outcome as durable telemetry -- proving the mechanics
(environment validation -> Laypa -> line extraction -> HTR -> PAGE XML -> parse -> telemetry -> report
regeneration) work end to end, without claiming a full Swedish-vs-Dutch comparison result.

**What this module does not do.** It does not run a large corpus, does not compute CER/WER, does not
create a `ResearchFinding`, and does not exclude Loghi after one bad page -- a systematic failure is
classified (`SmokeTestOutcome.failure_taxonomy_category`), never silently swallowed into "Loghi doesn't
work." Human plausibility review of the resulting text/PAGE XML happens through this codebase's
existing Review Center (`review/`) against the `Evidence`/`Observation` records this smoke test
produces exactly like any other method run -- no separate review mechanism is built here.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from archivetrust.providers.htr_adapter import RecognitionInput
from archivetrust.providers.loghi.adapter import LoghiAdapter, build_evidence, build_failure_record

DEFAULT_SWEDISH_PAGE_COUNT = 5
DEFAULT_DUTCH_PAGE_COUNT = 5


class SmokeTestPageRef(BaseModel):
    """One page to run through the smoke test -- a reference, never embedded image bytes."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    image_path: str
    corpus_language: str
    """`"sv"` or `"nl"` -- which pool this page belongs to, for the report's per-corpus breakdown."""


class SmokeTestPageOutcome(BaseModel):
    """One page's real result -- success or a classified failure, never silently dropped."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    corpus_language: str
    ok: bool
    failure_category: str | None = None
    failure_message: str | None = None
    text_preview: str | None = None
    """First 200 characters of recognized text, for a human skimming the report -- never the whole
    text (that lives in the full `Evidence`/`Observation` record this smoke test also creates)."""
    execution_time_ms: float | None = None
    evidence_id: str | None = None


class LoghiSmokeTestReport(BaseModel):
    """The full, real result of one smoke-test run -- reconstructible from durable telemetry alone
    (brief: "Demonstrate that completed Loghi results can be reconstructed and reported without
    rerunning the containers"), never requiring the adapter/facade to re-render."""

    model_config = ConfigDict(frozen=True)

    environment_valid: bool
    environment_messages: tuple[str, ...]
    page_outcomes: tuple[SmokeTestPageOutcome, ...]
    started_at: str
    completed_at: str

    def succeeded_count(self) -> int:
        return sum(1 for o in self.page_outcomes if o.ok)

    def failed_count(self) -> int:
        return sum(1 for o in self.page_outcomes if not o.ok)

    def success_rate_for(self, corpus_language: str) -> float | None:
        """Fraction of `corpus_language` pages that succeeded, or `None` if that pool was empty --
        never a fabricated `0.0` for "no pages attempted."""
        pool = [o for o in self.page_outcomes if o.corpus_language == corpus_language]
        if not pool:
            return None
        return sum(1 for o in pool if o.ok) / len(pool)

    def failure_categories(self) -> dict[str, int]:
        """A count per `FailureRecord.category` among failed pages -- the classification the brief
        requires before ever concluding "Loghi doesn't work": integration defect vs. environment
        defect vs. model-domain limitation vs. pipeline limitation vs. dataset incompatibility vs.
        unknown. This method only tallies the adapter's own reported categories (`KNOWN_FAILURE_
        CATEGORIES`); mapping those onto the six higher-level buckets is a human classification step,
        not automated here."""
        counts: dict[str, int] = {}
        for outcome in self.page_outcomes:
            if not outcome.ok and outcome.failure_category:
                counts[outcome.failure_category] = counts.get(outcome.failure_category, 0) + 1
        return counts


def run_loghi_smoke_test(
    *,
    adapter: LoghiAdapter,
    swedish_pages: tuple[SmokeTestPageRef, ...],
    dutch_pages: tuple[SmokeTestPageRef, ...],
    store=None,
) -> LoghiSmokeTestReport:
    """Runs every page through `adapter.recognize()` once, records real telemetry via `store` (a
    `DurableHtrResearchStore`; `None` skips telemetry recording, used by tests that only want the
    report). Never fabricates a page's outcome -- `adapter.recognize()` is genuinely called for each
    one, whether that resolves to success, a classified failure, or (against this machine, right now)
    an honest `environment_unavailable` failure for every page.
    """
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    validation = adapter.validate_environment()
    if store is not None:
        store.record_loghi_environment_validated(
            method_id="loghi",
            valid=validation.valid,
            environment_report={"messages": list(validation.messages)},
        )

    outcomes: list[SmokeTestPageOutcome] = []
    for page in (*swedish_pages, *dutch_pages):
        result = adapter.recognize(RecognitionInput(page_image_ref=page.image_path))

        if store is not None:
            method_run_id = f"smoke_{page.page_id}"
            store.record_loghi_pipeline_started(
                method_run_id=method_run_id,
                input_image_ref=page.image_path,
                component_versions=adapter.component_versions.model_dump(),
            )
            pipeline_result = result.raw_response.get("pipeline_result")
            if pipeline_result:
                for stage in pipeline_result.get("stages", []):
                    if stage.get("ok"):
                        store.record_loghi_stage_completed(method_run_id=method_run_id, stage_result=stage)
                    else:
                        store.record_loghi_stage_failed(method_run_id=method_run_id, stage_result=stage)
            if result.text is not None:
                parsed_page = result.raw_response.get("parsed_page") or {}
                store.record_loghi_page_xml_generated(
                    method_run_id=method_run_id,
                    source_xml_hash=parsed_page.get("source_xml_hash", ""),
                    page_schema_version=parsed_page.get("page_schema_version"),
                )

        if result.text is None:
            failure = build_failure_record(result, method_run_id=f"smoke_{page.page_id}")
            outcomes.append(
                SmokeTestPageOutcome(
                    page_id=page.page_id,
                    corpus_language=page.corpus_language,
                    ok=False,
                    failure_category=failure.category if failure else None,
                    failure_message=failure.reason if failure else "no failure record produced",
                )
            )
            continue

        evidence = build_evidence(result)
        outcomes.append(
            SmokeTestPageOutcome(
                page_id=page.page_id,
                corpus_language=page.corpus_language,
                ok=True,
                text_preview=result.text[:200],
                execution_time_ms=result.execution_time_ms,
                evidence_id=evidence.evidence_id if evidence is not None else None,
            )
        )

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return LoghiSmokeTestReport(
        environment_valid=validation.valid,
        environment_messages=validation.messages,
        page_outcomes=tuple(outcomes),
        started_at=started_at,
        completed_at=completed_at,
    )
