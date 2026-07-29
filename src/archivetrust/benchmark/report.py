"""Benchmark report generation (Benchmark & Measurement Framework, 2026-07-13).

Pure text/markdown rendering over an already-computed `BenchmarkRun` — no telemetry is read here,
no metric is computed here; this module only formats what `aggregate.py` already produced. Every
cell renders a `Metric` honestly: `UNAVAILABLE` prints as "unavailable", never as a blank, a zero,
or an omitted row — the report must never look more complete than the measurement actually is.
"""

from __future__ import annotations

from archivetrust.benchmark.metrics import Metric, MetricStatus
from archivetrust.benchmark.run import BenchmarkRun


def _cell(metric: Metric, *, precision: int = 2) -> str:
    if metric.status == MetricStatus.UNAVAILABLE:
        return "unavailable"
    value = metric.value
    if isinstance(value, float):
        text = f"{value * 100:.1f}%" if 0.0 <= value <= 1.0 and metric.unit is None else f"{value:.{precision}f}"
    else:
        text = str(value)
    if metric.unit and not isinstance(value, float):
        text += f" {metric.unit}"
    marker = {
        MetricStatus.MEASURED: "",
        MetricStatus.DERIVED: " ~",
        MetricStatus.ESTIMATED: " (est.)",
    }.get(metric.status, "")
    return text + marker


def executive_summary(run: BenchmarkRun) -> str:
    d, t, h = run.dataset, run.trust, run.human_review
    lines = [
        f"# Benchmark: {run.metadata.dataset_id}",
        f"Run `{run.metadata.run_id}` at {run.metadata.run_at}"
        + (f" — software {run.metadata.software_version}" if run.metadata.software_version else ""),
        "",
        "## Executive Summary",
        f"- Documents processed: {_cell(d.documents_processed)}",
        f"- Pages processed: {_cell(d.pages_processed)}",
        f"- Average canonical confidence: {_cell(t.average_canonical_confidence)}",
        f"- Human review: {_cell(h.review_percentage)} of slots ({_cell(h.review_items_total)} items across "
        f"{_cell(h.documents_entering_review)} document(s))",
        f"- Review reduction vs. Triage Policy v1 baseline: {_cell(h.review_reduction_vs_baseline)}",
        f"- Processing duration: {_cell(d.processing_duration_seconds)}",
    ]
    return "\n".join(lines)


def provider_comparison_table(run: BenchmarkRun) -> str:
    runtime_by_provider = {(r.provider_id, r.provider_version): r for r in run.runtime}
    lines = [
        "## Provider Comparison",
        "",
        "| Provider | Pages/hr | Avg Confidence | Review Rate | Agreement | Failures | Avg Runtime |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in run.quality:
        rt = runtime_by_provider.get((q.provider_id, q.provider_version))
        pages_hr = _cell(rt.throughput_pages_per_hour) if rt else "unavailable"
        avg_runtime = _cell(rt.avg_runtime_per_document_seconds) if rt else "unavailable"
        failures = _cell(rt.failure_rate) if rt else "unavailable"
        lines.append(
            f"| {q.provider_id} ({q.provider_version}) | {pages_hr} | {_cell(q.average_confidence)} | "
            f"{_cell(q.review_trigger_rate)} | {_cell(q.agreement_with_canonical)} | {failures} | {avg_runtime} |"
        )
    return "\n".join(lines)


def observation_type_table(run: BenchmarkRun) -> str:
    lines = [
        "## Observation-Type Comparison",
        "",
        "| Observation | Best Provider | Agreement | Confidence |",
        "|---|---|---|---|",
    ]
    for row in run.observation_types:
        lines.append(
            f"| {row.observation_type} | {_cell(row.best_provider)} | {_cell(row.agreement)} | "
            f"{_cell(row.confidence)} |"
        )
    return "\n".join(lines)


def reliability_section(run: BenchmarkRun) -> str:
    op = run.operational
    return "\n".join(
        [
            "## Reliability",
            f"- Container restarts: {_cell(op.container_restarts)}",
            f"- Provider startup failures: {_cell(op.provider_startup_failures)}",
            f"- Timeouts: {_cell(op.timeouts)}",
            f"- Telemetry completeness: {_cell(op.telemetry_completeness)}",
            f"- Pipeline completion rate: {_cell(op.pipeline_completion_rate)}",
        ]
    )


def trust_section(run: BenchmarkRun) -> str:
    t = run.trust
    return "\n".join(
        [
            "## Trust",
            f"- Agreement distribution: {_cell(t.agreement_distribution)}",
            f"- Conflicts: {_cell(t.conflict_count)}",
            f"- Evidence density: {_cell(t.evidence_density)}",
            f"- Average providers agreeing: {_cell(t.average_providers_agreeing)}",
            f"- Average providers disagreeing: {_cell(t.average_providers_disagreeing)}",
            f"- Low-confidence observations: {_cell(t.low_confidence_observations)}",
            f"- High-confidence observations: {_cell(t.high_confidence_observations)}",
        ]
    )


def performance_section(run: BenchmarkRun) -> str:
    lines = ["## Performance", "", "| Provider | GPU peak | Cold startup | CPU util. |", "|---|---|---|---|"]
    for r in run.runtime:
        lines.append(
            f"| {r.provider_id} ({r.provider_version}) | {_cell(r.peak_gpu_memory_bytes)} | "
            f"{_cell(r.cold_startup_seconds)} | {_cell(r.cpu_utilization_percent)} |"
        )
    return "\n".join(lines)


def calibration_section(run: BenchmarkRun) -> str | None:
    """`None` for a run persisted before Milestone 11 (`run.calibration` is `None`) -- omitted
    from `full_report` entirely rather than rendered as a fabricated "unavailable" section."""
    c = run.calibration
    if c is None:
        return None
    return "\n".join(
        [
            "## Confidence Calibration",
            f"- Evidence patterns: {_cell(c.patterns_total)}",
            f"- Patterns with calibration evidence: {_cell(c.patterns_with_calibration_evidence)}",
            f"- Patterns statistically mature (+/-2%): {_cell(c.patterns_statistically_mature)}",
            f"- Calibration-intent reviews recorded: {_cell(c.calibration_reviews_total)}",
            f"- Average progress toward +/-10% requirement: {_cell(c.average_progress_pct_of_loosest_requirement)}",
            f"- Blind spots (zero calibration evidence): {_cell(c.blind_spots_total)} "
            f"({_cell(c.blind_spot_corpus_share)} of corpus)",
        ]
    )


def full_report(run: BenchmarkRun) -> str:
    sections = [
        executive_summary(run),
        provider_comparison_table(run),
        observation_type_table(run),
        reliability_section(run),
        trust_section(run),
        performance_section(run),
    ]
    calibration = calibration_section(run)
    if calibration is not None:
        sections.append(calibration)
    return "\n\n".join(sections)


def public_benchmark_text(run: BenchmarkRun) -> str:
    """A concise, publication-suitable summary — aggregate statistics only. Never includes a
    document ref, archive object ref, provider raw output, or any other content-bearing field;
    only counts and rates from already-aggregated metric groups pass through this function."""
    d, t, h = run.dataset, run.trust, run.human_review

    def line_or_skip(label: str, metric: Metric) -> str | None:
        if metric.status == MetricStatus.UNAVAILABLE:
            return None
        return f"{label}:\n{_cell(metric)}\n"

    parts = [
        "ArchiveTrust Benchmark",
        "",
        f"Dataset:\n{run.metadata.dataset_id}",
        "",
    ]
    for label, metric in (
        ("Documents processed", d.documents_processed),
        ("Pages processed", d.pages_processed),
        ("Overall pipeline completion", run.operational.pipeline_completion_rate),
        ("Human review", h.review_percentage),
        ("Average canonical confidence", t.average_canonical_confidence),
    ):
        rendered = line_or_skip(label, metric)
        if rendered:
            parts.append(rendered)
    return "\n".join(parts).strip() + "\n"
