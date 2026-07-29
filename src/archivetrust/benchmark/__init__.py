"""ArchiveTrust Benchmark & Measurement Framework (2026-07-13).

Measurement, not optimization: every function in this package reads telemetry ArchiveTrust already
records and reports what happened. Nothing here invokes a provider, changes provider behaviour, or
tunes anything. See `docs/BENCHMARK_FRAMEWORK_2026-07-13.md` for the full design and validation
report.

Typical use:

    from archivetrust.benchmark.aggregate import (
        dataset_metrics, provider_runtime_metrics, provider_quality_metrics,
        observation_type_comparison, trust_metrics, human_review_metrics, operational_metrics,
    )
    from archivetrust.benchmark.run import BenchmarkMetadata, BenchmarkRun, BenchmarkRunStore
    from archivetrust.benchmark.report import full_report, public_benchmark_text
    from archivetrust.benchmark.export import to_csv, to_json

    run = BenchmarkRun(
        metadata=BenchmarkMetadata(run_id=..., run_at=..., dataset_id=...),
        dataset=dataset_metrics(source, progress_events=progress),
        runtime=provider_runtime_metrics(source, runtime_events=runtime_events),
        quality=provider_quality_metrics(source),
        observation_types=observation_type_comparison(source),
        trust=trust_metrics(source),
        human_review=human_review_metrics(source),
        operational=operational_metrics(source, runtime_events=runtime_events, progress_events=progress),
    )
    store.append(run)
    print(full_report(run))
"""

from __future__ import annotations
