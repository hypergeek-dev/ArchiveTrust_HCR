"""Runs `SwedishLionAdapter` over the exact same 600 `InputCrop` files the sealed
`reliability-2026-07-31` run (SATRN + Florence-2) consumed, producing a directly comparable
reliability report -- no new segmentation, no accuracy metric, same heuristics and thresholds.

## Why this is a dedicated script rather than a third method inside `ReliabilityRunner`

`htr/screening/runner.py` and `htr/screening/run_state.py` hardcode
`RECOGNITION_METHOD_IDS = (SATRN_METHOD_ID, FLORENCE2_METHOD_ID)` in several places (the execution
loop, `RunProgress.progress()`, `summarize_pending()`), rather than reading it from
`ResolvedConfiguration.method_ids`. Generalizing that harness to an arbitrary method set is a real,
separately-scoped change touching the module that produced the already-sealed run's evidence; doing
it under time pressure for a one-off comparison risks that run's reproducibility guarantees for no
real benefit. This script instead composes the same underlying, already-trusted primitives directly:
`TaskIdentity` (task_identity.py), `RunState`/`RunPaths` (run_state.py, used read-only for the
original run and for this run's own resumability), `DurableHtrResearchStore`/`FileTelemetrySink`
(the same durable, hash-chained storage), `reliability_signals.evaluate_output` +
`ReliabilityThresholds` (the *same* default thresholds and heuristics version as the original run,
which is what makes the two reports genuinely comparable), and `run_report.write_report` (fully
generic over method_id -- no changes needed there at all).

## Crop reuse, not re-segmentation

The 600 `InputCrop` objects (byte-identical PNG files, real content hashes) are read directly from
`reliability-2026-07-31`'s own sealed telemetry via `RunState.load` + `selected_crops_for_page` --
the same evenly-spaced-over-detected-lines selection that run already made. This script re-hashes
every crop file from disk immediately before feeding it to the adapter (same integrity guard
`ReliabilityRunner._recognize` uses) but performs **no new Florence-2 line detection**. The crops'
`InputCrop` records are re-registered into this run's own new telemetry stream verbatim (same
crop_id, hash, storage_path -- pointing at the original run's `crops/` directory, not copied) so
this run's own log is self-contained and independently replayable.

## What is identical to the original run, and what differs

Copied verbatim from `reliability-2026-07-31`'s recorded `ResolvedConfiguration`: dataset identity,
corpus digest, the 60 sampled pages, crops-per-page, crop selection rule, segmentation adapter name
and configuration hash, preprocessing version, reliability heuristics version and thresholds hash.
Only `method_ids` (`("swedish_lion",)`) and `model_revisions` differ -- which correctly gives this
run its own `configuration_hash`/`experiment_version_id`, a genuinely different task-identity space,
never mistaken for a resume of the original.

## Resumability

Every unit of work's completion is a deterministic fact derivable from THIS run's own
`events.jsonl` (`TaskIdentity.method_run_id`), exactly as the interactive harness's runs are.
Re-running this script re-derives what is already done via `RunState.load` and skips it -- safe to
interrupt (Ctrl+C) and re-run.

Never touches `reliability-2026-07-31/events.jsonl` or any of its files.
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.htr.corpus.models import InputCrop, Page, ResearchProject  # noqa: E402
from archivetrust.htr.experiment.baseline_execution import InputCropHashMismatchError  # noqa: E402
from archivetrust.htr.experiment.models import (  # noqa: E402
    Experiment,
    ExperimentRun,
    ExperimentVersion,
    MethodRun,
    ReproducibilityManifest,
)
from archivetrust.htr.persistence.durable_store import DurableHtrResearchStore  # noqa: E402
from archivetrust.htr.screening.reliability_signals import (  # noqa: E402
    ReliabilityThresholds,
    evaluate_output,
)
from archivetrust.htr.screening.run_report import write_report  # noqa: E402
from archivetrust.htr.screening.run_state import (  # noqa: E402
    CONFIGURATION_MANIFEST_KEY,
    IntegrityError,
    RunPaths,
    RunState,
)
from archivetrust.infrastructure.storage.integrity import verify_hash_chain_sidecar  # noqa: E402
from archivetrust.infrastructure.storage.telemetry_sink import FileTelemetrySink  # noqa: E402
from archivetrust.providers.htr_adapter import RecognitionInput  # noqa: E402
from archivetrust.providers.swedish_lion.adapter import (  # noqa: E402
    ADAPTER_VERSION as SWEDISH_LION_ADAPTER_VERSION,
    SwedishLionAdapter,
    build_evidence,
    build_failure_record,
    build_observation_payloads,
)

ORIGINAL_RUN_ID = "reliability-2026-07-31"
COMPARISON_RUN_ID = "reliability-2026-07-31-swedish-lion"
SWEDISH_LION_METHOD_ID = "swedish_lion"
ACTOR_ID = "swedish-lion-comparison-script"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _runs_root() -> Path:
    return REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "full-run"


def main() -> int:
    runs_root = _runs_root()
    original_paths = RunPaths.for_run(runs_root, ORIGINAL_RUN_ID)
    original_state = RunState.load(original_paths)
    if not original_state.sealed:
        print(f"REFUSING: original run {ORIGINAL_RUN_ID!r} is not sealed.")
        return 1
    original_config = original_state.recorded_configuration
    if original_config is None:
        print(f"REFUSING: original run {ORIGINAL_RUN_ID!r} has no recorded configuration.")
        return 1

    adapter = SwedishLionAdapter()
    model_revision = adapter.get_metadata().model_revision

    new_config = original_config.model_copy(
        update={
            "method_ids": (SWEDISH_LION_METHOD_ID,),
            "model_revisions": {SWEDISH_LION_METHOD_ID: model_revision},
        }
    )

    comparison_paths = RunPaths.for_run(runs_root, COMPARISON_RUN_ID)
    sink = FileTelemetrySink(comparison_paths.events, blob_dir=comparison_paths.blobs)
    store = DurableHtrResearchStore.open(sink, actor_id=ACTOR_ID)
    state = RunState.load(comparison_paths)

    if state.sealed:
        print(f"{COMPARISON_RUN_ID} is already sealed -- regenerating report only.")
        write_report(comparison_paths)
        print(f"Report written to {comparison_paths.report}")
        return 0

    thresholds = ReliabilityThresholds()  # identical defaults to the original run
    created_at = _now_iso()
    project_id = _stable_id("research_project", "technical-reliability-screening")
    experiment_id = _stable_id(
        "experiment",
        "swedish-historical-htr-technical-reliability-screening-swedish-lion-comparison",
    )

    with store.correlated_to(comparison_paths.run_id):
        if store.project(project_id) is None:
            store.register_project(
                ResearchProject(
                    project_id=project_id,
                    name="Swedish Historical HTR Technical Reliability Screening",
                    description=(
                        "Reference-free technical reliability screening on real Swedish "
                        "witchcraft-trial court records. No ground truth exists for this corpus; "
                        "no accuracy metric is computed."
                    ),
                    created_at=created_at,
                )
            )
        if store.experiment(experiment_id) is None:
            store.register_experiment(
                Experiment(
                    experiment_id=experiment_id,
                    name="Swedish Lion comparison against reliability-2026-07-31",
                    description=(
                        f"Runs SwedishLionAdapter (Riksarkivet/trocr-base-handwritten-hist-swe-2) "
                        f"over the exact same 600 InputCrop files (byte-identical, re-verified by "
                        f"hash before every call) that {ORIGINAL_RUN_ID}'s sealed SATRN/Florence-2 "
                        "run consumed. No new segmentation is performed. Same dataset sample, same "
                        "preprocessing version, same reliability heuristics version and thresholds "
                        "as that run -- only the recognition method and its model revision differ, "
                        "so the two reports' per_method blocks are directly comparable. "
                        "Reference-free technical reliability screening only -- no CER/WER, no "
                        "ground truth."
                    ),
                    research_project_id=project_id,
                    created_at=created_at,
                )
            )
        version_id = new_config.experiment_version_id
        if store.experiment_version(version_id) is None:
            store.register_experiment_version(
                ExperimentVersion(
                    experiment_version_id=version_id,
                    experiment_id=experiment_id,
                    version=1,
                    dataset_version_id=new_config.dataset_version_id,
                    method_ids=new_config.method_ids,
                    segmentation_configuration_ref=new_config.segmentation_configuration_hash,
                    pipeline_configuration_ref=new_config.configuration_hash,
                    created_at=created_at,
                )
            )
        if not state.exists:
            run = ExperimentRun(
                experiment_run_id=comparison_paths.run_id,
                experiment_version_id=version_id,
                is_end_to_end=False,
                started_at=created_at,
            )
            store.register_experiment_run(run)
            store.register_manifest(
                ReproducibilityManifest(
                    manifest_id=_stable_id("reproducibility_manifest", comparison_paths.run_id),
                    experiment_run_id=comparison_paths.run_id,
                    software_environment={
                        CONFIGURATION_MANIFEST_KEY: new_config.model_dump(mode="json"),
                        "configuration_hash": new_config.configuration_hash,
                        "dataset_fingerprint": new_config.dataset_fingerprint,
                        "reused_crops_from_run_id": ORIGINAL_RUN_ID,
                        "reused_crops_note": (
                            "InputCrop bytes and content hashes are reused verbatim from "
                            f"{ORIGINAL_RUN_ID}'s sealed segmentation output -- no new Florence-2 "
                            "line detection was run. Verified by re-hashing each crop file from "
                            "disk immediately before recognition."
                        ),
                        "adapter_version": SWEDISH_LION_ADAPTER_VERSION,
                    },
                    pipeline_configuration_hash=new_config.configuration_hash,
                    created_at=created_at,
                )
            )
            state.experiment_run_id = comparison_paths.run_id
            state.started_at = created_at

        total = new_config.page_count * new_config.crops_per_page
        executed = 0
        skipped = 0
        failures: list[str] = []
        started = time.monotonic()

        for page_index, page in enumerate(new_config.pages, start=1):
            if store.page(page.page_id) is None:
                store.register_page(
                    Page(
                        page_id=page.page_id,
                        archive_object_ref=page.document_id,
                        page_number=page.page_number,
                        width=page.width,
                        height=page.height,
                    )
                )

            selected = original_state.selected_crops_for_page(original_config, page)
            if selected is None:
                print(f"REFUSING: original run has no completed segmentation for page {page.page_id}")
                return 1

            seg_identity = state.segmentation_identity(new_config, page)
            if seg_identity.segmentation_run_id not in state.completed_segmentation_run_ids:
                for _, crop in selected:
                    if store.input_crop(crop.crop_id) is None:
                        store.register_input_crop(crop)
                store.record_segmentation_run(
                    page_id=page.page_id,
                    segmentation_adapter_name=new_config.segmentation_adapter_name,
                    region_ids=(),
                    text_line_ids=(),
                    input_crop_ids=tuple(crop.crop_id for _, crop in selected),
                    segmentation_run_id=seg_identity.segmentation_run_id,
                )
                state.completed_segmentation_run_ids.add(seg_identity.segmentation_run_id)

            for position, (crop_index, crop) in enumerate(selected, start=1):
                identity = state.recognition_identity(
                    new_config,
                    page_id=page.page_id,
                    crop_index=crop_index,
                    crop_hash=crop.hash,
                    method_id=SWEDISH_LION_METHOD_ID,
                )
                if identity.method_run_id in state.completed_method_run_ids:
                    skipped += 1
                    continue

                crop_path = Path(crop.storage_path)
                if not crop_path.is_file():
                    raise IntegrityError(f"crop {crop.crop_id} file missing at {crop_path}")
                on_disk = crop_path.read_bytes()
                recomputed = InputCrop.compute_hash(on_disk)
                if recomputed != crop.hash:
                    raise InputCropHashMismatchError(
                        f"crop {crop.crop_id} on disk hashes to {recomputed!r}, recorded {crop.hash!r}"
                    )

                started_at = _now_iso()
                wall_started = time.monotonic()
                result = adapter.recognize(RecognitionInput(input_crop_id=str(crop_path)))
                wall_seconds = time.monotonic() - wall_started
                completed_at = _now_iso()

                evidence = build_evidence(result)
                method_run = MethodRun(
                    method_run_id=identity.method_run_id,
                    experiment_run_id=comparison_paths.run_id,
                    method_id=SWEDISH_LION_METHOD_ID,
                    model_version_id=result.model_revision,
                    input_crop_id=crop.crop_id,
                    evidence_id=(
                        evidence.evidence_id
                        if evidence is not None
                        else _stable_id("evidence_missing", identity.task_key)
                    ),
                    outcome="succeeded" if result.text is not None else "failed",
                    started_at=started_at,
                    completed_at=completed_at,
                )
                failure = build_failure_record(result, method_run_id=method_run.method_run_id)

                started_event = store.register_method_run(method_run)
                if evidence is not None:
                    store.record_evidence(
                        evidence,
                        document_ref=page.document_id,
                        invocation_id=method_run.method_run_id,
                        caused_by=started_event,
                    )
                payloads = build_observation_payloads(result)
                if payloads is not None:
                    raw_payload, parsed_payload, normalized_payload = payloads
                    from archivetrust.htr.research_store import MethodRunTranscript  # noqa: PLC0415

                    store.register_transcript(
                        MethodRunTranscript(
                            method_run_id=method_run.method_run_id,
                            raw_text=raw_payload.text,
                            parsed_text=parsed_payload.text,
                            normalized_text=normalized_payload.text,
                        ),
                        caused_by=started_event,
                        evidence_id=method_run.evidence_id,
                    )

                signals = evaluate_output(
                    result.text,
                    method=SWEDISH_LION_METHOD_ID,
                    raw_response=result.raw_response,
                    thresholds=thresholds,
                )
                for classification, flagged, detail in (
                    ("execution_failed", not signals.produced_output, signals.failure_message),
                    ("empty_or_whitespace_only", signals.empty_or_whitespace_only, None),
                    ("degenerate_repetition", signals.degenerate_repetition, None),
                    ("very_short_output", signals.very_short_output, None),
                    ("truncation_suspected", signals.truncation_suspected, None),
                ):
                    if flagged:
                        store.record_reliability_issue(
                            method_run_id=method_run.method_run_id,
                            classification=classification,
                            detail=detail,
                            caused_by=started_event,
                        )
                crop_aspect = (crop.width / crop.height) if (crop.width and crop.height) else None
                if crop_aspect is not None and crop_aspect < thresholds.crop_min_aspect_ratio:
                    store.record_reliability_issue(
                        method_run_id=method_run.method_run_id,
                        classification="crop_geometry_implausible",
                        detail=f"width/height = {crop_aspect:.3f}",
                        caused_by=started_event,
                    )

                store.complete_method_run(
                    method_run,
                    caused_by=started_event,
                    failure_reason=failure.reason if failure is not None else None,
                )
                state.completed_method_run_ids.add(method_run.method_run_id)
                executed += 1
                if result.text is None:
                    failures.append(f"crop {crop.crop_id[:20]}: {result.raw_response.get('message')}")

                done = executed + skipped
                if done % 25 == 0 or done == total:
                    elapsed = time.monotonic() - started
                    print(
                        f"[{done}/{total}] page {page_index}/{len(new_config.pages)} "
                        f"crop {position}/{len(selected)} {'ok' if result.text is not None else 'FAIL'} "
                        f"{wall_seconds:.2f}s (elapsed {elapsed:.0f}s, skipped {skipped})"
                    )

        completed_run = ExperimentRun(
            experiment_run_id=comparison_paths.run_id,
            experiment_version_id=version_id,
            is_end_to_end=False,
            started_at=state.started_at or created_at,
            completed_at=_now_iso(),
        )
        store.complete_experiment_run(completed_run)

    verification = verify_hash_chain_sidecar(comparison_paths.events)
    print(f"Hash chain verified: {verification.ok}")
    report_path = write_report(comparison_paths)
    print(f"Executed {executed}, skipped {skipped} (already done), {len(failures)} failures.")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
