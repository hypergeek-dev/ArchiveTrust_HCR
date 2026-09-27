"""Runs one frozen model over one frozen benchmark and writes its predictions as immutable evidence.

Fairness rules enforced here:
- Both models read the *same files*: the frozen benchmark's `lines/`, re-verified against the
  manifest hashes immediately before and after the run.
- Every manifest line gets exactly one prediction record. A line the model failed on, or that the
  backend silently skipped (Loghi drops unreadable images from its list), is recorded with status
  `failed` / `missing` and scored later as an empty prediction -- never dropped.
- The model's files are verified against the pinned identity, and the decoding profile is written
  in full into the run record.
- Prediction files are never overwritten.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.benchmark.build import verify_frozen
from archivetrust.htr.benchmark.contract import BenchmarkLine, sha256_file
from archivetrust.htr.benchmark.model_registry import (
    LION,
    LOGHI,
    DecodingProfile,
    ModelIdentity,
    loghi_checkpoint_dir,
    resolve_model,
    verify_loghi_checkpoint,
)
from archivetrust.htr.benchmark.normalization import canonicalize_prediction
from archivetrust.htr.benchmark.provenance import provenance, utc_now

PREDICTION_SCHEMA = "benchmark-prediction/1"
Status = Literal["ok", "empty", "failed", "missing"]


class RunError(RuntimeError):
    pass


class PredictionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_id: Literal["benchmark-prediction/1"] = PREDICTION_SCHEMA
    benchmark_id: str
    manifest_sha256: str
    dataset_id: str
    document_id: str
    page_id: str
    line_id: str
    image_sha256: str
    model_id: str
    decoding_profile: str
    status: Status
    prediction_raw: str | None
    """Exactly what the model emitted (None when it emitted nothing usable)."""
    prediction: str
    """`canonicalize_prediction(prediction_raw)`; "" for failed/missing lines. What gets scored."""
    confidence: float | None = None
    duration_seconds: float | None = None
    duration_kind: Literal["per_line", "batch_amortized"] | None = None
    error_category: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BackendResult:
    line_id: str
    status: Status
    text: str | None = None
    confidence: float | None = None
    duration_seconds: float | None = None
    duration_kind: Literal["per_line", "batch_amortized"] | None = None
    error_category: str | None = None
    error_message: str | None = None


class RecognizerBackend(Protocol):
    model: ModelIdentity
    profile: DecodingProfile

    def environment(self) -> dict: ...

    def recognize(self, items: Sequence[tuple[str, Path]], work_dir: Path) -> Iterable[BackendResult]: ...


# --- Loghi ---------------------------------------------------------------------------------------


class LoghiBackend:
    """Runs the pinned Loghi container over the frozen line images (mounted read-only). The checkpoint
    is verified, then *copied* into the work dir because the container writes tokenizer.json back into
    its model directory -- the same staging discipline as scripts/evaluate_full_run_lap.py."""

    model = LOGHI

    def __init__(self, profile: str | None = None, *, checkpoint_dir: Path | None = None, frozen_dir: Path,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, gpu_flag: str = "all") -> None:
        self.profile = LOGHI.profile(profile)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else loghi_checkpoint_dir()
        self.frozen_dir = Path(frozen_dir)
        self.runner = runner
        self.gpu_flag = gpu_flag
        self._verified: dict | None = None
        self._argv: list[str] | None = None

    def environment(self) -> dict:
        return {"checkpoint": self._verified, "container_image": LOGHI.pinned["container_image"], "argv": self._argv}

    def decoding_args(self) -> list[str]:
        p = self.profile.parameters
        args = ["--beam_width", str(p["beam_width"]), "--seed", str(p["seed"])]
        return args + (["--greedy"] if p["greedy"] else [])

    def recognize(self, items: Sequence[tuple[str, Path]], work_dir: Path) -> Iterable[BackendResult]:
        from archivetrust.htr.training.full_run.lap_evaluation import build_inference_argv, parse_results_file  # noqa: PLC0415

        self._verified = verify_loghi_checkpoint(self.checkpoint_dir)
        staged = work_dir / "staged_checkpoint"
        shutil.copytree(self.checkpoint_dir, staged)
        verify_loghi_checkpoint(staged)
        lists = work_dir / "lists"
        lists.mkdir()
        container_path: dict[str, str] = {}
        for line_id, image in items:
            relative = image.resolve().relative_to(self.frozen_dir.resolve()).as_posix()
            container_path[f"/benchmark/{relative}"] = line_id
        (lists / "inference_list.txt").write_text("".join(f"{p}\n" for p in container_path), encoding="utf-8")
        output = work_dir / "output"
        output.mkdir()
        self._argv = build_inference_argv(
            image_ref=LOGHI.pinned["container_image"], model_host=staged, output_host=output,
            inference_list_host=lists / "inference_list.txt", batch_size=self.profile.parameters["batch_size"],
            gpu_flag=self.gpu_flag, extra_volumes=[f"{self.frozen_dir.resolve()}:/benchmark:ro"], extra_args=self.decoding_args(),
        )
        started = time.monotonic()
        completed = self.runner(self._argv, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace")
        elapsed = time.monotonic() - started
        (work_dir / "container_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (work_dir / "container_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        results_path = output / "results.txt"
        if completed.returncode != 0 or not results_path.is_file():
            raise RunError(f"Loghi container exited {completed.returncode} without results; see {work_dir / 'container_stderr.txt'}")
        results = parse_results_file(results_path.read_text(encoding="utf-8"))
        per_line = elapsed / max(1, len(items))
        for path, line_id in container_path.items():
            if path not in results:
                yield BackendResult(line_id, "missing", error_category="not_in_results",
                                    error_message="the container produced no result for this image (skipped or unreadable)")
                continue
            confidence, text = results[path]
            yield BackendResult(line_id, "ok" if canonicalize_prediction(text) else "empty", text=text, confidence=confidence,
                                duration_seconds=per_line, duration_kind="batch_amortized")


# --- Lion ----------------------------------------------------------------------------------------


class LionBackend:
    """In-process TrOCR through the existing Swedish Lion facade, with every generation parameter
    passed explicitly (`generation_kwargs`)."""

    model = LION

    def __init__(self, profile: str | None = None, *, facade=None, device: str = "auto") -> None:
        self.profile = LION.profile(profile)
        self.device = device
        self._facade = facade
        self._env: dict = {}

    def environment(self) -> dict:
        return self._env

    def recognize(self, items: Sequence[tuple[str, Path]], work_dir: Path) -> Iterable[BackendResult]:
        if self._facade is None:
            from archivetrust.providers.swedish_lion.facade import real_swedish_lion_facade  # noqa: PLC0415

            self._facade = real_swedish_lion_facade()
        pinned, params = LION.pinned, self.profile.parameters
        for line_id, image in items:
            result = self._facade.run_inference(
                image_path=str(image), device_request=self.device, model_id=pinned["hf_model"],
                model_revision=pinned["hf_model_revision"], processor_model_id=pinned["hf_processor"],
                processor_revision=pinned["hf_processor_revision"], num_beams=params["num_beams"], generation_kwargs=dict(params),
            )
            if result.get("ok"):
                if not self._env:
                    self._env = {k: result.get(k) for k in ("device_used", "gpu_name", "software_environment", "model_revision")}
                yield BackendResult(line_id, "ok", text=result.get("text"), duration_seconds=result.get("elapsed_seconds"),
                                    duration_kind="per_line")
            elif result.get("category") == "empty_output":
                yield BackendResult(line_id, "empty", text="", error_category="empty_output")
            else:
                yield BackendResult(line_id, "failed", error_category=result.get("category"), error_message=result.get("message"))


# --- runner --------------------------------------------------------------------------------------


def prediction_filename(model: ModelIdentity, profile: DecodingProfile) -> str:
    short = {LOGHI.model_id: "loghi", LION.model_id: "lion"}.get(model.model_id, model.model_id)
    return f"{short}.jsonl" if profile.profile_id == model.default_profile else f"{short}@{profile.profile_id}.jsonl"


def make_backend(model_name: str, profile: str | None, *, frozen_dir: Path, **kwargs) -> RecognizerBackend:
    model = resolve_model(model_name)
    if model is LOGHI:
        return LoghiBackend(profile, frozen_dir=frozen_dir, **kwargs)
    return LionBackend(profile, **kwargs)


def run_model(frozen_dir: Path, report_dir: Path, backend: RecognizerBackend, *, official: bool = False,
              limit: int | None = None) -> dict:
    frozen_dir, report_dir = Path(frozen_dir), Path(report_dir)
    record, lines, findings = verify_frozen(frozen_dir)
    if findings:
        raise RunError(f"frozen benchmark failed verification: {[f.code for f in findings][:5]}")
    if limit is not None:
        if official:
            raise RunError("an official run covers the whole benchmark; --limit is for smoke tests only")
        lines = lines[:limit]
    run_provenance = provenance(official=official)

    predictions_dir = report_dir / "predictions"
    target = predictions_dir / prediction_filename(backend.model, backend.profile)
    if target.exists():
        raise RunError(f"{target} exists; prediction files are immutable evidence -- use a new report directory")
    work_dir = report_dir / "work" / target.stem
    if work_dir.exists():
        raise RunError(f"{work_dir} exists from an earlier attempt; inspect and remove it first")
    work_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    by_id: dict[str, BenchmarkLine] = {line.line_id: line for line in lines}
    items = [(line.line_id, frozen_dir / line.line_image_path) for line in lines]
    started_at = utc_now()
    results: dict[str, BackendResult] = {}
    for result in backend.recognize(items, work_dir):
        if result.line_id not in by_id or result.line_id in results:
            raise RunError(f"backend returned an unexpected or duplicate line_id {result.line_id!r}")
        results[result.line_id] = result
    finished_at = utc_now()

    rows: list[PredictionRecord] = []
    for line in lines:
        result = results.get(line.line_id) or BackendResult(line.line_id, "missing", error_category="not_returned")
        rows.append(PredictionRecord(
            benchmark_id=record["benchmark_id"], manifest_sha256=record["manifest_sha256"], dataset_id=line.dataset_id,
            document_id=line.document_id, page_id=line.page_id, line_id=line.line_id, image_sha256=line.image_sha256,
            model_id=backend.model.model_id, decoding_profile=backend.profile.profile_id, status=result.status,
            prediction_raw=result.text, prediction=canonicalize_prediction(result.text) if result.status in ("ok", "empty") and result.text else "",
            confidence=result.confidence, duration_seconds=result.duration_seconds, duration_kind=result.duration_kind,
            error_category=result.error_category, error_message=result.error_message,
        ))

    _, _, after = verify_frozen(frozen_dir)
    if after:
        raise RunError("frozen benchmark changed during the run; predictions discarded")
    partial = target.with_suffix(".jsonl.partial")
    partial.write_text("".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                       encoding="utf-8")
    partial.rename(target)
    os.chmod(target, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

    run_record = {
        "benchmark": {"benchmark_id": record["benchmark_id"], "manifest_sha256": record["manifest_sha256"],
                      "frozen_dir": str(frozen_dir), "frozen_record_sha256": sha256_file(frozen_dir / "FROZEN.json")},
        "model": backend.model.record(backend.profile.profile_id),
        "decoding_rationale": backend.profile.rationale,
        "backend_environment": backend.environment(),
        "predictions_file": target.name,
        "predictions_sha256": sha256_file(target),
        "lines": len(rows),
        "limited_to": limit,
        "status_counts": dict(Counter(r.status for r in rows)),
        "prediction_outer_whitespace_stripped": sum(1 for r in rows if r.prediction_raw and r.prediction_raw != r.prediction_raw.strip()),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "provenance": run_provenance,
    }
    (predictions_dir / f"{target.stem}.run.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                                           encoding="utf-8")
    return run_record


def read_predictions(path: Path) -> list[PredictionRecord]:
    return [PredictionRecord.model_validate(json.loads(raw)) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]
