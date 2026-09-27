"""`python -m archivetrust.htr.benchmark <command>` (also `archivetrust-benchmark`).

    inspect  <source_id>                          report on incoming/<source_id> (never modifies it)
    build    <source_id> --dataset-id ID          candidate manifest + review queue (uses work/<source_id>/decisions.jsonl)
    freeze   <source_id> <benchmark_id>           immutable benchmark/<benchmark_id>
    verify   <benchmark_id>                       re-hash a frozen benchmark
    run      <benchmark_id> <run_id> --model M    predictions for one model (loghi | lion)
    score    <benchmark_id> <run_id>              scores.json + report.md from both prediction files
    overlap-index <parquet_dir>                   index the HF training corpus for contamination checks
    overlap  <benchmark_id>                       check a frozen benchmark against that index
    check    [--benchmark ID]                     readiness check of environment, model and data
    dryrun-build <pages_dir> <segmentation.jsonl> <dryrun_id>
                                                  NO-GT mechanical dry run: imported segmentation -> crops -> manifest
    dryrun-run <dryrun_id> --model M --limit N    smoke inference on a dry run (never scored)

Data root: benchmark-data/ in the repo, or $ARCHIVETRUST_BENCHMARK_ROOT, or --root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from archivetrust.htr.benchmark.layout import BenchmarkLayout


def _layout(args) -> BenchmarkLayout:
    return BenchmarkLayout(Path(args.root)) if args.root else BenchmarkLayout.default()


def _dump(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def cmd_inspect(args) -> int:
    from archivetrust.htr.benchmark.inspection import inspect_source, write_inspection  # noqa: PLC0415
    from archivetrust.htr.benchmark.report import default_loghi_charset  # noqa: PLC0415

    layout = _layout(args)
    result = inspect_source(layout.incoming_source(args.source_id), adapter_id=args.adapter, charset=default_loghi_charset())
    out = layout.inspection_dir(args.source_id)
    write_inspection(result, out)
    print((out / "inspection.md").read_text(encoding="utf-8").split("## Findings")[0])
    print(f"Full report: {out / 'inspection.md'}\nFindings:    {out / 'findings.jsonl'}")
    return 0


def cmd_build(args) -> int:
    from archivetrust.htr.benchmark.build import build_candidate  # noqa: PLC0415

    layout = _layout(args)
    summary = build_candidate(layout.incoming_source(args.source_id), layout.candidate_dir(args.source_id),
                              dataset_id=args.dataset_id, source_id=args.source_id, adapter_id=args.adapter,
                              crop_policy=args.crop_policy, decisions_path=layout.decisions_file(args.source_id),
                              overwrite=args.overwrite)
    print(f"included {summary.included}, review queue {summary.review}, excluded {summary.excluded}, "
          f"unresolved dataset-level findings {len(summary.unresolved_dataset_findings)}")
    print(f"candidate: {summary.candidate_dir}  (manifest sha256 {summary.manifest_sha256})")
    if summary.review or summary.unresolved_dataset_findings:
        print(f"Next: decide the items in review_queue.jsonl / unresolved_dataset_findings.jsonl in "
              f"{layout.decisions_file(args.source_id)}, then rebuild with --overwrite.")
    return 0


def cmd_freeze(args) -> int:
    from archivetrust.htr.benchmark.build import freeze  # noqa: PLC0415

    layout = _layout(args)
    card = json.loads(Path(args.dataset_card).read_text(encoding="utf-8")) if args.dataset_card else None
    record = freeze(layout.candidate_dir(args.source_id), layout.frozen_dir(args.benchmark_id), benchmark_id=args.benchmark_id,
                    exclude_unresolved=args.exclude_unresolved, official=args.official, dataset_card=card)
    print(f"frozen {record['lines']} lines -> {layout.frozen_dir(args.benchmark_id)}\nmanifest sha256 {record['manifest_sha256']}")
    return 0


def cmd_verify(args) -> int:
    from archivetrust.htr.benchmark.build import verify_frozen  # noqa: PLC0415

    record, lines, findings = verify_frozen(_layout(args).frozen_dir(args.benchmark_id))
    print(f"{args.benchmark_id}: {len(lines)} lines, manifest {record['manifest_sha256']}")
    for f in findings:
        print(f"  {f.severity} {f.code} {f.line_id or f.path or ''}: {f.message}")
    print("OK" if not findings else f"FAILED ({len(findings)} problems)")
    return 0 if not findings else 1


def cmd_run(args) -> int:
    from archivetrust.htr.benchmark.inference import make_backend, run_model  # noqa: PLC0415

    layout = _layout(args)
    frozen = layout.frozen_dir(args.benchmark_id)
    kwargs = {"device": args.device} if args.model == "lion" else {}
    backend = make_backend(args.model, args.profile, frozen_dir=frozen, **kwargs)
    record = run_model(frozen, layout.report_dir(args.run_id), backend, official=args.official, limit=args.limit)
    print(f"{record['model']['model_id']} [{record['model']['decoding_profile']}]: {record['status_counts']} -> "
          f"{layout.report_dir(args.run_id) / 'predictions' / record['predictions_file']}")
    return 0


def cmd_dryrun_build(args) -> int:
    from archivetrust.htr.benchmark.dryrun import build_dryrun  # noqa: PLC0415

    layout = _layout(args)
    record = build_dryrun(Path(args.pages_dir), Path(args.segmentation), layout.dryrun_dir(args.dryrun_id),
                          dryrun_id=args.dryrun_id, max_pages=args.max_pages)
    _dump({k: record[k] for k in ("label", "dryrun_id", "pages_requested", "pages_cropped", "lines", "findings", "manifest_sha256")})
    return 0


def cmd_dryrun_run(args) -> int:
    from archivetrust.htr.benchmark.dryrun import run_dryrun  # noqa: PLC0415
    from archivetrust.htr.benchmark.inference import make_backend  # noqa: PLC0415

    layout = _layout(args)
    directory = layout.dryrun_dir(args.dryrun_id)
    kwargs = {"device": args.device} if args.model == "lion" else {}
    backend = make_backend(args.model, args.profile, frozen_dir=directory, **kwargs)
    record = run_dryrun(directory, backend, limit=args.limit)
    print(f"[{record['label']}] {record['model']['model_id']} [{record['model']['decoding_profile']}]: "
          f"{record['status_counts']} -> {directory / 'predictions' / record['predictions_file']}")
    return 0


def cmd_score(args) -> int:
    from archivetrust.htr.benchmark.report import default_loghi_charset, score_run  # noqa: PLC0415

    layout = _layout(args)
    files = {"loghi": args.loghi, "lion": args.lion}
    score_run(layout.frozen_dir(args.benchmark_id), layout.report_dir(args.run_id), prediction_files=files,
              official=args.official, loghi_charset=default_loghi_charset())
    print((layout.report_dir(args.run_id) / "report.md").read_text(encoding="utf-8"))
    return 0


def cmd_overlap_index(args) -> int:
    from archivetrust.htr.benchmark.overlap import build_training_index  # noqa: PLC0415

    layout = _layout(args)
    shards = sorted(Path(args.parquet_dir).rglob("*.parquet"))
    if not shards:
        print(f"no .parquet files under {args.parquet_dir}")
        return 1
    _dump(build_training_index(shards, Path(args.index) if args.index else layout.root / "overlap" / "training_index.sqlite"))
    return 0


def cmd_overlap(args) -> int:
    from archivetrust.htr.benchmark.overlap import check_overlap  # noqa: PLC0415

    layout = _layout(args)
    index = Path(args.index) if args.index else layout.root / "overlap" / "training_index.sqlite"
    if not index.is_file():
        print(f"no training index at {index}; build one with `overlap-index <parquet_dir>` (the HF corpus is required)")
        return 1
    result = check_overlap(layout.frozen_dir(args.benchmark_id), index, layout.root / "overlap" / args.benchmark_id)
    _dump({k: result[k] for k in ("lines_checked", "lines_with_strong_signal", "signal_counts", "caveat")})
    return 0


# --- readiness check -----------------------------------------------------------------------------


def _probe(argv: list[str], timeout: int = 60) -> tuple[bool, str]:
    if shutil.which(argv[0]) is None:
        return False, f"{argv[0]} not found on PATH"
    try:
        done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return done.returncode == 0, (done.stdout or done.stderr).strip().splitlines()[0] if (done.stdout or done.stderr).strip() else ""


def readiness(layout: BenchmarkLayout, benchmark_id: str | None = None) -> list[tuple[str, str, str]]:
    """(status PASS/WARN/FAIL, check, detail). FAIL = cannot run an official benchmark."""
    from archivetrust.htr.benchmark.model_registry import LOGHI, ModelIdentityError, verify_loghi_checkpoint  # noqa: PLC0415
    from archivetrust.htr.benchmark.provenance import git_state  # noqa: PLC0415

    rows: list[tuple[str, str, str]] = []
    rows.append(("PASS" if sys.version_info >= (3, 11) else "FAIL", "python >= 3.11", sys.version.split()[0]))
    for module, needed_for in (("PIL", "everything"), ("pydantic", "everything"), ("pyarrow", "parquet sources, overlap index")):
        ok = importlib.util.find_spec(module) is not None
        rows.append(("PASS" if ok else ("FAIL" if needed_for == "everything" else "WARN"), f"import {module}", needed_for))
    try:
        verify_loghi_checkpoint()
        rows.append(("PASS", "Loghi checkpoint", f"{LOGHI.model_id}: 3 files match pinned SHA-256"))
    except (ModelIdentityError, OSError) as exc:
        rows.append(("FAIL", "Loghi checkpoint", str(exc)[:300]))
    ok, detail = _probe(["docker", "info", "--format", "{{.ServerVersion}}"])
    # `docker info` exits 0 with an empty version when the engine pipe exists but the backend is down.
    ok = ok and bool(detail.strip())
    rows.append(("PASS" if ok else "FAIL", "docker daemon",
                 f"server {detail}" if ok else (detail or "not reachable -- start Docker Desktop")))
    if ok:
        ok_img, detail_img = _probe(["docker", "image", "inspect", "--format", "{{.Id}}", LOGHI.pinned["container_image"]])
        rows.append(("PASS" if ok_img else "FAIL", "Loghi container image", LOGHI.pinned["container_image"] if ok_img else
                     f"not present locally: docker pull {LOGHI.pinned['container_image']}"))
    ok, detail = _probe(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    rows.append(("PASS" if ok else "WARN", "NVIDIA GPU", detail))
    lion_deps = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
    rows.append(("PASS" if not lion_deps else "FAIL", "Lion dependencies (torch, transformers)",
                 "importable" if not lion_deps else f"missing {lion_deps} (pip install -e .[transformers])"))
    git = git_state()
    rows.append(("PASS" if git["dirty"] is False else "WARN", "benchmark code committed",
                 f"commit {str(git['commit'])[:12]}" + (f", {git['dirty_path_count']} uncommitted code paths (official runs refused)"
                                                        if git["dirty"] else "")))
    rows.append(("PASS" if layout.root.is_dir() else "WARN", "data root", str(layout.root)))
    if layout.incoming.is_dir():
        for source in sorted(p for p in layout.incoming.iterdir() if p.is_dir()):
            inspection = layout.inspection_dir(source.name) / "inspection.json"
            verdict = json.loads(inspection.read_text(encoding="utf-8"))["verdict"] if inspection.is_file() else "not inspected"
            rows.append(("PASS" if verdict == "ready_to_build" else "WARN", f"incoming/{source.name}", verdict))
    if benchmark_id:
        from archivetrust.htr.benchmark.build import verify_frozen  # noqa: PLC0415

        try:
            _, lines, findings = verify_frozen(layout.frozen_dir(benchmark_id))
            rows.append(("PASS" if not findings else "FAIL", f"benchmark/{benchmark_id}",
                         f"{len(lines)} lines verified" if not findings else f"{len(findings)} integrity problems"))
        except (OSError, ValueError) as exc:
            rows.append(("FAIL", f"benchmark/{benchmark_id}", str(exc)[:200]))
    return rows


def cmd_check(args) -> int:
    rows = readiness(_layout(args), args.benchmark)
    width = max(len(r[1]) for r in rows)
    for status, name, detail in rows:
        print(f"[{status:4}] {name.ljust(width)}  {detail}")
    failed = [r for r in rows if r[0] == "FAIL"]
    print(f"\n{'READY' if not failed else 'NOT READY'}: {len(failed)} fail, {sum(r[0] == 'WARN' for r in rows)} warn")
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="archivetrust-benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="benchmark data root (default: benchmark-data/ or $ARCHIVETRUST_BENCHMARK_ROOT)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("source_id")
    p.add_argument("--adapter", choices=["page_xml", "alto_xml", "line_pairs", "tabular"])
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("build")
    p.add_argument("source_id")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--adapter", choices=["page_xml", "alto_xml", "line_pairs", "tabular"])
    p.add_argument("--crop-policy", default="bbox_v1", choices=["bbox_v1", "polygon_mask_v1"])
    p.add_argument("--overwrite", action="store_true", help="replace the existing candidate (derived data only)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("freeze")
    p.add_argument("source_id")
    p.add_argument("benchmark_id")
    p.add_argument("--exclude-unresolved", action="store_true")
    p.add_argument("--official", action="store_true")
    p.add_argument("--dataset-card", help="JSON file of model-independent reference facts/caveats, embedded in FROZEN.json")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("verify")
    p.add_argument("benchmark_id")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("run")
    p.add_argument("benchmark_id")
    p.add_argument("run_id")
    p.add_argument("--model", required=True, choices=["loghi", "lion"])
    p.add_argument("--profile", help="decoding profile (default: the model's registered default)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Lion only")
    p.add_argument("--limit", type=int, help="smoke test on the first N lines (never official, never scorable)")
    p.add_argument("--official", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score")
    p.add_argument("benchmark_id")
    p.add_argument("run_id")
    p.add_argument("--loghi", default="loghi.jsonl")
    p.add_argument("--lion", default="lion.jsonl")
    p.add_argument("--official", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("overlap-index")
    p.add_argument("parquet_dir")
    p.add_argument("--index")
    p.set_defaults(func=cmd_overlap_index)

    p = sub.add_parser("overlap")
    p.add_argument("benchmark_id")
    p.add_argument("--index")
    p.set_defaults(func=cmd_overlap)

    p = sub.add_parser("check")
    p.add_argument("--benchmark")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("dryrun-build", help="NO_GROUND_TRUTH mechanical dry run from pages + imported segmentation")
    p.add_argument("pages_dir")
    p.add_argument("segmentation", help="JSONL line segmentation (see archivetrust.htr.benchmark.dryrun)")
    p.add_argument("dryrun_id")
    p.add_argument("--max-pages", type=int)
    p.set_defaults(func=cmd_dryrun_build)

    p = sub.add_parser("dryrun-run", help="smoke inference on a dry run (not an accuracy measurement)")
    p.add_argument("dryrun_id")
    p.add_argument("--model", required=True, choices=["loghi", "lion"])
    p.add_argument("--profile", help="decoding profile (default: the model's registered primary)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Lion only")
    p.add_argument("--limit", type=int, help="first N lines only")
    p.set_defaults(func=cmd_dryrun_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
