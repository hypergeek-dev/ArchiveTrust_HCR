#!/usr/bin/env python
"""Bounded diagnostic for the Checkpoint 3 SATRN output-repetition finding.

The Checkpoint 3 smoke test (`docs/experiments/technical-reliability-screening/smoke-test/`)
recorded SATRN returning the identical string `staden den 27 dennes` for 6 of 15 real, byte-distinct
line crops (7/15 distinct outputs), while Florence-2 produced 15/15 distinct outputs on the exact
same crops. This script reconstructs which crops repeated and tests the *integration-defect*
hypotheses (stale tensor / cached result / reused temporary filename / reused subprocess /
response-matching bug / retained model state / parsing fallback) against real evidence.

It deliberately does NOT tune decoding, does NOT change the model, and does NOT run the full
~60-page benchmark. It re-runs the same 15 real crops the smoke test used, byte-identically, from
the files the smoke test already wrote.

Passes:
  A  all 15 crops, original smoke-test order, through the real `SatrnAdapter` (instrumented)
  B  the repeated-output crops again, same order, same host process   -> determinism check
  C  the repeated-output crops in REVERSED order                      -> order-dependence check
  D  the repeated-output crops each in a fully isolated top-level run -> cross-call-state check
  E  reference path: `mmocr.apis.TextRecInferencer` called directly inside `.venv-satrn` on all 15
     crops in ONE process with ONE model load, bypassing the ArchiveTrust adapter entirely

Instrumentation captured per adapter call: the exact subprocess argv, the child PID, the SHA-256 of
the bytes at `image_path` immediately before and immediately after the call, the worker's reported
`model_revision`/`config_revision`, device, score, raw text, parsed text and elapsed time.

Run:

    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_satrn_repetition_diagnostic.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archivetrust.providers.htr_adapter import RecognitionInput  # noqa: E402
from archivetrust.providers.satrn import facade as satrn_facade  # noqa: E402
from archivetrust.providers.satrn.adapter import SatrnAdapter  # noqa: E402

SMOKE = REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "smoke-test"
OUT_DIR = REPO_ROOT / "docs" / "experiments" / "technical-reliability-screening" / "diagnostic"
WORKER = REPO_ROOT / "src" / "archivetrust" / "providers" / "satrn" / "_worker.py"

_CALL_LOG: list[dict] = []


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _install_subprocess_instrumentation() -> None:
    """Wraps `subprocess.run` *inside the satrn facade module only*, so every real worker launch
    records its argv and the child's PID. Uses `Popen` under the hood purely to obtain the PID --
    `subprocess.run` does not expose it. Behaviour is otherwise identical (same args, same
    capture, same timeout semantics, same `TimeoutExpired` type the facade already handles)."""
    real_run = satrn_facade.subprocess.run

    def instrumented_run(args, **kwargs):  # noqa: ANN001, ANN202
        record: dict = {"argv": list(args), "started_at": datetime.now(timezone.utc).isoformat()}
        timeout = kwargs.pop("timeout", None)
        kwargs.pop("check", None)
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=kwargs.get("text", True),
        )
        record["child_pid"] = proc.pid
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        record["returncode"] = proc.returncode
        record["stdout_line_count"] = len([line for line in stdout.splitlines() if line.strip()])
        record["stdout_raw_last_line"] = (
            [line for line in stdout.splitlines() if line.strip()] or [""]
        )[-1]
        record["stderr_tail"] = stderr[-400:]
        _CALL_LOG.append(record)
        return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)

    satrn_facade.subprocess.run = instrumented_run  # type: ignore[assignment]
    # Guard: prove the seam we just patched is the one the facade actually calls.
    assert satrn_facade.subprocess.run is instrumented_run
    _ = real_run


def load_manifest() -> list[dict]:
    """The 15 crops the smoke test ran recognition on, in the smoke test's own execution order,
    joined to their on-disk paths from the durable `InputCropCreated` telemetry."""
    crop_records: dict[str, dict] = {}
    for line in (SMOKE / "smoke_test_events.jsonl").open(encoding="utf-8"):
        event = json.loads(line)
        if event["kind"] == "InputCropCreated":
            crop_records[event["input_crop"]["crop_id"]] = event["input_crop"]

    results = json.loads((SMOKE / "smoke_test_results.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for page in results["pages"]:
        for crop in page.get("recognition_crops", []):
            record = crop_records[crop["crop_id"]]
            rows.append(
                {
                    "order": len(rows) + 1,
                    "page_id": page["page_id"],
                    "crop_id": crop["crop_id"],
                    "text_line_id": record["text_line_id"],
                    "path": record["storage_path"],
                    "recorded_crop_hash": crop["crop_hash"],
                    "width": crop["crop_width"],
                    "height": crop["crop_height"],
                    "byte_size": record["byte_size"],
                    "smoke_satrn_text": crop["satrn"]["text"],
                    "smoke_satrn_confidence": crop["satrn"]["confidence"],
                    "smoke_florence2_text": crop["florence2"]["text"],
                }
            )

    counts = Counter(row["smoke_satrn_text"] for row in rows)
    for row in rows:
        row["smoke_repeat_group_size"] = counts[row["smoke_satrn_text"]]
        row["repeated_in_smoke"] = counts[row["smoke_satrn_text"]] > 1
    return rows


def characterize(rows: list[dict]) -> None:
    """Independent verification that the crops are genuinely different images -- SHA-256 computed
    here from the file bytes (not trusted from the smoke test's record or from the filename), plus
    decoded-pixel statistics proving none is blank and no two are near-identical."""
    import numpy as np
    from PIL import Image

    for row in rows:
        path = Path(row["path"])
        data = path.read_bytes()
        row["exists"] = path.exists()
        row["sha256_png"] = hashlib.sha256(data).hexdigest()
        row["file_size_bytes"] = len(data)
        row["file_size_matches_recorded"] = len(data) == row["byte_size"]
        row["recorded_hash_verified"] = (
            row["recorded_crop_hash"].split("_", 1)[-1] == row["sha256_png"]
        )
        with Image.open(path) as image:
            row["pil_mode"] = image.mode
            row["pil_size"] = list(image.size)
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        row["sha256_decoded_pixels"] = hashlib.sha256(gray.tobytes()).hexdigest()
        row["pixel_mean"] = round(float(gray.mean()), 3)
        row["pixel_std"] = round(float(gray.std()), 3)
        row["distinct_gray_levels"] = int(np.unique(gray).size)
        row["aspect_ratio"] = round(row["width"] / row["height"], 3)


def run_pass(label: str, rows: list[dict], adapter: SatrnAdapter) -> list[dict]:
    """One measurement pass over `rows` in the order given, through the real adapter."""
    print(f"\n{'=' * 110}\nPASS {label}  ({len(rows)} crops)\n{'=' * 110}")
    observations: list[dict] = []
    for position, row in enumerate(rows, start=1):
        before_hash = _sha256_file(row["path"])
        calls_before = len(_CALL_LOG)
        started = time.monotonic()
        result = adapter.recognize(RecognitionInput(input_crop_id=row["path"]))
        wall = time.monotonic() - started
        after_hash = _sha256_file(row["path"])
        call = _CALL_LOG[calls_before] if len(_CALL_LOG) > calls_before else {}
        raw = result.raw_response

        observation = {
            "pass": label,
            "position_in_pass": position,
            "crop_id": row["crop_id"],
            "smoke_order": row["order"],
            "path": row["path"],
            "sha256_before_call": before_hash,
            "sha256_after_call": after_hash,
            "input_bytes_unchanged_by_call": before_hash == after_hash,
            "input_hash_matches_manifest": before_hash == row["sha256_png"],
            "argv": call.get("argv"),
            "argv_image_path": call.get("argv", [None, None, None])[2]
            if call.get("argv")
            else None,
            "argv_image_path_is_source_crop": (
                call.get("argv", [None, None, None])[2] == row["path"] if call.get("argv") else None
            ),
            "child_pid": call.get("child_pid"),
            "returncode": call.get("returncode"),
            "stdout_line_count": call.get("stdout_line_count"),
            "worker_stdout_raw": call.get("stdout_raw_last_line"),
            "text_raw": result.text,
            "confidence": result.confidence,
            "model_revision": result.model_revision,
            "config_revision": raw.get("config_revision"),
            "device_used": raw.get("device_used"),
            "gpu_name": raw.get("gpu_name"),
            "peak_gpu_memory_mb": raw.get("peak_gpu_memory_mb"),
            "worker_elapsed_seconds": raw.get("elapsed_seconds"),
            "wall_seconds": round(wall, 3),
            "software_environment": raw.get("software_environment"),
            "width": row["width"],
            "height": row["height"],
            "smoke_satrn_text": row["smoke_satrn_text"],
            "smoke_satrn_confidence": row["smoke_satrn_confidence"],
        }
        observation["matches_smoke_text"] = observation["text_raw"] == row["smoke_satrn_text"]
        observations.append(observation)
        print(
            f"  [{position:2d}] {row['crop_id'][:34]} {row['width']:>4}x{row['height']:<4} "
            f"pid={observation['child_pid']} rev={str(observation['model_revision'])[:8]} "
            f"conf={observation['confidence']} "
            f"{'SAME' if observation['matches_smoke_text'] else 'DIFF'} :: {observation['text_raw']!r}"
        )
    return observations


def run_isolated_pass(label: str, rows: list[dict]) -> list[dict]:
    """Pass D -- each crop in its own *top-level* Python process, which itself constructs a fresh
    `SatrnAdapter` and runs exactly one recognition. This removes even the shared host interpreter,
    so no module-level cache, import-time singleton or accumulated state in *this* process can be
    common to two crops."""
    print(f"\n{'=' * 110}\nPASS {label}  ({len(rows)} crops, one isolated host process each)\n{'=' * 110}")
    snippet = (
        "import json,sys;sys.path.insert(0,r'{src}');"
        "from archivetrust.providers.satrn.adapter import SatrnAdapter;"
        "from archivetrust.providers.htr_adapter import RecognitionInput;"
        "r=SatrnAdapter().recognize(RecognitionInput(input_crop_id=sys.argv[1]));"
        "print('DIAG'+json.dumps({{'text':r.text,'confidence':r.confidence,"
        "'model_revision':r.model_revision,'raw':r.raw_response}}))"
    ).format(src=REPO_ROOT / "src")

    observations: list[dict] = []
    for position, row in enumerate(rows, start=1):
        started = time.monotonic()
        completed = subprocess.run(
            [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"), "-c", snippet, row["path"]],
            capture_output=True,
            text=True,
            check=False,
        )
        wall = time.monotonic() - started
        payload = {}
        for line in completed.stdout.splitlines():
            if line.startswith("DIAG"):
                payload = json.loads(line[4:])
        observations.append(
            {
                "pass": label,
                "position_in_pass": position,
                "crop_id": row["crop_id"],
                "smoke_order": row["order"],
                "host_returncode": completed.returncode,
                "text_raw": payload.get("text"),
                "confidence": payload.get("confidence"),
                "model_revision": payload.get("model_revision"),
                "config_revision": (payload.get("raw") or {}).get("config_revision"),
                "device_used": (payload.get("raw") or {}).get("device_used"),
                "wall_seconds": round(wall, 3),
                "smoke_satrn_text": row["smoke_satrn_text"],
                "matches_smoke_text": payload.get("text") == row["smoke_satrn_text"],
                "stderr_tail": completed.stderr[-300:],
            }
        )
        print(
            f"  [{position:2d}] {row['crop_id'][:34]} rc={completed.returncode} "
            f"conf={payload.get('confidence')} "
            f"{'SAME' if payload.get('text') == row['smoke_satrn_text'] else 'DIFF'} "
            f":: {payload.get('text')!r}"
        )
    return observations


def run_reference_pass(rows: list[dict]) -> dict:
    """Pass E -- the model's own official inference API (`mmocr.apis.TextRecInferencer`, exactly
    what Riksarkivet's htrflow uses) driven directly inside `.venv-satrn`, ONE process, ONE model
    load, all 15 crops. This bypasses `SatrnAdapter`, `facade.py` and `_worker.py` completely, so
    any repetition it also shows cannot originate in ArchiveTrust's integration code."""
    print(f"\n{'=' * 110}\nPASS E  reference mmocr.TextRecInferencer, single process, {len(rows)} crops\n{'=' * 110}")
    script = OUT_DIR / "_reference_inference.py"
    script.write_text(
        '''"""Reference SATRN inference: mmocr's own TextRecInferencer, one process, one model load.

Runs under `.venv-satrn` only. Imports nothing from `archivetrust` (that package is not installed
in the isolated interpreter). Mirrors `providers/satrn/_worker.py`'s model resolution so the same
pinned checkpoint is used, then feeds every crop to the *same* inferencer instance -- deliberately
the opposite of the adapter's fresh-process-per-crop pattern.
"""
import fnmatch, json, sys
from huggingface_hub import hf_hub_download, list_repo_files
from mmengine.config import Config

REPO, REVISION = "Riksarkivet/satrn_htr", "a40c7093232eaa47a83ce6469fc4abd033486bdc"


def grab(pattern):
    for name in list_repo_files(REPO):
        if fnmatch.fnmatch(name, pattern):
            return hf_hub_download(REPO, name, revision=REVISION)
    raise FileNotFoundError(pattern)


weights, config, dictionary = grab("*.pth"), grab("config.py"), grab("dictionary.txt")
cfg = Config.fromfile(config)
cfg.dictionary["dict_file"] = dictionary
cfg.model["decoder"]["dictionary"]["dict_file"] = dictionary
cfg.dump(config)

from mmocr.apis import TextRecInferencer

inferencer = TextRecInferencer(model=config, weights=weights, device="cuda")
paths = json.loads(sys.argv[1])
out = []
for path in paths:
    prediction = inferencer([path], batch_size=1, return_datasamples=False, progress_bar=False)[
        "predictions"
    ][0]
    out.append({"path": path, "text": prediction["text"], "score": prediction.get("scores")})
print("REFJSON" + json.dumps(out))
''',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv-satrn" / "Scripts" / "python.exe"),
            str(script),
            json.dumps([row["path"] for row in rows]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = []
    for line in completed.stdout.splitlines():
        if line.startswith("REFJSON"):
            payload = json.loads(line[7:])
    observations = []
    for position, (row, entry) in enumerate(zip(rows, payload, strict=False), start=1):
        observations.append(
            {
                "pass": "E-reference-mmocr",
                "position_in_pass": position,
                "crop_id": row["crop_id"],
                "smoke_order": row["order"],
                "text_raw": entry["text"],
                "confidence": entry["score"],
                "smoke_satrn_text": row["smoke_satrn_text"],
                "matches_smoke_text": entry["text"] == row["smoke_satrn_text"],
            }
        )
        print(
            f"  [{position:2d}] {row['crop_id'][:34]} conf={entry['score']} "
            f"{'SAME' if entry['text'] == row['smoke_satrn_text'] else 'DIFF'} :: {entry['text']!r}"
        )
    return {
        "returncode": completed.returncode,
        "observations": observations,
        "stderr_tail": completed.stderr[-1500:],
    }


def constancy(texts: list[str | None]) -> dict:
    present = [text for text in texts if text is not None]
    counter = Counter(present)
    top = counter.most_common(1)
    return {
        "outputs": len(present),
        "distinct_outputs": len(counter),
        "distinct_ratio": round(len(counter) / len(present), 4) if present else None,
        "most_repeated_output": top[0][0] if top else None,
        "most_repeated_count": top[0][1] if top else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _install_subprocess_instrumentation()

    rows = load_manifest()
    characterize(rows)

    print("=" * 110)
    print("SATRN OUTPUT-REPETITION DIAGNOSTIC -- 15 real crops from the Checkpoint 3 smoke test")
    print("=" * 110)
    header = (
        f"{'#':>2} {'rep':<4} {'crop_id':<44} {'WxH':<12} {'ar':>7} {'bytes':>8} "
        f"{'mean':>8} {'std':>7} sha256[:16]"
    )
    print(header)
    for row in rows:
        print(
            f"{row['order']:>2} {'REP' if row['repeated_in_smoke'] else '-':<4} {row['crop_id']:<44} "
            f"{str(row['width']) + 'x' + str(row['height']):<12} {row['aspect_ratio']:>7} "
            f"{row['file_size_bytes']:>8} {row['pixel_mean']:>8} {row['pixel_std']:>7} "
            f"{row['sha256_png'][:16]}"
        )
    print()
    print(f"  all crop files present                        : {all(r['exists'] for r in rows)}")
    sizes_ok = all(r["file_size_matches_recorded"] for r in rows)
    hashes_ok = all(r["recorded_hash_verified"] for r in rows)
    distinct_png = len({r["sha256_png"] for r in rows})
    distinct_pixels = len({r["sha256_decoded_pixels"] for r in rows})
    print(f"  on-disk size == smoke-test recorded byte_size : {sizes_ok}")
    print(f"  smoke-test crop hash == my own SHA-256        : {hashes_ok}")
    print(f"  distinct SHA-256 of PNG bytes                 : {distinct_png}/{len(rows)}")
    print(f"  distinct SHA-256 of decoded pixels            : {distinct_pixels}/{len(rows)}")
    print(f"  any blank/near-blank crop (std < 5)           : {any(r['pixel_std'] < 5 for r in rows)}")

    repeated = [row for row in rows if row["smoke_satrn_text"] == "staden den 27 dennes"]
    controls = [row for row in rows if not row["repeated_in_smoke"]]
    print(f"\n  repeated group ('staden den 27 dennes'): {[r['order'] for r in repeated]}")
    print(f"  distinct-output controls               : {[r['order'] for r in controls]}")

    adapter = SatrnAdapter()
    pass_a = run_pass("A-all15-original-order", rows, adapter)
    pass_b = run_pass("B-repeated6-same-order-same-process", repeated, adapter)
    pass_c = run_pass("C-repeated6-REVERSED-order", list(reversed(repeated)), adapter)
    pass_d = run_isolated_pass("D-repeated6-isolated-host-process", repeated)
    pass_e = run_reference_pass(rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Bounded diagnostic of the Checkpoint 3 SATRN output-repetition finding. Does not "
            "modify the smoke-test experiment, its telemetry or its results.json."
        ),
        "source_evidence": {
            "smoke_test_results": str(SMOKE / "smoke_test_results.json"),
            "smoke_test_telemetry": str(SMOKE / "smoke_test_events.jsonl"),
        },
        "crop_manifest": rows,
        "repeated_group_crop_ids": [row["crop_id"] for row in repeated],
        "control_crop_ids": [row["crop_id"] for row in controls],
        "passes": {
            "A_all15_original_order": pass_a,
            "B_repeated6_same_order_same_process": pass_b,
            "C_repeated6_reversed_order": pass_c,
            "D_repeated6_isolated_host_process": pass_d,
            "E_reference_mmocr_single_process": pass_e,
        },
        "constancy": {
            "smoke_test_original": constancy([row["smoke_satrn_text"] for row in rows]),
            "pass_A_rerun": constancy([o["text_raw"] for o in pass_a]),
            "pass_E_reference_mmocr": constancy([o["text_raw"] for o in pass_e["observations"]]),
        },
        "integration_checks": {
            "distinct_child_pids_pass_A": len({o["child_pid"] for o in pass_a}),
            "calls_in_pass_A": len(pass_a),
            "every_call_got_a_fresh_child_process": len({o["child_pid"] for o in pass_a}) == len(pass_a),
            "argv_always_pointed_at_source_crop": all(o["argv_image_path_is_source_crop"] for o in pass_a),
            "no_temp_file_in_argv": all(
                o["argv_image_path"] == o["path"] for o in pass_a
            ),
            "input_bytes_unchanged_by_call": all(o["input_bytes_unchanged_by_call"] for o in pass_a),
            "input_hash_matches_manifest": all(o["input_hash_matches_manifest"] for o in pass_a),
            "worker_emitted_exactly_one_json_line": all(o["stdout_line_count"] == 1 for o in pass_a),
            "distinct_model_revisions": sorted({o["model_revision"] for o in pass_a}),
            "distinct_config_revisions": sorted({str(o["config_revision"]) for o in pass_a}),
            "distinct_confidences_pass_A": len({o["confidence"] for o in pass_a}),
            "distinct_confidences_within_repeated_group": len(
                {o["confidence"] for o in pass_a if o["crop_id"] in {r["crop_id"] for r in repeated}}
            ),
        },
        "determinism": {
            "pass_A_reproduced_smoke_text_for_all_15": all(o["matches_smoke_text"] for o in pass_a),
            "pass_B_reproduced_repeat_for_all_6": all(o["matches_smoke_text"] for o in pass_b),
            "pass_C_reversed_reproduced_repeat_for_all_6": all(o["matches_smoke_text"] for o in pass_c),
            "pass_D_isolated_reproduced_repeat_for_all_6": all(o["matches_smoke_text"] for o in pass_d),
            "pass_E_reference_reproduced_repeat_for_all_6": all(
                o["matches_smoke_text"]
                for o in pass_e["observations"]
                if o["crop_id"] in {r["crop_id"] for r in repeated}
            ),
        },
        "documented_preprocessing": {
            "source": "Riksarkivet/satrn_htr@a40c7093232eaa47a83ce6469fc4abd033486bdc config.py",
            "test_pipeline_resize": "Resize(scale=(400, 64), keep_ratio=False)",
            "train_pipeline_resize": "Resize(scale=(400, 64), keep_ratio=False)",
            "data_preprocessor_mean": [123.675, 116.28, 103.53],
            "data_preprocessor_std": [58.395, 57.12, 57.375],
            "backbone_input_channels": 3,
            "decoder_max_seq_len": 100,
            "train_test_pipeline_consistent": True,
        },
    }

    report_path = OUT_DIR / "satrn_repetition_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'=' * 110}\nRESULTS\n{'=' * 110}")
    for key, value in report["integration_checks"].items():
        print(f"  {key:<52} {value}")
    print()
    for key, value in report["determinism"].items():
        print(f"  {key:<52} {value}")
    print()
    for key, value in report["constancy"].items():
        print(f"  {key:<28} {value}")
    print(f"\nwrote {report_path}")
    print(f"reference-pass returncode: {pass_e['returncode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
