"""Generates both machine-readable and human-readable session reports (Work Package 16) --
reconstructed entirely from durable state (checkpoint index, session state, manifests), never by
re-running training. Mirrors the existing Loghi smoke-test module's "report regeneration without
rerunning" discipline.
"""

from __future__ import annotations

import json
from pathlib import Path

from archivetrust.htr.training.checkpoint_index import (
    best_validation_checkpoint,
    latest_resumable_checkpoint,
    load_index,
)
from archivetrust.htr.training.pilot_split import load_pilot_split_summary
from archivetrust.htr.training.training_session import load_session_state


def generate_session_reports(
    *,
    run_state_dir: str | Path,
    checkpoint_index_path: str | Path,
    manifests_dir: str | Path,
    reports_dir: str | Path,
    dataset_root: str | Path,
) -> dict:
    """Writes `reports_dir/session-report.json` and `.md`. Returns the JSON-serializable dict that
    was written, for callers that want it without re-reading the file."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    state = load_session_state(run_state_dir)
    entries = load_index(checkpoint_index_path)
    latest = latest_resumable_checkpoint(checkpoint_index_path)
    best = best_validation_checkpoint(checkpoint_index_path)

    split_summary = None
    split_summary_path = Path(manifests_dir) / "pilot_split_summary.json"
    if split_summary_path.exists():
        split_summary = load_pilot_split_summary(manifests_dir).model_dump()

    char_report = None
    char_report_path = Path(reports_dir) / "character-compatibility.json"
    if char_report_path.exists():
        char_report = json.loads(char_report_path.read_text(encoding="utf-8"))

    memory_probe = None
    memory_probe_path = Path(reports_dir) / "memory-probe.json"
    if memory_probe_path.exists():
        memory_probe = json.loads(memory_probe_path.read_text(encoding="utf-8"))

    resume_proof = None
    resume_proof_path = Path(reports_dir) / "resume-proof.json"
    if resume_proof_path.exists():
        resume_proof = json.loads(resume_proof_path.read_text(encoding="utf-8"))

    report = {
        "dataset_root": str(dataset_root),
        "run_id": state.run_id if state else None,
        "cumulative_epoch": state.cumulative_epoch if state else 0,
        "global_step": state.global_step if state else 0,
        "session_count": state.session_count if state else 0,
        "cumulative_training_seconds": state.cumulative_training_seconds if state else 0.0,
        "validation_history": list(state.validation_history) if state else [],
        "best_val_cer": state.best_val_cer if state else None,
        "last_stop_reason": state.last_stop_reason if state else None,
        "last_stopped_mid_epoch": state.last_stopped_mid_epoch if state else None,
        "last_stop_boundary": state.last_stop_boundary if state else None,
        "latest_checkpoint": latest.model_dump() if latest else None,
        "best_checkpoint": best.model_dump() if best else None,
        "checkpoint_index_entry_count": len(entries),
        "pilot_split_summary": split_summary,
        "character_compatibility": char_report,
        "memory_probe": memory_probe,
        "resume_proof": resume_proof,
    }

    (reports_dir / "session-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "session-report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict) -> str:
    lines = ["# Swedish Loghi Fine-Tuning -- Session Report", ""]
    lines.append(f"- Dataset root: `{report['dataset_root']}`")
    lines.append(f"- Run id: `{report['run_id']}`")
    lines.append(f"- Cumulative epoch: {report['cumulative_epoch']}")
    lines.append(f"- Global step: {report['global_step']}")
    lines.append(f"- Sessions completed: {report['session_count']}")
    lines.append(f"- Cumulative training time: {report['cumulative_training_seconds']:.1f}s")
    lines.append(f"- Best validation CER: {report['best_val_cer']}")
    if report["last_stop_reason"] is not None:
        lines.append(f"- Last session stop reason: `{report['last_stop_reason']}`")
        lines.append(f"- Stopped mid-epoch: {report['last_stopped_mid_epoch']}")
        lines.append(f"- Stop boundary: `{report['last_stop_boundary']}`")
    lines.append("")

    if report["pilot_split_summary"]:
        s = report["pilot_split_summary"]
        lines.append("## Pilot split")
        lines.append(f"- Train: {s['actual_train']} (target {s['target_train']})")
        lines.append(f"- Val: {s['actual_val']} (target {s['target_val']})")
        lines.append(f"- Test reserved: {s['actual_test_reserved']} (target {s['target_test_reserved']})")
        if s["shortfall_warnings"]:
            lines.append(f"- Shortfalls: {s['shortfall_warnings']}")
        lines.append("")

    if report["character_compatibility"]:
        c = report["character_compatibility"]
        lines.append("## Character compatibility")
        lines.append(f"- Lines examined: {c['total_lines_examined']}")
        lines.append(f"- Compatible: {c['compatible']}")
        lines.append(f"- Unsupported characters: {c['unsupported_character_count']}")
        lines.append("")

    if report["memory_probe"]:
        m = report["memory_probe"]
        lines.append("## RTX 3070 memory probe")
        lines.append(f"- Chosen batch size: {m['chosen_batch_size']}")
        lines.append(f"- Peak VRAM: {m['chosen_peak_vram_mb']} MB / {m['total_vram_mb']} MB total")
        lines.append(f"- Safety margin: {m['safety_margin_mb']} MB")
        lines.append("")

    if report["resume_proof"]:
        r = report["resume_proof"]
        lines.append("## Full-state resume proof")
        lines.append(f"- Proof passed: {r['proof_passed']}")
        lines.append(f"- Epoch continued (not restarted): {r['epoch_continued_not_restarted']}")
        lines.append(f"- Global step continued (not restarted): {r['global_step_continued_not_restarted']}")
        lines.append("")

    if report["validation_history"]:
        lines.append("## Validation history")
        lines.append("| Epoch | Train CER | Val CER | Duration (s) |")
        lines.append("|---|---|---|---|")
        for entry in report["validation_history"]:
            lines.append(
                f"| {entry['epoch']} | {entry.get('train_cer')} | {entry.get('val_cer')} | "
                f"{entry.get('duration_seconds')} |"
            )
        lines.append("")

    if report["latest_checkpoint"]:
        lc = report["latest_checkpoint"]
        lines.append("## Latest verified resumable checkpoint")
        lines.append(f"- Checkpoint id: `{lc['checkpoint_id']}`")
        lines.append(f"- Epoch: {lc['epoch']}")
        lines.append(f"- Path: `{lc['checkpoint_dir']}`")
        lines.append(f"- Resumable: {lc['resumable']}")
        lines.append("")

    if report["best_checkpoint"]:
        bc = report["best_checkpoint"]
        lines.append("## Best validation checkpoint")
        lines.append(f"- Checkpoint id: `{bc['checkpoint_id']}`")
        lines.append(f"- Val CER: {bc['validation_metrics'].get('val_CER_metric')}")
        lines.append(f"- Path: `{bc['checkpoint_dir']}`")
        lines.append("")

    lines.append("## Continuing later")
    lines.append("")
    lines.append("```")
    lines.append("PYTHONPATH=src .venv/Scripts/python.exe scripts/train_loghi_swedish.py")
    lines.append("# then choose option 6 (Start or resume five-hour session)")
    lines.append("```")
    lines.append("")
    lines.append(
        "**Note:** falling training loss/CER here is evidence the training mechanics work, not a "
        "quality claim. `loghi_swedish_finetuned_v1` has not been evaluated on the reserved test set "
        "or the Lion-vs-Loghi comparison; do not interpret these numbers as benchmark quality."
    )
    return "\n".join(lines)
