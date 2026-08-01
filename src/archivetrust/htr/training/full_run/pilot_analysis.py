"""Analyzes a completed (or in-progress) pilot run's real, recorded behavior into `PilotAnalysis` --
the sole input `monitoring_config.py` uses to derive full-corpus monitoring policy. Every number here
comes from files the pilot already wrote (`session_state.json`'s `validation_history`,
`checkpoint_index.json`, `memory-probe.json`, `pilot_split_summary.json`, `training_identity.json`,
and any real `telemetry/gpu_samples.jsonl`) -- nothing is invented, and every derived value's exact
method is recorded on the artifact itself (`assumptions_and_method`), never left implicit.

**A real, disclosed data gap**: `train_loss`/`val_loss` were only added to `validation_history` entries
partway through this project's real pilot run (epochs 1-19 predate the fix; epochs 20-22 have them).
Loss-based analysis here only ever uses epochs where both are actually present, and reports how many
that was -- never interpolated or backfilled.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from statistics import median, mean, pstdev

from pydantic import BaseModel, ConfigDict

from archivetrust.htr.training.checkpoint_index import load_index
from archivetrust.htr.training.pilot_split import load_pilot_split_summary
from archivetrust.htr.training.training_session import load_session_state

DEFAULT_TRAILING_WINDOW = 5
"""How many of the most recent epoch-to-epoch transitions define the "noise floor"
(`meaningful_improvement_threshold`) -- large enough to smooth one noisy epoch, small enough to stay
representative of *current*, not historical, training dynamics."""
STABILITY_WINDOW = 3
"""Consecutive epochs that must all sit at-or-below the threshold before a plateau is called real,
not a single lucky small delta."""
STABILITY_TOLERANCE = 1.5
"""Multiplier on `meaningful_improvement_threshold` used for the stability check -- slightly looser
than the threshold itself so a plateau isn't missed by noise sitting just above it."""
MAX_EXTRAPOLATION_EPOCHS = 60
"""Upper bound on how far forward an unobserved plateau may be projected before this module gives up
and honestly reports `None` rather than an unbounded guess."""


class PilotEpochObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    epoch: int
    train_cer: float | None
    val_cer: float | None
    train_wer: float | None
    val_wer: float | None
    train_loss: float | None
    val_loss: float | None
    duration_seconds: float
    val_cer_delta: float | None
    """`val_cer[epoch] - val_cer[epoch - 1]` -- `None` for the first epoch (no prior epoch to
    compare against) or when either epoch's `val_cer` is missing."""
    val_cer_relative_delta: float | None
    """`val_cer_delta / val_cer[epoch - 1]` -- the same improvement expressed as a fraction of the
    prior epoch's CER, since a fixed absolute delta means something very different at CER=0.5 than
    at CER=0.05."""
    rolling_mean_val_cer_delta: float | None
    """Mean of `|val_cer_delta|` over the trailing `DEFAULT_TRAILING_WINDOW` transitions ending at
    this epoch -- the same rolling window `meaningful_improvement_threshold` uses, exposed per-epoch
    so a report can show the trend, not just the final aggregate."""


class PilotAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    pilot_run_id: str
    base_checkpoint_identity: str | None
    """The real pinned parent checkpoint this pilot fine-tuned from (`training_identity.json`'s own
    `identity.parent_checkpoint`), e.g. `"generic-2023-02-15@0da2c00a..."` -- the same identity string
    the full run's own preflight checks against."""
    source_metric_files: tuple[str, ...]
    pilot_epoch_count: int
    train_line_count: int | None
    batch_size: int | None
    steps_per_pilot_epoch: int | None
    best_pilot_epoch: int | None
    best_pilot_val_cer: float | None
    last_checkpoint_epoch: int | None
    meaningful_improvement_threshold: float | None
    estimated_plateau_epoch: int | None
    plateau_is_extrapolated: bool
    plateau_estimation_note: str
    max_near_flat_streak: int
    """Longest observed run of *consecutive* epochs whose `|val_cer_delta|` sat at or below
    `meaningful_improvement_threshold` before a later epoch recovered with a larger improvement --
    real evidence `monitoring_config.py` uses to size `recommended_patience`, not a guessed default."""
    near_flat_epochs: tuple[int, ...]
    """Every epoch (not just the longest streak) whose `|val_cer_delta|` sat at or below
    `meaningful_improvement_threshold` -- the raw evidence `max_near_flat_streak` is computed from."""
    validation_noise_stdev: float | None
    """Population stdev of `|val_cer_delta|` over the trailing window -- how noisy the signal actually
    is near the end of the observed run, the real basis for judging whether a future small delta is
    "noise" or a genuine slowdown."""
    genuine_plateau_occurred: bool
    """`True` only when `estimated_plateau_epoch` was a real, observed stable window
    (`plateau_is_extrapolated is False`) -- this project's own real pilot never actually reached one
    before its wall-clock cap, so this is honestly `False` for it."""
    early_stopping_triggered: bool
    early_stopping_note: str
    throughput_lines_per_second: float | None
    """`train_line_count / epoch_duration_mean_seconds` -- real, measured throughput."""
    gpu_observations_available: bool
    checkpoint_behavior_summary: str
    stop_reason: str | None
    """The pilot's own real, final `TrainingSessionState.last_stop_reason` -- honestly `None` if the
    pilot run never recorded one (e.g. it predates that field being tracked)."""
    measured_vs_extrapolated_summary: str
    """One human-readable line stating plainly which headline numbers above are real measurements and
    which are extrapolations -- so a reader never has to cross-reference `plateau_is_extrapolated`
    against every other field to know what to trust."""
    train_val_divergence: dict
    gpu_memory_high_water_mb: float | None
    checkpoint_size_bytes: dict
    runtime_stability: dict
    epoch_observations: tuple[PilotEpochObservation, ...]
    assumptions_and_method: str


def _val_cer_deltas(observations: list[dict]) -> list[tuple[int, float]]:
    """`[(epoch, delta), ...]` for every consecutive pair where both `val_cer` values are present."""
    deltas: list[tuple[int, float]] = []
    for prev, curr in zip(observations, observations[1:]):
        if prev.get("val_cer") is not None and curr.get("val_cer") is not None:
            deltas.append((curr["epoch"], curr["val_cer"] - prev["val_cer"]))
    return deltas


def _meaningful_improvement_threshold(deltas: list[tuple[int, float]]) -> float | None:
    if not deltas:
        return None
    window = deltas[-DEFAULT_TRAILING_WINDOW:]
    return median(abs(d) for _, d in window)


def _max_near_flat_streak(deltas: list[tuple[int, float]], threshold: float | None) -> int:
    if not deltas or threshold is None:
        return 0
    longest = current = 0
    for _, d in deltas:
        if abs(d) <= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _near_flat_epochs(deltas: list[tuple[int, float]], threshold: float | None) -> tuple[int, ...]:
    if threshold is None:
        return ()
    return tuple(epoch for epoch, d in deltas if abs(d) <= threshold)


def _validation_noise_stdev(deltas: list[tuple[int, float]]) -> float | None:
    window = deltas[-DEFAULT_TRAILING_WINDOW:]
    if len(window) < 2:
        return None
    return pstdev(abs(d) for _, d in window)


def _rolling_mean_deltas(deltas: list[tuple[int, float]]) -> dict[int, float]:
    """`{epoch: rolling_mean(|delta|) over the trailing DEFAULT_TRAILING_WINDOW transitions ending at
    that epoch}` -- one value per real transition, not just the final aggregate."""
    result: dict[int, float] = {}
    magnitudes = [abs(d) for _, d in deltas]
    for i, (epoch, _) in enumerate(deltas):
        window = magnitudes[max(0, i - DEFAULT_TRAILING_WINDOW + 1) : i + 1]
        result[epoch] = sum(window) / len(window)
    return result


def _relative_deltas(observations: list[dict]) -> dict[int, float]:
    """`{epoch: delta / val_cer[epoch - 1]}` for every consecutive pair where both `val_cer` values
    are present and the prior value is non-zero -- the same pairing `_val_cer_deltas` uses, expressed
    as a fraction of the prior epoch's CER."""
    result: dict[int, float] = {}
    for prev, curr in zip(observations, observations[1:]):
        prev_val, curr_val = prev.get("val_cer"), curr.get("val_cer")
        if prev_val is not None and curr_val is not None and prev_val != 0:
            result[curr["epoch"]] = (curr_val - prev_val) / prev_val
    return result


def _early_stopping_behavior(state, deltas: list[tuple[int, float]]) -> tuple[bool, str]:
    """Whether the pilot's own real session ever recorded `stop_reason ==
    "no_val_cer_improvement"` -- the actual patience-triggered stop, distinct from a wall-clock or
    epoch-cap stop."""
    stop_reason = getattr(state, "last_stop_reason", None)
    if stop_reason == "no_val_cer_improvement":
        return True, (
            f"Early stopping (patience-based) genuinely triggered -- recorded stop_reason "
            f"'no_val_cer_improvement' (epochs_since_improvement={getattr(state, 'epochs_since_improvement', 'unknown')})."
        )
    return False, (
        f"Early stopping never triggered -- the pilot's real stop_reason was "
        f"{stop_reason!r}, not 'no_val_cer_improvement'. The pilot was still improving (or stopped "
        "for an unrelated reason, e.g. a wall-clock cap) when it last ran."
    )


def _checkpoint_behavior_summary(entries) -> str:
    kinds = {}
    for e in entries:
        kinds[e.checkpoint_kind] = kinds.get(e.checkpoint_kind, 0) + 1
    unresumable = sum(1 for e in entries if not e.resumable)
    return (
        f"{len(entries)} real checkpoint index entries recorded ({kinds}); "
        f"{unresumable} failed verification (never marked resumable)."
    )


def _estimate_plateau_epoch(
    deltas: list[tuple[int, float]], threshold: float | None
) -> tuple[int | None, bool, str]:
    """Returns `(estimated_plateau_epoch, is_extrapolated, note)`. First tries to find a real,
    observed stable window (`STABILITY_WINDOW` consecutive epochs all at-or-below `threshold *
    STABILITY_TOLERANCE`); if none exists in the observed data (true for this project's own real
    pilot run, which never actually plateaued before stopping at its wall-clock cap), falls back to
    a log-linear extrapolation of the trailing window, clearly labeled as such.
    """
    if threshold is None or len(deltas) < 2:
        return None, False, "Fewer than 2 recorded epoch-to-epoch deltas -- nothing to estimate from."

    for i in range(len(deltas) - STABILITY_WINDOW + 1):
        window = deltas[i : i + STABILITY_WINDOW]
        if all(abs(d) <= threshold * STABILITY_TOLERANCE for _, d in window):
            return window[0][0], False, (
                f"Real, observed plateau: epochs {window[0][0]}-{window[-1][0]} all had "
                f"|val_cer delta| <= {threshold * STABILITY_TOLERANCE:.5f} "
                f"(threshold {threshold:.5f} x {STABILITY_TOLERANCE} stability tolerance)."
            )

    # No observed plateau -- extrapolate from the trailing window's decay rate (log-linear fit on
    # |delta|, the same method used to estimate this project's own full-corpus epoch-1 measurement
    # extrapolation, see docs/methods/loghi-training-dashboard.md's §9).
    fit_window = deltas[-min(6, len(deltas)) :]
    if len(fit_window) < 3:
        return None, False, (
            "No observed plateau and too few trailing epochs (< 3) to extrapolate a decay trend "
            "responsibly."
        )
    xs = [epoch for epoch, _ in fit_window]
    ys = [math.log(abs(d)) if d != 0 else math.log(1e-12) for _, d in fit_window]
    n = len(xs)
    mean_x, mean_y = mean(xs), mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None, False, "Trailing window has identical epoch numbers -- cannot fit a trend."
    slope = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / denom
    intercept = mean_y - slope * mean_x

    if slope >= 0:
        return None, True, (
            f"Trailing-window log-linear fit over epochs {xs[0]}-{xs[-1]} shows a non-decaying "
            "trend (slope >= 0) -- the pilot has not shown any sign of plateauing yet; no responsible "
            "extrapolated epoch to report."
        )

    last_epoch = xs[-1]
    for k in range(1, MAX_EXTRAPOLATION_EPOCHS + 1):
        candidate_epoch = last_epoch + k
        predicted = math.exp(intercept + slope * candidate_epoch)
        if predicted <= threshold:
            return candidate_epoch, True, (
                f"No observed plateau within {last_epoch} recorded epochs (still improving at the "
                f"real pilot's final epoch). Extrapolated via a log-linear fit of |val_cer delta| "
                f"over epochs {xs[0]}-{xs[-1]} (slope={slope:.5f}, i.e. delta shrinks by a factor of "
                f"{math.exp(slope):.4f} per epoch): projected to cross the threshold "
                f"({threshold:.5f}) at epoch {candidate_epoch}. This is an extrapolation, not an "
                "observation -- treat it as a starting point, not a certainty."
            )

    return None, True, (
        f"Log-linear extrapolation did not cross the threshold within "
        f"{MAX_EXTRAPOLATION_EPOCHS} epochs past the last observed one -- refusing to report an "
        "unbounded guess."
    )


def _train_val_divergence(observations: list[dict]) -> dict:
    paired = [o for o in observations if o.get("train_loss") is not None and o.get("val_loss") is not None]
    if not paired:
        return {"sample_epoch_count": 0, "note": "No epochs with both train_loss and val_loss recorded."}
    gaps = [(o["epoch"], o["val_loss"] - o["train_loss"]) for o in paired]
    half = max(1, len(gaps) // 2)
    # Compared by |gap| (divergence magnitude), never the signed value -- a gap of -4.0 shrinking to
    # -2.0 means train and val are converging (narrowing), even though -2.0 > -4.0 as signed numbers.
    first_half_mean = mean(abs(g) for _, g in gaps[:half])
    second_half_mean = mean(abs(g) for _, g in gaps[half:]) if len(gaps) > half else first_half_mean
    if second_half_mean > first_half_mean * 1.1:
        trend = "widening"
    elif second_half_mean < first_half_mean * 0.9:
        trend = "narrowing"
    else:
        trend = "stable"
    return {
        "sample_epoch_count": len(paired),
        "epochs_used": [e for e, _ in gaps],
        "final_gap": gaps[-1][1],
        "trend": trend,
    }


def _runtime_stability(observations: list[dict]) -> dict:
    durations = [o["duration_seconds"] for o in observations if o.get("duration_seconds") is not None]
    if not durations:
        return {"epoch_duration_mean_seconds": None, "epoch_duration_stdev_seconds": None}
    return {
        "epoch_duration_mean_seconds": mean(durations),
        "epoch_duration_stdev_seconds": pstdev(durations) if len(durations) > 1 else 0.0,
        "epoch_count_observed": len(durations),
    }


def _checkpoint_sizes(entries) -> dict:
    sizes: dict[str, int | None] = {}
    for kind in ("latest", "best_val"):
        matching = [e for e in entries if e.checkpoint_kind == kind]
        if not matching:
            sizes[kind] = None
            continue
        newest = max(matching, key=lambda e: (e.epoch, e.created_at))
        keras_files = list(Path(newest.checkpoint_dir).glob("*.keras")) if Path(newest.checkpoint_dir).exists() else []
        sizes[kind] = keras_files[0].stat().st_size if keras_files else None
    return sizes


def _gpu_memory_high_water_mb(pilot_run_dir: Path) -> float | None:
    samples_path = pilot_run_dir / "run-state" / "telemetry" / "gpu_samples.jsonl"
    if not samples_path.exists():
        return None
    peak: float | None = None
    with samples_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            used = row.get("gpu", {}).get("memory_used_mb")
            if used is not None and (peak is None or used > peak):
                peak = used
    return peak


def analyze_pilot_run(pilot_run_dir: str | Path) -> PilotAnalysis:
    pilot_run_dir = Path(pilot_run_dir)
    run_state_dir = pilot_run_dir / "run-state"

    identity_path = run_state_dir / "training_identity.json"
    identity_payload = json.loads(identity_path.read_text(encoding="utf-8"))
    pilot_run_id = identity_payload["identity"]["run_id"]
    base_checkpoint_identity = identity_payload["identity"].get("parent_checkpoint")

    state = load_session_state(run_state_dir)
    if state is None:
        raise FileNotFoundError(f"No session_state.json under {run_state_dir} -- pilot has no recorded epochs.")
    observations_raw = list(state.validation_history)

    memory_probe_path = pilot_run_dir / "reports" / "memory-probe.json"
    memory_probe = json.loads(memory_probe_path.read_text(encoding="utf-8")) if memory_probe_path.exists() else {}
    batch_size = memory_probe.get("chosen_batch_size")

    split_summary_path = pilot_run_dir / "manifests" / "pilot_split_summary.json"
    train_line_count = (
        load_pilot_split_summary(pilot_run_dir / "manifests").actual_train if split_summary_path.exists() else None
    )
    steps_per_pilot_epoch = math.ceil(train_line_count / batch_size) if train_line_count and batch_size else None

    checkpoint_index_path = run_state_dir / "checkpoint_index.json"
    entries = load_index(checkpoint_index_path)
    best_entries = [e for e in entries if e.checkpoint_kind == "best_val" and "val_CER_metric" in e.validation_metrics]
    best_entry = min(best_entries, key=lambda e: e.validation_metrics["val_CER_metric"], default=None)
    latest_entries = [e for e in entries if e.checkpoint_kind == "latest"]
    last_entry = max(latest_entries, key=lambda e: (e.epoch, e.created_at), default=None)

    deltas = _val_cer_deltas(observations_raw)
    threshold = _meaningful_improvement_threshold(deltas)
    plateau_epoch, is_extrapolated, note = _estimate_plateau_epoch(deltas, threshold)
    streak = _max_near_flat_streak(deltas, threshold)
    near_flat = _near_flat_epochs(deltas, threshold)
    noise_stdev = _validation_noise_stdev(deltas)
    genuine_plateau = plateau_epoch is not None and not is_extrapolated
    early_stopping_triggered, early_stopping_note = _early_stopping_behavior(state, deltas)

    delta_by_epoch = dict(deltas)
    relative_delta_by_epoch = _relative_deltas(observations_raw)
    rolling_mean_by_epoch = _rolling_mean_deltas(deltas)
    epoch_observations = tuple(
        PilotEpochObservation(
            epoch=o["epoch"],
            train_cer=o.get("train_cer"),
            val_cer=o.get("val_cer"),
            train_wer=o.get("train_wer"),
            val_wer=o.get("val_wer"),
            train_loss=o.get("train_loss"),
            val_loss=o.get("val_loss"),
            duration_seconds=o["duration_seconds"],
            val_cer_delta=delta_by_epoch.get(o["epoch"]),
            val_cer_relative_delta=relative_delta_by_epoch.get(o["epoch"]),
            rolling_mean_val_cer_delta=rolling_mean_by_epoch.get(o["epoch"]),
        )
        for o in observations_raw
    )

    runtime_stability = _runtime_stability(observations_raw)
    mean_epoch_duration = runtime_stability.get("epoch_duration_mean_seconds")
    throughput = (
        train_line_count / mean_epoch_duration
        if train_line_count and mean_epoch_duration else None
    )
    gpu_high_water = _gpu_memory_high_water_mb(pilot_run_dir)
    checkpoint_sizes = _checkpoint_sizes(entries)

    measured_vs_extrapolated_summary = (
        f"MEASURED: {len(observations_raw)} real pilot epochs, best_pilot_epoch="
        f"{best_entry.epoch if best_entry else None}, best_pilot_val_cer="
        f"{best_entry.validation_metrics.get('val_CER_metric') if best_entry else None}, "
        f"max_near_flat_streak={streak}. "
        + (
            f"EXTRAPOLATED: estimated_plateau_epoch={plateau_epoch} ({note})"
            if plateau_epoch is not None and is_extrapolated
            else (
                "MEASURED: estimated_plateau_epoch is a real, observed stable window, not an "
                "extrapolation."
                if genuine_plateau
                else "No plateau estimate available (see plateau_estimation_note)."
            )
        )
    )

    source_files = tuple(
        str(p) for p in (
            run_state_dir / "session_state.json",
            checkpoint_index_path,
            memory_probe_path,
            split_summary_path,
            identity_path,
        ) if p.exists()
    )

    assumptions = (
        f"meaningful_improvement_threshold = median(|val_cer delta|) over the trailing "
        f"{DEFAULT_TRAILING_WINDOW} epoch-to-epoch transitions. estimated_plateau_epoch: {note} "
        f"steps_per_pilot_epoch = ceil(train_line_count / batch_size) "
        f"({train_line_count} / {batch_size})." if train_line_count and batch_size else
        f"meaningful_improvement_threshold = median(|val_cer delta|) over the trailing "
        f"{DEFAULT_TRAILING_WINDOW} epoch-to-epoch transitions. estimated_plateau_epoch: {note} "
        "steps_per_pilot_epoch could not be computed (missing train line count or batch size)."
    )

    return PilotAnalysis(
        pilot_run_id=pilot_run_id,
        base_checkpoint_identity=base_checkpoint_identity,
        source_metric_files=source_files,
        pilot_epoch_count=len(observations_raw),
        train_line_count=train_line_count,
        batch_size=batch_size,
        steps_per_pilot_epoch=steps_per_pilot_epoch,
        best_pilot_epoch=best_entry.epoch if best_entry else None,
        best_pilot_val_cer=best_entry.validation_metrics.get("val_CER_metric") if best_entry else None,
        last_checkpoint_epoch=last_entry.epoch if last_entry else None,
        meaningful_improvement_threshold=threshold,
        estimated_plateau_epoch=plateau_epoch,
        plateau_is_extrapolated=is_extrapolated,
        plateau_estimation_note=note,
        max_near_flat_streak=streak,
        near_flat_epochs=near_flat,
        validation_noise_stdev=noise_stdev,
        genuine_plateau_occurred=genuine_plateau,
        early_stopping_triggered=early_stopping_triggered,
        early_stopping_note=early_stopping_note,
        throughput_lines_per_second=throughput,
        gpu_observations_available=gpu_high_water is not None,
        checkpoint_behavior_summary=_checkpoint_behavior_summary(entries),
        stop_reason=getattr(state, "last_stop_reason", None),
        measured_vs_extrapolated_summary=measured_vs_extrapolated_summary,
        train_val_divergence=_train_val_divergence(observations_raw),
        gpu_memory_high_water_mb=gpu_high_water,
        checkpoint_size_bytes=checkpoint_sizes,
        runtime_stability=runtime_stability,
        epoch_observations=epoch_observations,
        assumptions_and_method=assumptions,
    )


def write_pilot_analysis(analysis: PilotAnalysis, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(output_path.parent), prefix=".tmp-pilot-analysis-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(analysis.model_dump_json(indent=2))
        os.replace(tmp_path, str(output_path))
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
