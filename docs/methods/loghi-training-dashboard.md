# Loghi Training Dashboard — Read-Only Run Visualization

Status: Complete — read-only, epoch-granularity, pilot-and-future-full-corpus training runs
Governs: How the dashboard reads and displays real training-run state; it never starts, stops,
pauses, or resumes a training process.

## 1. Architectural boundary

The trainer (`scripts/train_loghi_swedish.py`) and the dashboard
(`scripts/train_loghi_swedish_dashboard.py`) are two independent processes that never call into each
other. The trainer writes durable files; the dashboard only reads them.

```
Trainer  -> writes training/<run>/run-state/{session_state.json, checkpoint_index.json,
            training_identity.json, telemetry/{status.json, gpu_samples.jsonl}, events.jsonl}
Dashboard -> reads those files, computes status/health, renders charts
```

Killing or restarting either process never affects the other. There is no button, form, or code path
in the dashboard that touches a training process — verified by `tests/presentation/
test_training_dashboard_viewmodel.py` and by direct inspection: every dashboard method is a pure read.

## 2. Starting the dashboard

```
pip install -e .[dashboard]
streamlit run scripts/train_loghi_swedish_dashboard.py
```

The `dashboard` extra (`pyproject.toml`) is optional — the rest of the system installs and its full
test suite runs without it, same as `gui`/`transformers`/`watch`.

## 3. Where run data lives

Every run is a directory under the repository's `training/` root (e.g. `training/loghi-swedish-v1/`),
discovered by the presence of `run-state/training_identity.json`:

```
training/<run-name>/
  run-state/
    training_identity.json      # run_id, base model, training_phase, created_at (immutable)
    session_state.json          # cumulative progress, launch-budget echo, last stop info
    checkpoint_index.json       # append-only, atomic — latest/best_val/end_of_session entries
    telemetry/
      status.json               # liveness heartbeat, latest GPU/system reading
      gpu_samples.jsonl         # append-only GPU/system/disk samples, ~5s cadence
    epoch_output/epoch_N/...    # real Loghi container output per epoch
  manifests/pilot_split_summary.json
  reports/{character-compatibility,memory-probe,resume-proof}.json
  prepared-data/preparation_report.json
```

No new "run.json" source of truth was invented — `run_status.py::compute_run_status` assembles
`RunStatus` entirely from these existing files.

## 4. How the trainer writes telemetry

`htr/training/telemetry_sampler.py::TelemetrySampler` runs as a background thread *around* (not
instead of) the trainer's existing blocking container call — it does not read the container's
stdout and cannot affect what the container does. Every ~5s it samples GPU (`nvidia-smi`), system
(`psutil`, optional), and disk (`shutil.disk_usage`), appending to `gpu_samples.jsonl` and atomically
rewriting `status.json`. All fields degrade to `None` on absence (no GPU, no `psutil`) rather than
guessing.

## 5. Refresh behavior

The dashboard polls every 5–15s (configurable in the sidebar) via `time.sleep` + `st.rerun()` — no
extra JS-timer dependency. `gpu_samples.jsonl` is never re-read from the start: the dashboard keeps a
`(byte_offset, accumulated_rows)` cursor per run in Streamlit's session state and reads only the
bytes appended since the last refresh (`htr/training/incremental_jsonl_reader.py`), so it stays
responsive across a multi-day run. A malformed individual line is skipped and counted, never fatal.

## 6. Inspecting a completed or interrupted run

Open the **Run detail** tab and select the run. `RunStatus.status` is derived entirely from real
signals:

- **`interrupted`** means a telemetry heartbeat existed at some point but has gone stale (no update
  in 60s) with no clean `last_stop_reason` ever recorded — the process was killed without a chance
  to record why.
- **`running`** with no heartbeat at all but real recorded progress means this session predates
  `TelemetrySampler` being wired in (or ran with telemetry disabled) — an inference from last-known
  progress, not a live confirmation. Every session launched after this feature shipped gets a real
  heartbeat.
- **`completed`** / **`failed`** are only ever set once a session's `run_training_session()` call
  returns and records a terminal `last_stop_reason`.

## 7. How to resume a run

The dashboard has no resume button by design. Resume a run from its own launcher:

```
PYTHONPATH=src .venv/Scripts/python.exe scripts/train_loghi_swedish.py
# choose option 6, "Start or resume a training session"
```

## 8. Meaning of each stop reason

| `stop_reason` | Meaning |
|---|---|
| `time_budget_reached` | The session's configured wall-clock cap was reached. |
| `stop_requested` | A safe stop was requested via the `STOP_REQUESTED` sentinel file. |
| `epoch_would_not_fit` | The next epoch's estimated duration would not fit the remaining budget. |
| `epoch_failed` | The container/process for one epoch failed before producing a verified checkpoint — the only stop reason classified `stopped_mid_epoch=True`. |
| `target_epochs_reached` | The session's configured epoch cap was reached. |
| `no_val_cer_improvement` | Validation CER did not improve for the configured early-stopping patience. |

Every stop reason except `epoch_failed` only ever occurs *between* epochs, after that epoch's
validation ran and its checkpoint was verified — there is no mid-epoch stop capability by design
("one epoch = one container invocation", `training_session.py`'s own module docstring).

## 9. Limitations of ETA estimates

This phase is epoch-granularity only. There is no per-batch/step telemetry — "current batch,"
"lines processed within the current epoch," and true loss-by-training-step charts are out of scope
(they would require replacing the trainer's blocking `subprocess.run()` with a streaming `Popen` and
live stdout parse of Keras's progress bar, a real architecture change deliberately deferred past this
phase). Learning-rate-by-step is the one step-level series shown, and it is *computed* from each
epoch's real recorded schedule parameters (`htr/training/lr_schedule.py`, ported from the pinned
`LoghiLearningRateSchedule` formula) rather than scraped from output — no stdout parsing was needed
for it.

Epoch-duration and time-remaining estimates use the run's own `recent_epoch_durations` once at least
one real epoch has completed; before that, they fall back to the run profile's default estimate
(§10). Both are estimates, not commitments — a `WARNING` is shown whenever the configured wall-clock
cap is shorter than the estimated duration of a single epoch.

## 10. Full-corpus vs. measured pilot timings

`htr/training/run_profile.py` is the single source of truth for both profiles' timing assumptions —
the dashboard never hardcodes them elsewhere:

- **Pilot** (`pilot_profile()`): 9999 train + 1000 val lines, **~547.7s/epoch — real, measured** on
  this host (2026-08-01, batch_size=16), default 3h cap / patience 5.
- **Full corpus** (`full_corpus_profile()`): 563,933 valid lines, **~7.17h/epoch — extrapolated, not
  measured**, fit from two real timing points (a 25-line smoke probe and the 10,999-line pilot
  epoch). Refuses a silent wall-clock default; a first full-corpus run's suggested starting point is
  3 epochs / ~27h, shown as an explicit, chosen starting budget rather than derived silently from the
  per-epoch estimate. `cap_shorter_than_one_epoch_warning()` fires whenever a configured cap would not
  fit even one epoch of whichever profile is active.

Neither figure is a hard-coded universal truth — once a run has completed at least one real epoch,
its own `recent_epoch_durations` supersede the profile's static estimate for that run's own ETA
display.
