# Loghi Full-Corpus Training — Operator-Controlled Workflow

Status: Implemented, not yet started — everything below is prepared and validated; the real
full-corpus training run has not been launched (an explicit operator decision, per the brief's
"Do not start the real full-corpus training automatically").
Governs: How the pilot's real, observed behavior becomes the full-corpus run's monitoring policy,
how the model is reset for that run, and how to prepare/preflight/start/stop/resume it via the CLI
or GUI without opening VS Code.

## 1. What was learned from the pilot

The pilot (`training/loghi-swedish-v1/`) ran 22 real epochs over a bounded 9999-line subset. Its
real, recorded `validation_history` shows:

- val_CER improved every single epoch, from 0.248 (epoch 1) to 0.169 (epoch 22) — it never
  triggered its own patience-5 early stopping.
- The *rate* of improvement shrank steadily and then got genuinely noisy in the final epochs
  (epoch 15's delta was −0.0002, almost flat, before epoch 16 recovered to −0.0027) — the first real
  sign of a plateau approaching, not yet a plateau itself.
- `train_loss`/`val_loss` were only recorded for the last 7 of 22 epochs (added mid-project); over
  that window, the gap between them was consistently negative (val_loss below train_loss) and
  **narrowing** — real evidence used directly by `pilot_analysis.py`, not glossed over.

## 2. Which pilot observations became monitoring rules

`htr/training/full_run/pilot_analysis.py::analyze_pilot_run` turns the above into real numbers
(`training/config/pilot_analysis.json`); `monitoring_config.py::derive_monitoring_config` turns
those into full-run policy (`training/config/pilot_derived_monitoring.json`):

| Pilot observation | Full-run rule |
|---|---|
| No real, observed plateau within 22 epochs (still improving at the end) | `estimated_plateau_epoch` is **extrapolated** (log-linear fit of the trailing deltas), explicitly labeled as such — never presented as measured. |
| `steps_per_pilot_epoch = ceil(9999 / 16) = 625` | `min_exposure_steps = steps_per_pilot_epoch * plateau_epoch` — a **step count**, not a raw epoch number, so it translates correctly to a full-corpus "epoch" (one shard) that is a materially different size. |
| Longest observed near-flat streak before recovery | `recommended_patience = max(3, streak + 2)` — real evidence, not a fixed default. |
| No `--steps_per_epoch` reliance (upstream-documented broken, `modes/training.py:81` `# FIXME`) | One full-run "epoch" = one **shard** (`corpus_sharding.py`), sized by default to match the pilot's own 9999-line epoch — checkpoint/validation frequency is therefore always "every shard," never coarser. |

## 3. Why the model is reset before the full run

The full run always starts from the pristine pinned `generic-2023-02-15` checkpoint, never from any
pilot epoch's output — enforced structurally, not by convention (§5 below). This also happens to be
what gives a **genuinely fresh optimizer/scheduler state** for free: the pristine checkpoint is an
old-format Dutch checkpoint that Keras's own loader (`model/management.py::load_model_from_directory`,
read directly, not assumed) falls back to a weights-only reconstruction path for — optimizer state is
never touched by that path. No separate "reset optimizer" code was needed once the pilot-checkpoint
rejection was in place.

The pilot's **training** data (its 9999-line train split) *does* remain part of the full corpus —
only its `val_manifest`/`test_reserved_manifest` line IDs are excluded from full-corpus shards, so
validation never overlaps training and the reserved test set stays untouched for the eventual sealed
benchmark.

## 4. Launching the GUI outside VS Code

```
.\start_full_training.ps1
```

or directly:

```
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m archivetrust.htr.training.full_run gui
```

Requires the `gui` extra: `pip install -e ".[gui]"`.

## 5. Running the same workflow from PowerShell (CLI)

```powershell
$env:PYTHONPATH = "src"

# 1. Analyze the pilot -> training/config/pilot_analysis.json + pilot_derived_monitoring.json
python -m archivetrust.htr.training.full_run analyze-pilot --pilot-run training/loghi-swedish-v1

# 2. Inspect/edit training/config/pilot_derived_monitoring.json by hand if desired, or re-run
#    analyze-pilot with --shard-line-count / --patience / --max-epochs / --corpus-line-count overrides.

# 3. Preflight (requires Docker running)
python -m archivetrust.htr.training.full_run preflight --config training/config/pilot_derived_monitoring.json

# 4. Prepare a fresh run (never overwrites an existing run directory)
python -m archivetrust.htr.training.full_run prepare --config training/config/pilot_derived_monitoring.json --run-name my-full-run

# 5. Start (explicit, separate step -- never automatic)
python -m archivetrust.htr.training.full_run start --run training/my-full-run --hours 5

# 6. Check status any time
python -m archivetrust.htr.training.full_run status --run training/my-full-run
```

## 6. Stopping and resuming safely

- **Stop**: `python -m archivetrust.htr.training.full_run stop --run <path>` writes a
  `STOP_REQUESTED` sentinel file. The run finishes its **current shard** (validates, checkpoints,
  verifies) and stops there — there is no mid-shard stop capability by design, matching the pilot's
  own `training_session.py` architecture ("one epoch = one container invocation").
- **Force kill** (GUI only, a separately-confirmed second action): terminates the subprocess
  immediately. The last verified checkpoint is preserved regardless; only progress since then is lost.
- **Resume**: `python -m archivetrust.htr.training.full_run resume --run <path> --hours 5` continues
  from the real persisted `session_state.json`/`run_state.json` — the same shard index, optimizer
  state (via the checkpoint), `epochs_since_improvement` counter, and patience gate it left off with.
  `resume` refuses to run against a `"prepared"` (never-started) run; use `start` for that.

## 7. Identifying the best final checkpoint

`run_state.json`'s `best_checkpoint` field, or:

```
python -m archivetrust.htr.training.full_run status --run <path>
```

`best_checkpoint` is tracked independently of `latest_checkpoint` throughout — a later, worse epoch
never overwrites it (`checkpoint_index.py`'s existing `"best_val"` kind, reused unchanged).

## 8. Distinguishing a plateau from overfitting

Both are shown in the GUI's "Monitoring state" field and computable via
`monitoring_state.py::classify_monitoring_state`:

- **Plateau candidate**: `epochs_since_improvement > 0` — the most recent shard did not set a new
  best val_CER. This is the *only* signal used; a raw comparison against the best value was tried
  and rejected (a shard that just set a new best always ties the best value exactly, which a
  gap-based check misreads as "close to the best" — i.e. a false plateau signal).
- **Likely overfitting**: `val_loss > train_loss * 1.5` — requires *both* real values; **never**
  inferred from training loss alone (the brief's explicit rule, "never declare a plateau using
  training loss alone," extended here to overfitting).

## 9. Where things are stored

```
training/config/
  pilot_analysis.json              # analyze-pilot's real, observed pilot behavior
  pilot_derived_monitoring.json    # derived full-run policy (editable before prepare)

training/<run-name>/
  run-state/
    training_identity.json         # run_id, base model, "full_corpus" phase, created_at
    session_state.json             # cumulative shard progress (training_session.py's engine)
    run_state.json                 # the WP6 audit file -- status, PID, heartbeat, metrics, stop reason
    checkpoint_index.json          # append-only, atomic -- latest/best_val/end_of_session entries
    STOP_REQUESTED                 # present only while a graceful stop is pending
  shards/
    shard_00000.parquet ...        # deterministic, seeded full-corpus shards (excl. pilot val/test)
    sharding_summary.json
  config/
    full_run_monitoring.json       # the monitoring config this specific run was prepared with
```

## Remaining risks and assumptions (honestly disclosed)

- `estimated_plateau_epoch` for this project's real pilot is **extrapolated**, not observed — the
  pilot never actually plateaued before its wall-clock cap. Treat the derived `min_exposure_steps`/
  `recommended_patience` as a documented starting point, editable before `prepare`.
- The full-corpus per-epoch (per-shard-lap) time estimate (~7.2h for a true full pass) is itself an
  extrapolation from two real timing points (`run_profile.py`'s own docstring) — shard-level timing
  will be far more accurate once real full-corpus shards actually run, since a shard is sized to
  match the pilot's own measured ~9-minute epoch.
- This phase is epoch(shard)-granularity only — no per-batch/step telemetry, matching the read-only
  dashboard's own documented scope decision (`docs/methods/loghi-training-dashboard.md`).
- Preflight's smoke test and the checkpoint round-trip check both require Docker; they report a
  clear, actionable failure (not a crash) when Docker is unreachable.
