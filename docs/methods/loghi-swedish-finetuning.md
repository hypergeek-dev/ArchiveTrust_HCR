# Swedish Loghi Fine-Tuning — Pilot Resumable Training Run

Status: In progress — pilot phase (`loghi_swedish_finetuned_v1`, `training_phase=pilot_10k`)
Governs: How the generic Dutch Loghi-HTR checkpoint is fine-tuned on real Swedish training data,
resumably, in ~5-hour sessions.

## 1. Reused vs. extended (Work Package 1)

Everything below was inspected before writing any new code. Nothing in the "Reuse unchanged" column is
modified by this work.

| Component | Status | Notes |
|---|---|---|
| `providers/loghi/adapter.py`, `environment.py`, `facade.py`, `page_xml.py`, `stage_results.py` | Reuse unchanged | Inference-time concerns; fine-tuning is a materially different lifecycle, built alongside, not inside, this package. |
| `providers/loghi/pinned_versions.py` | **Extend** | Placeholders replaced with real resolved values (§2) — the file's own docstring anticipates this as "itself a new, reviewable commit." |
| `providers/loghi/models.py::LoghiComponentVersions` | Reuse unchanged | Still the one place pipeline pins live; training pins reuse its shape (`training_identity.py` composes rather than duplicates it). |
| `htr/research_status.py` | Reuse unchanged | Fine-tuning produces a *candidate checkpoint*, not a new active method — `loghi`'s active status is untouched. |
| Four-cell experiment builder (`htr/screening/lion_loghi_experiment.py`) | Reuse unchanged | Not invoked by this work; the fine-tuned checkpoint is not benchmarked here. |
| Loghi smoke-test module (`htr/screening/loghi_smoke_test.py`) | Reuse unchanged | Inference smoke test; training has its own (§ resumable training wrapper). |
| `scripts/run_lion_loghi_comparison.py` | Reuse unchanged | Not duplicated — `scripts/train_loghi_swedish.py` is a separate launcher for a separate concern (training identity, not method comparison). |
| `domain/telemetry/events.py`, `htr/persistence/durable_store.py` | **Extend** | Same emit-then-project pattern the Loghi inference events already established; only 3 new kinds for the training-session lifecycle (§ telemetry). |
| The 14-category Loghi failure taxonomy | Reuse unchanged | Inference failures; training failures are a distinct, smaller taxonomy scoped to this package. |
| `.gitignore` | **Extend** | New `training/loghi-swedish-v1/{source-inventory,prepared-data,checkpoints,session-records,run-state}/` and `.loghi-upstream/` paths. |

## 2. Real, pinned Loghi environment (Work Package 2)

Resolved from the actual installed environment — nothing below is invented, and every field below
either names a real value or is not yet filled in (never floating `latest`/`main`).

*(Filled in as installation proceeds — see `providers/loghi/pinned_versions.py::CURRENT_PINNED_VERSIONS`
for the authoritative, machine-readable record.)*
