# Swedish Lion I vs. Loghi: In-Domain and Reciprocal Cross-Domain Comparison

Status: Current — active research phase (`htr/research_status.py::CURRENT_RESEARCH_PHASE`)
Active methods: `swedish_lion`, `loghi`
Archived from this phase: `satrn`, `florence2_htr` (sealed evidence:
`docs/experiments/technical-reliability-screening/`)
Inactive (out of scope, not archived): `transkribus_swedish_lion_1`

## Why the active pair changed

The prior technical reliability screening benchmark (SATRN, Florence-2) is complete and sealed. Nothing
about that evidence is invalidated — see `docs/loghi-integration-audit.md` for the full audit of what
was reused, extended, or parked. This phase narrows the active comparison to two methods and one new
research question: how do a local Swedish-specific model and a local, containerized, general pipeline
compare in-domain and cross-domain on Swedish and Dutch historical handwriting?

## A correction, stated once, load-bearing throughout this directory

"Swedish Lion I" in this phase is `swedish_lion` (Swedish Lion **Libre**, `Riksarkivet/
trocr-base-handwritten-hist-swe-2`) — a real, **local** TrOCR model, not the external, manual-upload
Transkribus workflow the phrase "Swedish Lion I" describes elsewhere in this codebase
(`transkribus_swedish_lion_1`). This was an explicit choice confirmed with the user during this
integration (`docs/loghi-integration-audit.md` §0). Consequently:

* Both active methods execute **locally** — there is no external upload/queue/wait time in this phase.
  See `docs/EVALUATION_PROTOCOL.md` §12.5 for how the timing-category discussion was adapted
  accordingly.
* `transkribus_swedish_lion_1` remains fully supported and documented
  (`docs/methods/transkribus-swedish-lion-1.md`), simply not part of the active pair.

## Datasets

* `dataset-rgb/` = **Swedish corpus** (unchanged, not renamed — 766 witch-trial-record pages,
  1584–1764).
* `dataset-dutch-rgb/republic7/` = **Dutch corpus** (515 pages, Nationaal Archief scans with
  Transkribus ground-truth PAGE XML — see `dataset-provenance-dutch.md` for what is and is not known
  about its origin and licensing, and `dataset-comparability.md` for how the two corpora compare).

Both corpora pass through the same, unmodified, versioned RGB-normalization stage
(`htr/preprocessing/`) before either method sees them — the shared page-input policy:

```
Original page
  → canonical RGB-normalized derivative
    ├── swedish_lion workflow (in-process TrOCR call)
    └── loghi workflow (Laypa → Loghi Tooling → Loghi HTR, containerized)
```

## Four-cell design

`htr/screening/lion_loghi_experiment.py::build_lion_loghi_comparison` constructs four structurally
separate `Experiment`/`ExperimentVersion` pairs, never one `ExperimentRun` covering all four, grouped
under one `ExperimentComparisonGroup`:

| Cell | Method | Corpus | Domain relationship |
|---|---|---|---|
| Swedish Lion I on Swedish pages | `swedish_lion` | Swedish | `in_domain` |
| Swedish Lion I on Dutch pages | `swedish_lion` | Dutch | `cross_domain` |
| Loghi on Swedish pages | `loghi` | Swedish | `cross_domain` |
| Loghi on Dutch pages | `loghi` | Dutch | `in_domain` (assumes Loghi's Dutch-trained checkpoint) |

Report `Method × Corpus` and `Method × Domain relationship` tables — never one merged score.

## What has, and has not, been done

**Done in this integration pass:** research-status model, Loghi adapter (environment probing, Docker/
WSL2-aware facade, PAGE XML parsing, failure taxonomy), capability matrix regeneration, telemetry, the
four-cell experiment builder, the Dutch dataset extracted with provenance recorded, the feasibility
smoke-test tooling (verified against a fake facade), and this documentation set.

**Not done, deliberately:** no real Loghi Docker environment has been installed (pins are placeholders
— `providers/loghi/pinned_versions.py`), no smoke test has been run against real containers, and no
Lion-vs-Loghi result — technical, plausibility, or accuracy — is claimed anywhere in this repository.
See `RUNNING_THE_COMPARISON.md` for what happens next and who triggers it.

## Documents in this directory

| File | Contents |
|---|---|
| `dataset-provenance-dutch.md` | What is known and unknown about the Dutch corpus's origin and licensing |
| `dataset-comparability.md` | Swedish-vs-Dutch corpus comparison, dimension by dimension |
| `RUNNING_THE_COMPARISON.md` | How to actually run the smoke test and, later, the full comparison |
| `dataset-dutch-manifest.json` / `dataset-dutch-inventory.csv` | Real per-page hashes/dimensions for the Dutch corpus |
