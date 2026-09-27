# Datasets

Snapshot: 2026-09-27, machine `H:\ArchiveTrust_HCR 2`. See `docs/BENCHMARK_PROTOCOL.md` for how an
external dataset becomes a benchmark.

| Dataset | Where | Status | Role |
| --- | --- | --- | --- |
| Riksarkivet "Training data for Swedish Lion Libre" (HF parquet, 11 collections, 565,146 lines) | `F:\huggingface_dataset` on the old machine | **Not on this machine** | Training corpus of the Loghi model. Also Lion's training source. Needed for the overlap index. |
| Loghi training splits (train / val 1000 lines / sealed test 810 lines) | Manifests deleted in commit f0de635 | Only hashes survive: dataset `fc21a708…`, train `e12796a0…`, val `92f74011c568…` | Val produced CER 0.0998 / WER 0.3273. The sealed test was never evaluated and cannot be rebuilt without the corpus. |
| `dataset-rgb/` | Repo root (gitignored) | 766 page PNGs in 31 volume folders (14 GB, RGB, mostly double-page spreads). **No transcriptions.** | **`NO_GROUND_TRUTH / NOT_AN_ACCURACY_BENCHMARK`.** Used only for the mechanical dry run (see below). Inspection correctly reports `gt.none_found`. No CER/WER may ever be computed or quoted on it. |
| Reliability-screening segmentation and crops | `docs/experiments/technical-reliability-screening/full-run/reliability-2026-07-31/` (crops gitignored) | Florence-2 line boxes for 60 `dataset-rgb` pages (2,678 lines). No GT. | Segmentation input for the dry run (`scripts/dryrun_segmentation_from_reliability.py`). |
| External benchmark set (from a colleague): `svea-hovratt-2026-09` | `benchmark-data/incoming/svea-hovratt-2026-09/` | **Delivered and frozen 2026-09-27** as `svea-hovratt-2026-09-primary` (6,486 lines); provenance partly unavailable, see below. | The independent Loghi-vs-Lion benchmark. See the section below. |

## dataset-rgb mechanical dry run (NO_GROUND_TRUTH / NOT_AN_ACCURACY_BENCHMARK)

```powershell
python scripts/dryrun_segmentation_from_reliability.py <tmp>\dataset-rgb-segmentation.jsonl
python -m archivetrust.htr.benchmark dryrun-build dataset-rgb <tmp>\dataset-rgb-segmentation.jsonl dataset-rgb-dryrun
python -m archivetrust.htr.benchmark dryrun-run dataset-rgb-dryrun --model lion --limit 5
python -m archivetrust.htr.benchmark dryrun-run dataset-rgb-dryrun --model loghi --limit 5
```

The output goes to `benchmark-data/dryrun/dataset-rgb-dryrun/`. On 2026-09-27 the build produced
60 pages → 2,678 crops with no findings, and manifest SHA-256 `eb57f9bc…`.

- Two independent builds were byte-identical.
- The crops are pixel-identical to the reliability run's own crops.
- Both smoke runs (5 lines each, on the same crops) returned `ok` for every line:
  - Lion ran with `generation_config` on CUDA.
  - Loghi ran with beam 10 and seed 42 in the pinned container digest; its checkpoint hashes
    were verified before the run.

The predictions only show that each model loads and emits text. **They are not an accuracy
measurement.**

## svea-hovratt-2026-09 (external GT, Transkribus export)

The delivery is 14 Transkribus export jobs (PAGE-XML and ALTO with JPG pages), 237 pages and
10,557 lines, delivered as one macOS zip. Its tree digest is `ae80127b…` over 1,618 files,
including the `__MACOSX` metadata files. The decisions (D1–D5, with D4b extended to unbracketed `??`/`???`) are
in `docs/BENCHMARK_PROTOCOL.md` §3.

The candidate primary set, rebuilt 2026-09-27 after the D4b extension, is **5 documents, 140
pages, 6,486 lines, 213,824 characters, 37,545 words**:

- lines per export job: 4502442: 1,661; 4502443: 595; 4502444: 776; 4502445: 2,137; 4502446: 1,317;
- 59 lines are held in the editorial-markup review queue.

- characters are NFC code points of canonical GT, including inner spaces;
- words are split on whitespace, as the scorer does.

Caveat: all 140 pages have Transkribus status `IN_PROGRESS`; none is `GT` or `FINAL`.

It was **frozen** on 2026-09-27, before any model run, as `benchmark/svea-hovratt-2026-09-primary`:
manifest `7c47b9ee…`, decisions `24c815be…`, `FROZEN.json` `ffd2a22f…`.

- The transcriptions were made by students.
- Provenance status: PROVENANCE UNAVAILABLE / CANNOT BE RESOLVED FROM SOURCE. The remaining
  convention questions cannot be answered.
- The benchmark measures agreement with the supplied reference transcription.
- Details are in `benchmark-data/work/svea-hovratt-2026-09/PROVENANCE.md`, and in `FROZEN.json`
  under `dataset_card`.

## Training corpus search

On 2026-09-27 no `huggingface_dataset` folder was found on any mounted drive (C:, D:, E:, F:,
G:, H:, I:, searched two levels deep). The contamination check (`overlap`) is therefore
**partial**. The benchmark can still run, and the provider's answers about provenance (below)
become the main defence against overlap.

## Provenance checklist (ask the provider, record the answers before `freeze`)

Record the answers in `benchmark-data/work/<delivery-id>/PROVENANCE.md`, together with:

- who delivered the data, the delivery date and the delivery medium;
- the SHA-256 tree digest that `build.json` computes.

1. Where exactly did the material come from? Which archive and collection? Give reference codes or
   volumes if possible.
2. Was it downloaded from Riksarkivet, and if so, from which service or dataset?
3. Did it originate from, or pass through, Transkribus?
4. Was any of it part of a public HTR training corpus? In particular, Riksarkivet's HF datasets,
   or "Training data for Swedish Lion Libre".
5. Was Lion, or any related Riksarkivet model, trained or fine-tuned on this material?
6. Who created the ground truth (person or project), and how was it checked?
7. Is the transcription diplomatic or normalized? How are abbreviations, special characters,
   superscripts, deletions and insertions, and editorial brackets handled?
8. Is the transcription page-level or line-level?
9. Are line polygons or baselines included? From which tool, and were they manually corrected?
10. Are writer and document IDs available?
11. Are there licensing or reuse restrictions? May results and example lines be published?

A "yes" or "don't know" to question 4 or 5 does not stop the benchmark. It has to be reported next
to the results, and the `overlap` check becomes more important.

## Rules for the external benchmark set

- Put the delivery in `benchmark-data/incoming/<source_id>/` exactly as received. If it is an
  archive, extract it into a new source folder. Nothing in `incoming/` is ever modified.
- It must never be used for training, fine-tuning, charset changes or hyper-parameter choices.
- Record who provided it, under what terms, and its provenance (see the checklist above; archive, volumes, who transcribed
  it, and the transcription conventions) in `benchmark-data/work/<source_id>/README.md`. The
  inspection will ask about bracketed markup and special characters. Only the provider can say
  whether they are diplomatic or editorial.
- Ask the provider whether any of it came from Riksarkivet's published HF data or from Transkribus
  collections that fed it. That is the most likely route to contamination.
