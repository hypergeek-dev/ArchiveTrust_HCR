# Loghi Swedish HTR fine-tuning baselines: preservation package

Preserved 2026-08-04. Covers two completed, immutable training runs.
Neither run's original artifacts were modified, renamed, deleted, or reinterpreted by this package --
everything here is either a small derived summary/index or an unmodified copy of a small text/JSON/CSV
artifact. Model weights are never copied; checkpoints are referenced by path + independently
re-verified sha256.

## Experiment 0 -- the shard-reset baseline

Generic Loghi checkpoint fine-tuned over the full corpus through 57 separate shard-level trainer invocations (the shard-reset baseline). NOT one continuous epoch: each of the 57 invocations was an independent container process.

- Held-out corpus CER: **0.1728**
- Held-out corpus WER: **0.4945**
- Run directory: `D:\ArchiveTrust_HCR\training\full-corpus-20260802T044250Z`

## Experiment 1 -- the continuous-epoch (corrected-method) run

Generic Loghi checkpoint fine-tuned for one complete, uninterrupted full-corpus epoch in a single trainer/container lifecycle -- the corrected-method run.

- Held-out corpus CER: **0.1747**
- Held-out corpus WER: **0.4956**
- Best checkpoint: `D:\ArchiveTrust_HCR\training\experiment-1-continuous-epoch-20260803T192146Z\run-state\epoch_output\epoch_1\model_new10\best_val`
- Run directory: `D:\ArchiveTrust_HCR\training\experiment-1-continuous-epoch-20260803T192146Z`

## Authoritative interpretation

Correcting optimizer and learning-rate continuity was methodologically necessary, but it did not materially improve held-out generalisation after one full epoch for this corpus, architecture, optimizer, and learning-rate configuration. The difference between the experiments is within observed noise (CER delta +0.0019, WER delta +0.0011 for Experiment 1 relative to Experiment 0). This does not mean Experiment 1 is meaningfully worse, does not mean Experiment 0 is superior, and does not mean optimizer continuity never matters -- the supported conclusion is limited to this exact comparison and training budget.

## Package structure

```
experiment_0/   metadata.json, artifact_index.json, reports/, configs/, metrics/, checkpoint_references/
experiment_1/   (same shape)
comparison/     experiment_0_vs_1.{md,json}, per_collection_comparison.csv, metric_summary.csv
provenance/     code_revision.txt, repository_diff.patch, container_image.json,
                dataset_identity.json, evaluation_identity.json
checksums.sha256   sha256 of every file in this package (verify with: sha256sum -c checksums.sha256)
preservation_manifest.json   top-level index + build provenance
```

## What is NOT in this package

- Model weights (`.keras` files) -- referenced by path + hash in `checkpoint_references/`, never copied.
- The sealed 810-line reserved test set -- neither experiment's training or evaluation touched it, and
  this preservation pass did not open, read, or hash it. Its path is recorded for lineage only.
- The 57 per-shard `.keras` checkpoints of Experiment 0 beyond `best_val`/`end_of_epoch` -- referenced
  collectively via `checkpoint_index.json` in the original run directory, not individually indexed here.

## How to cite these runs

Cite `experiment_0/metadata.json` / `experiment_1/metadata.json` directly, or the comparison in
`comparison/experiment_0_vs_1.json`. Both are self-contained and do not require reopening the original
run directories or re-deriving the comparison.
