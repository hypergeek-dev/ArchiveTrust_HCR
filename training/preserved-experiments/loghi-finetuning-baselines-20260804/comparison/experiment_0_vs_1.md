# Experiment 0 vs. Experiment 1: controlled comparison

Generated 2026-08-04T04:49:09Z by `scripts/build_preservation_package.py`.

## Controlled comparison status: CONTROLLED

Sole intended independent variable: **trainer lifecycle / optimizer and LR continuity**.

| field | Experiment 0 | Experiment 1 | identical? |
|---|---|---|---|
| parent_checkpoint_sha256 | `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95` | `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95` | yes |
| architecture | `new10` | `new10` | yes |
| training_dataset_hash | `fc21a70825650927edcae753a20411cef8e4344e2c22bcbcaed1c0038b9b97b0` | `fc21a70825650927edcae753a20411cef8e4344e2c22bcbcaed1c0038b9b97b0` | yes |
| train_manifest_hash | `e12796a02c868d849b9e113a4b415f94ff483fe2148c1c58d17cd36eba4af46f` | `e12796a02c868d849b9e113a4b415f94ff483fe2148c1c58d17cd36eba4af46f` | yes |
| validation_manifest_hash | `92f74011c568b04798c9c53725311633f2df2ea4fb8c408e1d73b4f197cea85c` | `92f74011c568b04798c9c53725311633f2df2ea4fb8c408e1d73b4f197cea85c` | yes |
| character_vocabulary_hash | `f1877392e09591cf40517c5ef636d6011a1c2b61e4a240b508167d0c94ff6754` | `f1877392e09591cf40517c5ef636d6011a1c2b61e4a240b508167d0c94ff6754` | yes |
| batch_size | `16` | `16` | yes |
| optimizer | `adam` | `adam` | yes |
| initial_learning_rate | `0.0001` | `0.0001` | yes |
| precision_policy | `mixed_float16 (loghi-htr's own setup/environment.py applies this automatically whenever a GPU is used and --use_float32 is not passed; neither experiment passed --use_float32)` | `mixed_float16 (loghi-htr's own setup/environment.py applies this automatically whenever a GPU is used and --use_float32 is not passed; neither experiment passed --use_float32)` | yes |
| seed | `42` | `42` | yes |
| docker_image_digest | `sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8` | `sha256:414fc89ac574a61fd745836ad9852842759bf3cf7046e4549ae96315ff9132e8` | yes |
| gpu_model | `NVIDIA GeForce RTX 3070` | `NVIDIA GeForce RTX 3070` | yes |

### Disclosed differences

- **code_commit** (not material): 'b518e51bc5578edc3ce58b041383c23cfd4e16db' vs '26fdf29105f26d3c8394de10528e9a8866ffcc41' -- 4 commits landed between the two runs, confined to src/archivetrust/htr/training/full_run/ (shard-orchestration bookkeeping, the new end-of-lap evaluation suite, CLI/preflight additions). None touch container_epoch_runner.py, training_session.py, checkpoint_index.py, or training_identity.py -- the code paths that actually build the docker invocation and training configuration. See provenance/repository_diff.patch.
- **trainer_lifecycle / optimizer and LR continuity** (material): '57 separate trainer invocations, optimizer/scheduler reset each time' vs '1 continuous trainer invocation, optimizer/scheduler never reset' -- This is the sole INTENDED independent variable of this comparison.

## Held-out result (same evaluation code, same 1,000-line validation set)

| | Experiment 0 | Experiment 1 | delta (Exp1 - Exp0) |
|---|---|---|---|
| corpus CER | 0.1728 | 0.1747 | +0.0019 |
| corpus WER | 0.4945 | 0.4956 | +0.0011 |

## Authoritative interpretation

Correcting optimizer and learning-rate continuity was methodologically necessary, but it did not materially improve held-out generalisation after one full epoch for this corpus, architecture, optimizer, and learning-rate configuration. The difference between the experiments is within observed noise (CER delta +0.0019, WER delta +0.0011 for Experiment 1 relative to Experiment 0). This does not mean Experiment 1 is meaningfully worse, does not mean Experiment 0 is superior, and does not mean optimizer continuity never matters -- the supported conclusion is limited to this exact comparison and training budget.

## Per-collection detail

See `per_collection_comparison.csv` in this directory.
