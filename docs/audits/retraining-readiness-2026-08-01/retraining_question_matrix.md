# Retraining Question Matrix

Status legend: **FACT** = verified fact (direct evidence). **INFERRED** = conclusion drawn from
evidence but not directly observed. **ASSUMPTION** = accepted without direct verification.
**UNANSWERED** = genuinely unresolved by this audit, requires a human/governance decision or
work this audit did not perform. Severity applies only where the question exposes a gap; blank
severity means the question is answered with no residual concern.

---

## A. Objective and success criteria

| # | Question | Answer | Evidence | Status | Severity | Required action |
|---|---|---|---|---|---|---|
| 1 | Exact problem the retrained model solves | Improve HTR (CER) on Swedish historical handwritten documents beyond the pinned generic Dutch-trained Loghi checkpoint, using the Riksarkivet Swedish Lion Libre training data | `docs/methods/loghi-swedish-finetuning.md`, `TRAINING_DATASET_ID = "riksarkivet_swedish_lion_libre_training_data"` | FACT | | |
| 2 | Domains/languages/periods/document types/scan qualities in scope | Swedish historical handwriting, scope defined by the source inventory's 11 collections | `pilot_split.py` docstring listing collection names | INFERRED (collection names known; no formal scope document read) | LOW | Document a formal in/out-of-scope statement |
| 3 | Explicitly out of scope | Not stated anywhere in the repository | — | UNANSWERED | LOW | Governance decision |
| 4 | What "better" means | Not formally defined anywhere in the repository | — | UNANSWERED | MEDIUM | Governance decision before Gate 7 |
| 5 | Authoritative metric | `val_CER` is used throughout as the operational metric (`validation_metric="val_CER"`, `validation_mode="minimize"` in `FullRunMonitoringConfig`); no explicit statement that this is the *final acceptance* metric vs. test CER/WER/human review effort | `monitoring_config.py` | INFERRED | MEDIUM | Explicitly declare the Gate 7 acceptance metric |
| 6 | Baseline model performance on the fixed held-out test set | Not found — the reserved test set (810 lines) has never been evaluated per this audit's evidence (`test_cer: None` field explicitly documented as "never evaluated" in `training_dashboard_viewmodel.py::RunSummaryRow`) | `training_dashboard_viewmodel.py` | UNANSWERED | HIGH | Must be established before Gate 7 |
| 7 | Minimum improvement required to justify retraining | Not defined | — | UNANSWERED | MEDIUM | Governance decision |
| 8 | Unacceptable regression threshold | Not defined | — | UNANSWERED | MEDIUM | Governance decision |
| 9 | Is aggregate CER sufficient, or subgroup breakdown required | Not enforced anywhere in code; `RunSummaryRow` has no subgroup fields | — | UNANSWERED | HIGH (given R-009's leakage risk for specific collections) | Require subgroup CER at Gate 7 |
| 10 | Research artifact / prototype / production / publication | Not stated | — | UNANSWERED | LOW | Governance decision |
| 11 | Evidence that would qualify the model for each intended use | Not defined | — | UNANSWERED | LOW | Governance decision |
| 12 | Who has authority to approve the final model | Not defined anywhere in the repository | — | UNANSWERED | MEDIUM | Governance decision, needed for Gate 7 |
| 13 | What would cause rejection even if CER improves | Not defined | — | UNANSWERED | MEDIUM | Governance decision |

## B. Model identity and lineage

| # | Question | Answer | Evidence | Status | Severity |
|---|---|---|---|---|---|
| 1 | Exact Loghi implementation | Pinned `loghi/docker.htr` image, digest `sha256:414fc89a...` | `pinned_versions.py`; `docker image inspect` | FACT | |
| 2 | Exact architecture | `new10` VGSL-style architecture | `TrainingConfiguration.model_architecture="new10"` | FACT | |
| 3 | Original checkpoint path | `.loghi-upstream/pretrained-models/loghi-htr/generic-2023-02-15/model.keras` | `PARENT_CHECKPOINT_DIR` | FACT | |
| 4 | Cryptographic hash | `0da2c00ab2b12b23e9f64c01ec67ad29724f275eeadb4561843f8a29ff6fff95` | Recomputed twice, matches pin both times | FACT | |
| 5 | Origin repository/release/version/commit | `generic-2023-02-15` — a KNAW-HuC Loghi generic release identifier; exact upstream git commit/release tag not independently re-verified in this audit | `pinned_versions.py::model_checkpoint_id` | ASSUMPTION | LOW |
| 6 | License compatibility (retraining/redistribution/publication/commercial/municipal/derivative) | Not verified — no LICENSE file for the checkpoint or `loghi-htr` was located and read as part of this audit | — | UNANSWERED | HIGH |
| 7 | Is the tokenizer/charset embedded in the model | The old-format pristine checkpoint uses a separate `charlist.txt`; the new10-format checkpoints (post-conversion) carry a `tokenizer.json` | Verified via directory listing before/after conversion | FACT | |
| 8 | Tokenizer version verified | `charlist_hash` is included in `configuration_hash` computation; no separate "tokenizer version" concept beyond that hash | `TrainingConfiguration.charlist_hash` | FACT | |
| 9 | Architecture/checkpoint tensor compatibility | Delegated entirely to the pinned container's own `model/management.py::load_or_create_model`; not independently re-verified beyond the real, passing smoke test (a genuine forward+backward pass on the real checkpoint) | Real preflight `smoke_test_forward_backward` PASS | FACT (compatibility demonstrated for at least one real epoch) | |
| 10 | Missing/unexpected/silently-ignored state-dict keys | Not independently checked; `tf.keras.models.load_model()`'s own default behavior applies, not overridden | — | UNANSWERED | LOW |
| 11 | Strict checkpoint loading enabled | Not overridden by ArchiveTrust — pinned container's own default | `.loghi-upstream/loghi-htr/src/model/management.py` (not fully read in this audit) | UNANSWERED | LOW |
| 12 | All layers trainable | Not overridden — no freezing flags passed anywhere in `container_epoch_runner.py::_build_argv` | Direct inspection of the argv builder | FACT (no freezing configured) | |
| 13 | Any layers intentionally frozen | No | Same as above | FACT | |
| 14 | Fine-tuned / continued-pretrained / from-scratch | Fine-tuned from the real pinned checkpoint (`--model <staged copy>` always passed; never a VGSL spec string) | `container_epoch_runner.py` docstring: "do not train from random initialization is enforced structurally here" | FACT | |
| 15 | Pilot checkpoint excluded from final parent lineage | Yes, structurally | `identity.py::assert_parent_checkpoint_is_not_a_pilot_path`; `launch_manifest.json: pilot_checkpoint_used_as_parent=false` | FACT | |
| 16 | Can the launch guard prove the selected parent is the original approved checkpoint | Yes — hash-verified at preflight and recorded in the launch manifest | `preflight.py::_check_base_checkpoint_hash_matches_pin` | FACT | |
| 17 | Can a wrong checkpoint with the same filename bypass validation | No — hash-based, not filename-based | Same check | FACT | |
| 18 | Is the parent checkpoint copied/referenced/mounted/mutated | Copied once to a disposable staging directory before any container ever sees it (`_stage_writable_checkpoint`); the pristine source is never mounted read-write in the production path | `training_session.py::_stage_writable_checkpoint` | FACT | |
| 19 | Can training overwrite the original model | Not via the production path (verified). It did happen via a preflight-smoke-test bug — found and fixed in this audit (R-001) | `launch_guard.py`/`preflight.py` before/after this audit | FACT (post-fix) | |
| 20 | Is model lineage stored in immutable run metadata | Yes — `launch_manifest.json`, `training_identity.json` | Direct file inspection | FACT | |

## C. Dataset provenance and governance

| # | Question | Answer | Evidence | Status | Severity |
|---|---|---|---|---|---|
| 1 | Exact source of every training record | Riksarkivet Swedish Lion Libre collections, enumerated by `collection`/`source_parquet_file` columns in the inventory | `swedish_dataset_inventory.py` | FACT | |
| 2 | Inventory by archive/collection/document/page/line | By collection and source file, yes; by document/page, no true identifier exists | `pilot_split.py` docstring | FACT (partial) | MEDIUM |
| 3 | Who created the transcriptions | Not documented in the repository's own artifacts read during this audit | — | UNANSWERED | MEDIUM |
| 4 | Manual / corrected-HTR / automatic / mixed | Not documented | — | UNANSWERED | MEDIUM |
| 5 | Transcription provenance per line | Not found as a per-line field in the inventory schema reviewed | — | UNANSWERED | MEDIUM |
| 6 | Confidence/review status per line | Not found | — | UNANSWERED | MEDIUM |
| 7 | What qualifies a line as valid | `valid` boolean column, computed from real, disclosed exclusion reasons (duplicate, malformed, etc.) | `swedish_dataset_inventory.py` | FACT | |
| 8 | What causes rejection | `exclusion_reasons` list, includes `duplicate_image`, `duplicate_image_transcription_pair`, and others not fully enumerated in this audit | Same | FACT (partial enumeration) | LOW |
| 9 | Total raw lines | 565,146 | Direct parquet read | FACT | |
| 10 | Valid lines | 563,933 | Direct parquet read | FACT | |
| 11 | Train/val/test assignment counts | Full-corpus train 562,123 / val 1,000 / test-reserved 810 | `sharding_summary.json`, manifest reads | FACT | |
| 12 | Excluded count | 1,810 from full-corpus shards (val+test); 1,213 invalid rows excluded from the inventory entirely (565,146-563,933) | Direct computation | FACT | |
| 13 | Why excluded | Val/test: reserved for evaluation, structurally excluded from shards. Invalid rows: duplicate/malformed per `exclusion_reasons` | `corpus_sharding.py`, `swedish_dataset_inventory.py` | FACT | |
| 14 | Duplicate images present | Yes, detected and excluded (exact hash match) | `swedish_dataset_inventory.py::duplicate_status` | FACT | |
| 15 | Duplicate transcriptions present | Detected via (image_hash, transcription) pairs | Same | FACT | |
| 16 | Near-duplicate crops present | Unknown — no perceptual/near-duplicate detection exists | — | UNANSWERED (R-010) | MEDIUM |
| 17 | Same page regions in multiple splits | Cannot be fully ruled out for 7/11 collections (line-level split granularity) | `pilot_split.py` | INFERRED risk, disclosed | MEDIUM-HIGH (R-009) |
| 18 | Augmented versions of the same line in different splits | No augmentation is used at all (`augmentation_policy="none"`) — question is moot | `TrainingConfiguration.augmentation_policy` | FACT | |
| 19 | Document-level leakage | Possible for 7/11 single-source-file collections | R-009 | INFERRED risk, disclosed | MEDIUM-HIGH |
| 20 | Writer-level leakage | Possible, same root cause (no writer identifier exists at all, for any collection) | `pilot_split.py` docstring | INFERRED risk, disclosed | MEDIUM |
| 21 | Collection-level leakage | No — collection is the split's primary grouping dimension, verified zero exact-ID overlap | `sharding_summary.json` cross-checks | FACT | |
| 22 | Temporal leakage | Not assessed — no timestamp/date field reviewed in this audit | — | UNANSWERED | LOW |
| 23 | Source-system leakage | Not assessed | — | UNANSWERED | LOW |
| 24 | Deterministic split assignment | Yes — seeded, reproducible | `pilot_split.py`, `corpus_sharding.py`, reproducibility confirmed via independent rebuild | FACT | |
| 25 | Can the exact split be regenerated from source | Yes | Same | FACT | |
| 26 | Is the split stored immutably | Yes — `pilot_split_summary.json`, `sharding_summary.json`, both write-once/refuse-to-overwrite | `corpus_sharding.py::build_full_corpus_shards`'s own "refuses to overwrite" docstring | FACT | |
| 27 | Line-level random sampling where document-level grouping would be safer | Exactly this, for 7/11 collections, disclosed | R-009 | FACT (disclosed limitation) | MEDIUM-HIGH |
| 28 | Does validation realistically represent expected use | Not independently assessed against a deployment-distribution definition (none exists) | — | UNANSWERED | MEDIUM |
| 29 | Does the test set remain untouched by training decisions | Yes, structurally — never included in any shard, never referenced by monitoring/stopping logic | `corpus_sharding.py` exclusion; `orchestrator.py`'s stopping logic only reads `val_cer` | FACT | |
| 30 | Has the test set already influenced hyperparameter selection | No evidence found that it has been evaluated at all yet | — | FACT (never evaluated, per Q6 above) | |
| 31 | Are pilot training lines included in the full training split | Yes | `corpus_sharding.py` only excludes pilot val/test, not pilot train | FACT | |
| 32 | Is that inclusion correct and documented | Yes, explicitly, by design (`cli.py::cmd_prepare`'s own comment) and matches the audit brief's own instruction not to exclude it | `cli.py` | FACT | |
| 33 | Are pilot val lines part of the fixed full-run validation set | Yes, reused unmodified | `cli.py::cmd_prepare`: `val_manifest_path = DEFAULT_PILOT_RUN_DIR / "manifests" / "val_manifest.parquet"` | FACT | |
| 34 | Is any pilot result contaminating final test evaluation | No evidence of this — the reserved test manifest (810 lines) has never been evaluated by anything found in this audit | — | FACT (no contamination found) | |
| 35-46 | Unicode/malformed sequences, normalization, whitespace/punctuation rules, abbreviations, line-breaks, illegible chars, empty/short/long labels, image-label mismatches, missing/corrupt images, dimension/channel validation | Not independently re-verified in this audit; delegated to `swedish_dataset_inventory.py`'s build-time validation, whose exact rule set was not exhaustively read line-by-line | `swedish_dataset_inventory.py` (partially reviewed) | UNANSWERED (partial) | MEDIUM |
| 47-49 | Missing/corrupt images, dimension/channel validation, crop boundaries | Same as above | Same | UNANSWERED (partial) | MEDIUM |
| 50 | Reading order preserved | Not assessed | — | UNANSWERED | LOW |
| 51 | Licenses/permissions for source documents and transcriptions recorded | Not found in this audit's scope | — | UNANSWERED | MEDIUM |
| 52 | Personally sensitive information present | Not assessed — historical archival documents may contain names/personal data; no PII review was found or performed | — | UNANSWERED | MEDIUM |
| 53 | Lawful/policy-compliant processing | Not assessed by this audit | — | UNANSWERED | MEDIUM |
| 54 | Dataset versioned | Yes, via content hash | `dataset_hash_record.json` | FACT | |
| 55 | Stable dataset hash | Yes | Same | FACT | |
| 56 | Does the hash cover file contents, paths, metadata, labels | Covers the inventory parquet file's full byte content (which includes labels/metadata as columns), not external file paths separately | `cli.py::_dataset_hash` — sha256 of the whole inventory file | FACT | |
| 57 | Can files change without changing the recorded identity | No, for the inventory file itself — a content-hash. Individual source images referenced *by* the inventory are not independently hashed per-file in this workflow | Same | FACT (for the inventory); UNANSWERED (for individual source images) | LOW |
| 58 | Storage medium reliability for a multi-day run | Local NTFS drive (D:); no network storage in the training path itself | Observed paths throughout | FACT | |
| 59 | Protection against partial copies/network interruptions | N/A — local disk, no network dependency in the training path | Same | FACT | |
| 60 | Backup/recovery procedures defined | Not defined in this repository | — | UNANSWERED | MEDIUM |

## D. Dataset representativeness

| # | Question | Answer | Status | Severity |
|---|---|---|---|---|
| 1 | Distribution across time/style/language/region/archive/type/writer/quality | Not measured in this audit; `pilot_split.py` tracks per-collection distribution but no cross-cutting analysis was performed | UNANSWERED | MEDIUM |
| 2-3 | Largest source dominating / rare sources underrepresented | Partially mitigated by `max_collection_fraction=0.25` capping any one collection's share of train+val; not independently verified against the real distribution in this audit | INFERRED (mitigated by design) | LOW |
| 4 | Val/test distribution aligned with deployment distribution | Not assessed (no deployment distribution defined) | UNANSWERED | MEDIUM |
| 5 | Subgroups too small for reliable evaluation | Not assessed | UNANSWERED | MEDIUM |
| 6-9 | Pilot representativeness, deterministic sampling, easy-line/document overrepresentation | Pilot sampling is deterministic and seeded (real); representativeness of the *sample* vs. the full corpus was not independently measured | INFERRED (sampling method sound; representativeness unmeasured) | MEDIUM |
| 10-12 | Pilot-derived thresholds transferring; which are safe defaults vs. need recalibration | Addressed in the audit report §4/§14 — explicitly labeled pilot-derived defaults, recommended recalibration at Gate 5 | FACT (design intent) + RECOMMENDATION | |
| 13-14 | Source/writer balancing | Not used (`augmentation_policy="none"`, no resampling logic found) | FACT (not used) | |
| 15-16 | Curriculum learning / long-line filtering artificially improving CER | No curriculum learning; no line-length filtering beyond validity checks found | FACT (not present) | |
| 17 | Vocabulary coverage sufficiency | Not independently assessed | UNANSWERED | LOW |

## E. Image preprocessing

| # | Question | Answer | Status | Severity |
|---|---|---|---|---|
| 1-30 | All preprocessing questions (resizing, padding, normalization, deskew, binarization, augmentation, etc.) | Preprocessing is performed entirely inside the pinned `loghi-htr` container, not by ArchiveTrust code; this audit did not read the container's internal preprocessing pipeline (`.loghi-upstream/loghi-htr/src/`'s data-loading modules) in sufficient depth to answer these individually. `augmentation_policy="none"` is confirmed at the ArchiveTrust configuration layer (Q23-30 largely moot as a result — no augmentation is configured). Image writing from source parquet is confirmed "byte-identical" (`preprocessing_version="byte_identical_from_source_parquet"`). | FACT (augmentation=none; source images byte-identical from parquet) / UNANSWERED (all internal container preprocessing details) | MEDIUM |

## F. Text preprocessing and vocabulary

| # | Question | Answer | Status | Severity |
|---|---|---|---|---|
| 1 | Exact character vocabulary | Defined by `charlist.txt`, hashed into `configuration_hash` | FACT (hashed, not enumerated in this audit) | |
| 2-18 | Character counts, coverage, normalization, CER implementation verification | Not independently re-derived or verified in this audit; delegated to the pinned container's own CER computation (`log.csv`'s `CER_metric`/`val_CER_metric` columns, parsed verbatim, never recomputed by ArchiveTrust) | UNANSWERED | MEDIUM-HIGH (CER implementation correctness is foundational and was not independently verified against known examples) |

## G. Train/validation/test split integrity

| # | Question | Answer | Evidence | Status | Severity |
|---|---|---|---|---|---|
| 1 | Split unit | File-group `(collection, source_parquet_file)` where 2+ files exist; line-level for 7/11 single-file collections | `pilot_split.py` | FACT | |
| 2 | Why appropriate | Best available proxy for document-level grouping given no true document/writer/volume ID exists | Same, disclosed | FACT (with disclosed limitation) | |
| 3 | Leakage checked at every grouping level | Checked at line-ID level (real, zero overlap); not checked at document/writer level (no identifier exists) | R-009 | FACT (partial) | MEDIUM-HIGH |
| 4 | Train/val overlap exactly zero | **Yes, verified** across all 171 shards | Direct full read of every shard's `line_id` vs. val manifest | FACT | |
| 5 | Train/test overlap exactly zero | Zero at prepare time (structural exclusion); never re-verified at launch time | R-007 | FACT (at prepare) / GAP (at launch) | HIGH |
| 6 | Val/test overlap exactly zero | Yes, verified (real preflight check `val_test_no_overlap`) | `preflight.py` | FACT | |
| 7-9 | Hash-based exact duplicate / near-duplicate / repeated-transcription detection | Exact-hash: yes. Near-duplicate: no. Repeated transcription: yes (pair-based) | R-010 | FACT (partial) | MEDIUM |
| 10 | Different crops from the same physical line split across sets | Cannot be ruled out for the same 7/11-collection reason | R-009 | INFERRED risk | MEDIUM-HIGH |
| 11 | Validation set large enough for stable CER | 1,000 lines — reasonable but no formal power analysis / CI computation found in this audit | — | UNANSWERED | MEDIUM |
| 12-13 | Confidence intervals / statistical noise | Not computed anywhere in the codebase for val CER deltas beyond the pilot's own `validation_noise_stdev` (a real, computed value) | `pilot_analysis.py` | FACT (noise floor computed) / UNANSWERED (formal CI) | MEDIUM |
| 14-15 | Test set size/diversity adequate; frozen test manifest exists | 810 lines, frozen (`test_reserved_manifest.parquet`, excluded from all splitting logic downstream) | Direct file inspection | FACT | |
| 16 | Does launch prevent test manifest access during training | Yes, structurally — never referenced by `orchestrator.py`'s training/monitoring loop | Direct code inspection | FACT | |
| 17 | Is the test set excluded from dashboards that could influence stopping | Yes — dashboards are read-only and stopping logic only ever reads `val_cer`, never any test-manifest-derived value | Direct code inspection | FACT | |
| 18 | Is evaluation code isolated from training code | No dedicated "evaluation" module for the reserved test set was found in this audit — this appears to be a genuine gap: there is no code path in this repository that evaluates the reserved test set at all yet | — | UNANSWERED | HIGH (needed before Gate 7) |
| 19 | Can accidental path globbing include val/test files | No glob-based manifest discovery exists — every manifest path is explicit, hardcoded, or passed as a parameter | Direct code inspection | FACT | |
| 20 | Are split hashes revalidated immediately before launch | Dataset hash and training-manifest hash: yes (launch guard). Val/test manifest hashes specifically: recorded in `launch_manifest.json` but not independently re-verified by the guard at launch time beyond the val-overlap re-check | `launch_guard.py` | FACT (partial) | MEDIUM |

## H. Training configuration

Covered in depth in the main audit report §5. Key answers not repeated in full table form here to
avoid duplication:

| # | Question | Answer | Status |
|---|---|---|---|
| 1-4 | Batch size 16, per-device (no gradient accumulation, `gradient_accumulation=1`), effective batch size = 16 | FACT |
| 5-8 | Mixed precision `mixed_float16`; loss-scaling/numerical-instability monitoring is internal to the pinned container, not independently verified | FACT (precision) / UNANSWERED (internal loss scaling) |
| 9-14 | Adam, LR 0.0001, hardcoded (not scaled for effective batch size), chosen ad hoc / not explicitly derived from the pilot in the code (the pilot itself likely used the same fixed value, not independently reconfirmed) | FACT / ASSUMPTION |
| 15-19 | Exponential decay per real source-code analysis (§5, R-017), no warmup (`warmup_ratio=0.0`), so "50x warmup expansion" concern does not apply | FACT (verified via pinned source) |
| 20-23 | Weight decay/bias exclusion: internal to pinned container, not independently verified; gradient clipping/threshold: not configured by ArchiveTrust, internal container default applies | UNANSWERED |
| 24-28 | Gradient norm logging: not present; NaN/Inf detection: yes, real, post-shard (§9 of the audit report); causes safe stop | FACT (NaN/Inf) / UNANSWERED (gradient norm) |
| 29-31 | Label smoothing, dropout: internal to pinned container, not independently verified | UNANSWERED |
| 32-37 | CTC loss / blank-token / alignment validation: internal to pinned container, not independently verified; skipped-sample counting: not implemented at the ArchiveTrust layer (§ Failure-mode review, scenario 5) | UNANSWERED |
| 38 | Acceptable skip count | Not defined | UNANSWERED |
| 39-46 | Shuffling: seeded per-shard (`epoch_seed = random_seed + cumulative_epoch + 1`), deterministic given a fixed seed (see R-006 for the resume caveat); shuffling is per-shard, not global across the whole 562,123-line corpus in one shuffle buffer; shard boundaries themselves are NOT randomized (deterministic shard order within a lap, real-verified) | FACT (with R-006 caveat) |
| 47-49 | "Epoch" under sharding = one shard, not one full corpus pass; steps-per-shard verified real (625 = ceil(9999/16)) | FACT |
| 50-51 | Max-epoch policy explicit (171-shard ceiling); staged training supported via bounded `--hours` invocations | FACT |
| 52-57 | Early-stopping semantics, min_delta, patience — all derived from real pilot evidence, explicitly labeled provisional (§4 of the audit report) | FACT |
| 58-61 | Wall-clock cap: real, cannot interrupt mid-shard by construction (verified: `epoch_would_not_fit` estimator refuses to start a shard that won't fit) | FACT |
| 62-63 | Best-checkpoint restoration for final export: tracked separately, but the CLI does not automate "export the best checkpoint" — an operator/export-time decision | FACT (tracked) / UNANSWERED (automated export) |
| 64-65 | Configuration immutable after launch (via `configuration_hash` enforcement); resolved configuration persisted, not just a template | FACT |

## I. Pilot interpretation

Covered in full in the main audit report §4. All 29 questions are answered there with real,
measured values (epoch count, CER trajectory, plateau evidence, timing, throughput) except:
"How many optimizer updates will occur per full epoch" (steps_per_shard=625, real, computed) and
"is comparing epoch numbers between pilot and full run misleading" — **yes, explicitly**, and this
is exactly why the workflow defines "epoch" as "one shard" rather than reusing the pilot's own
epoch semantics (FACT, by design).

## J. Sharding and corpus traversal

Covered in full in the main audit report §3/§9 (shard validation). Summary of key answers:

| # | Question | Answer | Status |
|---|---|---|---|
| 1-4 | Sharding needed because `--steps_per_epoch` is upstream-documented broken in the pinned commit; 171 shards planned, 9999 lines/shard, final shard of each lap smaller (2,179 lines, real, verified) | FACT |
| 5-7 | Every intended line present exactly once per lap (verified, zero duplicates within any lap, zero records lost — aggregate line count per lap = usable_line_count exactly) | FACT |
| 8-11 | Shard order deterministic within a lap; re-shuffled with a re-derived seed at each lap boundary; reproducible (independently confirmed) | FACT |
| 12-13 | Related lines can cluster within a shard (no anti-clustering logic); whether this biases early training is unmeasured | ASSUMPTION / UNANSWERED |
| 14-17 | Validation runs after every shard, not after a full corpus pass — "epoch" terminology is used to mean "shard," disclosed and consistent throughout the codebase's own docstrings, not silently inconsistent | FACT |
| 18 | Can LR scheduling treat each shard as a full epoch — addressed in depth, R-017: yes, and this is architecturally correct/intentional given `decay_steps` resolves to one shard's own step count | FACT (verified via pinned source) |
| 19-23 | Optimizer state continuity across shards: asserted from source + small proof, not empirically proven at scale (R-004/R-005); model reloaded between shards (yes, via checkpoint chaining); Docker restarted between shards (yes, `--rm` per invocation); fixed startup overhead paid per shard (yes, real, unquantified in this audit) | FACT (mechanism) / UNANSWERED (scale-proven continuity) |
| 24-26 | Resume identifies exact next shard via `cumulative_epoch` (real, verified); cannot skip shards (sequential); could theoretically repeat a shard's data if interrupted mid-shard, since a shard isn't "consumed" until fully completed — acceptable, not harmful (occasional shard repetition is not the same as train/val leakage) | FACT |
| 27-30 | Shard manifests hashed (`line_id_set_hash`); tied to the training configuration hash; shard files could change post-preparation without detection (R-007); no automatic pre-launch integrity recheck beyond the self-referential summary-hash comparison | FACT (partial) / GAP (R-007) |
| 31-32 | Temporary vs. authoritative shard files: shards ARE the authoretative training manifest (not temporary); cleanup is not automatic/needed since shards are meant to persist for the run's lifetime | FACT |
| 33 | Sufficient disk space for shards+checkpoints+logs+temp+recovery margin | **No** — this is R-018, the audit's primary blocking finding | GAP | CRITICAL |
| 34-35 | Shard generation deterministic (yes); does not modify source data (yes, read-only against the inventory) | FACT |

## K. Reproducibility

| # | Question | Answer | Status | Severity |
|---|---|---|---|---|
| 1-4 | Code commit recorded (yes); repo dirty state recorded (yes); uncommitted changes captured as a boolean flag only, not a diff/archive | FACT (partial — dirty flag, not diff content) | LOW |
| 5-6 | Container image pinned by digest (yes); Python package versions locked | FACT (image) / UNANSWERED (Python env — no lockfile hash was checked in this audit) | LOW |
| 7-11 | CUDA/cuDNN/GPU model/driver/OS versions recorded | Not recorded anywhere in this workflow's own artifacts (fixed inside the pinned image, but the host driver stack outside the container is unrecorded) | UNANSWERED | LOW-MEDIUM |
| 12-13 | Random seeds explicit | Partially — `random_seed` is explicit but not cross-validated on resume (R-006); no explicit control over NumPy/framework-level seeds beyond what the pinned container does with `--seed` | FACT (partial) | HIGH (R-006) |
| 14-16 | Deterministic framework options / nondeterministic operations / documentation of nondeterminism | Not addressed anywhere in this codebase (cuDNN nondeterminism on GPU is a known TF/CUDA reality, not disclaimed or controlled for here) | UNANSWERED | LOW-MEDIUM |
| 17-19 | Can the exact run be repeated / exact CER expected / tolerance | Shard plan and configuration are reproducible; exact numerical CER reproducibility is not claimed or tested (GPU nondeterminism likely precludes bit-exact reproduction even with matching seeds) | INFERRED | LOW |
| 20-23 | Manifests immutable (yes); hashes persisted (yes); resolved configuration persisted, not just a template (yes) | FACT | |
| 24-25 | Environment variables persisted safely / secrets excluded | No environment-variable dump found anywhere in persisted run metadata — secrets exposure risk is effectively zero since none are used | FACT | |
| 26-28 | Timestamps timezone-explicit (yes, UTC `Z`-suffixed throughout); schema versioning present for telemetry (a `TELEMETRY_SCHEMA_VERSION` constant exists, `"telemetry-standard-v1"`) and manifests (implicit via pydantic model shape, no explicit version field found on most manifest JSON files) | FACT (telemetry) / PARTIAL (manifests) | LOW |
| 29 | Is the audit result itself stored with the run | Not automatically — this audit's artifacts live under `docs/audits/`, not inside the run directory itself | FACT (by choice, not a defect) | |

## L. Checkpointing and resume

Covered in depth in the main audit report §6 and the risk register (R-004, R-005, R-006, R-008,
R-011, R-013). Key summary answers not otherwise stated: checkpoint writing is atomic at the index
level (verified) but the underlying `.keras` file write itself is the pinned container's own
responsibility, not independently made atomic by ArchiveTrust (UNANSWERED whether the container's
own save is atomic — HIGH, ties directly into R-018's disk-full scenario). Latest and best are
tracked as distinct index entries (verified, cannot silently overwrite each other). No checkpoint
retention/deletion policy exists (R-018) — so "how many checkpoints retained" is **unbounded**, and
"is available disk sufficient under worst-case retention" is **no** (R-018).

## M. Validation and evaluation correctness

| # | Question | Answer | Status | Severity |
|---|---|---|---|---|
| 1-11 | Validation timing, eval-mode, gradient-disabled, augmentation-disabled, determinism, full-set evaluation, sampling, skip reporting, CER aggregation method, documentation | All delegated to the pinned container's own `--do_validate` behavior; not independently re-verified in this audit beyond confirming the flag is always passed | UNANSWERED (internal container behavior) | MEDIUM |
| 12-13 | Empty-reference handling, normalization consistency with CER calc | Not independently verified | UNANSWERED | MEDIUM |
| 14 | Confidence intervals available | No | FACT (absent) | LOW |
| 15 | Subgroup metrics calculated | No | FACT (absent) | MEDIUM |
| 16-19 | Best-checkpoint selection metric correctness (verified: `val_cer < state.best_val_cer`, strict improvement, real code); min_delta application (via `meaningful_improvement_threshold`, real); lower-CER-is-better (verified, correct direction in code); NaN CER treated as improvement | Verified: `val_cer_improved = result.val_cer is not None and (state.best_val_cer is None or result.val_cer < state.best_val_cer)` — a `None` or `NaN` val_cer would NOT satisfy `result.val_cer < state.best_val_cer` in Python's float comparison semantics for NaN (NaN comparisons are always False), so NaN cannot be silently treated as an improvement; `None` is explicitly excluded by the `is not None` check | FACT | |
| 20 | Can a missing validation result advance patience | No — `orchestrator.py` explicitly stops with `missing_validation_result` rather than silently continuing | FACT | |
| 21 | Is validation failure a hard stop | Yes | FACT | |
| 22 | Does the dashboard distinguish "not evaluated" from zero | Yes, for the reserved test set specifically (`test_cer: str | None = None` with an explicit "always None this phase" docstring) | FACT | |
| 23-30 | Final test evaluation timing/independence, baseline comparison methodology, paired comparisons, statistical significance, qualitative review, regression set, human-review measurement | **No code path evaluates the reserved test set at all** — this entire area is UNANSWERED, a real gap for Gate 7 | UNANSWERED | HIGH |

## N. Monitoring and observability

Covered in depth in the audit report §7. All 40 questions are substantively answered there
(deterministic 11-state classifier, real telemetry, real staleness detection, GPU/VRAM/temp/CPU/RAM/
disk all recorded, throughput not explicitly separated from raw duration, ETA/ETA-recalculation not
implemented as a distinct feature, ETA/estimates clearly labeled as pilot-derived where they appear).
Remaining explicit gaps: I/O throughput and data-loader-bottleneck visibility (UNANSWERED — not
instrumented separately from overall duration); checkpoint-duration and validation-duration are not
recorded as separate timing fields from overall shard duration (UNANSWERED, LOW).

## O. Hardware and infrastructure

Covered in depth in the audit report §8. Primary open item: **disk capacity is insufficient for the
full prepared plan** (R-018, CRITICAL). GPU sharing/thermal/power/host-sleep/reboot-prevention/UPS
questions are UNANSWERED (host/OS policy, outside this repository's own configuration surface).
Docker restart policy: containers run with `--rm`, no restart policy configured (correct — an
auto-restarting training container would be actively dangerous, matching the audit brief's own
concern about "could that cause an unsafe duplicate run").

## P. Runtime and duration planning

Covered in depth in the audit report §4/§9/§14, including the specific, quantified disk-cost
extrapolation this audit performed. Timing estimates for 1/2/3/5/10/20 epochs (shards) are directly
computable from the real measured per-shard duration (482.3s) but were not pre-tabulated as a formal
table in this audit beyond the per-lap (57-shard) and full-plan (171-shard) figures already given.
20 pilot epochs are **not** meaningfully comparable to 20 full-corpus shards in absolute progress
through the corpus (20 pilot epochs = 20 passes over the *same* 9,999 lines; 20 full-corpus shards =
20/57 ≈ 35% of *one* pass over the full 562,123-line corpus) — a real, important distinction, FACT,
already reflected in the workflow's own "epoch = shard" terminology discipline.

## Q. Experiment design

| # | Question | Answer | Status | Severity |
|---|---|---|---|---|
| 1-24 | Controlled experiment vs. single run; frozen baseline evaluated; ablations; experiment registry; hypothesis; preregistration; test-set peeking prevention | This workflow is designed and operated as a **single production retraining run**, not a controlled experiment — no ablation plan, no experiment registry, no baseline-on-the-same-test-set evaluation was found anywhere in the repository. Test-set peeking is structurally prevented (no code path touches it). If a defensible research comparison or publication is intended, this entire section is a real, unaddressed gap. | UNANSWERED (by design choice, not oversight — this appears to be intentionally a production run, not a study) | LOW for a pure production use case / HIGH if publication or rigorous before/after comparison is intended |

## R. Production and downstream integration

| # | Question | Answer | Status | Severity |
|---|---|---|---|---|
| 1-24 | Export format, inference runtime, vocabulary packaging, versioning/rollback, model card, inference speed/memory, segmentation-vs-recognition error attribution, confidence calibration, post-processing | No export/deployment pipeline for the full-corpus output was found in this repository as of this audit — this entire section is UNANSWERED. The original Loghi and pilot models are both retained on disk (verified, never deleted by any code path in `full_run/`). | UNANSWERED | MEDIUM (not blocking for the training run itself; blocking for eventual deployment) |

## S. Security and supply chain

Covered in depth in the audit report §11. All "yes" answers are FACT, directly verified. No
UNANSWERED items in this section beyond "audit trail of who launched a run" (informational gap for
a single-operator context).

## T. Failure-mode analysis

Covered in `failure_recovery_runbook.md` — all 40 scenarios addressed, with explicit **(verified)**
vs. **(unverified)** labeling per scenario. Scenario 9 (disk nearly/completely full) is the audit's
primary, quantified finding (R-018).

## U. Human decision gates

Covered in full in the audit report §13, with required evidence, pass/fail criteria, and the
explicit confirmation that no gate advances automatically.
