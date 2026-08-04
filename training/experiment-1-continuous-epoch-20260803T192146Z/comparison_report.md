# Experiment 0 (shard-based) vs. Experiment 1 (single continuous epoch)

- Experiment 0 directory: `D:\ArchiveTrust_HCR\training\full-corpus-20260802T044250Z`
- Experiment 1 directory: `D:\ArchiveTrust_HCR\training\experiment-1-continuous-epoch-20260803T192146Z`

## Method

| | Experiment 0 | Experiment 1 |
|---|---|---|
| Learning-rate policy | `per_shard_fresh_adam_base_0.0001_intra_shard_decay_0.99_no_cross_shard_continuity` | `single_continuous_epoch_fresh_adam_base_0.0001_continuous_decay_0.99_across_full_corpus` |
| Shard count | 171 | 1 |
| Batch size | 16 | 16 |
| Optimizer | adam | adam |
| Learning rate | 0.0001 | 0.0001 |
| Base checkpoint hash | `0da2c00ab2b12b23...` | `0da2c00ab2b12b23...` |

## Run outcome

| | Experiment 0 | Experiment 1 |
|---|---|---|
| status | stopped | completed |
| epochs_completed | 1 | 1 |
| stop_reason | epoch_boundary_reached | target_epochs_reached |
| best_metrics.val_cer (in-training) | 0.1742 | 0.1755 |

## Held-out evaluation (`lap_evaluation.json`, best_val checkpoint, same code/metric implementation)

| | Experiment 0 | Experiment 1 |
|---|---|---|
| corpus_cer | 0.1728 | 0.1747 |
| corpus_wer | 0.4945 | 0.4956 |
| line_error_rate | 0.8950 | 0.8960 |
| mean_per_line_cer | 0.1736 | 0.1764 |
| scored_line_count | 1000 | 1000 |
| worst_collection | alvsborgs_losen | alvsborgs_losen |
| best_collection | bergskollegium_relationer_och_skrivelser | bergskollegium_relationer_och_skrivelser |

### Per-collection CER (Experiment 0 vs Experiment 1)

| collection | Exp 0 corpus_cer | Exp 1 corpus_cer |
|---|---|---|
| alvsborgs_losen | 0.2760 | 0.2860 |
| jonkopings_radhusratt_och_magistrat | 0.2658 | 0.2593 |
| gota_hovratt | 0.2282 | 0.2399 |
| trolldomskommissionen | 0.2226 | 0.2205 |
| bergmastaren_i_nora_htr | 0.2195 | 0.2329 |
| krigshovrattens_dombocker | 0.1646 | 0.1666 |
| goteborgs_poliskammare_fore_1900 | 0.1337 | 0.1342 |
| carl_fredrik_pahlmans_resejournaler | 0.1186 | 0.1173 |
| svea_hovratt | 0.1172 | 0.1153 |
| frihetstidens_utskottshandlingar | 0.0995 | 0.0987 |
| bergskollegium_relationer_och_skrivelser | 0.0564 | 0.0571 |

## Experiment 1 optimizer/LR continuity proof

- Trace file: `D:\ArchiveTrust_HCR\training\experiment-1-continuous-epoch-20260803T192146Z\run-state\epoch_output\epoch_1\optimizer_trace.csv` (72 rows)
- optimizer.iterations: 0 -> 35099
- monotonically non-decreasing across the whole epoch: **True**
- learning_rate: 0.0001 -> 0.0001

## Experiment 1 GPU/system telemetry

- samples: 3264
- utilization_pct: mean 27.4000, max 100.0000
- memory_used_mb: mean 6916.1000, max 7534.0000
- max temperature_c: 66.0000

## Caveat

This isolates trainer continuity (one variable) -- corpus, validation set, architecture, optimizer, and learning rate are held identical between the two runs by construction. It is not a comparison against Swedish Lion or any other architecture.