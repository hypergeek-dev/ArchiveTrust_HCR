# Experiment 0 vs 1 vs 2 comparison

Generated 2026-08-04T09:43:17Z.

## Headline

| Experiment | Initialization | Architecture | Trainer lifecycle | Epochs | CER | True WER |
|---|---|---|---|---:|---:|---:|
| 0 | Generic pretrained Loghi | Generic checkpoint architecture (new10) | 57 resets | 1 | 0.1728 | 0.4945 |
| 1 | Generic pretrained Loghi | Generic checkpoint architecture (new10) | Continuous | 1 | 0.1747 | 0.4956 |
| 2 | Random initialization | Recommended VGSL | Continuous | 1 | 0.1918 | 0.5470 |

## Runtime, throughput, and scale

| | Experiment 0 | Experiment 1 | Experiment 2 |
|---|---|---|---|
| Runtime (s) | 25093 | 16543 | 15460 |
| Optimizer steps | n/a (predates instrumentation) | 35099 | 35111 |
| Throughput (lines/s) | n/a (Experiment 0 predates the optimizer_trace instrumentation) | 33.9 | 36.3 |
| Parameter count | (Exp0 shares Exp1's architecture) | 31,033,929 | 30,694,654 |
| Peak GPU memory (MB) | 7400.0 | 7534.0 | 7835.0 |
| Checkpoint size (bytes) | 372594085 | 372594089 | 368525198 |

## Training trajectory

- Experiment 1 train CER: 0.33475178480148315 -> 0.1949 (fine-tune, started already competent)
- Experiment 2 train CER: 1.030945897102356 -> 0.2749 (scratch, started near-random ~1.0)
- Experiment 2's train CER was still declining at the last logged step (no plateau observed) -- see optimizer_trace.csv.

## Per-collection CER (held-out)

| collection | Exp 0 | Exp 1 | Exp 2 |
|---|---|---|---|
| alvsborgs_losen | 0.2760 | 0.2860 | 0.3489 |
| jonkopings_radhusratt_och_magistrat | 0.2658 | 0.2593 | 0.2956 |
| gota_hovratt | 0.2282 | 0.2399 | 0.2384 |
| trolldomskommissionen | 0.2226 | 0.2205 | 0.2193 |
| bergmastaren_i_nora_htr | 0.2195 | 0.2329 | 0.2311 |
| krigshovrattens_dombocker | 0.1646 | 0.1666 | 0.1724 |
| goteborgs_poliskammare_fore_1900 | 0.1337 | 0.1342 | 0.1497 |
| carl_fredrik_pahlmans_resejournaler | 0.1186 | 0.1173 | 0.1540 |
| svea_hovratt | 0.1172 | 0.1153 | 0.1184 |
| frihetstidens_utskottshandlingar | 0.0995 | 0.0987 | 0.1244 |
| bergskollegium_relationer_och_skrivelser | 0.0564 | 0.0571 | 0.0736 |

## Caveat

Experiment 0/1 vs Experiment 2 is NOT a single-variable comparison: both initialization (random vs pretrained) AND architecture (recommended VGSL vs the generic checkpoint's new10) differ simultaneously. Describe this comparison as *native recommended architecture trained from scratch versus generic-checkpoint fine-tuning*, not as an isolated ablation of either variable alone.