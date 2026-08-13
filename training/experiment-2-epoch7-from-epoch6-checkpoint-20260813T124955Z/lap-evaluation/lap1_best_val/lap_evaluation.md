# End-of-lap evaluation -- lap 1

- run: `loghi_training_run_e45c88d9c7a94f149cd53438751bae1b`
- checkpoint: `C:\Users\denni\OneDrive\Desktop\ArchiveTrust_HCR\training\experiment-2-epoch7-from-epoch6-checkpoint-20260813T124955Z\run-state\epoch_output\epoch_1\recommended\best_val`
- checkpoint sha256: `7a11df10b943330fe93c7fbf2ddb0584a75539ecb6b1da06dcaa3a5c39f17572`
- position: 1 shards = 1 full corpus lap(s) of 1 shards each
- validation lines: 1000 scored of 1000

## Overall

| metric | value |
| --- | --- |
| corpus CER | **0.0998** |
| corpus WER (true word error rate) | 0.3273 |
| line error rate (what the container calls "WER") | 0.7490 |
| mean per-line CER | 0.1061 |
| total edits | 3,148 |
| total reference chars | 31,554 |

Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure comparable with the container's own `CERMetric`. The per-line mean is shown beside it because they diverge when line lengths vary.

## Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| jonkopings_radhusratt_och_magistrat | 0.1645 | 0.5245 | 91 | 3,386 | 0.840 |
| alvsborgs_losen | 0.1614 | 0.3355 | 91 | 1,797 | 0.850 |
| bergmastaren_i_nora_htr | 0.1459 | 0.4726 | 91 | 2,756 | 0.851 |
| gota_hovratt | 0.1326 | 0.4103 | 91 | 3,243 | 0.870 |
| trolldomskommissionen | 0.1245 | 0.3860 | 90 | 3,302 | 0.876 |
| goteborgs_poliskammare_fore_1900 | 0.0777 | 0.2604 | 91 | 2,445 | 0.930 |
| krigshovrattens_dombocker | 0.0775 | 0.2834 | 91 | 2,917 | 0.915 |
| carl_fredrik_pahlmans_resejournaler | 0.0735 | 0.2792 | 91 | 3,973 | 0.931 |
| svea_hovratt | 0.0548 | 0.2279 | 91 | 2,593 | 0.948 |
| frihetstidens_utskottshandlingar | 0.0510 | 0.2049 | 91 | 2,412 | 0.928 |
| bergskollegium_relationer_och_skrivelser | 0.0311 | 0.1325 | 91 | 2,730 | 0.974 |

## Confidence calibration

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 - 0.20 | 8 | 0.6988 | 0.6069 |
| 0.2 - 0.40 | 8 | 0.5606 | 0.4950 |
| 0.4 - 0.60 | 22 | 0.3620 | 0.3439 |
| 0.6 - 0.80 | 85 | 0.2607 | 0.2444 |
| 0.8 - 1.01 | 877 | 0.0752 | 0.0536 |

### Rejection coverage

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1061 |
| 0.50 | 97.5% | 975 | 0.0944 |
| 0.70 | 94.3% | 943 | 0.0860 |
| 0.80 | 87.7% | 877 | 0.0752 |
| 0.90 | 68.4% | 684 | 0.0527 |
| 0.95 | 46.0% | 460 | 0.0348 |

## Inspection sample

Stratified: the same number of lines from every collection, selected by a stable hash seeded on the checkpoint identity, so this sample is reproducible and comparable across laps.

**alvsborgs_losen** -- CER 0.000, confidence 0.953

- truth: `Koor — 2 mk`
- pred:  `Koor — 2 mk`

**alvsborgs_losen** -- CER 0.000, confidence 0.987

- truth: `Får — 2`
- pred:  `Får — 2`

**alvsborgs_losen** -- CER 0.042, confidence 0.934

- truth: `Koor — 4. Stutar och Quigor om 4 år — 2. om 2 år`
- pred:  `Koor — 4. Stutor och auigor om 4 år — 2. om 2 år`

**bergmastaren_i_nora_htr** -- CER 0.078, confidence 0.898

- truth: `blifwit skyldig den tijden han wijd Qwarnbacka Bruk`
- pred:  `blifwit skyldig den tyjden han wijd 1 Swarnbacka Bruk`

**bergmastaren_i_nora_htr** -- CER 0.457, confidence 0.533

- truth: `För B: T: R: framträdde malmletaren`
- pred:  `Fi 1mt. R. framrädde qaten latare`

**bergmastaren_i_nora_htr** -- CER 0.200, confidence 0.771

- truth: `13. 9`
- pred:  `13.9`

**bergskollegium_relationer_och_skrivelser** -- CER 0.000, confidence 0.982

- truth: `af de till Rom förde aquaducter.`
- pred:  `af de till Rom förde aquaducter.`

**bergskollegium_relationer_och_skrivelser** -- CER 0.030, confidence 0.987

- truth: `Alla desse orter äro malmslag och`
- pred:  `Alla desse orter äro Malmslag och`

**bergskollegium_relationer_och_skrivelser** -- CER 0.029, confidence 0.969

- truth: `at igenombryta och sällan felar att`
- pred:  `at igenombryta och sällan folar att`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.385, confidence 0.845

- truth: `De 15de Mars.`
- pred:  `Den 15te Mon.`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.023, confidence 0.967

- truth: `kände det såsom varande Cousin till honom så`
- pred:  `kände det såsom varande Cousen till honom så`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.049, confidence 0.934

- truth: `för vår Prest min skrämsel och upptäckten`
- pred:  `för vår Prest min skrämsel och dyptäckten`

**frihetstidens_utskottshandlingar** -- CER 0.038, confidence 0.975

- truth: `hushållningen och sparsam¬`
- pred:  `hushållningen och spersam¬`

**frihetstidens_utskottshandlingar** -- CER 0.000, confidence 0.998

- truth: `anmärkning, ändring och til¬`
- pred:  `anmärkning, ändring och til¬`

**frihetstidens_utskottshandlingar** -- CER 0.000, confidence 0.841

- truth: `är ett`
- pred:  `är ett`

**gota_hovratt** -- CER 0.029, confidence 0.958

- truth: `kyrkioplikt, och afbedia sitt brott`
- pred:  `kyrckioplikt, och afbedia sitt brott`

**gota_hovratt** -- CER 0.089, confidence 0.885

- truth: `man doch icke hafwer någon expresie lagh con¬`
- pred:  `man doch icke hafwer någon expreste lass. con¬`

**gota_hovratt** -- CER 0.000, confidence 0.994

- truth: `Biörnsdotter, som han dock påstår wara`
- pred:  `Biörnsdotter, som han dock påstår wara`

**goteborgs_poliskammare_fore_1900** -- CER 0.129, confidence 0.888

- truth: `Härmed meddelas att förre extra`
- pred:  `Härmed meddelas att förre vt`

**goteborgs_poliskammare_fore_1900** -- CER 0.143, confidence 0.930

- truth: `Per Cederborg.`
- pred:  `r Cederborg.`

**goteborgs_poliskammare_fore_1900** -- CER 0.000, confidence 0.982

- truth: `denne`
- pred:  `denne`

**jonkopings_radhusratt_och_magistrat** -- CER 0.348, confidence 0.434

- truth: `Unnder wårtth Secretth.`
- pred:  `Bnder ubrth Seret.`

**jonkopings_radhusratt_och_magistrat** -- CER 0.050, confidence 0.878

- truth: `man michel siffrison`
- pred:  `man miches siffrison`

**jonkopings_radhusratt_och_magistrat** -- CER 0.160, confidence 0.943

- truth: `och bod dem vara welkomne`
- pred:  `och bad dem wara welkoma`

**krigshovrattens_dombocker** -- CER 0.171, confidence 0.850

- truth: `Miner Ch: Paul. Sierg. Blixtenfelt.`
- pred:  `Miner Cl: Sacil. Serg. Blixterfelt.`

**krigshovrattens_dombocker** -- CER 0.023, confidence 0.968

- truth: `dröyde med andra Salvan till des Sången och`
- pred:  `dröjde med andra Salvan till des Sången och`

**krigshovrattens_dombocker** -- CER 0.065, confidence 0.965

- truth: `des lämnades företräde åt henne`
- pred:  `des lämnades företorde åt henne`

**svea_hovratt** -- CER 0.044, confidence 0.941

- truth: `ses och Güldenhoffs resolutioner, af then 6te`
- pred:  `ses och Guldenhofs resolutioner, af then 6te`

**svea_hovratt** -- CER 0.000, confidence 0.998

- truth: `men Zetterman och Schmidt un¬`
- pred:  `men Zetterman och Schmidt un¬`

**svea_hovratt** -- CER 0.083, confidence 0.853

- truth: `Then 4 Julii`
- pred:  `Ten 4 Julii`

**trolldomskommissionen** -- CER 0.000, confidence 0.987

- truth: `Stina Debois tillstånd werkeligen sig så`
- pred:  `Stina Debois tillstånd werkeligen sig så`

**trolldomskommissionen** -- CER 0.035, confidence 0.932

- truth: `röyer sin harm emoot menniskiors barn. Men the som i then`
- pred:  `röger sin harm emoot menniskiors barn: Men the som i then`

**trolldomskommissionen** -- CER 0.333, confidence 0.871

- truth: `genast bartguff`
- pred:  `genast wortgått`

## Comparison with prior reference points

```json
{
  "pilot_best_val_cer": 0.1691,
  "pilot_best_val_cer_note": "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, shown for orientation rather than as a like-for-like target.",
  "pilot_single_pass_cer": 0.2481,
  "pilot_single_pass_cer_note": "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps to one full-corpus shard, not to a lap.",
  "checkpoint_kind_evaluated": "best_val",
  "checkpoint_recorded_at_shard": 1,
  "run_best_val_cer_during_lap": 0.10076732933521271,
  "final_shard_metrics": {
    "train_cer": 0.05623653903603554,
    "val_cer": 0.10076732933521271,
    "train_wer": 0.6122164130210876,
    "val_wer": 0.7549999952316284,
    "train_loss": 6.820504188537598,
    "val_loss": 12.660988807678223
  },
  "final_shard_metrics_note": "Train and validation CER/WER/loss as the container itself reported them at the end of the lap's final shard. The train/val gap here is the overfitting signal; the corpus CER above is an independent recomputation from raw predictions.",
  "in_training_val_cer_note": "The in-training figures above come from the container's own validation at the end of each shard, on this same 1,000-line set. They are the right thing to compare this pass against; the pilot numbers are context, not a target.",
  "inference_seconds": 44.4
}
```
