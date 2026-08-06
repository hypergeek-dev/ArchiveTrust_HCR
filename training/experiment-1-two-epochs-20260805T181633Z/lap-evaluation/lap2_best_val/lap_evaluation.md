# End-of-lap evaluation -- lap 2

- run: `loghi_training_run_0d0e1322897948d6b3c0fe7682bd9cf9`
- checkpoint: `D:\ArchiveTrust_HCR\training\experiment-1-two-epochs-20260805T181633Z\run-state\epoch_output\epoch_1\model_new10\best_val`
- checkpoint sha256: `cee57f2e29653c13dd5060e4a4e7ba08138f4f3c038bc0811b7dd4549a72755a`
- position: 1 shards = 2 full corpus lap(s) of 1 shards each
- validation lines: 1000 scored of 1000

## Overall

| metric | value |
| --- | --- |
| corpus CER | **0.1522** |
| corpus WER (true word error rate) | 0.4458 |
| line error rate (what the container calls "WER") | 0.8640 |
| mean per-line CER | 0.1518 |
| total edits | 4,801 |
| total reference chars | 31,554 |

Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure comparable with the container's own `CERMetric`. The per-line mean is shown beside it because they diverge when line lengths vary.

## Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| jonkopings_radhusratt_och_magistrat | 0.2434 | 0.6751 | 91 | 3,386 | 0.761 |
| alvsborgs_losen | 0.2398 | 0.4692 | 91 | 1,797 | 0.770 |
| gota_hovratt | 0.2063 | 0.5560 | 91 | 3,243 | 0.804 |
| bergmastaren_i_nora_htr | 0.1948 | 0.5821 | 91 | 2,756 | 0.796 |
| trolldomskommissionen | 0.1938 | 0.5175 | 90 | 3,302 | 0.816 |
| krigshovrattens_dombocker | 0.1443 | 0.4451 | 91 | 2,917 | 0.854 |
| goteborgs_poliskammare_fore_1900 | 0.1170 | 0.3612 | 91 | 2,445 | 0.901 |
| carl_fredrik_pahlmans_resejournaler | 0.1050 | 0.3699 | 91 | 3,973 | 0.908 |
| svea_hovratt | 0.0991 | 0.3431 | 91 | 2,593 | 0.910 |
| frihetstidens_utskottshandlingar | 0.0829 | 0.3122 | 91 | 2,412 | 0.900 |
| bergskollegium_relationer_och_skrivelser | 0.0436 | 0.1801 | 91 | 2,730 | 0.959 |

## Confidence calibration

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 - 0.20 | 8 | 0.6876 | 0.6113 |
| 0.2 - 0.40 | 15 | 0.5077 | 0.5000 |
| 0.4 - 0.60 | 36 | 0.4524 | 0.4224 |
| 0.6 - 0.80 | 175 | 0.2722 | 0.2639 |
| 0.8 - 1.01 | 766 | 0.0977 | 0.0800 |

### Rejection coverage

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1518 |
| 0.50 | 96.6% | 966 | 0.1377 |
| 0.70 | 89.2% | 892 | 0.1196 |
| 0.80 | 76.6% | 766 | 0.0977 |
| 0.90 | 49.0% | 490 | 0.0679 |
| 0.95 | 24.8% | 248 | 0.0441 |

## Inspection sample

Stratified: the same number of lines from every collection, selected by a stable hash seeded on the checkpoint identity, so this sample is reproducible and comparable across laps.

**alvsborgs_losen** -- CER 0.182, confidence 0.906

- truth: `Koor — 5 mk`
- pred:  `Koor — 3 mkr`

**alvsborgs_losen** -- CER 0.231, confidence 0.858

- truth: `Kopp — 2 tb :`
- pred:  `Kopp — 2 ℔b.`

**alvsborgs_losen** -- CER 0.200, confidence 0.797

- truth: `Khop — 4 ℔`
- pred:  `Klop — 4 ℔b`

**bergmastaren_i_nora_htr** -- CER 0.364, confidence 0.680

- truth: `10 Dr 28 /:`
- pred:  `10 D. 2/.`

**bergmastaren_i_nora_htr** -- CER 0.310, confidence 0.723

- truth: `3. Framtedde kåhlmätaren Olof Perßon bewis`
- pred:  `3: Frantidde Kapmåtaren Olf Perßon, beiens`

**bergmastaren_i_nora_htr** -- CER 0.071, confidence 0.914

- truth: `giort deras mästar¬stycken, samt undergått`
- pred:  `giort deras mästar: stycken, samt undergålt`

**bergskollegium_relationer_och_skrivelser** -- CER 0.000, confidence 0.990

- truth: `de ungerske Bergwärken att`
- pred:  `de ungerske Bergwärken att`

**bergskollegium_relationer_och_skrivelser** -- CER 0.103, confidence 0.911

- truth: `Thut 5816 Cent:r darin 181 qv:r 12 loth`
- pred:  `Thet 5816 Cent:r därin 181 mvr 12 loth`

**bergskollegium_relationer_och_skrivelser** -- CER 0.000, confidence 0.956

- truth: `Detta warder om lördagen i 2:ne`
- pred:  `Detta warder om lördagen i 2:ne`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.000, confidence 0.901

- truth: `Sådane äro`
- pred:  `Sådane äro`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.115, confidence 0.874

- truth: `man är vittne till bära alla stämpeln af lättsinnig¬`
- pred:  `mnn är vittne till tära alla stämpeln af lättsening`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.167, confidence 0.873

- truth: `domar och prydnader. Dessutom kan man väl säga att det`
- pred:  `domar och sryd neder. Desutom han man nät sagd att det`

**frihetstidens_utskottshandlingar** -- CER 0.111, confidence 0.930

- truth: `uti 1 arts 4 §. ålagde äro,`
- pred:  `uti 1 art:o 4 §. alagde äro,`

**frihetstidens_utskottshandlingar** -- CER 0.103, confidence 0.936

- truth: `skior och Kreatyr äro anförde`
- pred:  `skior och kreatuer äro anförde`

**frihetstidens_utskottshandlingar** -- CER 0.097, confidence 0.892

- truth: `ingen Censor Publicus tillstän¬`
- pred:  `ingen Cenfor Publicus tillstår¬`

**gota_hovratt** -- CER 0.026, confidence 0.952

- truth: `Nilßon och Botwid Larßon til döden, Men`
- pred:  `Nilßon och Botwid Larßon til döden, Me-`

**gota_hovratt** -- CER 0.415, confidence 0.750

- truth: `dhet sielf giordt och således jure Struto`
- pred:  `dhet saktegwore, och således wrl skenka`

**gota_hovratt** -- CER 0.412, confidence 0.658

- truth: `böterne dhen 19. 20 Octorbti 1687.`
- pred:  `botene Gen ut vctoori 1687.`

**goteborgs_poliskammare_fore_1900** -- CER 0.167, confidence 0.972

- truth: `No 37.`
- pred:  `No 37`

**goteborgs_poliskammare_fore_1900** -- CER 0.171, confidence 0.904

- truth: `å Jonas Johannesson öfverensstämmer`
- pred:  `i Johas Johannesson öberemstämmer`

**goteborgs_poliskammare_fore_1900** -- CER 0.172, confidence 0.902

- truth: `under sitt vistande härstädes`
- pred:  `under sitt vistande hos stade`

**jonkopings_radhusratt_och_magistrat** -- CER 0.167, confidence 0.837

- truth: `Suenson y Rommesiö bleff`
- pred:  `suenßon y kommesio bleff`

**jonkopings_radhusratt_och_magistrat** -- CER 0.273, confidence 0.776

- truth: `och inge peinge, thett`
- pred:  `och nya pennge, ther`

**jonkopings_radhusratt_och_magistrat** -- CER 0.623, confidence 0.559

- truth: `III: 29 A 103 finnes som emot Suerigis lagn wåre wtgångne me¬`
- pred:  `finmas, somt denor Hhurdiges last. wåro rörtångnr att`

**krigshovrattens_dombocker** -- CER 0.000, confidence 0.989

- truth: `sig till minnes wid hwad tid the`
- pred:  `sig till minnes wid hwad tid the`

**krigshovrattens_dombocker** -- CER 0.167, confidence 0.807

- truth: `uppwyt affskiedz Paß om hans wälför¬`
- pred:  `uppwiyt afskjedz Paß om hans walfoa¬`

**krigshovrattens_dombocker** -- CER 0.032, confidence 0.945

- truth: `des lämnades företräde åt henne`
- pred:  `des lämnades företrede åt henne`

**svea_hovratt** -- CER 0.270, confidence 0.598

- truth: `S. d. uplästes af.ne H.r CommercieRå¬`
- pred:  `d. upläses af e H.r Commaricdå¬`

**svea_hovratt** -- CER 0.188, confidence 0.834

- truth: `sw: at han bor i Wästergiöthland`
- pred:  `sw: at han bor i wastergiotsla`

**svea_hovratt** -- CER 0.129, confidence 0.943

- truth: `dingen som uti Kong. Hofrätten,`
- pred:  `omeen som uti Kong. Hofrätten,`

**trolldomskommissionen** -- CER 0.294, confidence 0.810

- truth: `wärk iryggen; och för öfrigit icke`
- pred:  `warLiryggen, byh för öfrigit nu¬`

**trolldomskommissionen** -- CER 0.051, confidence 0.954

- truth: `ian och Strömberg, stundom och för Hans`
- pred:  `lar och Strömberg, stundom och för Hans`

**trolldomskommissionen** -- CER 0.146, confidence 0.850

- truth: `Ellfäntige huadh farliget ähr, befara medh månge`
- pred:  `tillfärlige huadh farliget ähr befara, medhmånge`

## Comparison with prior reference points

```json
{
  "pilot_best_val_cer": 0.1691,
  "pilot_best_val_cer_note": "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, shown for orientation rather than as a like-for-like target.",
  "pilot_single_pass_cer": 0.2481,
  "pilot_single_pass_cer_note": "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps to one full-corpus shard, not to a lap.",
  "checkpoint_kind_evaluated": "best_val",
  "checkpoint_recorded_at_shard": 1,
  "run_best_val_cer_during_lap": 0.15337054431438446,
  "final_shard_metrics": {
    "train_cer": 0.1351269632577896,
    "val_cer": 0.15337054431438446,
    "train_wer": 0.8271514773368835,
    "val_wer": 0.8659999966621399,
    "train_loss": 16.248558044433594,
    "val_loss": 19.53328514099121
  },
  "final_shard_metrics_note": "Train and validation CER/WER/loss as the container itself reported them at the end of the lap's final shard. The train/val gap here is the overfitting signal; the corpus CER above is an independent recomputation from raw predictions.",
  "in_training_val_cer_note": "The in-training figures above come from the container's own validation at the end of each shard, on this same 1,000-line set. They are the right thing to compare this pass against; the pilot numbers are context, not a target.",
  "inference_seconds": 845.8
}
```
