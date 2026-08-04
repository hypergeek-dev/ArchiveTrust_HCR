# End-of-lap evaluation -- lap 1

- run: `loghi_training_run_d20dc825910945acbabc3ba7162d340b`
- checkpoint: `D:\ArchiveTrust_HCR\training\experiment-2-loghi-recommended-scratch-20260804T051959Z\run-state\epoch_output\epoch_1\recommended\best_val`
- checkpoint sha256: `232ac54eddbe5d3c9f9ebc71f6f4899ce29d3e1f0e60e1dd9c830d3c8ee9c72a`
- position: 1 shards = 1 full corpus lap(s) of 1 shards each
- validation lines: 1000 scored of 1000

## Overall

| metric | value |
| --- | --- |
| corpus CER | **0.1918** |
| corpus WER (true word error rate) | 0.5470 |
| line error rate (what the container calls "WER") | 0.9350 |
| mean per-line CER | 0.1998 |
| total edits | 6,051 |
| total reference chars | 31,554 |

Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure comparable with the container's own `CERMetric`. The per-line mean is shown beside it because they diverge when line lengths vary.

## Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| alvsborgs_losen | 0.3489 | 0.6221 | 91 | 1,797 | 0.580 |
| jonkopings_radhusratt_och_magistrat | 0.2956 | 0.7580 | 91 | 3,386 | 0.653 |
| gota_hovratt | 0.2384 | 0.6341 | 91 | 3,243 | 0.716 |
| bergmastaren_i_nora_htr | 0.2311 | 0.6740 | 91 | 2,756 | 0.736 |
| trolldomskommissionen | 0.2193 | 0.5947 | 90 | 3,302 | 0.749 |
| krigshovrattens_dombocker | 0.1724 | 0.5449 | 91 | 2,917 | 0.786 |
| carl_fredrik_pahlmans_resejournaler | 0.1540 | 0.4898 | 91 | 3,973 | 0.834 |
| goteborgs_poliskammare_fore_1900 | 0.1497 | 0.4693 | 91 | 2,445 | 0.829 |
| frihetstidens_utskottshandlingar | 0.1244 | 0.4220 | 91 | 2,412 | 0.800 |
| svea_hovratt | 0.1184 | 0.4436 | 91 | 2,593 | 0.837 |
| bergskollegium_relationer_och_skrivelser | 0.0736 | 0.2795 | 91 | 2,730 | 0.922 |

## Confidence calibration

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 - 0.20 | 18 | 0.7728 | 0.6948 |
| 0.2 - 0.40 | 31 | 0.5723 | 0.5455 |
| 0.4 - 0.60 | 106 | 0.3829 | 0.3563 |
| 0.6 - 0.80 | 304 | 0.2358 | 0.2168 |
| 0.8 - 1.01 | 541 | 0.1033 | 0.0938 |

### Rejection coverage

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1998 |
| 0.50 | 92.3% | 923 | 0.1673 |
| 0.70 | 73.7% | 737 | 0.1316 |
| 0.80 | 54.1% | 541 | 0.1033 |
| 0.90 | 22.8% | 228 | 0.0586 |
| 0.95 | 8.1% | 81 | 0.0300 |

## Inspection sample

Stratified: the same number of lines from every collection, selected by a stable hash seeded on the checkpoint identity, so this sample is reproducible and comparable across laps.

**alvsborgs_losen** -- CER 0.571, confidence 0.512

- truth: `Suma aff Grenge Sochn`
- pred:  `Enna oht Bengs Trche`

**alvsborgs_losen** -- CER 0.000, confidence 0.589

- truth: `Får — 2`
- pred:  `Får — 2`

**alvsborgs_losen** -- CER 0.067, confidence 0.822

- truth: `Sölff — 10 lott`
- pred:  `Sölff — 10 ott`

**bergmastaren_i_nora_htr** -- CER 0.707, confidence 0.614

- truth: `begärdt mera pgr, Och therhos stod på, at`
- pred:  `hjert men ue i der Josos sinladt Hages at`

**bergmastaren_i_nora_htr** -- CER 0.200, confidence 0.681

- truth: `13. 9`
- pred:  `13.9`

**bergmastaren_i_nora_htr** -- CER 0.444, confidence 0.842

- truth: `H. Robsam`
- pred:  `H. Rossam 1.`

**bergskollegium_relationer_och_skrivelser** -- CER 0.032, confidence 0.893

- truth: `Sålunda warder med Glantzertzen`
- pred:  `Sålunda warder med glantzertzen`

**bergskollegium_relationer_och_skrivelser** -- CER 0.050, confidence 0.938

- truth: `sig bör igenomslagne`
- pred:  `sig bör igenomslagnr`

**bergskollegium_relationer_och_skrivelser** -- CER 0.103, confidence 0.856

- truth: `Thut 5816 Cent:r darin 181 qv:r 12 loth`
- pred:  `Thut 5816 Cent: darin 181 mkr 12 loth`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.180, confidence 0.886

- truth: `långt skägg etc. Cardinalerne äfven kläddda i dyr¬`
- pred:  `länst skägg. etp Cardinaterne äfven klädda i Sypr¬`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.167, confidence 0.876

- truth: `den aflidnes`
- pred:  `den aflidras`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.204, confidence 0.802

- truth: `domar och prydnader. Dessutom kan man väl säga att det`
- pred:  `Gonar och pngdnader. Desertom han man väd sag att de`

**frihetstidens_utskottshandlingar** -- CER 0.333, confidence 0.678

- truth: `§: 12.`
- pred:  `8o 12.`

**frihetstidens_utskottshandlingar** -- CER 0.067, confidence 0.887

- truth: `6 Exemplar aflemnas, Af hwilka`
- pred:  `6 Exemplar aflomna, Af hwilka`

**frihetstidens_utskottshandlingar** -- CER 0.032, confidence 0.949

- truth: `to förebygga, men theremot icke`
- pred:  `to förebygga, men theremet icke`

**gota_hovratt** -- CER 0.333, confidence 0.713

- truth: `öfwer unge drängen Håkan olofzon an¬`
- pred:  `öfwer mgedringen Haken losfens en¬`

**gota_hovratt** -- CER 0.114, confidence 0.848

- truth: `Tingzrätten har altså efter 1653 års straff¬`
- pred:  `Tingrätten har altså efter 1673 ärs skraff`

**gota_hovratt** -- CER 0.237, confidence 0.779

- truth: `gande, om denna Biörnen Larßon för be¬`
- pred:  `gande, om dedtrig Biänng Larßon för be¬`

**goteborgs_poliskammare_fore_1900** -- CER 0.000, confidence 0.967

- truth: `Per Cederborg.`
- pred:  `Per Cederborg.`

**goteborgs_poliskammare_fore_1900** -- CER 0.057, confidence 0.849

- truth: `hvars befälhafvare jag i skrifvelse`
- pred:  `hvers befälhafvare jg i skrifvelse`

**goteborgs_poliskammare_fore_1900** -- CER 0.211, confidence 0.736

- truth: `Angereds Sn för omkring 3 veckor sedan`
- pred:  `Aegerede dn för omking 3 äcks sedan`

**jonkopings_radhusratt_och_magistrat** -- CER 0.148, confidence 0.841

- truth: `städes våre, och effter han`
- pred:  `städes wäre, och effwe han`

**jonkopings_radhusratt_och_magistrat** -- CER 0.276, confidence 0.715

- truth: `Ten Andre dömdes och för dett`
- pred:  `sra Ander döndes och få dett`

**jonkopings_radhusratt_och_magistrat** -- CER 0.194, confidence 0.667

- truth: `serck, et båndekläde, och möket`
- pred:  `serk, et bändakläd, och möe`

**krigshovrattens_dombocker** -- CER 0.100, confidence 0.888

- truth: `han hit till Stockholm blifwit`
- pred:  `han het till Stockhjolm blifvit`

**krigshovrattens_dombocker** -- CER 0.162, confidence 0.732

- truth: `Compagnie No 26 Erik Eklund förekommo`
- pred:  `Compagne No 26 Eit: Elund förekomma`

**krigshovrattens_dombocker** -- CER 0.032, confidence 0.905

- truth: `des lämnades företräde åt henne`
- pred:  `des lämnades företrade åt henne`

**svea_hovratt** -- CER 0.000, confidence 0.930

- truth: `Större Rummet.`
- pred:  `Större Rummet.`

**svea_hovratt** -- CER 0.133, confidence 0.844

- truth: `må anses hafwa sitt wad förlo¬`
- pred:  `mte anser hafwa sett wad förlo¬`

**svea_hovratt** -- CER 0.032, confidence 0.924

- truth: `äfwen sig anmält, ej ännu wijst`
- pred:  `äfwen sig anmält, ej ännu wyjst`

**trolldomskommissionen** -- CER 0.326, confidence 0.669

- truth: `och gitte skuk låte saningen koma i dagzliuset`
- pred:  `och Eyfte skätt låte sinnign kona i dagtiusok`

**trolldomskommissionen** -- CER 0.158, confidence 0.835

- truth: `bekjennelsse kunde marit nu icke brin¬`
- pred:  `tekjennelse kunde marit nu icke tem¬`

**trolldomskommissionen** -- CER 0.300, confidence 0.572

- truth: `och dwahla per prastigias inbilla något sitt tien¬`
- pred:  `och derehl pe pnstigie inbella något sik tä¬`

## Comparison with prior reference points

```json
{
  "pilot_best_val_cer": 0.1691,
  "pilot_best_val_cer_note": "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, shown for orientation rather than as a like-for-like target.",
  "pilot_single_pass_cer": 0.2481,
  "pilot_single_pass_cer_note": "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps to one full-corpus shard, not to a lap.",
  "checkpoint_kind_evaluated": "best_val",
  "checkpoint_recorded_at_shard": 1,
  "run_best_val_cer_during_lap": 0.19398820400238037,
  "final_shard_metrics": {
    "train_cer": 0.2749086022377014,
    "val_cer": 0.19398820400238037,
    "train_wer": 0.9423084855079651,
    "val_wer": 0.9350000023841858,
    "train_loss": 32.00843811035156,
    "val_loss": 23.21910858154297
  },
  "final_shard_metrics_note": "Train and validation CER/WER/loss as the container itself reported them at the end of the lap's final shard. The train/val gap here is the overfitting signal; the corpus CER above is an independent recomputation from raw predictions.",
  "in_training_val_cer_note": "The in-training figures above come from the container's own validation at the end of each shard, on this same 1,000-line set. They are the right thing to compare this pass against; the pilot numbers are context, not a target.",
  "inference_seconds": 75.0
}
```
