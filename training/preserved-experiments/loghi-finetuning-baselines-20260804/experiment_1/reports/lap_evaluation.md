# End-of-lap evaluation -- lap 1

- run: `loghi_training_run_6b2d57d1b30240aa9f4c0982375f4d6e`
- checkpoint: `D:\ArchiveTrust_HCR\training\experiment-1-continuous-epoch-20260803T192146Z\run-state\epoch_output\epoch_1\model_new10\best_val`
- checkpoint sha256: `a1fbb832b6610f91c833db41bdb36949e1652cd7ac082eff870356ba58ee142f`
- position: 1 shards = 1 full corpus lap(s) of 1 shards each
- validation lines: 1000 scored of 1000

## Overall

| metric | value |
| --- | --- |
| corpus CER | **0.1747** |
| corpus WER (true word error rate) | 0.4956 |
| line error rate (what the container calls "WER") | 0.8960 |
| mean per-line CER | 0.1764 |
| total edits | 5,513 |
| total reference chars | 31,554 |

Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure comparable with the container's own `CERMetric`. The per-line mean is shown beside it because they diverge when line lengths vary.

## Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| alvsborgs_losen | 0.2860 | 0.5202 | 91 | 1,797 | 0.705 |
| jonkopings_radhusratt_och_magistrat | 0.2593 | 0.7073 | 91 | 3,386 | 0.726 |
| gota_hovratt | 0.2399 | 0.6181 | 91 | 3,243 | 0.773 |
| bergmastaren_i_nora_htr | 0.2329 | 0.6674 | 91 | 2,756 | 0.758 |
| trolldomskommissionen | 0.2205 | 0.5807 | 90 | 3,302 | 0.783 |
| krigshovrattens_dombocker | 0.1666 | 0.4930 | 91 | 2,917 | 0.818 |
| goteborgs_poliskammare_fore_1900 | 0.1342 | 0.4054 | 91 | 2,445 | 0.883 |
| carl_fredrik_pahlmans_resejournaler | 0.1173 | 0.3947 | 91 | 3,973 | 0.884 |
| svea_hovratt | 0.1153 | 0.3824 | 91 | 2,593 | 0.889 |
| frihetstidens_utskottshandlingar | 0.0987 | 0.3683 | 91 | 2,412 | 0.879 |
| bergskollegium_relationer_och_skrivelser | 0.0571 | 0.2319 | 91 | 2,730 | 0.946 |

## Confidence calibration

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 - 0.20 | 10 | 0.6436 | 0.6227 |
| 0.2 - 0.40 | 19 | 0.5242 | 0.5190 |
| 0.4 - 0.60 | 57 | 0.4697 | 0.4286 |
| 0.6 - 0.80 | 231 | 0.2591 | 0.2500 |
| 0.8 - 1.01 | 683 | 0.1075 | 0.0893 |

### Rejection coverage

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1764 |
| 0.50 | 95.6% | 956 | 0.1595 |
| 0.70 | 83.4% | 834 | 0.1300 |
| 0.80 | 68.3% | 683 | 0.1075 |
| 0.90 | 39.8% | 398 | 0.0713 |
| 0.95 | 16.8% | 168 | 0.0487 |

## Inspection sample

Stratified: the same number of lines from every collection, selected by a stable hash seeded on the checkpoint identity, so this sample is reproducible and comparable across laps.

**alvsborgs_losen** -- CER 0.300, confidence 0.730

- truth: `Hustru Barebro i bih`
- pred:  `Hustru Sarsbro i N: C`

**alvsborgs_losen** -- CER 0.857, confidence 0.305

- truth: `Kumerum`
- pred:  `Suurränn`

**alvsborgs_losen** -- CER 0.455, confidence 0.442

- truth: `Tomas Ibidh`
- pred:  `Tomar 9Ki8l`

**bergmastaren_i_nora_htr** -- CER 0.368, confidence 0.663

- truth: `ne lådepenningar 10 dr: SöllfwerMyndt.`
- pred:  `ir Rådpenningar 10 D:r Sellför Wygdt`

**bergmastaren_i_nora_htr** -- CER 0.200, confidence 0.783

- truth: `13. 9`
- pred:  `13.9`

**bergmastaren_i_nora_htr** -- CER 0.385, confidence 0.681

- truth: `spective Lands höfdinge Embetets i Öre¬`
- pred:  `pectioe, Lands serdinget attett i Oce`

**bergskollegium_relationer_och_skrivelser** -- CER 0.214, confidence 0.899

- truth: `mera C. giör migliare i Ven:`
- pred:  `mera C: gior Migliare. ven:`

**bergskollegium_relationer_och_skrivelser** -- CER 0.043, confidence 0.909

- truth: `med en skärfsåg i sådan`
- pred:  `red en skärfsåg i sådan`

**bergskollegium_relationer_och_skrivelser** -- CER 0.093, confidence 0.957

- truth: `och under Noberg med Eecheberget Schieffern`
- pred:  `och under Noberg med Eeckeberget Schistern`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.128, confidence 0.902

- truth: `nåd för mina ögon. Så litet var den med`
- pred:  `nad för mins ögon. Så tikt var den med`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.073, confidence 0.938

- truth: `marmor, hvilka man endast på knä kan upp¬`
- pred:  `marmor, hvilka man endast på kna kanna upp¬`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.190, confidence 0.810

- truth: `just den 2ra Aug, ty man hade sagt oss att`
- pred:  `jast den 2e Aug. ty man hade sagt afs ot`

**frihetstidens_utskottshandlingar** -- CER 0.333, confidence 0.806

- truth: `de 3ne första §.§.`
- pred:  `i de 3ne första§d`

**frihetstidens_utskottshandlingar** -- CER 0.000, confidence 0.917

- truth: `le afhålla sig ifrån the ytterligare`
- pred:  `le afhålla sig ifrån the ytterligare`

**frihetstidens_utskottshandlingar** -- CER 0.119, confidence 0.942

- truth: `och förfarenhet trodde sig wara sig wuxne,`
- pred:  `och förfaranhet trodde sig wara sig wig ne¬`

**gota_hovratt** -- CER 0.282, confidence 0.718

- truth: `höfdingen dömbdt effter Landz lagh rätt`
- pred:  `söfdingen dörelet efter Fordslagh rätt`

**gota_hovratt** -- CER 0.267, confidence 0.689

- truth: `som uthi dhet 9 cap. Kongb: Ll. säger att om `
- pred:  `som por: dhet 9. cap. Bongk. 1l säger at om`

**gota_hovratt** -- CER 0.227, confidence 0.727

- truth: `skylt Sadellmakaren Hans Krak och des hustru`
- pred:  `sylt Sedellmäkaren Wars woak och det hufru`

**goteborgs_poliskammare_fore_1900** -- CER 0.071, confidence 0.904

- truth: `förhör förnekat all kännedom`
- pred:  `förhör, farnekat all kännedom`

**goteborgs_poliskammare_fore_1900** -- CER 0.133, confidence 0.933

- truth: `måtte varda till mig öfverlem¬`
- pred:  `mälle varda till mig öfverbem¬`

**goteborgs_poliskammare_fore_1900** -- CER 0.038, confidence 0.925

- truth: `pantsatt en del af det här`
- pred:  `pantsatt en delaf det här`

**jonkopings_radhusratt_och_magistrat** -- CER 0.207, confidence 0.665

- truth: `Westerårs 5 January Anno r 51`
- pred:  `vsterärs 5 Januaiy Anno E51`

**jonkopings_radhusratt_och_magistrat** -- CER 0.143, confidence 0.817

- truth: `han hade och stålet en hest.`
- pred:  `han hade och stöler re hest.`

**jonkopings_radhusratt_och_magistrat** -- CER 0.156, confidence 0.806

- truth: `Den ädelig welbördig Par Ribbing`
- pred:  `Deni ädelig workbördig Par pibbing`

**krigshovrattens_dombocker** -- CER 0.086, confidence 0.836

- truth: `kade och swor Iemmerligen sigh all¬`
- pred:  `tade och swor Jemmarligen sigh all¬`

**krigshovrattens_dombocker** -- CER 0.000, confidence 0.981

- truth: `sig till minnes wid hwad tid the`
- pred:  `sig till minnes wid hwad tid the`

**krigshovrattens_dombocker** -- CER 0.162, confidence 0.943

- truth: `dem emellan der det skiedde; allenast`
- pred:  `dem emellan det det skadde, allenaf`

**svea_hovratt** -- CER 0.065, confidence 0.953

- truth: `wit skuldebref, samt til Ausul¬`
- pred:  `wit skuldebref, samt til Ansul-`

**svea_hovratt** -- CER 0.000, confidence 0.948

- truth: `selius. Pehr Franc. Johan`
- pred:  `selius. Pehr Franc. Johan`

**svea_hovratt** -- CER 0.067, confidence 0.912

- truth: `Anders Johansson i Norrby, ge¬`
- pred:  `Anders Johansson i Nloreby, ge¬`

**trolldomskommissionen** -- CER 0.400, confidence 0.638

- truth: `Högwälborne Herrar, Sweriges Rykes Rådh;`
- pred:  `Högswalboser herne, berngs Rykes Adh,`

**trolldomskommissionen** -- CER 0.618, confidence 0.070

- truth: `Men finner intet så oftta falska skiähl tillförende som`
- pred:  `Man ine nt te t a ifide o`

**trolldomskommissionen** -- CER 0.256, confidence 0.788

- truth: `trenne särskilte personer åt hwilka hon`
- pred:  `trgnn färstate persener, at hwilka han`

## Comparison with prior reference points

```json
{
  "pilot_best_val_cer": 0.1691,
  "pilot_best_val_cer_note": "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, shown for orientation rather than as a like-for-like target.",
  "pilot_single_pass_cer": 0.2481,
  "pilot_single_pass_cer_note": "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps to one full-corpus shard, not to a lap.",
  "checkpoint_kind_evaluated": "best_val",
  "checkpoint_recorded_at_shard": 1,
  "run_best_val_cer_during_lap": 0.17553427815437317,
  "final_shard_metrics": {
    "train_cer": 0.19487333297729492,
    "val_cer": 0.17553427815437317,
    "train_wer": 0.8947499394416809,
    "val_wer": 0.8949999809265137,
    "train_loss": 23.543516159057617,
    "val_loss": 22.073938369750977
  },
  "final_shard_metrics_note": "Train and validation CER/WER/loss as the container itself reported them at the end of the lap's final shard. The train/val gap here is the overfitting signal; the corpus CER above is an independent recomputation from raw predictions.",
  "in_training_val_cer_note": "The in-training figures above come from the container's own validation at the end of each shard, on this same 1,000-line set. They are the right thing to compare this pass against; the pilot numbers are context, not a target.",
  "inference_seconds": 194.3
}
```
