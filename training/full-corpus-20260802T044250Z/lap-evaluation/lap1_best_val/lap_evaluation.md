# End-of-lap evaluation -- lap 1

- run: `loghi_training_run_8847b9fcbbdf4c6b8256481944f5bded`
- checkpoint: `training\full-corpus-20260802T044250Z\run-state\epoch_output\epoch_48\model_new10\best_val`
- checkpoint sha256: `ade57f9f28250731d8bcbabcf03e2b362a6ae7075e56afc2be44f4d9d6c07dfd`
- position: 57 shards = 1 full corpus lap(s) of 57 shards each
- validation lines: 1000 scored of 1000

## Overall

| metric | value |
| --- | --- |
| corpus CER | **0.1728** |
| corpus WER (true word error rate) | 0.4945 |
| line error rate (what the container calls "WER") | 0.8950 |
| mean per-line CER | 0.1736 |
| total edits | 5,452 |
| total reference chars | 31,554 |

Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure comparable with the container's own `CERMetric`. The per-line mean is shown beside it because they diverge when line lengths vary.

## Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| alvsborgs_losen | 0.2760 | 0.5032 | 91 | 1,797 | 0.729 |
| jonkopings_radhusratt_och_magistrat | 0.2658 | 0.7327 | 91 | 3,386 | 0.740 |
| gota_hovratt | 0.2282 | 0.5986 | 91 | 3,243 | 0.780 |
| trolldomskommissionen | 0.2226 | 0.5877 | 90 | 3,302 | 0.796 |
| bergmastaren_i_nora_htr | 0.2195 | 0.6389 | 91 | 2,756 | 0.769 |
| krigshovrattens_dombocker | 0.1646 | 0.4850 | 91 | 2,917 | 0.832 |
| goteborgs_poliskammare_fore_1900 | 0.1337 | 0.4054 | 91 | 2,445 | 0.886 |
| carl_fredrik_pahlmans_resejournaler | 0.1186 | 0.4020 | 91 | 3,973 | 0.889 |
| svea_hovratt | 0.1172 | 0.4044 | 91 | 2,593 | 0.889 |
| frihetstidens_utskottshandlingar | 0.0995 | 0.3659 | 91 | 2,412 | 0.884 |
| bergskollegium_relationer_och_skrivelser | 0.0564 | 0.2277 | 91 | 2,730 | 0.946 |

## Confidence calibration

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 - 0.20 | 8 | 0.6697 | 0.6390 |
| 0.2 - 0.40 | 17 | 0.5573 | 0.5000 |
| 0.4 - 0.60 | 44 | 0.4217 | 0.3923 |
| 0.6 - 0.80 | 235 | 0.2735 | 0.2593 |
| 0.8 - 1.01 | 696 | 0.1090 | 0.0938 |

### Rejection coverage

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1736 |
| 0.50 | 95.8% | 958 | 0.1573 |
| 0.70 | 85.5% | 855 | 0.1338 |
| 0.80 | 69.6% | 696 | 0.1090 |
| 0.90 | 40.2% | 402 | 0.0716 |
| 0.95 | 17.9% | 179 | 0.0489 |

## Inspection sample

Stratified: the same number of lines from every collection, selected by a stable hash seeded on the checkpoint identity, so this sample is reproducible and comparable across laps.

**alvsborgs_losen** -- CER 0.214, confidence 0.835

- truth: `Sölff — 1 Lodh`
- pred:  `Solff — 1 lode`

**alvsborgs_losen** -- CER 0.125, confidence 0.860

- truth: `Sölff — 1½ qutt:`
- pred:  `Sölff — 1 ½ qutt.`

**alvsborgs_losen** -- CER 0.500, confidence 0.756

- truth: `Kiör — 2`
- pred:  `Kidn — 5r`

**bergmastaren_i_nora_htr** -- CER 0.059, confidence 0.885

- truth: `höfdingen Funk, den efterrättelse,`
- pred:  `höfdingen Fins, den efterrättelse,`

**bergmastaren_i_nora_htr** -- CER 0.056, confidence 0.967

- truth: `gen med flera widlöftigheter uti den`
- pred:  `gen med flera wid löstigheter uti den`

**bergmastaren_i_nora_htr** -- CER 0.211, confidence 0.792

- truth: `den låfliga Hammartingzrätten begiäran`
- pred:  `den läfliga Hammartingtaclen begiän`

**bergskollegium_relationer_och_skrivelser** -- CER 0.120, confidence 0.828

- truth: `Spanske Wästindien til at`
- pred:  `Sianske wästindien til att`

**bergskollegium_relationer_och_skrivelser** -- CER 0.100, confidence 0.972

- truth: `blifwer försänd, hwarefter han`
- pred:  `blifwor försand, hwareffter han`

**bergskollegium_relationer_och_skrivelser** -- CER 0.087, confidence 0.899

- truth: `med en skärfsåg i sådan`
- pred:  `red en stärfsåg i sådan`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.097, confidence 0.882

- truth: `kunnat bestämma. Det är wäl im¬`
- pred:  `kunnat bestämne. Det är väl im¬`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.130, confidence 0.888

- truth: `som de besjungit. Hvarthän man vänder blicken,`
- pred:  `som de besjungit. Hvartkon man sänder bllken`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.238, confidence 0.855

- truth: `just den 2ra Aug, ty man hade sagt oss att`
- pred:  `jart den 2de Aug. ty man hade sagt af octt`

**frihetstidens_utskottshandlingar** -- CER 0.056, confidence 0.977

- truth: `gar: Lagar utan minsta attention och`
- pred:  `gar. Lagar utan minsta altention och`

**frihetstidens_utskottshandlingar** -- CER 0.138, confidence 0.868

- truth: `trycket utgifwa något Lexicon`
- pred:  `trycket utgifwa något Ledercor`

**frihetstidens_utskottshandlingar** -- CER 0.194, confidence 0.842

- truth: `les förqwäfjas, at mången före¬`
- pred:  `rkes förgwafjas, at mången for¬`

**gota_hovratt** -- CER 0.250, confidence 0.707

- truth: `öfwer unge drängen Håkan olofzon an¬`
- pred:  `öfwe nngedrängen Haken oleffe an¬`

**gota_hovratt** -- CER 0.474, confidence 0.579

- truth: `Förehades en Criminal Sak från Tuna oh`
- pred:  `föresades en stemmalket för Mun f`

**gota_hovratt** -- CER 0.194, confidence 0.823

- truth: `hon intet annat tänkt, än skaf¬`
- pred:  `hon intet annat tådelt, en shaf¬`

**goteborgs_poliskammare_fore_1900** -- CER 0.029, confidence 0.979

- truth: `blifvit derstädes häktad och inne¬`
- pred:  `blifvit derstädes häktad och inni¬`

**goteborgs_poliskammare_fore_1900** -- CER 0.294, confidence 0.816

- truth: `rödlätt ansigtsfärg och ena tummen`
- pred:  `rödlått ansigtserg och ea tienna¬`

**goteborgs_poliskammare_fore_1900** -- CER 0.118, confidence 0.944

- truth: `vid "Mellangränd" i Stockholm haf¬`
- pred:  `vid Mellångrand i Stockholm haf¬`

**jonkopings_radhusratt_och_magistrat** -- CER 0.245, confidence 0.720

- truth: `eder föge achtes, uten i mene, att wy anten skole`
- pred:  `eder fögt acher, wlen i mens, att won anlln,skoll`

**jonkopings_radhusratt_och_magistrat** -- CER 0.294, confidence 0.474

- truth: `July. Anno re 48.`
- pred:  `Jali Anno re 49`

**jonkopings_radhusratt_och_magistrat** -- CER 0.259, confidence 0.710

- truth: `komme till någenn förmheringh och förbättringh, uthenn`
- pred:  `köns till någoen förengerngh och förbättingh, wchen`

**krigshovrattens_dombocker** -- CER 0.083, confidence 0.890

- truth: `de hon det Catharina Fagg war henne skylldig 50,`
- pred:  `de hon det Catharina Dagg war hemna skylldig 50`

**krigshovrattens_dombocker** -- CER 0.093, confidence 0.890

- truth: `dröyde med andra Salvan till des Sången och`
- pred:  `wröjde mer andra Salvan till des Sungen och`

**krigshovrattens_dombocker** -- CER 0.105, confidence 0.840

- truth: `tahlning förbunden.`
- pred:  `lachlning förbunden.`

**svea_hovratt** -- CER 0.000, confidence 0.990

- truth: `Större Rummet.`
- pred:  `Större Rummet.`

**svea_hovratt** -- CER 0.069, confidence 0.946

- truth: `att henne i så måtto biträda;`
- pred:  `att henne i så måtte biträda,`

**svea_hovratt** -- CER 0.179, confidence 0.846

- truth: `klara, wid 10 D.r Smtz wite.`
- pred:  `klara, wid 10 D:r Mig wite.`

**trolldomskommissionen** -- CER 0.286, confidence 0.671

- truth: `et grufwel. Sathans Wärck och anfächtning, som länder hans ryke`
- pred:  `at grufwel: Sachand wirk och ankäcknna, son länder sam yte`

**trolldomskommissionen** -- CER 0.081, confidence 0.952

- truth: `digheter, samt efter den anledning 11`
- pred:  `digheter, samt etter den anlednng 1,`

**trolldomskommissionen** -- CER 0.032, confidence 0.915

- truth: `ingen annan orsak, än then Olof`
- pred:  `ingen annan or sak, än then Olof`

## Comparison with prior reference points

```json
{
  "pilot_best_val_cer": 0.1691,
  "pilot_best_val_cer_note": "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, shown for orientation rather than as a like-for-like target.",
  "pilot_single_pass_cer": 0.2481,
  "pilot_single_pass_cer_note": "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps to one full-corpus shard, not to a lap.",
  "checkpoint_kind_evaluated": "best_val",
  "checkpoint_recorded_at_shard": 48,
  "run_best_val_cer_during_lap": 0.17423425614833832,
  "final_shard_metrics": {
    "train_cer": 0.14631116390228271,
    "val_cer": 0.17559769749641418,
    "train_wer": 0.8371350169181824,
    "val_wer": 0.8930000066757202,
    "train_loss": 17.686532974243164,
    "val_loss": 21.741981506347656
  },
  "final_shard_metrics_note": "Train and validation CER/WER/loss as the container itself reported them at the end of the lap's final shard. The train/val gap here is the overfitting signal; the corpus CER above is an independent recomputation from raw predictions.",
  "in_training_val_cer_note": "The in-training figures above come from the container's own validation at the end of each shard, on this same 1,000-line set. They are the right thing to compare this pass against; the pilot numbers are context, not a target.",
  "inference_seconds": 106.1
}
```
