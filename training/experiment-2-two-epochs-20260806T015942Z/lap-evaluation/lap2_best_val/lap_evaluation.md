# End-of-lap evaluation -- lap 2

- run: `loghi_training_run_5be4c4ab0ceb444a9b333f0913b8f215`
- checkpoint: `D:\ArchiveTrust_HCR\training\experiment-2-two-epochs-20260806T015942Z\run-state\epoch_output\epoch_1\recommended\best_val`
- checkpoint sha256: `1f8a164e86fb9e1882ca0ea341b61129ef73918f41fff643dfd8c3dfa847e339`
- position: 1 shards = 2 full corpus lap(s) of 1 shards each
- validation lines: 1000 scored of 1000

## Overall

| metric | value |
| --- | --- |
| corpus CER | **0.1487** |
| corpus WER (true word error rate) | 0.4546 |
| line error rate (what the container calls "WER") | 0.8780 |
| mean per-line CER | 0.1543 |
| total edits | 4,691 |
| total reference chars | 31,554 |

Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure comparable with the container's own `CERMetric`. The per-line mean is shown beside it because they diverge when line lengths vary.

## Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| alvsborgs_losen | 0.2499 | 0.4968 | 91 | 1,797 | 0.712 |
| jonkopings_radhusratt_och_magistrat | 0.2499 | 0.7022 | 91 | 3,386 | 0.723 |
| bergmastaren_i_nora_htr | 0.1985 | 0.5974 | 91 | 2,756 | 0.776 |
| gota_hovratt | 0.1819 | 0.5169 | 91 | 3,243 | 0.795 |
| trolldomskommissionen | 0.1684 | 0.4895 | 90 | 3,302 | 0.803 |
| krigshovrattens_dombocker | 0.1286 | 0.4351 | 91 | 2,917 | 0.835 |
| goteborgs_poliskammare_fore_1900 | 0.1166 | 0.3514 | 91 | 2,445 | 0.871 |
| carl_fredrik_pahlmans_resejournaler | 0.1107 | 0.3977 | 91 | 3,973 | 0.876 |
| svea_hovratt | 0.0960 | 0.3578 | 91 | 2,593 | 0.887 |
| frihetstidens_utskottshandlingar | 0.0867 | 0.3415 | 91 | 2,412 | 0.858 |
| bergskollegium_relationer_och_skrivelser | 0.0531 | 0.2277 | 91 | 2,730 | 0.947 |

## Confidence calibration

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 - 0.20 | 13 | 0.7078 | 0.6393 |
| 0.2 - 0.40 | 18 | 0.5153 | 0.5000 |
| 0.4 - 0.60 | 55 | 0.4103 | 0.3898 |
| 0.6 - 0.80 | 217 | 0.2444 | 0.2281 |
| 0.8 - 1.01 | 697 | 0.0865 | 0.0727 |

### Rejection coverage

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1543 |
| 0.50 | 94.7% | 947 | 0.1332 |
| 0.70 | 84.8% | 848 | 0.1117 |
| 0.80 | 69.7% | 697 | 0.0865 |
| 0.90 | 41.6% | 416 | 0.0555 |
| 0.95 | 17.1% | 171 | 0.0313 |

## Inspection sample

Stratified: the same number of lines from every collection, selected by a stable hash seeded on the checkpoint identity, so this sample is reproducible and comparable across laps.

**alvsborgs_losen** -- CER 0.429, confidence 0.555

- truth: `Domtha:`
- pred:  `Tomsza:`

**alvsborgs_losen** -- CER 1.000, confidence 0.070

- truth: `bärke`
- pred:  `Lakön`

**alvsborgs_losen** -- CER 0.429, confidence 0.390

- truth: `Daler — 1 stKe`
- pred:  `Saker — 2 silr`

**bergmastaren_i_nora_htr** -- CER 0.270, confidence 0.962

- truth: `hwardärföre blifwa answarig, med mera`
- pred:  `därföre blifwa answarig, med mera hwar¬`

**bergmastaren_i_nora_htr** -- CER 0.396, confidence 0.611

- truth: `Commissarien thet pris, hwartil en tylik extraso`
- pred:  `CComussoier tht sigs, hwrtil i tid efterso`

**bergmastaren_i_nora_htr** -- CER 0.154, confidence 0.859

- truth: `Transporterat`
- pred:  `Pransponterat`

**bergskollegium_relationer_och_skrivelser** -- CER 0.000, confidence 0.988

- truth: `tillförenne haft ansenlige`
- pred:  `tillförenne haft ansenlige`

**bergskollegium_relationer_och_skrivelser** -- CER 0.143, confidence 0.959

- truth: `pen Klyfwas i trenne stycken`
- pred:  `gen klyfwas i trenne, stycken-`

**bergskollegium_relationer_och_skrivelser** -- CER 0.133, confidence 0.942

- truth: `et öpet öga uti fördegelen som`
- pred:  `öpet äga uti fördegelen som`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.179, confidence 0.843

- truth: `axlar. Hjulen äro af järn. Hastigheten är aldeles otro¬ `
- pred:  `axdar. Hjulen ära af järn. Nastigheten äro aldelesatne.`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.122, confidence 0.912

- truth: `för vår Prest min skrämsel och upptäckten`
- pred:  `lin vår Prest min skrämsel och depptäckten`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.200, confidence 0.762

- truth: `så lämna de den med en högmögen sjelfsvåldig¬`
- pred:  `så lämma de den möd en kögningen sjulsväldig¬`

**frihetstidens_utskottshandlingar** -- CER 0.100, confidence 0.863

- truth: `deras efterrättelse, som af o¬`
- pred:  `derar efterrättelse, som afe¬`

**frihetstidens_utskottshandlingar** -- CER 0.100, confidence 0.907

- truth: `Såframt de icke förut af någon`
- pred:  `döframt desicke förut af någon`

**frihetstidens_utskottshandlingar** -- CER 0.077, confidence 0.921

- truth: `theß redan tagne författningar granskar`
- pred:  `theß redan tagen författningar gränskar`

**gota_hovratt** -- CER 0.268, confidence 0.688

- truth: `skall både böndernaß och Soldaternas lösa`
- pred:  `skall Fud böndranast och holdat mas lösa`

**gota_hovratt** -- CER 0.421, confidence 0.525

- truth: `Förehades en Criminal Sak från Tuna oh`
- pred:  `förshades en hwrmelkal frå Slåa h`

**gota_hovratt** -- CER 0.282, confidence 0.705

- truth: `henneß beskylningh, och henne trulkonan`
- pred:  `semnest beskylningh, och henne Wwilben`

**goteborgs_poliskammare_fore_1900** -- CER 0.050, confidence 0.922

- truth: `jemte tillägg af 900 kr. för ett biträde`
- pred:  `jemte tillegg af 900 Kr. för ett biträde`

**goteborgs_poliskammare_fore_1900** -- CER 0.038, confidence 0.945

- truth: `pantsatt en del af det här`
- pred:  `pantsatt en del af det har`

**goteborgs_poliskammare_fore_1900** -- CER 0.100, confidence 0.916

- truth: `Köpenhamn.`
- pred:  `Köpenhamn`

**jonkopings_radhusratt_och_magistrat** -- CER 0.042, confidence 0.833

- truth: `hon stålit, et lakan, en`
- pred:  `hon stålin, et lakan, en`

**jonkopings_radhusratt_och_magistrat** -- CER 0.212, confidence 0.703

- truth: `hwad deel i oss på Cronone wågne skyllige och plich¬`
- pred:  `hwad det i ost på Tronoms wågna skylige och pluf¬`

**jonkopings_radhusratt_och_magistrat** -- CER 0.654, confidence 0.138

- truth: `Wy Gustaff medt Gudz nådi Swerigis, Göti och Wend zr`
- pred:  `Sy öf et båd, wih nig o on t `

**krigshovrattens_dombocker** -- CER 0.143, confidence 0.918

- truth: `ock som Hustru Lundstedts föreburne`
- pred:  `och som hustru Landstedts företärne`

**krigshovrattens_dombocker** -- CER 0.091, confidence 0.931

- truth: `sedel och adress, hade ingen miß¬`
- pred:  `sedel och Adress, hade ingen mist¬`

**krigshovrattens_dombocker** -- CER 0.211, confidence 0.749

- truth: `tahlning förbunden.`
- pred:  `sahlning förkundn`

**svea_hovratt** -- CER 0.077, confidence 0.962

- truth: `Månad ingifne skrift äfwen`
- pred:  `månad ingifne skritt äfwen`

**svea_hovratt** -- CER 0.094, confidence 0.952

- truth: `för en annan simpel fordran, sex`
- pred:  `för en annan Simpel fordran, sig`

**svea_hovratt** -- CER 0.071, confidence 0.908

- truth: `Protocoll från Westeråhs hit`
- pred:  `Protocoll från Wester åhs sit`

**trolldomskommissionen** -- CER 0.104, confidence 0.861

- truth: `sätet enfaldigh min meningh tillkienna, hwarföre`
- pred:  `atet infaldigt min meningh tillkenna, hwarföre`

**trolldomskommissionen** -- CER 0.025, confidence 0.952

- truth: `wardßon förmådt henne dertil, och om hon`
- pred:  `werdßon förmådt henne dertil, och om hon`

**trolldomskommissionen** -- CER 0.120, confidence 0.921

- truth: `collet, utan hwilkas när¬`
- pred:  `collet, utom hwilkaas när¬`

## Comparison with prior reference points

```json
{
  "pilot_best_val_cer": 0.1691,
  "pilot_best_val_cer_note": "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, shown for orientation rather than as a like-for-like target.",
  "pilot_single_pass_cer": 0.2481,
  "pilot_single_pass_cer_note": "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps to one full-corpus shard, not to a lap.",
  "checkpoint_kind_evaluated": "best_val",
  "checkpoint_recorded_at_shard": 1,
  "run_best_val_cer_during_lap": 0.15029488503932953,
  "final_shard_metrics": {
    "train_cer": 0.11036395281553268,
    "val_cer": 0.15029488503932953,
    "train_wer": 0.8133716583251953,
    "val_wer": 0.8790000081062317,
    "train_loss": 13.302531242370605,
    "val_loss": 17.787019729614258
  },
  "final_shard_metrics_note": "Train and validation CER/WER/loss as the container itself reported them at the end of the lap's final shard. The train/val gap here is the overfitting signal; the corpus CER above is an independent recomputation from raw predictions.",
  "in_training_val_cer_note": "The in-training figures above come from the container's own validation at the end of each shard, on this same 1,000-line set. They are the right thing to compare this pass against; the pilot numbers are context, not a target.",
  "inference_seconds": 100.5
}
```
