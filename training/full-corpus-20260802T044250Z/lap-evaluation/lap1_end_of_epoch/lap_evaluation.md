# End-of-lap evaluation -- lap 1

- run: `loghi_training_run_8847b9fcbbdf4c6b8256481944f5bded`
- checkpoint: `d:\ArchiveTrust_HCR\training\full-corpus-20260802T044250Z\run-state\epoch_output\epoch_57\model_new10\epoch_0_CER_0.1463_val_0.1756`
- checkpoint sha256: `265fb3fb6e4e48d6332bdf7c07b422aaa9d95deae52301331021bb2583a1c5d3`
- position: 57 shards = 1 full corpus lap(s) of 57 shards each
- validation lines: 1000 scored of 1000

## Overall

| metric | value |
| --- | --- |
| corpus CER | **0.1735** |
| corpus WER (true word error rate) | 0.4878 |
| line error rate (what the container calls "WER") | 0.8930 |
| mean per-line CER | 0.1731 |
| total edits | 5,475 |
| total reference chars | 31,554 |

Corpus CER is edit-weighted (`total edits / total reference chars`) and is the figure comparable with the container's own `CERMetric`. The per-line mean is shown beside it because they diverge when line lengths vary.

## Per collection (worst first)

| collection | CER | WER | lines | ref chars | mean confidence |
| --- | --- | --- | --- | --- | --- |
| alvsborgs_losen | 0.2788 | 0.5074 | 91 | 1,797 | 0.720 |
| jonkopings_radhusratt_och_magistrat | 0.2679 | 0.7191 | 91 | 3,386 | 0.729 |
| gota_hovratt | 0.2350 | 0.6039 | 91 | 3,243 | 0.786 |
| bergmastaren_i_nora_htr | 0.2235 | 0.6433 | 91 | 2,756 | 0.767 |
| trolldomskommissionen | 0.2217 | 0.5737 | 90 | 3,302 | 0.791 |
| krigshovrattens_dombocker | 0.1656 | 0.4790 | 91 | 2,917 | 0.827 |
| goteborgs_poliskammare_fore_1900 | 0.1301 | 0.3931 | 91 | 2,445 | 0.885 |
| svea_hovratt | 0.1207 | 0.4044 | 91 | 2,593 | 0.892 |
| carl_fredrik_pahlmans_resejournaler | 0.1145 | 0.3801 | 91 | 3,973 | 0.892 |
| frihetstidens_utskottshandlingar | 0.0949 | 0.3537 | 91 | 2,412 | 0.885 |
| bergskollegium_relationer_och_skrivelser | 0.0582 | 0.2277 | 91 | 2,730 | 0.949 |

## Confidence calibration

| confidence bucket | lines | mean CER | median CER |
| --- | --- | --- | --- |
| 0.0 - 0.20 | 13 | 0.6313 | 0.6494 |
| 0.2 - 0.40 | 20 | 0.5219 | 0.4653 |
| 0.4 - 0.60 | 44 | 0.4133 | 0.3984 |
| 0.6 - 0.80 | 231 | 0.2749 | 0.2647 |
| 0.8 - 1.01 | 692 | 0.1051 | 0.0889 |

### Rejection coverage

| threshold | coverage | accepted lines | CER of accepted |
| --- | --- | --- | --- |
| 0.00 | 100.0% | 1000 | 0.1731 |
| 0.50 | 95.5% | 955 | 0.1555 |
| 0.70 | 86.1% | 861 | 0.1342 |
| 0.80 | 69.2% | 692 | 0.1051 |
| 0.90 | 41.6% | 416 | 0.0689 |
| 0.95 | 19.1% | 191 | 0.0506 |

## Inspection sample

Stratified: the same number of lines from every collection, selected by a stable hash seeded on the checkpoint identity, so this sample is reproducible and comparable across laps.

**alvsborgs_losen** -- CER 0.235, confidence 0.941

- truth: `pennr 73 mkr 6 ör`
- pred:  `penndr — 73 mk 6 ör`

**alvsborgs_losen** -- CER 0.371, confidence 0.651

- truth: `Säthe gård är Jolstada öd og brett.`
- pred:  `Oöthe gårdz är f olstade ök och be 44.`

**alvsborgs_losen** -- CER 0.077, confidence 0.886

- truth: `Sölff — 3 Lod`
- pred:  `Solff — 3 Lod`

**bergmastaren_i_nora_htr** -- CER 0.426, confidence 0.576

- truth: `skall bortstuulit 4 stn Bleck och bortskiänckt,`
- pred:  `skull bortsfrullit 4 st: 3 ud och beteyckuendt`

**bergmastaren_i_nora_htr** -- CER 0.312, confidence 0.887

- truth: `Anders Andersson`
- pred:  `anders andernen`

**bergmastaren_i_nora_htr** -- CER 0.100, confidence 0.912

- truth: `öhrfihlar af Patrone`
- pred:  `åhrfihlar af Patron`

**bergskollegium_relationer_och_skrivelser** -- CER 0.179, confidence 0.877

- truth: `mera C. giör migliare i Ven:`
- pred:  `mera C: giör Migliare. ven:`

**bergskollegium_relationer_och_skrivelser** -- CER 0.065, confidence 0.957

- truth: `Sålunda warder med Glantzertzen`
- pred:  `Sålunda warder med glantzerten`

**bergskollegium_relationer_och_skrivelser** -- CER 0.030, confidence 0.981

- truth: `Alla desse orter äro malmslag och`
- pred:  `Alla desse orter äro Malmslag och`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.154, confidence 0.913

- truth: `Redan på avstånd hörer man dånet af den nedstörtande`
- pred:  `Sedan på afståned hörer man danit af det ned stortande`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.023, confidence 0.972

- truth: `egen wilja. Calaset kostade honom emellertid`
- pred:  `egen wilja. Calaset kostade honom emedlertid`

**carl_fredrik_pahlmans_resejournaler** -- CER 0.244, confidence 0.772

- truth: `så lämna de den med en högmögen sjelfsvåldig¬`
- pred:  `så limna de den söd en kögnogen sjulfvätdig`

**frihetstidens_utskottshandlingar** -- CER 0.027, confidence 0.971

- truth: `mänhetens uplysning, Kan hafwa nödigt`
- pred:  `mänhetens uplysning, kan hafwa nödigt`

**frihetstidens_utskottshandlingar** -- CER 0.300, confidence 0.865

- truth: `ett ganska`
- pred:  `ett ganffra`

**frihetstidens_utskottshandlingar** -- CER 0.500, confidence 0.731

- truth: `digheter`
- pred:  `iaigherer,`

**gota_hovratt** -- CER 0.286, confidence 0.799

- truth: `att bloden begynt rinna, bediandes doch nu`
- pred:  `ärs bloden oggit nina, bediandes, doch er`

**gota_hovratt** -- CER 0.225, confidence 0.887

- truth: `på högra Kinbenet, et på näsan öfr högra`
- pred:  `på högra kinkenet, et på nåsan, och förra`

**gota_hovratt** -- CER 0.250, confidence 0.769

- truth: `Kongl. Rätten lätt sigh och föreläsa om `
- pred:  `Kongl. Rätte litl sigt och föarlicha om`

**goteborgs_poliskammare_fore_1900** -- CER 0.000, confidence 0.995

- truth: `samt`
- pred:  `samt`

**goteborgs_poliskammare_fore_1900** -- CER 0.091, confidence 0.849

- truth: `E Hellström`
- pred:  `E Hallström`

**goteborgs_poliskammare_fore_1900** -- CER 0.061, confidence 0.875

- truth: `ka natten till den 28/3 härstädes`
- pred:  `ka natten till den 2/g härstädes`

**jonkopings_radhusratt_och_magistrat** -- CER 0.161, confidence 0.892

- truth: `och giorde Aldrig Antingen bott`
- pred:  `och giore Aldin Antingan bolt`

**jonkopings_radhusratt_och_magistrat** -- CER 0.097, confidence 0.887

- truth: `strommen, och huar något sätter`
- pred:  `stromne, och huar någoe sätter`

**jonkopings_radhusratt_och_magistrat** -- CER 0.207, confidence 0.784

- truth: `såter och Tienere här medh förstå Att thenne breffwiiisere`
- pred:  `skner och Zienere har medh förstå An henne breffwörsen`

**krigshovrattens_dombocker** -- CER 0.000, confidence 0.970

- truth: `tagit wärfning; hwar¬`
- pred:  `tagit wärfning; hwar¬`

**krigshovrattens_dombocker** -- CER 0.000, confidence 0.927

- truth: `Sterbhuuß wägnar emoth Kong. Artollerie`
- pred:  `Sterbhuuß wägnar emoth Kong. Artollerie`

**krigshovrattens_dombocker** -- CER 0.027, confidence 0.898

- truth: `honom redan aflagde wittnes Ed, hwars`
- pred:  `honom redan aflagde rittnes Ed, hwars`

**svea_hovratt** -- CER 0.167, confidence 0.856

- truth: `Then 4 Julii`
- pred:  `Ten 4 Julli`

**svea_hovratt** -- CER 0.062, confidence 0.973

- truth: `penningepåster, som wid en efter`
- pred:  `penninge påster, som wid en efter¬`

**svea_hovratt** -- CER 0.000, confidence 0.938

- truth: `Nils Steenman.`
- pred:  `Nils Steenman.`

**trolldomskommissionen** -- CER 0.081, confidence 0.961

- truth: `digheter, samt efter den anledning 11`
- pred:  `digheter, samt etter den anlednng 1,`

**trolldomskommissionen** -- CER 0.107, confidence 0.780

- truth: `3 Cr Smt Testor Joh. Drachez`
- pred:  `3 Er Sml Testor Joh. Drache`

**trolldomskommissionen** -- CER 0.438, confidence 0.805

- truth: `Ny Sockes Kyrckan oct icke sof om natten, varan¬`
- pred:  `Aysoches Figritzan och uteso fomnatten, warar`

## Comparison with prior reference points

```json
{
  "pilot_best_val_cer": 0.1691,
  "pilot_best_val_cer_note": "Best validation CER reached by the 9,999-line pilot training pass. NOT a corpus lap: the pilot repeatedly revisited one fixed 9,999-line subset, so it is a different quantity, shown for orientation rather than as a like-for-like target.",
  "pilot_single_pass_cer": 0.2481,
  "pilot_single_pass_cer_note": "CER after a single 9,999-line pilot training pass -- roughly comparable in optimizer steps to one full-corpus shard, not to a lap.",
  "checkpoint_kind_evaluated": "end_of_epoch",
  "checkpoint_recorded_at_shard": 57,
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
  "inference_seconds": 111.2
}
```
