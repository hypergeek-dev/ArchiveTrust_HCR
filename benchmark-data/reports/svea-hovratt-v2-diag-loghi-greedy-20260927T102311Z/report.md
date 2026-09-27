# HTR benchmark: lion vs loghi on `svea-hovratt-2026-09-primary-v2`

**Status: UNOFFICIAL (not run with --official) -- not all stages were official runs.** Pre-adaptation results: the benchmark was frozen before any model output was seen.

## Benchmark identity

- Dataset `svea-hovratt-2026-09`, 4627 lines, 105 pages, 4 documents, 166201 reference characters; crop policy ['bbox_v1'].
- Manifest SHA-256 `e43ee895851036f47e92801ce9aa5ca102c22a1f67d8c15225d41261c5d5b8e1`; frozen 2026-09-27T09:35:38Z.
- Excluded before freeze: 5930 by recorded decision, 0 unresolved at freeze.
- GT normalization `gt-normalization/1` (rules: unicode_nfc, strip_file_line_terminator, human_correction); predictions: unicode_nfc, strip_outer_whitespace. Nothing else.

## Models

- **lion**: `riksarkivet-swedish-lion-libre`, decoding profile `generation_config` {"do_sample": false, "early_stopping": true, "length_penalty": 2.0, "max_length": 256, "no_repeat_ngram_size": 3, "num_beams": 4}; predictions `aef14754070f8ff0...`, status {'ok': 4627}, OFFICIAL.
- **loghi**: `loghi-swedish-scratch-exp2-epoch7`, decoding profile `greedy` {"batch_size": 16, "beam_width": 1, "greedy": true, "seed": 42}; predictions `f9f5e0f274f9ea05...`, status {'empty': 91, 'ok': 4536}, UNOFFICIAL (not run with --official).

## Headline (corpus-level; every line counted, failures scored as empty output)

Confidence intervals: 95% percentile cluster bootstrap over 105 pages, 2000 resamples, seed 20260927.

| Model | CER | 95% CI | WER | 95% CI | CER (ws-norm) | Exact lines | S / I / D chars | Failed+missing | Empty |
|---|---|---|---|---|---|---|---|---|---|
| lion | 18.42% | [17.29%, 19.65%] | 43.63% | [42.25%, 45.07%] | 18.41% | 7.33% | 17926 / 3220 / 9468 | 0 | 0 |
| loghi | 33.00% | [31.43%, 34.58%] | 67.61% | [66.05%, 69.11%] | 33.00% | 0.82% | 20973 / 809 / 33065 | 0 | 91 |

Sensitivity score (pre-registered; **not** the primary result): line-end hyphen harmonized (a line-final `¬` counts as `-` on both sides; nothing else changed).

| Model | CER raw | CER line-end hyphen harmonized | WER raw | WER line-end hyphen harmonized |
|---|---|---|---|---|
| lion | 18.42% | 18.23% | 43.63% | 42.98% |
| loghi | 33.00% | 32.86% | 67.61% | 67.37% |

**Paired difference (lion minus loghi)**: CER 95% CI [-15.56%, -13.58%], WER 95% CI [-25.32%, -22.69%]. An interval that contains 0 means the benchmark does not separate the models at this size.

## Line-level comparison

| Category | Lines |
|---|---|
| both_correct | 26 |
| both_wrong_lion_better | 3603 |
| both_wrong_loghi_better | 357 |
| both_wrong_equal_edits | 280 |
| both_wrong_identically | 36 |
| only_lion_correct | 313 |
| only_loghi_correct | 12 |

Catastrophic lines (failed/missing or line CER >= 50%): {'lion': 481, 'loghi': 984, 'both': 442}.

### Error types (character edits)

| Tag | lion | loghi |
|---|---|---|
| abbreviation_word | 2116 | 3444 |
| case | 832 | 643 |
| diacritic | 277 | 431 |
| digit | 707 | 1194 |
| letter | 19824 | 39129 |
| punctuation | 3808 | 4007 |
| spacing | 2474 | 5137 |
| special_character | 576 | 862 |

Top confusions, lion: ` →∅` x981, `e→∅` x819, `s→∅` x731, `.→∅` x703, `a→∅` x563, `.→:` x557, `a→e` x541, `l→∅` x504, `n→∅` x469, `r→∅` x454, `∅→i` x428, `y→j` x348, `i→∅` x347, `o→∅` x333, `d→∅` x326

Top confusions, loghi: ` →∅` x3952, `e→∅` x2991, `a→∅` x2030, `n→∅` x1895, `s→∅` x1841, `r→∅` x1743, `t→∅` x1704, `l→∅` x1685, `i→∅` x1420, `.→∅` x1371, `d→∅` x1197, `o→∅` x1142, `a→e` x1016, `h→∅` x922, `f→∅` x830

### Largest disagreements: lion_much_better (top 10 of 30 in scores.json)

- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0007_DSC_0440/line_1638889589548_2023` REF `att intächten effter giord förwandling måtte skie på det` | lion `att intächten effter giord förwandling warit, kiöpa uti` | loghi `b` | lion_edits `12` | loghi_edits `56`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0008_DSC_0456/r1l13` REF `hoos Probsten att undergå förhör och åthwarning tillbät-` | lion `hoos Probsten att undergå förgds och utewarning till båt` | loghi `håg  wällarnglabe` | lion_edits `8` | loghi_edits `46`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0007_DSC_0440/line_1638889589533_2015` REF `wara rätt underrättade om denne Proriant Skiutz dhe` | lion `förra rätt underrättade om denne proviant skiutz hu¬` | loghi `eltfimd` | lion_edits `9` | loghi_edits `47`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0013_DSC_0446/r2l30` REF `Officerarnes ordres hålla Husesyn, men enär der till kommit så finns bra nog.` | lion `Officerarnes ordres hålla husesyn, men enär der till Knut så fins öre nog` | loghi `fiurarus onus fåll fisammäretill Kntsifusbma` | lion_edits `10` | loghi_edits `47`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0003_DSC_0436/r1l46` REF `effter sluten Rächning och undersatte bomärcke d. 2. Febr.` | lion `effter sluten Räckning och undersatte bomännen d. 2 febr.` | loghi `sån Rnlingoch annat banh` | lion_edits `7` | loghi_edits `42`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0005_DSC_0453/r1l12` REF `Hansdotters arf och Joen Hemmingssons fordran till 32 Daler.` | lion `hans dotters arf och Joen Hemmingzons föräldt hafwa` | loghi `hamn wam` | lion_edits `21` | loghi_edits `55`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0003_DSC_0451/r1l14` REF `af Erich Ehrsson och Elias Swensson, öfwer så` | lion `af Erich Eliasson och Elia Swensson än welat` | loghi `` | lion_edits `12` | loghi_edits `45`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0003_DSC_0451/r1l20` REF `finnes det Pehr Joensson slagit 2.ne åhr in uppå` | lion `finnes det Pehr Joensson slagit 2ne inne.` | loghi `sagt S` | lion_edits `10` | loghi_edits `43`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0003_DSC_0451/r1l17` REF `baka å wästra sydan af åkerhagarna af Pehr Joenson` | lion `laga å wästra sidan af åkerhagarna af the Jonsse` | loghi `ålla re smölesgnamå` | lion_edits `9` | loghi_edits `41`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0007_DSC_0440/line_1638889589548_2017` REF `Ländzman effter Rättens resolution å tinget in Decemb.` | lion `Ländzman efter Rättens resolution å tinget in Humb` | loghi `r oos tesklatten i tillagt ideas` | lion_edits `6` | loghi_edits `38`

### Largest disagreements: loghi_much_better (top 10 of 30 in scores.json)

- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0011_DSC_0383/line_1638269466359_17` REF `till ångermanland, och sedan aldrig` | lion `sig bekommit. Iembandes Borge` | loghi `iigemelend, och seden aldag` | lion_edits `28` | loghi_edits `13`
- `Svea_Hovraett_etc_export_job_4502446_842250_Svea_Hovraett_-_Advokatfiskalens_Jaemtlands_laen__1702__-7a829e7f/0024_DSC_0358/r2l6` REF `genheet, nekade nu der till, och eme-` | lion `till bemälte hafans att wijd samma` | loghi `et, nekade n der till n` | lion_edits `30` | loghi_edits `15`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0024_DSC_0401/r2l5` REF `blandelse med honom fry, och så-` | lion `... och så¬` | loghi `ahr med honom tag, och så` | lion_edits `25` | loghi_edits `12`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0009_DSC_0442/line_1638890141661_2347` REF `13.` | lion `Såsom Sigrid Markusdotter tillslädes kommen examinerande` | loghi `Eu Sgud markusdot tillsads kommen examnarali` | lion_edits `56` | loghi_edits `44`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0018_DSC_0392/r1l1` REF `skillige giäldenärne här i tingelaget` | lion `beswärde` | loghi `fingeGådmnären hig i tage` | lion_edits `33` | loghi_edits `21`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0029_DSC_0407/r1l6` REF `ka alla tillyka uti denne gierning` | lion `i alla tillwärk för slikt giöra` | loghi `te alla tillyka uti denne gen¬` | lion_edits `19` | loghi_edits `7`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0034_DSC_0413/r2l6` REF `till Swedie, hwilken Bonde Capitein` | lion `sa kiort, som Bonden` | loghi `oden, hwilken wande Capten` | lion_edits `25` | loghi_edits `14`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0039_DSC_0421/r2l7` REF `lofwat hoo. tiäna, allenast Mo-` | lion `fördt htt samtyckia och dyrkhu¬` | loghi `git hoo tiennwmat Mo¬` | lion_edits `28` | loghi_edits `17`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0041_DSC_0423/r3l2` REF `Bostället underlagdt, ey heller är det` | lion `Öfwerlaplikt, blef med stånden` | loghi `Hnderlagdt eykalle udet` | lion_edits `30` | loghi_edits `19`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0044_DSC_0426/r2l10` REF `hwar med han honom instämdt,` | lion `har med han begifwit, och dy saker` | loghi `ed hon honom instänkde,` | lion_edits `21` | loghi_edits `10`

## Caveats that bound these numbers

- Loghi's reference validation CER (9.98%) was measured in-distribution; it is not comparable to this external benchmark.
- Both models were trained on Riksarkivet's public HF line collections. Overlap between this benchmark and that material must be checked with the `overlap` command; no detected overlap is not proof of none.
- Loghi cannot emit characters outside its 124-character set: 1 reference characters (0.00%) are out of its vocabulary: ¼.
- Loghi durations are batch-amortized; Lion's are per line. Timing is not a like-for-like comparison.
- Only Loghi reports a per-line confidence.
