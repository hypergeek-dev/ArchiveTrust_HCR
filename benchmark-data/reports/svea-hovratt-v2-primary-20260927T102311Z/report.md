# HTR benchmark: lion vs loghi on `svea-hovratt-2026-09-primary-v2`

**Status: OFFICIAL.** Pre-adaptation results: the benchmark was frozen before any model output was seen.

## Benchmark identity

- Dataset `svea-hovratt-2026-09`, 4627 lines, 105 pages, 4 documents, 166201 reference characters; crop policy ['bbox_v1'].
- Manifest SHA-256 `e43ee895851036f47e92801ce9aa5ca102c22a1f67d8c15225d41261c5d5b8e1`; frozen 2026-09-27T09:35:38Z.
- Excluded before freeze: 5930 by recorded decision, 0 unresolved at freeze.
- GT normalization `gt-normalization/1` (rules: unicode_nfc, strip_file_line_terminator, human_correction); predictions: unicode_nfc, strip_outer_whitespace. Nothing else.

## Models

- **lion**: `riksarkivet-swedish-lion-libre`, decoding profile `generation_config` {"do_sample": false, "early_stopping": true, "length_penalty": 2.0, "max_length": 256, "no_repeat_ngram_size": 3, "num_beams": 4}; predictions `aef14754070f8ff0...`, status {'ok': 4627}, OFFICIAL.
- **loghi**: `loghi-swedish-scratch-exp2-epoch7`, decoding profile `validated` {"batch_size": 16, "beam_width": 10, "greedy": false, "seed": 42}; predictions `9aed0d6e2d3bcd73...`, status {'empty': 71, 'ok': 4556}, OFFICIAL.

## Headline (corpus-level; every line counted, failures scored as empty output)

Confidence intervals: 95% percentile cluster bootstrap over 105 pages, 2000 resamples, seed 20260927.

| Model | CER | 95% CI | WER | 95% CI | CER (ws-norm) | Exact lines | S / I / D chars | Failed+missing | Empty |
|---|---|---|---|---|---|---|---|---|---|
| lion | 18.42% | [17.29%, 19.65%] | 43.63% | [42.25%, 45.07%] | 18.41% | 7.33% | 17926 / 3220 / 9468 | 0 | 0 |
| loghi | 32.70% | [31.16%, 34.25%] | 67.53% | [65.95%, 69.07%] | 32.71% | 0.89% | 21080 / 762 / 32506 | 0 | 71 |

Sensitivity score (pre-registered; **not** the primary result): line-end hyphen harmonized (a line-final `¬` counts as `-` on both sides; nothing else changed).

| Model | CER raw | CER line-end hyphen harmonized | WER raw | WER line-end hyphen harmonized |
|---|---|---|---|---|
| lion | 18.42% | 18.23% | 43.63% | 42.98% |
| loghi | 32.70% | 32.56% | 67.53% | 67.29% |

**Paired difference (lion minus loghi)**: CER 95% CI [-15.25%, -13.30%], WER 95% CI [-25.28%, -22.60%]. An interval that contains 0 means the benchmark does not separate the models at this size.

## Line-level comparison

| Category | Lines |
|---|---|
| both_correct | 25 |
| both_wrong_lion_better | 3576 |
| both_wrong_loghi_better | 371 |
| both_wrong_equal_edits | 287 |
| both_wrong_identically | 38 |
| only_lion_correct | 314 |
| only_loghi_correct | 16 |

Catastrophic lines (failed/missing or line CER >= 50%): {'lion': 481, 'loghi': 982, 'both': 442}.

### Error types (character edits)

| Tag | lion | loghi |
|---|---|---|
| abbreviation_word | 2116 | 3429 |
| case | 832 | 647 |
| diacritic | 277 | 432 |
| digit | 707 | 1193 |
| letter | 19824 | 38863 |
| punctuation | 3808 | 4003 |
| spacing | 2474 | 4919 |
| special_character | 576 | 862 |

Top confusions, lion: ` →∅` x981, `e→∅` x819, `s→∅` x731, `.→∅` x703, `a→∅` x563, `.→:` x557, `a→e` x541, `l→∅` x504, `n→∅` x469, `r→∅` x454, `∅→i` x428, `y→j` x348, `i→∅` x347, `o→∅` x333, `d→∅` x326

Top confusions, loghi: ` →∅` x3744, `e→∅` x2978, `a→∅` x2014, `s→∅` x1829, `n→∅` x1814, `r→∅` x1724, `l→∅` x1666, `t→∅` x1649, `i→∅` x1402, `.→∅` x1346, `d→∅` x1189, `o→∅` x1131, `a→e` x1024, `h→∅` x905, `f→∅` x813

### Largest disagreements: lion_much_better (top 10 of 30 in scores.json)

- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0007_DSC_0440/line_1638889589548_2023` REF `att intächten effter giord förwandling måtte skie på det` | lion `att intächten effter giord förwandling warit, kiöpa uti` | loghi `b` | lion_edits `12` | loghi_edits `56`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0008_DSC_0456/r1l13` REF `hoos Probsten att undergå förhör och åthwarning tillbät-` | lion `hoos Probsten att undergå förgds och utewarning till båt` | loghi `hågn wällarnglabe` | lion_edits `8` | loghi_edits `47`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0013_DSC_0446/r2l30` REF `Officerarnes ordres hålla Husesyn, men enär der till kommit så finns bra nog.` | lion `Officerarnes ordres hålla husesyn, men enär der till Knut så fins öre nog` | loghi `firarus onusfåll fisammäretill Kntsifusbma` | lion_edits `10` | loghi_edits `48`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0007_DSC_0440/line_1638889589533_2015` REF `wara rätt underrättade om denne Proriant Skiutz dhe` | lion `förra rätt underrättade om denne proviant skiutz hu¬` | loghi `leletim d` | lion_edits `9` | loghi_edits `45`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0003_DSC_0436/r1l46` REF `effter sluten Rächning och undersatte bomärcke d. 2. Febr.` | lion `effter sluten Räckning och undersatte bomännen d. 2 febr.` | loghi `sån Rnlingoch annat banh` | lion_edits `7` | loghi_edits `42`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0003_DSC_0451/r1l17` REF `baka å wästra sydan af åkerhagarna af Pehr Joenson` | lion `laga å wästra sidan af åkerhagarna af the Jonsse` | loghi `ålla re smölesgnamå` | lion_edits `9` | loghi_edits `41`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0005_DSC_0453/r1l12` REF `Hansdotters arf och Joen Hemmingssons fordran till 32 Daler.` | lion `hans dotters arf och Joen Hemmingzons föräldt hafwa` | loghi `n hamn wam` | lion_edits `21` | loghi_edits `53`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0002_DSC_0450/r1l2` REF `näst boende tillhållet, då märeta tillstår hafft kundskap` | lion `näst boende tillhållet, då märta tillstör haf:r` | loghi `inte lisme uti hafln` | lion_edits `13` | loghi_edits `44`
- `Svea_Hovraett_etc_export_job_4502443_858057_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-f577f7f3/0003_DSC_0451/r1l14` REF `af Erich Ehrsson och Elias Swensson, öfwer så` | lion `af Erich Eliasson och Elia Swensson än welat` | loghi `en` | lion_edits `12` | loghi_edits `43`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0007_DSC_0440/line_1638889589548_2017` REF `Ländzman effter Rättens resolution å tinget in Decemb.` | lion `Ländzman efter Rättens resolution å tinget in Humb` | loghi `ir otocs tesklatten i tillagt ideas` | lion_edits `6` | loghi_edits `37`

### Largest disagreements: loghi_much_better (top 10 of 30 in scores.json)

- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0011_DSC_0383/line_1638269466359_17` REF `till ångermanland, och sedan aldrig` | lion `sig bekommit. Iembandes Borge` | loghi `iigemelend, och seden aldag` | lion_edits `28` | loghi_edits `13`
- `Svea_Hovraett_etc_export_job_4502446_842250_Svea_Hovraett_-_Advokatfiskalens_Jaemtlands_laen__1702__-7a829e7f/0024_DSC_0358/r2l6` REF `genheet, nekade nu der till, och eme-` | lion `till bemälte hafans att wijd samma` | loghi `et, nekade n der till n` | lion_edits `30` | loghi_edits `15`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0024_DSC_0401/r2l5` REF `blandelse med honom fry, och så-` | lion `... och så¬` | loghi `ahr med honom tag, och så` | lion_edits `25` | loghi_edits `12`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0039_DSC_0421/r2l7` REF `lofwat hoo. tiäna, allenast Mo-` | lion `fördt htt samtyckia och dyrkhu¬` | loghi `git hoo tienn amat Mo¬` | lion_edits `28` | loghi_edits `15`
- `Svea_Hovraett_etc_export_job_4502444_855910_Svea_Hovraett_OCo_Advokatfiskalen_Jaemtlands_laen__1713_-b58e9491/0009_DSC_0442/line_1638890141661_2347` REF `13.` | lion `Såsom Sigrid Markusdotter tillslädes kommen examinerande` | loghi `Eu Sgud markusdot tillsads kommen examnarali` | lion_edits `56` | loghi_edits `44`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0029_DSC_0407/r1l6` REF `ka alla tillyka uti denne gierning` | lion `i alla tillwärk för slikt giöra` | loghi `te alla tillyka uti denne gen¬` | lion_edits `19` | loghi_edits `7`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0018_DSC_0392/r1l1` REF `skillige giäldenärne här i tingelaget` | lion `beswärde` | loghi `fingeGådmären hig i tage` | lion_edits `33` | loghi_edits `22`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0026_DSC_0404/r2l3` REF `Yttermehra confronterades Drag. Broman` | lion `Swarandes det Pähr Olßon för honom` | loghi `hr cefentrade bagkon` | lion_edits `33` | loghi_edits `22`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0034_DSC_0413/r2l6` REF `till Swedie, hwilken Bonde Capitein` | lion `sa kiort, som Bonden` | loghi `oden, hwilken wande Capten` | lion_edits `25` | loghi_edits `14`
- `Svea_Hovraett_etc_export_job_4502445_845640_Svea_Hovraett_-_Advokatfiskalen_Jaemtlands_laen__1702__3-6edbb12a/0041_DSC_0423/r3l2` REF `Bostället underlagdt, ey heller är det` | lion `Öfwerlaplikt, blef med stånden` | loghi `Hnderlagdt eykalle udet` | lion_edits `30` | loghi_edits `19`

## Caveats that bound these numbers

- Loghi's reference validation CER (9.98%) was measured in-distribution; it is not comparable to this external benchmark.
- Both models were trained on Riksarkivet's public HF line collections. Overlap between this benchmark and that material must be checked with the `overlap` command; no detected overlap is not proof of none.
- Loghi cannot emit characters outside its 124-character set: 1 reference characters (0.00%) are out of its vocabulary: ¼.
- Loghi durations are batch-amortized; Lion's are per line. Timing is not a like-for-like comparison.
- Only Loghi reports a per-line confidence.
