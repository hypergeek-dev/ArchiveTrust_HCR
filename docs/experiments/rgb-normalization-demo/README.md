# RGB normalization stage — demonstration run

Real, reproducible output of the versioned `RgbNormalization` preprocessing stage
(`src/archivetrust/htr/preprocessing/`), produced by
`scripts/run_rgb_normalization_demo.py`.

Regenerate with:

```
QT_QPA_PLATFORM=offscreen PYTHONPATH=src .venv/Scripts/python.exe scripts/run_rgb_normalization_demo.py
```

## This is not the committed baseline, and does not touch it

`docs/experiments/baseline-comparison/` is a **separate** directory that this run neither reads nor
writes. That baseline predates this stage, and its Transkribus `MethodRun` had **no page image at
all** (`input_crop_id=None`; `tests/fixtures/transkribus/` contains only `.xml`/`.txt` files), so
there was nothing to normalize and **no normalization event was added to it retroactively**. That
absence is enforced as a test, not merely promised:
`tests/htr/preprocessing/test_baseline_history_not_fabricated.py`.

## What was run

| Page | Source | Source mode | Provenance |
|---|---|---|---|
| 1 | `trolldomskommissionen_sample_line.jpg` | `RGB` | **Real archival image.** One line from Riksarkivet's `trolldomskommissionen_lines` dataset (17th-century Swedish court records) — see `tests/fixtures/htr/README.md`. Used as a **stand-in page**: it is a line crop, not a full page. This repository contains no real full-page archival scan. |
| 2 | `synthetic_cmyk.tif` | `CMYK` | Synthesized in-process by `tests/htr/preprocessing/_images.py`. |
| 3 | `synthetic_i16.png` | `I;16` | Synthesized — 16-bit grayscale. |
| 4 | `synthetic_palette_transparency.png` | `P` + transparency | Synthesized — palette with a transparency index. |
| — | undecodable bytes | n/a | Synthesized — produces a real recorded failure. |

Pages 2–4 are synthetic because the real image cannot exercise the colour modes that matter most:
CMYK, 16-bit, and palette transparency are exactly where a naive conversion loses data, and no real
image in this repository is in any of them.

## Real telemetry emitted

Written to `normalization_events.jsonl` (with its hash-chain sidecar
`normalization_events.jsonl.chain.jsonl`) by a real `FileTelemetrySink`:

| Event kind | Count |
|---|---|
| `ImageNormalizationStarted` | 5 |
| `DerivedImageArtifactCreated` | 4 |
| `ImageNormalizationCompleted` | 4 |
| `ImageNormalizationFailed` | 1 |

Five starts and four completions is the point, not an inconsistency: the fifth attempt genuinely
failed (`undecodable_image`) and is preserved as evidence rather than swallowed. Every event carries
`correlation_id="rgb_normalization_demo_run_1"`, and each is `causation_id`-chained to the one before.

`demonstration_summary.json` records the full per-artifact provenance. Real hashes:

| Page | Original hash | Normalized hash |
|---|---|---|
| real line-as-page | `page_image_e59f301d…48261` | `normalized_page_e59f301d…48261` |
| CMYK | `page_image_770c9016…21b8` | `normalized_page_457c0ecb…b596` |
| 16-bit gray | `page_image_a7862c63…88ea` | `normalized_page_cd0812bf…f364` |
| palette+transparency | `page_image_b4689a0b…1770` | `normalized_page_fad584fe…0f1c` |

Configuration hash (all four):
`normalization_config_13abd1eef72e3a648fd5c1b9a165e28eeac0b5c37edfd246905fabdcd88f57e2`.

### An honest observation about page 1

The real image's original and normalized digests **are identical** (`e59f301d…48261` under both
prefixes). That is not a bug and not a skipped normalization — it is the documented
already-canonical case, and the demonstration is more useful for having hit it by accident than it
would be if every row differed.

Two real facts caused it, both worth recording:

1. **The file named `.jpg` is actually a PNG.** Its magic bytes are `\x89PNG`, and Pillow reports
   `format="PNG"`. `tests/fixtures/htr/README.md` says the fixture was "saved as-is (JPEG bytes)",
   which is inaccurate about this file. That is a pre-existing discrepancy in the fixture's
   documentation, discovered here and left alone rather than silently corrected — it is not this
   stage's artifact to change.
2. **It was already in this stage's exact canonical form**: 8-bit RGB, no alpha, no ICC profile, no
   EXIF, PNG. Re-encoding it therefore reproduced its bytes exactly.

The hashes remain distinct *strings* because their prefixes differ, so an original can never be
confused with a derived artifact in a lookup. An earlier revision of `NormalizedPageArtifact`
rejected equal digests outright; that check was wrong and was removed — see that model's
`_validate` for the reasoning, and
`tests/htr/preprocessing/test_idempotence.py::test_hashes_are_always_distinct_strings_even_when_the_bytes_coincide`.

The three synthetic pages all transformed genuinely, so the stage's real work is still demonstrated:
CMYK → RGB, 16-bit → linearly scaled 8-bit, and palette transparency → composited over white.

## Export package

`export-packages/export_package_rgb_normalization_demo/` contains the four normalized PNGs plus
`manifest.json`. The manifest records, per page: document/page identifiers, original hash, normalized
hash, output filename, dimensions, source colour mode, the normalization version and configuration
hash, the package identifier, and the creation timestamp.

The manifest also carries its own two caveats, so they travel with the package if it is handed to a
colleague:

- **`external_processing_boundary`** — ArchiveTrust performs no upload, holds no credential, and
  makes no network call. A researcher must upload these files to Transkribus themselves.
- **`correspondence_caveat`** — Transkribus does not preserve or return these hashes, so any later
  association is researcher-confirmed, not cryptographic proof.

## Import association

`demonstration_summary.json` records a real `ImportAssociation` with
`target_kind="normalized_page_artifact"` and `correspondence_basis="researcher_confirmed"`.

**No real Transkribus upload or export occurred.** The `ExternalImport` points at the hand-authored
PAGE XML test fixture (`tests/fixtures/transkribus/sample_page.xml`, itself not genuine Transkribus
output — see that directory's README). What is demonstrated is the association *mechanism* and its
honest epistemic labelling, not a real round trip through Transkribus, and the summary says so in its
own `note` field.

## Replay verification

The script deletes the store and service, replays `normalization_events.jsonl` through a brand-new
`FileTelemetrySink` and `HtrJournal`, and asserts the reconstructed `NormalizedPageArtifact`s equal
the in-process ones field for field. `"replay_verified": true` in the summary means that assertion
passed during generation.
