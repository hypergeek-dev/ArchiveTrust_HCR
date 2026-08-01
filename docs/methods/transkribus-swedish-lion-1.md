# Transkribus Swedish Lion I — page-level workflow

First file in `docs/methods/`. Documents the page-level, externally-processed Swedish Lion I workflow
and the versioned RGB-normalization stage it depends on.

Related: `src/archivetrust/providers/transkribus/README.md` (the manual-import adapter),
`docs/htr-domain-design.md` §7 (pipeline stages as independent concerns),
`docs/architecture/htr-telemetry.md` §9 (this stage's event kinds),
`docs/experiments/rgb-normalization-demo/` (a real run).

> **2026-08-01 clarification, not a rewrite of the content below.** A second, unrelated adapter,
> `providers/swedish_lion/` (`method_id=swedish_lion`, "Swedish Lion **Libre**"), was added after this
> document was written. It is a real, local TrOCR model — no export, no manual upload, no PAGE/ALTO
> import — and is registered under its own `method_id` because its Hugging Face model card states no
> confirmed lineage to this method's Transkribus branding. Everything below this note describes
> **this** method (`transkribus_swedish_lion_1`), which remains exactly as documented: external,
> manual-import-only, unmodified by the Lion-vs-Loghi research phase. `transkribus_swedish_lion_1`'s
> research status in that phase is `inactive` (never part of the retired benchmark, simply out of
> scope) — `swedish_lion` is the method that is `active`. See `docs/loghi-integration-audit.md` §0 and
> `docs/CAPABILITY_MATRIX_HTR.md` §6 for the full distinction.

## 1. The workflow

```
select source pages
  → RgbNormalization stage (versioned, content-addressed, telemetry-backed)
    → export package + manifest written to local disk
      ┄┄┄┄ ArchiveTrust's control ends here ┄┄┄┄
      → researcher manually uploads to Transkribus, runs Swedish Lion I, exports the result
      ┄┄┄┄ ArchiveTrust's control resumes here ┄┄┄┄
    → manual import of the PAGE/ALTO/plain-text result
  → researcher-confirmed association back to the normalized page artifact
```

**ArchiveTrust never uploads anything.** No socket, no credential, no code path to Transkribus exists
— enforced by adapter and module design, not by a runtime check. See
`providers/transkribus/README.md` for the same guarantee on the import side and
`htr/preprocessing/export_package.py` for the export side.

## 2. Why the normalization stage exists

Swedish Lion I is consumed as a page-level, external service. What is handed to it is the actual
experimental input, so its colour representation is an experimental variable. Left implicit, that
variable is decided by whatever the source files happened to carry — a CMYK print scan, a 16-bit
archival TIFF, a palette PNG with transparent margins — and by whichever library default happened to
convert it. Two runs could then differ in their input representation with nothing recording that they
did.

The stage makes that representation a **stated, versioned, hashed fact**. It is an independent
pipeline stage in exactly the sense §7 of the domain design establishes for segmentation: named,
versioned, content-addressed, and owned by the pipeline rather than by whichever method needed it
first.

## 3. What the stage guarantees

Output is exactly: **RGB, 3 channels, R-G-B order, 8 bits per channel**, no alpha, no embedded ICC
profile, no EXIF/XMP metadata, lossless PNG, and pixel geometry unchanged except for a physically
applied EXIF orientation.

Supported inputs, each with a passing test in `tests/htr/preprocessing/test_input_modes.py`: bilevel
(`1`), grayscale (`L`), grayscale+alpha (`LA`), palette (`P`), palette+transparency, CMYK, RGB, RGBA,
16-bit grayscale (`I`/`I;16`).

### No enhancement — checked, not promised

No contrast adjustment, autocontrast or level stretching, sharpening, denoising, thresholding or
binarization, resizing, deskewing, cropping, padding, colour balancing, gamma adjustment, or lossy
compression.

This is machine-checked. `rgb_normalization.py` declares `ALLOWED_PILLOW_OPERATIONS`, and
`tests/htr/preprocessing/test_no_enhancement_contract.py` parses the module's AST and fails if it
touches any Pillow attribute outside that allowlist — and separately fails if the allowlist itself is
widened to include a forbidden operation. The one pixel-changing function is length-capped so it stays
verifiable by eye, and `RgbNormalizationConfig` structurally refuses `enhancement != "none"`, a lossy
`output_format`, and `preserve_dimensions=False`.

### The two operations that do change pixel values

Both are required by the target representation, not aesthetic choices:

1. **Alpha compositing.** RGB has no alpha channel, so transparency must resolve against a background.
2. **16-bit → 8-bit reduction.** Done by explicit linear scaling (`value × 255/65535`), **never** by
   Pillow's `convert("L")`, which was verified on Pillow 10.4.0 to *clip*: samples 4096, 32768 and
   65535 all became 255, turning most of a 16-bit scan pure white. The regression test asserts the
   upstream clipping behaviour directly, so it documents the trap rather than a remembered claim.

## 4. Configuration and policy decisions

`RgbNormalizationConfig` (frozen, content-addressed via `configuration_hash`):

| Field | Default |
|---|---|
| `stage` | `"image_color_normalization"` |
| `version` | `"1.0.0"` (config schema) |
| `target_mode` | `"RGB"` |
| `bits_per_channel` | `8` |
| `channel_count` / `channel_order` | `3` / `"RGB"` |
| `output_format` | `"PNG"` (lossless only) |
| `alpha_policy` | `WHITE_BACKGROUND` |
| `icc_profile_policy` | `RECORD_AND_STRIP_WITHOUT_APPLYING` |
| `apply_exif_orientation` | `True` |
| `strip_metadata` | `True` |
| `preserve_dimensions` | `True` |
| `enhancement` | `"none"` |

`RGB_NORMALIZATION_VERSION` (`"1.0.0"`) versions the *implementation* separately from the config
schema; both are recorded on every artifact and both feed the pipeline configuration.

### Alpha compositing: white background

**No prior archival-image colour or compositing policy exists in this repository.**
`docs/DATA_HANDLING_POLICY.md` was checked in full and governs licensing, PII, telemetry-as-document
data, and release requirements — it says nothing about image colour handling, alpha, or bit depth.
Neither does `docs/SECURITY_AND_DATA_HANDLING.md` or `docs/EVALUATION_PROTOCOL.md`. So this is a new
decision, not a departure from one, and it is recorded in `DATA_HANDLING_POLICY.md` §7.

White is chosen because historical manuscript pages are dark ink on light support: transparent
regions in archival derivatives are overwhelmingly padding, removed borders, or masked areas, and
compositing those to black would introduce large black regions that no source page had — which an
HTR model may read as content. White is continuous with the paper.

It is a *policy value*, not a constant: `AlphaCompositingPolicy.BLACK_BACKGROUND` exists, is tested,
and changing it changes the configuration hash and therefore forces a new experiment version.

### ICC profiles: recorded, not applied, then stripped

A reasoned departure from the two options originally posed ("strip after applying" or
"apply-then-record"), both of which *apply* the profile:

1. **Applying is an enhancement.** An ICC transform rewrites pixel values to change appearance. This
   stage guarantees no colour balancing; applying a profile would breach that guarantee while
   claiming to uphold it.
2. **Applying is not deterministic in the required sense.** Pillow's ICC transform runs through
   LittleCMS, whose output depends on the linked liblcms version and rendering-intent implementation.
   Identical input could produce different output bytes on two machines, and determinism is a stated
   requirement.

Nothing is lost: presence, byte size, SHA-256, and the profile's declared description are all recorded
on the artifact before the profile is stripped, so the colour intent remains recoverable evidence.
Because the policy is a versioned enum, a future version may legitimately choose to apply — at a
different configuration hash, hence a different pipeline configuration.

### EXIF orientation

Physically applied first, then the metadata is normalized away, so stored pixels are the pixels a
reader sees and no downstream consumer can rotate a second time. Orientations 5–8 legitimately swap
width and height; that is the single sanctioned geometry change and is flagged on the artifact as
`dimensions_changed_by_orientation` so a dimension difference is never unexplained. A missing tag is
recorded as `None`, never as "orientation 1 assumed".

### Determinism and idempotence

`normalize(normalize(x)) == normalize(x)` by output content hash, asserted literally for every
supported input mode. Encoder parameters are pinned (`optimize=False`, `compress_level=6`) so a future
Pillow default cannot silently change output hashes.

An already-conforming image is still re-encoded to canonical form, and for a source that *already is*
the canonical form this produces byte-identical output — so its digest coincides with the source's.
That is correct and is not an error; the hashes remain distinct strings because their prefixes differ.
An earlier revision rejected equal digests and was wrong to. See the demonstration run, where the real
archival fixture hit exactly this case.

## 5. Failure handling

Failure is **recorded, then raised** — never swallowed, and there is no fallback that submits the
un-normalized original. `NormalizationFailureCategory`: `undecodable_image`,
`rgb_output_not_producible`, `invalid_output_dimensions`, `encode_failed`, `hash_not_computable`,
`artifact_not_persisted`, `manifest_not_written`.

`ImageNormalizationStarted` is appended **before** the transform runs, so a failed attempt still
leaves its source image identified in the durable log. A failing page aborts the whole export package
before any manifest is written, so an un-normalized original cannot reach a package directory — a
structural property, not a check a caller might forget.

## 6. Pipeline configuration and versioning

`ImageColorNormalizationRequirement` is a field on `BaselineExperimentDefinition`, so it lands inside
`ExperimentVersion.pipeline_configuration_ref` through the **existing** deterministic `to_ref()`.
No parallel mechanism was introduced.

That is what forces a new version: `to_ref()` is a sorted JSON dump including `configuration_hash` and
`version`, so changing the normalization configuration necessarily changes
`pipeline_configuration_ref`. Two runs with different normalization configs cannot share one, and
`assert_experiment_mutable` already refuses to edit an `ExperimentVersion` that has a run — so
recording a changed configuration requires a new `ExperimentVersion`. Proven in
`tests/htr/preprocessing/test_experiment_versioning.py`.

`swedish_lion_1_page_level_definition()` builds the page-level configuration with
`required=True`. It is a *separate* definition from `default_baseline_definition()` deliberately —
see §7.

## 7. The pre-existing baseline: no fabricated history

`docs/experiments/baseline-comparison/` predates this stage. Its Transkribus `MethodRun` records
`input_crop_id=None`, and `tests/fixtures/transkribus/` contains only `.xml`/`.txt` — **there was no
raw page image at all.** It was a PAGE XML text import, not an image workflow.

Therefore: **no normalization was performed for that run, no normalization event was recorded for it,
and none has been added retroactively.** `default_baseline_definition()` keeps
`image_color_normalization=None`, which is the truthful value.

Enforced by `tests/htr/preprocessing/test_baseline_history_not_fabricated.py`, which scans every
committed baseline telemetry file for the four normalization event kinds, replays the baseline through
`HtrJournal` and asserts zero preprocessing entities, and asserts the fixture directory still contains
no image.

The demonstration lives in its own directory, `docs/experiments/rgb-normalization-demo/`.

## 8. Florence-2's inline RGB conversion is a different thing

`providers/florence2_htr/facade.py` contains `Image.open(image_path).convert("RGB")`. That is **not**
this stage, and the distinction is now documented at that line:

- **Florence-2's conversion** is its own model input requirement (its processor demands a 3-channel
  tensor, and shared line crops are often grayscale). Unversioned and unrecorded because nothing
  downstream reasons about it: the converted pixels live only inside one worker call and are never
  stored, hashed, or handed to anyone.
- **This stage** is versioned, content-addressed, telemetry-backed, and produces a durable artifact,
  for *page* images leaving ArchiveTrust for external processing.

Florence-2 was deliberately **not** refactored to use the stage. The stage is page-granular and emits
a stored artifact plus page-scoped telemetry per call; routing every line crop of every method run
through it would emit a derived artifact per crop per run and would silently change the bytes
Florence-2 sees relative to the committed baseline (the stage composites alpha over white and
re-encodes to canonical PNG), invalidating that baseline's recorded results. The stage must not be
applied retroactively to controlled line crops unless their own documented input requirements demand
it; Florence-2's requirement is satisfied by that one line. What *was* wrong — that the conversion was
undocumented — is fixed.

SATRN's and Florence-2's recognition logic are untouched by this work.

## 9. Known limitations

- No real Transkribus round trip has been performed. No Transkribus account or network access was
  available, so the import side is exercised against the hand-authored PAGE XML fixture. The
  association *mechanism* is real and tested; a real end-to-end Swedish Lion I round trip is not.
- This repository contains no real full-page archival scan. The demonstration uses a real line crop
  as a stand-in page and says so; the multi-mode coverage is synthetic.
- The association back from an imported result is researcher testimony, not verification, and cannot
  be made stronger while Transkribus does not preserve or return content hashes.
- `strip_metadata` is a declared config field, but PNG output carries no EXIF/XMP regardless of it —
  it documents intent and feeds the configuration hash rather than switching behaviour. A future
  output format for which it mattered would need it honoured explicitly.
- No `MethodRun` is created by the normalization stage. Normalization is not a recognition method and
  is deliberately not counted as one.
