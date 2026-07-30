# ArchiveTrust Data Handling Policy

**Status:** C6 policy skeleton.  
**Applies to:** datasets, telemetry archives, benchmark corpora, exports, screenshots, review packets,
and any artifact that may contain source document content or derived document facts.

## 1. Code License Is Not A Data License

ArchiveTrust code is licensed under Apache License 2.0.

That license does not automatically apply to:

- archive documents;
- OCR/VLM raw outputs;
- Evidence records;
- telemetry archives containing document-derived content;
- benchmark corpora;
- exported canonical documents;
- screenshots of real documents;
- review packets or human corrections.

Each dataset or corpus needs an explicit license and release decision before publication.

## 2. Default Data Posture

Private, municipal, personal, copyrighted, or otherwise sensitive records are not public by default.
They may be processed locally for authorized operational or research purposes, but they must not be
committed, published, or shared externally without a documented release decision.

Synthetic fixtures and cleared public-domain records are preferred for tests, demos, and examples.

## 3. PII And Sensitive Content

Before a dataset, telemetry archive, benchmark export, or publication package leaves the controlled
workspace, it needs a documented PII review.

The review should record:

- corpus name and version;
- source authority or license;
- intended recipients;
- fields/artifacts included;
- PII classes checked;
- redaction or exclusion decisions;
- reviewer and date;
- residual risk.

## 4. Telemetry Archives

Telemetry can contain raw provider output, rejected evidence, canonical observations, review
outcomes, and document-derived text. Treat telemetry archives as document data unless proven
otherwise.

F3 blob indirection and C3 integrity sidecars reduce size and improve defensibility; they do not
sanitize content.

## 5. Public Dataset Release Requirements

A public dataset release must include:

- dataset license;
- datasheet;
- provenance summary;
- PII review record;
- scope of included artifacts;
- integrity manifest;
- replay instructions;
- contact point for takedown or correction requests.

## 7. Derived Image Artifacts And Colour Normalization

Added with the versioned RGB-normalization preprocessing stage
(`src/archivetrust/htr/preprocessing/`, `docs/methods/transkribus-swedish-lion-1.md`).

**Before this section, this repository had no archival-image colour or compositing policy at all.**
Sections 1-6 above govern licensing, PII, telemetry-as-document-data and release requirements, and say
nothing about image colour handling, alpha, or bit depth; neither does
`docs/SECURITY_AND_DATA_HANDLING.md`. The decisions below are therefore new, not a departure from an
existing policy.

### 7.1 Derived images are document data

A normalized page image is derived from a source document and must be treated exactly as the original
is under §2 and §4: private/municipal/personal/copyrighted source material stays private when
normalized. Normalization is not a sanitizing step. It removes an ICC profile and EXIF metadata, which
may incidentally remove camera or operator identifiers, but it is not a redaction mechanism and must
never be relied on as one — the source image, with all its metadata, remains in the content-addressed
blob store.

### 7.2 Export packages leave the controlled workspace

An export package (`htr/preprocessing/export_package.py`) is written specifically so a human can hand
it to an external service. It is therefore an artifact "leaving the controlled workspace" under §3 and
needs a documented PII review before it is uploaded anywhere. ArchiveTrust performs no upload and
cannot enforce this; the manifest records the external-processing boundary explicitly so the obligation
is visible to whoever does.

### 7.3 Alpha compositing: white background

Transparency is composited over **opaque white** (`#FFFFFF`) by default.

Historical manuscript pages are dark ink on light support. Transparent regions in archival derivatives
are overwhelmingly padding, removed borders, or masked areas; compositing them to black would introduce
large black regions no source page had, which an HTR model may read as content. White is continuous
with the paper.

This is a recorded policy value, not a hardcoded constant: the background actually used is stored on
every `NormalizedPageArtifact` (`compositing_background`), and changing the policy changes the
configuration hash and forces a new experiment version.

### 7.4 ICC profiles: recorded, then stripped, never applied

An embedded ICC profile's presence, byte size, SHA-256, and declared description are recorded on the
artifact; the profile is then stripped from the output and is **not** applied to pixel values.

Applying it would be a colour transform — an enhancement this stage guarantees it does not perform —
and its result depends on the linked LittleCMS version, breaking determinism. The colour intent stays
recoverable as recorded evidence rather than being baked irreversibly into the derived pixels. See
`docs/methods/transkribus-swedish-lion-1.md` §4 for the full reasoning.

### 7.5 Bit depth reduction is lossy and is recorded as such

Reducing 16-bit samples to 8 bits discards information. It is required by the target representation,
is performed by explicit linear scaling (never by a content-dependent contrast stretch), and the
source's true bit depth is recorded on the artifact so the loss is visible rather than implied. **The
original is never replaced** — both artifacts are retained, content-addressed, and independently
retrievable.

## 8. B7 Placeholder

B7 Dataset Export & Licensing/PII Framework will replace this skeleton with corpus-specific
templates and release tooling. Until then, this document is the minimum policy: no dataset leaves the
project without explicit license and PII review.
