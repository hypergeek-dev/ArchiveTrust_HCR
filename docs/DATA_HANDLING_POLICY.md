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

## 6. B7 Placeholder

B7 Dataset Export & Licensing/PII Framework will replace this skeleton with corpus-specific
templates and release tooling. Until then, this document is the minimum policy: no dataset leaves the
project without explicit license and PII review.
