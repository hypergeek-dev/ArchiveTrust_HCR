# ArchiveTrust Current System

Status: Current authoritative description  
Scope: Supported single-site system and runtime boundaries  
Governs: Architecture and supported entry points  
Applies to version: 0.1.0 / Production Closure working tree  
Supersedes: broad implementation-status claims  
Superseded by: **`docs/htr-transformation-audit.md`, `docs/htr-domain-design.md`, and `README.md`, as of the HTR Research Center transformation (see `docs/htr-migration-plan.md`)**  
Last verified against code: 2026-07-17

> **HTR transformation notice:** This document describes the ArchiveTrust architecture *before*
> the transformation into a Swedish Historical HTR Research Center. The provider set, pipeline
> stages, and "OCR/layout/vision-language providers" language below are historical — the five
> general OCR providers this document describes (Docling, Tesseract+LayoutParser, Qwen2.5-VL,
> PaddleOCR-VL, Surya) were deleted; ArchiveTrust now studies SATRN, Florence-2, and Transkribus
> Swedish Lion I. The retained Evidence/Observation/Comparison/Canonical substrate this document
> describes is still structurally accurate — see `docs/htr-domain-design.md` for how it was
> extended, and `README.md` for the current system description. Kept here, not archived, because
> the substrate-level description remains a useful reference for that unchanged core.

```text
Desktop v2 -> durable worker command -> local worker process -> providers
     |                                      |
     +-- current-state/read models <- append-only telemetry + blob store
                         |
                         +-- operational review -> superseding observation/document
                         +-- JSON / PAGE / ALTO / METS release
                         +-- replay and diagnostics

Separate: workspace/evaluation -> blinded assignments -> two reviews -> adjudication
```

Archive objects are immutable. Provider output becomes content-addressed Evidence and typed
Observations. Domain/application `CurrentStateService` is the one projection of latest canonical
observations/documents, corrections, contested state, lineage, policies, eligibility, and integrity.
Desktop views, exports, replay, evaluation, the headless facade, and diagnostics consume it or the
shared read-model facade.

Production desktop startup requires a local authenticated account. Processing uses HMAC-protected
durable command/control envelopes, durable result files bound to the verified command digest, and a
local spawned worker process; it is not an installed supervised service. Worker command freshness
and replay state are durable under the deployment worker directory. Each workspace owns archive,
derived, telemetry/blob, evaluation, config/model, report, and metadata paths. Events are
authoritative; read models/renders/caches are rebuildable. Evaluation is separate from operational
review and reconciliation.

Supported entry points are the Desktop v2 module, `archivetrust-worker`, `archivetrust-replay`,
`archivetrust-evaluate`, and `archivetrust-admin`. Admin covers regeneration, run reconciliation,
telemetry history freeze/active epoch reset, crash-resilient qualification evidence runs,
config lock/verify, environment report, backup/verify/restore, health, diagnostics, alerts, and
initial administrator bootstrap.
