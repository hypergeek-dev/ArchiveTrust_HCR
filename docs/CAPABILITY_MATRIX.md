# ArchiveTrust Capability Matrix

Status: Current evidence classification  
Scope: Production Closure  
Governs: Meaning of readiness claims  
Applies to version: 0.1.0 / Production Closure working tree  
Supersedes: undifferentiated implemented claims  
Superseded by: **[`docs/CAPABILITY_MATRIX_HTR.md`](CAPABILITY_MATRIX_HTR.md)** (SATRN / Florence-2 / Transkribus)  
Last verified against code: 2026-07-17; supersession pointer updated 2026-07-30

> **HTR transformation notice:** The `(provider_id, provider_version, observation_type)` ratings
> below are for the deleted OCR providers (Docling, Tesseract+LayoutParser) and are no longer
> applicable — those providers do not exist in the codebase anymore.
>
> **The HTR capability matrix now exists as a standalone document:
> [`docs/CAPABILITY_MATRIX_HTR.md`](CAPABILITY_MATRIX_HTR.md)** (2026-07-30). Its tables are
> *generated* from each adapter's `get_capabilities()`/`get_metadata()` by
> `scripts/generate_capability_matrix.py`, and `tests/providers/test_htr_capability_matrix.py` fails if
> they ever drift — specifically so it cannot go stale the way this file did. The adapters
> (`providers/satrn/adapter.py`, `providers/florence2_htr/adapter.py`,
> `providers/transkribus/adapter.py`) remain the machine-readable source of truth; that document is a
> rendering of them plus prose on where the boolean flags mislead.
>
> **This file is not deleted and its table below is not corrected.** It is an accurate historical
> record of ratings that were really assigned to providers that really existed, and it is still the
> reference for the Capability Matrix *concept* the Comparison Engine uses
> (`domain/comparison/capability_matrix_data.py`, deliberately retained in Stage 5 for
> `application/pipeline.py`'s still-live legacy path). Read the rows below as history, never as a claim
> about the current codebase.

| Capability | Implemented/runtime | UI | Operational evidence | Qualified |
|---|---|---|---|---|
| Current truth/correction/reassembly | yes | yes | copied real workspaces | P1 pass |
| Review lifecycle | yes | yes | automation | P2 mechanical pass |
| Evaluation/adjudication/separation | yes | yes | automation, no campaign | P3 mechanical pass |
| Archival releases | yes | yes | 22/67 copied documents | external viewer pending |
| Recovery/worker | yes | yes | copied history/one worker doc | P4 partial |
| Telemetry bounds | yes | partial | focused tests | no 500-doc run |
| Identity/security/audit | partial | partial | focused tests | no |
| Governance/backup | partial | little | small restore drill | no |
| Config reproducibility | partial | no | worker lock tests | no dependency/model lock |
| Health/alerts/diagnostics | partial | old health only | CLI tests | no incident exercise |
| Installer/upgrade/rollback | no | no | none | no |
| UX/accessibility/localization | UI exists | yes | developer automation | no |
| Scientific campaign | workflow only | yes | none | no |

Implemented never implies production- or scientifically-qualified.
