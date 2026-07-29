# ArchiveTrust Capability Matrix

Status: Current evidence classification  
Scope: Production Closure  
Governs: Meaning of readiness claims  
Applies to version: 0.1.0 / Production Closure working tree  
Supersedes: undifferentiated implemented claims  
Superseded by: **not yet rebuilt for SATRN/Florence-2/Transkribus** — see `docs/htr-migration-plan.md`  
Last verified against code: 2026-07-17

> **HTR transformation notice:** The `(provider_id, provider_version, observation_type)` ratings
> below are for the deleted OCR providers (Docling, Tesseract+LayoutParser) and are no longer
> applicable — those providers do not exist in the codebase anymore. A capability matrix for
> SATRN/Florence-2/Transkribus has not yet been built as a standalone document; the closest
> current equivalent is each adapter's `get_capabilities()` (`providers/satrn/adapter.py`,
> `providers/florence2_htr/adapter.py`, `providers/transkribus/adapter.py`) plus their READMEs.
> Kept here as a reference for the Capability Matrix *concept* the Comparison Engine still uses.

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
