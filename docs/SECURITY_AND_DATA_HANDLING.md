# ArchiveTrust Security and Data Handling

Status: Current controls and explicit gaps  
Scope: Documents, corrections, evaluation, exports, and backups  
Governs: Security/privacy decisions  
Applies to version: 0.1.0 / Production Closure working tree  
Supersedes: `docs/DATA_HANDLING_POLICY.md` for runtime claims  
Superseded by: none  
Last verified against code: 2026-07-17

Threats include malicious PDFs/images, bombs/oversize, embedded files, filenames/symlinks/traversal,
prompt injection, compromised model servers, unauthorized action/export, corruption, poisoned
annotations, denial of service, secrets, and raw-text exposure.

Production admission checks signature/type, 512 MiB size, 1,000 PDF pages, normalized filename, no
symlink, and embedded-file markers. Qwen treats pixels/context as untrusted and rejects tool-control
responses. Events are bounded; large raw output becomes protected blobs. Parser sandboxing,
antivirus, structural PDF sanitization, full resource/time limits, and network policy remain open.

Passwords are salted PBKDF2-SHA256 (600,000 iterations); sessions expire; disabled accounts fail;
roles are partially wired. Spawned-worker processing commands and pause/resume/cancel controls now
use deployment-bound HMAC-SHA256 envelopes with expiry, nonce freshness, command digest binding,
configuration-lock digest binding, actor/session attribution where available, and durable replay
state. Worker results record the verified command digest and verification context, and forged
success results without that binding are not trusted. Audit facts carry actor/roles/session/action/
target/result and a hash chain. Authorization is still not complete across every config/account/
policy mutation, and Windows ACL enforcement for worker directories still requires clean-machine
validation.

Workspaces carry classification, retention, legal hold, external-export policy, and encryption
requirement. Hold/retention block deletion. Public export is prohibited by default. Verified
evaluation needs legal/sampling basis. Encryption at rest/backup, key recovery, redaction/PII review,
and disposition certification remain absent. Secrets may load only from the launch directory `.env`
or process environment and are excluded from diagnostics; OS credential storage is absent.
