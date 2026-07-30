# Telemetry Retention and Reviewer-Privacy Policy

Companion to [`docs/architecture/htr-telemetry.md`](architecture/htr-telemetry.md) (what is recorded)
and [`docs/OBSERVABILITY_PHILOSOPHY.md`](OBSERVABILITY_PHILOSOPHY.md) (why). This document answers the
question neither of those does: **what may be deleted, and what may not.**

It exists because the two obvious defaults are both wrong. Treating every stream as an application log
would expire research evidence on a 30-day rotation. Treating every stream as research evidence would
retain personally identifying reviewer telemetry forever. The follow-up brief states both halves:

> Operational logs may expire. Research evidence may require long-term preservation. Do not delete
> valuable research telemetry solely because normal application logs are temporary. Do not retain
> personally identifying reviewer telemetry longer than necessary.

Ten classes are defined below, each with the **real path or event kind** it lives at, so a retention
decision is made against a file that exists rather than against a category.

## 0. What is enforced in code today, and what is not — read this first

Being precise about this matters more than the table that follows, because a policy document that
reads as if it were enforced is worse than no document.

**Enforced in code:**

| Control | Where | What it does |
|---|---|---|
| Workspace-level retention floor | `governance/policy.py::WorkspaceGovernance.retention_until` + `assert_permanent_deletion_allowed` | Refuses permanent deletion of a Workspace before a date, and unconditionally while `legal_hold` is set. Coarse: one date for the whole Workspace, not per class. |
| External-export gate | `governance/policy.py::assert_external_export_allowed` | Refuses external export unless the Workspace's `external_export_policy` permits it, and requires an approved PII review under `PII_REVIEW_REQUIRED`. Default is `PROHIBITED`. |
| Append-only, tamper-evident storage | `infrastructure/storage/telemetry_sink.py::FileTelemetrySink` + `integrity.py::HashChainAppender` | Makes silent partial deletion detectable: removing a line breaks the hash chain. Retention is enforced by *evidence of tampering*, not by a write lock. |
| Reviewer-identity stripping for external export | `htr/knowledge/redaction.py::strip_reviewer_identities` (new, §7) | Produces a pseudonymised derivative export; keeps the accountability mapping internal. |
| Operator-action audit chain | `admin/identity.py::AdminAuditLog` | Hash-chained append-only JSONL at `<app_root>/identity/admin-audit.jsonl`, with `verify()`. |

**Not enforced in code — this document is the only statement of it:**

* There is **no scheduled expiry job, no rotation policy, and no retention daemon anywhere in
  `src/`.** Nothing deletes anything on a timer. Every "may expire after N days" below describes what
  an operator or a future job is *permitted* to do, not what happens automatically. Verified by
  inspection: no `cron`, no scheduler, no `unlink` on a time predicate outside tests.
* There is **no per-class retention field** on any entity. Retention granularity in code is one
  Workspace.
* `review/blind_review/store.py::BlindReviewStore` is **in-memory**, so review and adjudication
  evidence has no durable retention story to enforce yet beyond the telemetry events that mirror it
  (`docs/architecture/htr-telemetry.md` §7 records the deferral).

Stating a class as "preserve indefinitely" therefore means: *nothing in this system will delete it, and
an operator who deletes it is destroying research evidence.* It does not mean a write-protected store.

## 1. Operational logs — **may expire**

**Where**: `<workspace>/logs/` (`runtime/deployment_layout.py::DeploymentLayout.logs_dir`), and
`<app_root>/telemetry/runtime_events.jsonl` (`FileRuntimeTelemetrySink` — model load/unload, device
selection, GPU reservation).

**What they are**: process-lifecycle facts. Which runtime warmed up, how long a model load took,
whether a container started, whether a queue worker crashed and restarted.

**Retention**: **90 days suggested, expiry permitted at any time.** These are diagnostic aids for a
live system. None of them is cited by any `EvidenceReference` and none participates in
`HtrJournal.replay`, so deleting them cannot make a research result unreconstructable.

**One real exception, and it is load-bearing**: a runtime event that was *quoted* into a research
record stops being an operational log. `docs/research-findings.md`'s
`florence2_environment_reproducible` rests on a peak-GPU-memory measurement; that measurement lives in
`Evidence.gpu_memory_mb` in the research stream (class 8), not in `runtime_events.jsonl`, which is
exactly why it survived. **Before expiring operational logs, check that no `EvidenceReference` names
them** — `EvidenceReferenceKind.TELEMETRY_EVENT` references are resolvable, so this is a query, not a
judgement call.

## 2. Security and operator-audit logs — **preserve for the accountability period; never silently**

**Where**: `<app_root>/identity/admin-audit.jsonl` (`admin/identity.py::AdminAuditLog`), plus
authentication events in `clients/desktop/authentication.py`.

**What they are**: who logged in, who changed a role, who released an archival package, who deleted a
Workspace. `AdminAuditEvent` is hash-chained (`previous_hash`/`event_hash`) and `AdminAuditLog.verify()`
re-checks the chain.

**Retention**: **preserve for at least as long as the Workspace whose actions it records**, and longer
where an external obligation applies. Truncation is permitted only from the head, only as a whole
prefix, and only with a recorded checkpoint — because removing an arbitrary line invalidates every
subsequent `previous_hash` and makes the log indistinguishable from a tampered one.

**Personal data**: this log names real operators by design; that is its entire function. It is
therefore the one class where identity is retained deliberately, and it is also the class that must
**never** be included in a research export. Nothing in `htr/` reads it.

## 3. Research telemetry — **preserve indefinitely; this is the class the brief protects**

**Where**: `<workspace>/telemetry/events.jsonl`, `<workspace>/telemetry/htr_research_events.jsonl` and
their `.chain.jsonl` sidecars; the committed
`docs/experiments/baseline-comparison/htr_research_events.jsonl`.

**What it is**: the append-only log from which `application/journal.py::Journal.replay` and
`application/htr_journal.py::HtrJournal.replay` rebuild the entire entity graph. Per
`docs/experiments/baseline-comparison/README.md`: *"If every file in this directory except the log were
deleted, the report could be regenerated; if the log were deleted, it could not."*

**Retention**: **indefinite.** This is the specific thing the brief means by *do not delete valuable
research telemetry solely because normal application logs are temporary*. The trap is concrete and
avoidable: these files sit under a directory named `telemetry/`, next to `logs/`, in the same layout, in
the same JSONL format, written by a class whose name ends in `Sink`. A rotation rule written by
filename pattern or directory would take them.

**Rule**: any retention automation must select streams by **class** (this document's ten), never by
directory, extension, or size. A stream from which `replay` reconstructs state is research evidence
regardless of where it is stored.

**`htr_coarse_entities.json` is the exception inside this class**: an explicitly derived cache
(`durable_store.py::HtrCoarseEntitySnapshot` — "If this file is deleted, nothing is lost. That is the
test of whether it is a cache"). Freely deletable, regenerable by `rebuild()`.

## 4. Source provenance — **preserve for the lifetime of every result derived from it**

**Where**: `PROVENANCE_CONTEXT_ESTABLISHED` events; `Evidence.source_*` fields; the immutable Archive
Object under `<workspace>/archive/`; `InputCrop.hash` and `Page`/`Region`/`TextLine` registrations in
the HTR stream.

**What it is**: what document this came from, which page, which crop, and the content hash proving the
bytes have not changed.

**Retention**: **as long as any result derived from it is retained, and never shorter.** A metric
without its provenance is a number; the whole evidence-first design (Constitution Article 6) collapses
if provenance can expire before the results it explains.

**Practical consequence**: source provenance is the class whose retention is *implied* by every other
class's. `tests/htr/persistence/test_real_baseline_reconstruction.py` recomputes
`InputCrop.hash` from the fixture on disk on every run — so if the source image were deleted, that test
would fail, which is the intended alarm rather than a fragile test.

## 5. Raw model output — **preserve; it is the only unmediated record of what the model said**

**Where**: `RawMethodResultRecorded` events; `Evidence.raw_output`; and for payloads over
`_DEFAULT_RAW_OUTPUT_BLOB_THRESHOLD` (4096 bytes) the content-addressed
`infrastructure/storage/blob_store.py` under `<workspace>/telemetry/blobs/`.

**What it is**: the model's literal output before any parsing. In the baseline run this is
`</s><s>Be Kindsf. SAGE och Doth Hoon Minnes Coon Pilla gonger werff</s>` — special tokens and all.

**Retention**: **indefinite while the method run is retained.** Constitution Article 6 (full exposure)
means the unmediated output is not a debugging convenience: a parser change is only assessable against
the raw text it parsed, and re-running inference to recover it is neither cheap nor guaranteed to
reproduce (a checkpoint can move).

**Blob-store caution**: `blobs/` holds externalized raw output referenced by hash from the log. It
*looks* like a cache and is not — deleting it leaves dangling references in an append-only log that
cannot be repaired by appending. The baseline's `blobs/` is legitimately empty only because nothing in
that run exceeded 4 KiB.

## 6. Parsed and normalized output — **may be regenerated, so may expire; in practice retain**

**Where**: `ParsedMethodResultRecorded`, `NormalizedMethodResultRecorded`.

**What it is**: raw output after each adapter's parse step, then after
`evaluation.metrics.normalize_text` (NFC + whitespace-run collapse, no case folding).

**Retention**: **deletable in principle, retained in practice.** These are deterministic functions of
class 5 plus a named code version, so they are the one research class that is genuinely regenerable
without inference. Two reasons not to delete them anyway:

1. Regeneration reproduces *today's* parser, not the one that ran. The stored value is the record of
   what the pipeline actually produced at that revision; a recomputed value is a different claim.
2. They are ~40 events in a 100-event log. There is no storage pressure to trade the distinction for.

**Rule**: if these are ever expired, the normalization profile id and the parser revision must be
retained even where the values are not, or the metrics in class 7 become uninterpretable.

## 7. Evaluation evidence — **preserve indefinitely**

**Where**: `MetricDefinitionRegistered`, `MetricCalculated`, `ReliabilityIssueClassified`,
`MethodRunFailed`; `MetricResult` and `FailureRecord` entities; `metric_results.csv` as a derived view.

**What it is**: every CER, WER, edit-operation count, segmentation rate, and reliability
classification, each bound to its `MetricDefinition` and normalization profile.

**Retention**: **indefinite.** This is what a finding cites. `research_finding_0a8f60fd…` names four
`metric_result` ids in `supporting_metrics`; deleting them turns a scoped, evidenced claim into an
unsupported assertion with no way to tell that anything is missing except the export's
`unresolved_supporting_observations` field.

**Notable**: the *absence* of a metric is also evidence here.
`transkribus_not_comparable` rests on no CER existing anywhere in the log, asserted by
`test_no_invalid_transkribus_cer_or_wer_exists_anywhere_in_the_log`. Retention of an absence means the
log must stay complete — a partial deletion could manufacture that absence for a method where it was
never true.

## 8. Review evidence — **preserve the judgement, minimise the identity (see §7 of this doc's
privacy section, §10 below)**

**Where**: `ReviewAssignment`, `ReviewSubmission` (`review/htr_models.py`), held by
`review/blind_review/store.py::BlindReviewStore` (in-memory today) and mirrored durably by
`ReviewSubmissionRecorded` / `REVIEW_OUTCOME_RECORDED`; interaction telemetry at
`<workspace>/telemetry/review_interactions.jsonl`.

**Retention, split in two, because the two halves have opposite requirements:**

* **The judgement** — `submitted_value`, `illegible`, `notes`, timing, the `AgreementResult` computed
  from two blind submissions: **preserve indefinitely.** This is ground-truth provenance. A benchmark
  dataset whose derivation cannot be inspected is not a benchmark.
* **The reviewer identity** — `reviewer_ref`: **retain internally only as long as accountability
  requires, and never in an external export.** Two reviewers agreeing must remain *linkable* (or
  inter-reviewer agreement is uncomputable), which is why the mechanism in §10 is pseudonymisation with
  a stable mapping, not deletion of the field.

**Blind isolation is not a retention control and must not be mistaken for one.**
`BlindReviewStore.submission_for_other_reviewer` refuses cross-reads until both submissions are
finalized — that governs *access during review*, not how long anything is kept.

## 9. Adjudication evidence — **preserve indefinitely, all three positions**

**Where**: `Adjudication` (`review/htr_models.py`), `AdjudicationRecorded`;
`review/blind_review/outcome.py::BenchmarkOutcome` as the derived view.

**What it is**: the third reviewer's `resolved_value` and required non-empty `rationale`, alongside both
original submissions — which `outcome.py` keeps independently readable rather than collapsing
(Constitution Article 15).

**Retention**: **indefinite, and specifically all three positions.** Retaining only the adjudicated
result would delete the evidence that the item was contested at all. `Adjudication` does not
retroactively reclassify a `REQUIRES_ADJUDICATION` item as `AGREED`, and retention must not achieve by
deletion what the code refuses to do by overwrite. `adjudicator_ref` is subject to §10 exactly as
`reviewer_ref` is.

## 10. Accepted findings — **preserve permanently, with their entire history and their status**

**Where**: `<workspace>/telemetry/htr_knowledge_events.jsonl`; the committed
`docs/experiments/baseline-comparison/htr_knowledge_events.jsonl` and
`htr_knowledge_feedback_events.jsonl`; `docs/experiments/baseline-comparison/knowledge-export/`.

**What it is**: `ResearchObservation`, `ResearchFinding` at any status, every `FindingRevision`, every
`ContradictoryEvidence`, every `ResearchQuestion` and `Hypothesis`.

**Retention**: **permanent.** Three things must survive together, and this is the class where partial
retention is most tempting and most damaging:

1. **The status.** A `Candidate` finding retained without its `review_status` becomes an established
   conclusion. This is why every export format in `htr/knowledge/export.py` carries
   `status_qualified_statement` in addition to the field.
2. **The full `revision_history`.** A `Disputed` finding that once reached `Supported` must keep the
   revision that got it there. `lifecycle.py::_assert_history_only_grew` enforces this at transition
   time; retention must not undo it afterwards.
3. **Both sides of every contradiction.** `docs/research-findings.md`: *"Disputing finding A does not
   delete, edit, or downgrade the record that disputed it."* Expiring the contradicting record would.

**Rejected and Superseded findings are retained too.** A `Rejected` finding is the record of a claim
this project examined and declined; deleting it loses the negative result, which in a research context
is a result.

---

## Reviewer privacy: what was actually checked, what was found, and what was built

The brief asks for a real check before a real mechanism: *"check whether
`review/blind_review/store.py` or the durable HTR store currently records any reviewer-identifying
field (e.g. a real username/email vs. an opaque reviewer id) — if real PII is stored, add a real,
tested function … If no real PII field exists today (check first, don't assume), document that finding
honestly rather than building a redaction mechanism for a field that isn't there."*

Both were checked. The answer is different for the two, and both answers are recorded here.

### Finding 1: the durable HTR knowledge store **does** hold a real personal identity

`docs/experiments/baseline-comparison/htr_knowledge_events.jsonl` — a committed artifact — contains the
literal string `hypergeek-dev`, the repository maintainer's real GitHub handle, **23 times**:

```
PYTHONPATH=src .venv/Scripts/python.exe -c "..."   # field-walk over the committed log
  6 ('actor',        'hypergeek-dev')     # FindingRevision.actor
  1 ('recorded_by',  'hypergeek-dev')     # ContradictoryEvidence.recorded_by
  4 ('reviewer',     'hypergeek-dev')     # ResearchFinding.reviewer
  8 ('actor_id',     'hypergeek-dev')     # TelemetryEvent envelope
 10 ('actor_id',     'htr.knowledge.baseline_knowledge')   # an automated component, not a person
  9 ('author',       'htr.knowledge.baseline_knowledge')
  5 ('author_or_source_component', 'htr.knowledge.baseline_knowledge')
```

This is **not a defect**. `scripts/register_baseline_knowledge.py::DEFAULT_REVIEWER` attributes it
deliberately, because `transition_finding_status` requires an attributable human and inventing a
plausible placeholder would have been the dishonest option. The finding is simply that a real identity
is durably stored, so a mechanism is warranted.

### Finding 2: `BlindReviewStore` holds **no** real identity today, but its field permits one

`review/htr_models.py::ReviewAssignment.reviewer_ref`, `ReviewSubmission.reviewer_ref` and
`Adjudication.adjudicator_ref` are **undecorated `str` fields with no docstring requiring opacity**, and
no code anywhere in `src/` mints an opaque value for them. `BlindReviewStore` is in-memory and no real
review has ever been run through it, so nothing personal is stored *yet* — but the first real caller
would decide by accident whether that field holds `reviewer_a7f3` or `anna.svensson@example.org`.

Reported as-is rather than resolved by fiat: constraining the field to an opaque format is a change to a
shared review model with its own tests, which is beyond this pass's scope. What was done instead is
that the redaction mechanism covers it (`redaction.py::redact_review_record_identities`, tested against
real `ReviewAssignment`/`ReviewSubmission`/`Adjudication` objects), so the field being permissive does
not also mean it is unhandled.

### What was built

`src/archivetrust/htr/knowledge/redaction.py`:

| Function | What it does |
|---|---|
| `pseudonymise(identity, salt)` | `reviewer_<16 hex>` = truncated `sha256(salt \|\| identity)`. Stable within a salt, so agreement between two reviewers stays analysable externally. |
| `identity_values(export)` | Every value in any `IDENTITY_BEARING_FIELDS` field. The input a human inspects. |
| `probable_component_identity` / `probable_human_identities` | A **named heuristic** (`COMPONENT_IDENTITY_PATTERN` — dotted lowercase module paths are components). Offered, never applied silently. |
| `strip_reviewer_identities(export, human_identities=…, salt=…)` | `(redacted export, IdentityLedger)`. `human_identities` has **no default**: guessing which authors are people is exactly the decision that must not be implicit. |
| `redact_review_record_identities(records, ledger=…)` | The same treatment for `reviewer_ref`/`adjudicator_ref`, against the same ledger, so pseudonyms line up across layers. |

**The internal accountability id is kept, and kept separate.** `IdentityLedger` maps pseudonym → real
identity and is returned *beside* the redacted export. Nothing in the module writes it to disk, and
`scripts/export_baseline_knowledge.py` prints it to stdout rather than into the directory the redacted
export sits in — because a ledger next to its own redacted export is not a redaction.

**Nothing else is touched.** `tests/htr/knowledge/test_export.py::test_redaction_preserves_status_statement_limitations_and_history`
asserts that a redacted export keeps every `review_status`, every statement, every limitation, every
scope, and every revision's `from_status`/`to_status`/`reasoning`. The status-preservation requirement
and the reviewer-privacy requirement are not in tension, and that test is where it is demonstrated
rather than claimed.

**Real output, real data**: `docs/experiments/baseline-comparison/knowledge-export/research-knowledge-external.json`
is the redacted variant of the committed baseline knowledge. `hypergeek-dev` occurs 0 times in it;
`reviewer_cb7c502be63f383d` occurs 9 times. The four non-redacted exports beside it keep the real
attribution, deliberately: they live in this repository next to the log that already carries it, and
stripping attribution from an *internal* research artifact would weaken accountability for no privacy
gain.

### Three limitations, stated rather than papered over

1. **Pseudonymisation is not anonymisation.** With a secret high-entropy salt the mapping is one-way in
   practice; with a guessable salt and a small known reviewer population it is trivially reversible by
   enumeration. The committed example uses the published `DEMONSTRATION_SALT` **so the artifact is
   reproducible**, and that constant's own docstring says it is not a privacy control.
2. **Free text is not redacted.** `IDENTITY_BEARING_FIELDS` lists only fields that *are* identities.
   A reviewer's name written into a `reasoning`, `description` or `notes` field survives redaction. A
   mechanism that scrubbed those would mangle research content; the honest trade is to redact the
   structured fields and disclose this.
3. **The durable log itself is never rewritten.** Redaction produces a *derivative* export. The
   append-only log keeps the real identity, per Constitution Article 15 (supersede, never erase) — so
   "do not retain personally identifying reviewer telemetry longer than necessary" is satisfied for
   what *leaves*, and for the log itself is satisfied only by the Workspace-level deletion control in
   §0. If a jurisdiction required erasure from the log, that would need a documented log-rewrite
   procedure that this system deliberately does not have.

## Summary table

| # | Class | Real location | Retention | Contains personal data? |
|---|---|---|---|---|
| 1 | Operational logs | `<ws>/logs/`, `telemetry/runtime_events.jsonl` | 90d suggested; may expire | No |
| 2 | Security / operator audit | `<app>/identity/admin-audit.jsonl` | ≥ Workspace lifetime; prefix-truncate only | **Yes, by design** — never exported |
| 3 | Research telemetry | `<ws>/telemetry/{events,htr_research_events}.jsonl` | **Indefinite** | No |
| 4 | Source provenance | `<ws>/archive/`, `PROVENANCE_CONTEXT_ESTABLISHED`, `InputCrop.hash` | ≥ every derived result | Depends on the source records |
| 5 | Raw model output | `RawMethodResultRecorded`, `telemetry/blobs/` | **Indefinite** while the run is retained | Depends on the transcribed content |
| 6 | Parsed / normalized output | `Parsed…`/`Normalized…MethodResultRecorded` | Regenerable; retain in practice | Depends on the content |
| 7 | Evaluation evidence | `MetricCalculated`, `ReliabilityIssueClassified` | **Indefinite** | No |
| 8 | Review evidence | `ReviewSubmission`, `review_interactions.jsonl` | Judgement indefinite; identity per §10 | **Yes** — `reviewer_ref` |
| 9 | Adjudication evidence | `Adjudication`, `AdjudicationRecorded` | **Indefinite**, all three positions | **Yes** — `adjudicator_ref` |
| 10 | Accepted findings | `htr_knowledge_events.jsonl`, `knowledge-export/` | **Permanent**, with status + history + contradictions | **Yes** — `reviewer`, `actor`, `recorded_by` |

Classes 8, 9 and 10 are the three that carry reviewer identity and therefore the three that
`strip_reviewer_identities` / `redact_review_record_identities` exist for. Class 2 also carries identity
and is deliberately **out of scope** for redaction: an audit log whose actors can be pseudonymised is
not an audit log.
