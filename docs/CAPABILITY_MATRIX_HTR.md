# HTR Capability Matrix — SATRN / Florence-2 / Transkribus

Status: Current
Scope: The three real HTR recognition methods in this codebase
Governs: What each method can and cannot produce, and what a comparison between them may claim
Supersedes: `docs/CAPABILITY_MATRIX.md` (OCR-era, Docling/Tesseract ratings — retained as a historical record)
Source of truth: each adapter's `get_capabilities()` / `get_metadata()`, not this file

> **The tables below are generated, not written.** Everything between the
> `BEGIN GENERATED FROM ADAPTERS` and `END GENERATED FROM ADAPTERS` markers is rendered from the live
> adapters by `scripts/generate_capability_matrix.py`, and
> `tests/providers/test_htr_capability_matrix.py` fails if this file and the adapters ever disagree.
> A flag flipped in an adapter, a checkpoint repinned, a hand-edit to a table, or a fourth method added
> all break a test rather than quietly leaving a document that lies.
>
> This is a direct response to how `docs/CAPABILITY_MATRIX.md` went stale: it carried a hand-maintained
> rating table for Docling and Tesseract+LayoutParser, both adapters were deleted in migration Stage 5,
> and **nothing failed**. The document was simply wrong for months until a human noticed.
>
> Regenerate: `PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_capability_matrix.py`
> Check: `… scripts/generate_capability_matrix.py --check`

<!-- BEGIN GENERATED FROM ADAPTERS -- do not edit by hand -->

### Method identity, as each adapter reports it

| `method_id` | Method | Vendor | `model_revision` (pinned) |
|---|---|---|---|
| `satrn` | SATRN (Riksarkivet) | Riksarkivet (Swedish National Archives) | `a40c7093232eaa47a83ce6469fc4abd033486bdc` |
| `florence2_htr` | Florence-2 (vlm-htr line OCR) | Uppsala University / Riksarkivet (hoanghapham/vlm-htr thesis project) | `nazounoryuu/florence_base__mixed__line_bbox__ocr@994f47e8...` |
| `transkribus_swedish_lion_1` | Transkribus Swedish Lion I | READ-COOP (Transkribus) | `unpinned (Transkribus manual export declares no fixed che...` |

### Capability flags, as each adapter reports them

| Capability | satrn | florence2_htr | transkribus_swedish_lion_1 |
|---|---|---|---|
| Confidence (`confidence_supported`) | yes | yes | yes |
| Geometry (`geometry_supported`) | no | no | yes |
| Line-level input (`line_level_supported`) | yes | yes | yes |
| Page-level input (`page_level_supported`) | no | no | yes |
| Runs locally (`local_execution_supported`) | yes | yes | no |
| Requires external upload (`external_upload_required`) | no | no | no |

<!-- END GENERATED FROM ADAPTERS -->

## What the flags do *not* say — read before comparing anything

`MethodCapabilities` has six boolean fields and every one is required, so an unsupported capability is
always an explicit `no` and never a silent omission (`providers/htr_adapter.py`). Booleans are still
booleans, and four of the cells above are true in a way that would mislead a reader who stopped at the
table. Each is documented on the adapter that reports it; they are collected here because a *matrix* is
exactly where the flattening happens.

### 1. All three report `confidence_supported = yes`, and the three numbers are not comparable

This is the single most important caveat on this page. `confidence_supported` says *a number is
produced*, not *a comparable number is produced*.

| Method | What its confidence actually is |
|---|---|
| `satrn` | The model's **own scalar** per line — mean decode probability from mmocr's `AttentionPostprocessor`. A real measured quantity, not per-character. |
| `florence2_htr` | A **proxy**: `exp(sequences_scores)`, i.e. the exponential of HuggingFace beam search's length-normalized sequence log-probability. Real and measured, but **not a calibrated probability** — see `providers/florence2_htr/facade.py::sequence_log_prob_to_confidence_proxy`. |
| `transkribus_swedish_lion_1` | The **vendor's** value, read from PAGE `@conf` or ALTO `WC` — and only when the specific file actually carries one. `RecognitionResult.confidence` is `None` when absent; it is never fabricated. |

The 2026-07-30 baseline run makes the consequence concrete: SATRN reported 0.66661 and Florence-2
reported 0.24904 on **the same byte-identical crop**, and Florence-2 was the more accurate of the two
(CER 0.4310 vs. 0.7931). Ranking those two confidence numbers against each other would have inverted the
truth. `docs/research-findings.md`'s `florence2_lower_error_rates` therefore rests on CER/WER only, and
states the non-comparability as one of its own limitations.

**Rule**: confidence may be compared to *accuracy for the same method* (that is what
`htr/evaluation/failures.py::classify_reliability` does, flagging
`confidence_calibration_disagreement`). It may not be compared *across methods*.

### 2. `transkribus_swedish_lion_1` reports `external_upload_required = no`, and the flag cannot express the truth

Nothing about Transkribus runs in this process, and the adapter makes **zero network calls** — it parses
a PAGE/ALTO export file the researcher obtained by uploading to Transkribus's own service themselves.
So:

* `local_execution_supported` = `no` is exactly right, and is the one flag in this section that is not
  misleading: no model runs in this process. It is also the only capability asymmetry on this page that
  is about *where execution happens* rather than about what is produced, which is why it is the flag a
  data-governance reader should key on rather than the upload flag below.
* `external_upload_required` = `no` is right *about this adapter* and misleading *about the workflow*.
  The output originated in an external service; this adapter simply did not put it there.

The adapter's own module docstring states this and calls `no` "the least-misleading available value".
The gap is in the flag vocabulary, not in the adapter — a faithful answer needs a third state
("external-service-originated output, zero network calls made by this adapter") that
`MethodCapabilities` does not have. **Recorded as a known modelling limitation rather than papered over
by flipping a boolean to a value that is also wrong.** Anyone reading this matrix for a data-governance
decision must read this section, not the cell: `docs/DATA_HANDLING_POLICY.md`'s external-upload consent
requirements apply to the researcher's Transkribus workflow regardless of what this flag says.

### 3. `geometry_supported` splits the three methods, and that is why segmentation metrics exist for one of them

`satrn` and `florence2_htr` both take a pre-cropped line image and return text. Neither returns
coordinates, so neither can be scored on `htr/evaluation/segmentation.py`'s metrics (`REGION_PRECISION`,
`LINE_RECALL`, `MEAN_MATCHED_IOU`, `MISSED_LINE_RATE`, `READING_ORDER_ACCURACY`, …) at all — not because
they scored badly, but because there is nothing to score.

`transkribus_swedish_lion_1` reports `yes` because PAGE `Coords`/`Baseline` and ALTO
`HPOS`/`VPOS`/`WIDTH`/`HEIGHT` are real geometry when the file states them. This asymmetry is a
capability difference, not a quality difference, and `docs/EVALUATION_PROTOCOL.md` §3 is where it becomes
a rule about what may be compared.

### 4. `page_level_supported` is why exactly one method is outside the controlled comparison

`satrn` and `florence2_htr` are line-level only; `transkribus_swedish_lion_1` is both. The baseline
experiment's controlled comparison is line-level over one hash-verified `InputCrop`, so Transkribus
could only participate at page level, with `input_crop_id = null` — a different unit of analysis on a
different fixture.

That, plus the fixture's content not corresponding to the shared ground-truth line, is the whole reason
**no CER or WER exists for Transkribus anywhere in the baseline log**. This is a capability-driven
comparison boundary declared in advance by the experiment's own `exclusion_criteria`, not a post-hoc
excuse. See `docs/EVALUATION_PROTOCOL.md` §7 and
`docs/research-findings.md`'s `transkribus_not_comparable`.

## Model revisions in full

The generated identity table truncates long revision strings at 60 characters. In full:

| `method_id` | `model_revision` |
|---|---|
| `satrn` | `a40c7093232eaa47a83ce6469fc4abd033486bdc` — the pinned `Riksarkivet/satrn_htr` commit this repository has actually run CUDA inference against. |
| `florence2_htr` | `nazounoryuu/florence_base__mixed__line_bbox__ocr@994f47e8a0e8d77cb2e11528665efd07a855c3af` — repo *and* commit, because the checkpoint is a fine-tune hosted separately from its base model. |
| `transkribus_swedish_lion_1` | `unpinned (Transkribus manual export declares no fixed checkpoint hash; see per-run RecognitionResult.model_revision / Evidence.supporting_metadata for any version string the specific import file states, if any)` |

**The third row is the honest one and the awkward one.** A Transkribus export carries no checkpoint
hash, so this method cannot satisfy the pinned-revision discipline every other measured claim in this
repository relies on. The adapter says so in prose in the field itself rather than emitting `"latest"` or
a plausible-looking hash. A `ResearchScope` naming this method therefore names an unpinnable model
version, which is one more reason its results sit outside the controlled comparison
(`ResearchScope._scope_is_genuinely_specific` requires a `model_version_id` per method, and it gets this
string).

## Environment readiness is separate from capability, and is not in this document

What a method *can produce* (this file) and whether it *can run here right now* are different questions
with different lifetimes — a capability is a property of the method, readiness is a property of this
machine this minute. Readiness comes from `validate_environment()`, is surfaced live by the HTR Methods
page and by `presentation/first_launch_viewmodel.py::htr_method_readiness`, and is deliberately not
frozen into a committed document.

| Method | What `validate_environment()` actually checks |
|---|---|
| `satrn` | `torch` importability in the orchestrating venv (informational only — inference runs in an isolated venv) and the presence of that isolated venv's interpreter (`satrn_python_available()`). The only one of the three that can fail on a missing *interpreter* rather than a missing library. |
| `florence2_htr` | `transformers`/`torch` importability (`florence2_dependencies_available()`). |
| `transkribus_swedish_lion_1` | That the configured import directory exists and is a directory. **No GPU or model check at all** — there is no model to check. Reports `valid=True` with a message when no directory is configured, since `recognize()` accepts a per-call path. |

## How this document is kept true

| Mechanism | What it catches |
|---|---|
| `tests/providers/test_htr_capability_matrix.py::test_the_generated_region_matches_what_the_adapters_report` | Any drift between the tables above and the live adapters, in either direction. |
| `…::test_the_matrix_covers_exactly_the_methods_composition_builds` | A method added to `composition.py::_build_htr_adapters` (or removed) without updating this matrix. |
| `…::test_every_capability_flag_has_a_column` | A field added to `MethodCapabilities` that the generated table would silently omit. |
| `…::test_the_prose_documents_every_capability_asymmetry` | A capability flag that differs across the three methods with no prose section explaining the asymmetry — the failure mode this document's whole second half exists to prevent. |
| `scripts/generate_capability_matrix.py --check` | The same as the first row, runnable outside pytest. |

Note what is **not** claimed: none of these tests verifies that an adapter's reported capability matches
its actual behaviour. `confidence_supported = yes` is checked to be what the adapter *says*, not that a
confidence is really returned — that is the adapter contract tests' job
(`tests/providers/*/test_adapter_contract.py`), and conflating the two would let this document claim a
verification it does not perform.
