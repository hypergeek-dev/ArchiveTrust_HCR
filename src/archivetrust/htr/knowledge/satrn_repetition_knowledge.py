"""The real research knowledge extracted from the SATRN output-repetition diagnostic (2026-07-30).

**Every id, string and number in this module was read out of two committed artifacts** -- the
Checkpoint 3 smoke test
(`docs/experiments/technical-reliability-screening/smoke-test/smoke_test_results.json` and its
hash-chained `smoke_test_events.jsonl`) and the bounded diagnostic that investigated its most
alarming signal
(`docs/experiments/technical-reliability-screening/diagnostic/satrn_repetition_diagnostic.json`,
produced by `scripts/run_satrn_repetition_diagnostic.py`). Nothing here is a placeholder and nothing
is recomputed from prose: `tests/htr/knowledge/test_satrn_repetition_knowledge.py` re-reads both
artifacts and fails if this module drifts from them.

**What was observed.** The smoke test ran SATRN on 15 real, byte-distinct line crops cut from 5 real
pages of the 766-page Swedish witchcraft-trial court-record corpus. SATRN returned the identical
string `staden den 27 dennes` for 6 of them and `talan att` for 4 more -- 7 distinct outputs over 15
distinct inputs. Florence-2, on the byte-identical crops, produced 15 distinct outputs.

**Why this is filed as a `MODEL_LIMITATION` and not a `REPRODUCIBILITY_ANOMALY` or an
`ENVIRONMENT_ISSUE`.** The diagnostic tested the integration-defect hypotheses directly and excluded
them with evidence, most decisively by reproducing the identical 7/15 constancy -- and every
individual confidence value to full float precision -- through `mmocr.apis.TextRecInferencer` called
directly inside `.venv-satrn`, in one process with one model load, bypassing `SatrnAdapter`,
`facade.py` and `_worker.py` entirely. A defect in ArchiveTrust's integration cannot survive being
removed from the code path. The behaviour is a property of the checkpoint on these inputs.

**What this observation does NOT claim.** It does not claim SATRN is unfit, and it must not be read
as an argument to drop SATRN from the forthcoming full-corpus benchmark. It is scoped to 15 crops
from 5 pages under one configuration, `sample_size` is derived from the enumerated crop ids, and the
operational question "is this method viable for this corpus" belongs to a `ScreeningPolicy`/
`ScreeningDecision` pair that `docs/experiments/technical-reliability-screening/design-audit.md` §3.3
records as **not yet existing**. Inventing a verdict here would be exactly the post-hoc
rationalization that audit warns against.
"""

from __future__ import annotations

from archivetrust.htr.knowledge.models import (
    EvidenceReference,
    EvidenceReferenceKind,
    ObservationConfidence,
    ObservationType,
    ResearchObservation,
    ResearchScope,
    ScopeUnit,
)

# -- Streams --------------------------------------------------------------------------------------

SMOKE_TEST_STREAM = (
    "docs/experiments/technical-reliability-screening/smoke-test/smoke_test_events.jsonl"
)
"""The Checkpoint 3 smoke test's own hash-chained telemetry. **Read-only for this diagnostic** --
not one byte of it was rewritten, and the diagnostic appends nothing to it."""

DIAGNOSTIC_STREAM = (
    "docs/experiments/technical-reliability-screening/diagnostic/htr_knowledge_events.jsonl"
)
"""Where this observation is durably recorded -- a separate file, for the reason
`baseline_knowledge.py` gives for its own split: the smoke test's log is a committed research
artifact whose description ("456 KB, N events of these kinds") appending to would falsify."""

DIAGNOSTIC_REPORT = (
    "docs/experiments/technical-reliability-screening/diagnostic/satrn_repetition_diagnostic.json"
)
DIAGNOSTIC_NARRATIVE = (
    "docs/experiments/technical-reliability-screening/satrn-repetition-diagnostic.md"
)

# -- Real identifiers -----------------------------------------------------------------------------

SATRN_METHOD_ID = "satrn"
SATRN_MODEL_REVISION = "a40c7093232eaa47a83ce6469fc4abd033486bdc"
"""`Riksarkivet/satrn_htr` at the pinned commit. The diagnostic logged this revision string back
from the worker on all 33 adapter invocations -- it is the revision that actually ran, not the one
`get_metadata()` advertises."""

FLORENCE2_METHOD_ID = "florence2"
FLORENCE2_MODEL_REVISION = (
    "nazounoryuu/florence_base__mixed__line_bbox__ocr@994f47e8a0e8d77cb2e11528665efd07a855c3af"
)

SEGMENTATION_ADAPTER = "nazounoryuu/florence_base__mixed__page__line_od"
CONFOUND_NOTE = (
    "The crops both methods read were detected by a Florence-2-family line detector, so one screened "
    "method's own model family controlled the input the other was judged on. Flagged at Checkpoint 1, "
    "not neutralized here, and restated because this observation quotes a SATRN-vs-Florence-2 contrast."
)

REPEATED_OUTPUT = "staden den 27 dennes"
SECOND_REPEATED_OUTPUT = "talan att"

REPEATED_CROP_IDS = (
    "input_crop_1d9a3a4f709a4472ac488189f154b5c4",
    "input_crop_50dc8a1c4cbb493c9af29979b467f7dc",
    "input_crop_377f189a2c5f491b81ec4964709a5a40",
    "input_crop_1af212a49c2145b98c36a2cd7f98049e",
    "input_crop_c1c74279326c4fa89c030fa55bb592af",
    "input_crop_fb0310f8ad5a481da6e2f9e2a19ac9ad",
)
"""The six crops that all returned `staden den 27 dennes`, in smoke-test execution order
(positions 1, 2, 3, 6, 10, 15 of 15)."""

SECOND_REPEATED_CROP_IDS = (
    "input_crop_c2253d155b8a4ae8acaccdb7587a2574",
    "input_crop_91595c063b0b4842b8f61d664d71f99c",
    "input_crop_368077e5975c47bc9e4c1661442eab93",
    "input_crop_5ea401d261f345de8ec874e634e925f6",
)
"""The four crops that all returned `talan att` (positions 11-14)."""

DISTINCT_OUTPUT_CROP_IDS = (
    "input_crop_4f034275f85f42d8bdbf6659998939f7",
    "input_crop_0a37475cf52942bba20b31918530feea",
    "input_crop_0c89a9e029fb443eb4c368b47ece51f8",
    "input_crop_2aab3b7726bc4fa5a1531687ace3f10a",
    "input_crop_9dc3a7f35d71438391d8f93c32efc49e",
)
"""The five control crops whose SATRN output was unique within the sample (positions 4, 5, 7, 8, 9).
Their outputs -- `staden den 22 dennes`, `tala till den 22 dennes`, `stånd till`,
`stånd till den till denna`, `stånd till den 27 dennes` -- are lexically adjacent to the repeated
ones, which is itself part of what was observed."""

ALL_CROP_IDS = (
    REPEATED_CROP_IDS[:3]
    + DISTINCT_OUTPUT_CROP_IDS[:2]
    + (REPEATED_CROP_IDS[3],)
    + DISTINCT_OUTPUT_CROP_IDS[2:]
    + (REPEATED_CROP_IDS[4],)
    + SECOND_REPEATED_CROP_IDS
    + (REPEATED_CROP_IDS[5],)
)
"""All 15 crops in the smoke test's own execution order -- the scope's `covered_unit_ids`."""

REPEATED_CROP_SHA256 = {
    "input_crop_1d9a3a4f709a4472ac488189f154b5c4": (
        "3e840fc5bc49e708fa219cedb22b7f3aab36dd994fdc135f9ca0f328dc9da97f"
    ),
    "input_crop_50dc8a1c4cbb493c9af29979b467f7dc": (
        "4a2061ba3756e73e9ea2c5ad4e4b791367f05db1b658e90f1a2b8938aee07f75"
    ),
    "input_crop_377f189a2c5f491b81ec4964709a5a40": (
        "021e75f0e849d80e8536fe963d3835e1f4e04cb452b0b658302da705fb7f0528"
    ),
    "input_crop_1af212a49c2145b98c36a2cd7f98049e": (
        "3952b1461a1233dacb9965d3d77c109a1456d258413e5dc29f79740760f4384e"
    ),
    "input_crop_c1c74279326c4fa89c030fa55bb592af": (
        "24c3a196f02caf735978b76dc1c66c1a67fda8a69acf3ae83e2b9f388f9085bd"
    ),
    "input_crop_fb0310f8ad5a481da6e2f9e2a19ac9ad": (
        "c1fe5627ee76eecf1edae8c6a01fa32fd4b54e5e349ef36d8ff56b3762399059"
    ),
}
"""SHA-256 of each repeated crop's PNG bytes, computed by the diagnostic from the files themselves
rather than trusted from the smoke test's own record (the diagnostic separately confirmed the smoke
test's `crop_...` hashes are the same digests). `verify_against_diagnostic_report` re-resolves every
entry here against the diagnostic JSON, so these are checked transcriptions, not an authority."""

# -- Measured facts -------------------------------------------------------------------------------

CROPS_IN_SAMPLE = 15
SATRN_DISTINCT_OUTPUTS = 7
SATRN_DISTINCT_RATIO = 0.4667
FLORENCE2_DISTINCT_OUTPUTS = 15
FLORENCE2_DISTINCT_RATIO = 1.0

REPEATED_GROUP_CONFIDENCES = {
    "input_crop_1d9a3a4f709a4472ac488189f154b5c4": 0.5366043906658888,
    "input_crop_50dc8a1c4cbb493c9af29979b467f7dc": 0.536318052560091,
    "input_crop_377f189a2c5f491b81ec4964709a5a40": 0.5346601057797671,
    "input_crop_1af212a49c2145b98c36a2cd7f98049e": 0.5660757340490818,
    "input_crop_c1c74279326c4fa89c030fa55bb592af": 0.5236199896782636,
    "input_crop_fb0310f8ad5a481da6e2f9e2a19ac9ad": 0.5265959832817316,
}
"""**The single most diagnostic number set in this module.** Six identical output strings carrying
six *different* confidence scores. A cached, stale or cross-wired result would necessarily have
carried the same score too; six distinct scores mean six genuinely distinct forward passes over six
genuinely distinct inputs, which happened to decode to the same argmax string."""

EXTREME_ASPECT_RATIO_SPAN = (0.701, 9.579)
"""Width/height of the narrowest and widest crop in the 15. The checkpoint's own `test_pipeline`
resizes every input to 400x64 with `keep_ratio=False`, so a 0.70 crop is stretched roughly 9x
horizontally and a 9.58 crop is squeezed -- both land on the same 6.25 target ratio."""

DOCUMENTED_RESIZE = "Resize(scale=(400, 64), keep_ratio=False)"
DOCUMENTED_MEAN = (123.675, 116.28, 103.53)
DOCUMENTED_STD = (58.395, 57.12, 57.375)

BASELINE_CORROBORATION = (
    "The committed baseline run independently recorded this same checkpoint returning "
    "'till den 23 Januarii' for a line whose reference text is "
    "'bekiendt. Säger och deth hon Minnes hoon Tuå gånger waritt' -- a short, generic, date-shaped "
    "phrase with no lexical overlap with the line, produced on a different image by a different "
    "pipeline with no repetition question in play (see htr/knowledge/baseline_knowledge.py, "
    "SATRN_OUTPUT_TEXT)."
)

RERUN_PASSES = (
    "A: all 15 crops re-run through SatrnAdapter in the smoke test's original order",
    "B: the 6 repeated crops re-run in the same order in the same host process",
    "C: the 6 repeated crops re-run in reversed order",
    "D: the 6 repeated crops each re-run in its own isolated top-level Python process",
    "E: all 15 crops through mmocr.apis.TextRecInferencer directly inside .venv-satrn, one process, "
    "one model load, bypassing SatrnAdapter/facade.py/_worker.py entirely",
)
"""All five passes returned byte-identical text and bit-identical confidence for every crop."""

EXTRACTED_AT = "2026-07-30T20:30:00+00:00"


def _ref(
    kind: EvidenceReferenceKind, reference_id: str, note: str | None = None, *, stream: str | None
) -> EvidenceReference:
    return EvidenceReference(kind=kind, reference_id=reference_id, stream=stream, note=note)


def satrn_repetition_scope(
    *,
    experiment_id: str,
    experiment_version_id: str,
    experiment_run_id: str,
    dataset_id: str,
    dataset_version_id: str,
) -> ResearchScope:
    """The scope this observation is bound to -- all 15 crops enumerated by id, both methods named
    with their exact revisions, and the dataset version pinned.

    The ids are parameters rather than module constants (the shape `baseline_knowledge.py` uses)
    because this observation's experiment records are created by the registration script at
    registration time, not read back out of a pre-existing committed run. The registration script
    writes the ids it generated into the durable stream, and the test resolves this scope against
    that stream rather than against hard-coded literals.
    """
    return ResearchScope(
        experiment_id=experiment_id,
        experiment_version_id=experiment_version_id,
        experiment_run_ids=(experiment_run_id,),
        unit_of_analysis=ScopeUnit.LINE_CROP,
        covered_unit_ids=ALL_CROP_IDS,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        method_ids=(SATRN_METHOD_ID, FLORENCE2_METHOD_ID),
        model_version_ids=(SATRN_MODEL_REVISION, FLORENCE2_MODEL_REVISION),
        configuration_ref=(
            f"device=auto/cuda; {DOCUMENTED_RESIZE}; mean={DOCUMENTED_MEAN}; std={DOCUMENTED_STD}; "
            f"segmentation={SEGMENTATION_ADAPTER}"
        ),
    )


def satrn_output_repetition_observation(
    *,
    experiment_id: str,
    experiment_version_id: str,
    experiment_run_id: str,
    dataset_id: str,
    dataset_version_id: str,
    at: str = EXTRACTED_AT,
) -> ResearchObservation:
    """The observation: SATRN returned one identical string for 6 of 15 byte-distinct crops, and the
    behaviour survives removal of ArchiveTrust's integration from the code path.

    Typed `MODEL_LIMITATION`. `REPRODUCIBILITY_ANOMALY` would be wrong in the precise sense the enum
    intends -- nothing here is irreproducible; the opposite is true, and that reproducibility is the
    evidence. `ENVIRONMENT_ISSUE` would be wrong because pass E removes the environment-specific
    integration and the behaviour persists.

    `observation_confidence` is HIGH: this is a statement about what the records show for these 15
    crops, confirmed five independent ways including through the model's own official inference API.
    The HIGH confidence attaches to the *fact*, not to any generalization -- the scope is what bounds
    the claim, and the causal explanation is quarantined in `unverified_hypothesis`.
    """
    scope = satrn_repetition_scope(
        experiment_id=experiment_id,
        experiment_version_id=experiment_version_id,
        experiment_run_id=experiment_run_id,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
    )
    return ResearchObservation.create(
        observation_type=ObservationType.MODEL_LIMITATION,
        title=(
            f"SATRN@{SATRN_MODEL_REVISION[:8]} returned the identical string {REPEATED_OUTPUT!r} "
            f"for 6 of {CROPS_IN_SAMPLE} byte-distinct line crops; not an integration defect"
        ),
        description=(
            f"On {CROPS_IN_SAMPLE} real line crops cut from 5 real pages of the Swedish "
            f"witchcraft-trial court-record corpus, SATRN at revision {SATRN_MODEL_REVISION} "
            f"returned the identical string {REPEATED_OUTPUT!r} for 6 crops "
            f"({', '.join(REPEATED_CROP_IDS)}) and {SECOND_REPEATED_OUTPUT!r} for 4 more "
            f"({', '.join(SECOND_REPEATED_CROP_IDS)}): {SATRN_DISTINCT_OUTPUTS} distinct outputs "
            f"over {CROPS_IN_SAMPLE} distinct inputs (ratio {SATRN_DISTINCT_RATIO}). Florence-2, "
            f"reading the byte-identical crops, produced {FLORENCE2_DISTINCT_OUTPUTS} distinct "
            f"outputs (ratio {FLORENCE2_DISTINCT_RATIO}). "
            f"The {CROPS_IN_SAMPLE} crops are genuinely different images: all "
            f"{CROPS_IN_SAMPLE} PNG-byte SHA-256 digests are distinct, all {CROPS_IN_SAMPLE} "
            "decoded-pixel SHA-256 digests are distinct, none is blank (per-crop grayscale standard "
            "deviation 29.8-84.3 over 227-256 distinct levels), and each file's size matches the "
            "byte_size the smoke test recorded. "
            "The integration-defect hypotheses were tested and excluded. Every adapter invocation "
            "received its own fresh child process (15 distinct PIDs for 15 calls); the worker argv "
            "carried the source crop path itself, so no temporary file is created and none can be "
            "reused or collide; each crop's bytes were unchanged by its own call and matched the "
            "manifest hash; the worker reported the same pinned model and config revision on every "
            "call. Most decisively, pass E reproduced the identical "
            f"{SATRN_DISTINCT_OUTPUTS}/{CROPS_IN_SAMPLE} constancy -- and every confidence value to "
            "full float precision -- through mmocr.apis.TextRecInferencer called directly inside "
            ".venv-satrn in ONE process with ONE model load, bypassing SatrnAdapter, facade.py and "
            "_worker.py entirely. "
            "The 6 identical strings carry 6 different confidence scores "
            f"({', '.join(str(v) for v in REPEATED_GROUP_CONFIDENCES.values())}); a cached, stale or "
            "cross-wired result would have carried an identical score too. All 5 passes (same order, "
            "reversed order, isolated processes, and the reference API) returned byte-identical text "
            "and bit-identical confidence for every crop, so the behaviour is fully deterministic and "
            "order-independent. "
            f"{BASELINE_CORROBORATION} "
            "Recorded for these 15 crops and this configuration only. This observation asserts "
            "nothing about SATRN's suitability for the full corpus: that is an operational decision "
            "requiring the ScreeningPolicy/ScreeningDecision entities design-audit.md §3.3 records "
            "as not yet existing, and SATRN remains in scope for the forthcoming benchmark. "
            f"{CONFOUND_NOTE}"
        ),
        scope=scope,
        supporting_evidence=(
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                DIAGNOSTIC_REPORT,
                "machine-readable diagnostic: crop manifest, all 5 passes, integration checks",
                stream=None,
            ),
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                DIAGNOSTIC_NARRATIVE,
                "the written diagnostic report",
                stream=None,
            ),
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                "docs/experiments/technical-reliability-screening/smoke-test/smoke_test_results.json",
                f"Checkpoint 3 smoke test: satrn_output_constancy = "
                f"{SATRN_DISTINCT_OUTPUTS}/{CROPS_IN_SAMPLE}, florence2 = "
                f"{FLORENCE2_DISTINCT_OUTPUTS}/{CROPS_IN_SAMPLE}",
                stream=None,
            ),
            _ref(
                EvidenceReferenceKind.EXTERNAL_DOCUMENT,
                "src/archivetrust/providers/satrn/README.md",
                "documented checkpoint, isolated-venv setup and prior measured inference behaviour",
                stream=None,
            ),
        )
        + tuple(
            _ref(
                EvidenceReferenceKind.INPUT_CROP,
                crop_id,
                f"returned {REPEATED_OUTPUT!r} at confidence "
                f"{REPEATED_GROUP_CONFIDENCES[crop_id]}",
                stream=SMOKE_TEST_STREAM,
            )
            for crop_id in REPEATED_CROP_IDS
        )
        + tuple(
            _ref(
                EvidenceReferenceKind.INPUT_CROP,
                crop_id,
                "distinct-output control crop",
                stream=SMOKE_TEST_STREAM,
            )
            for crop_id in DISTINCT_OUTPUT_CROP_IDS
        ),
        source_experiment_id=experiment_id,
        source_experiment_run_id=experiment_run_id,
        affected_method=SATRN_METHOD_ID,
        affected_model_version=SATRN_MODEL_REVISION,
        affected_dataset_id=dataset_id,
        affected_dataset_version_id=dataset_version_id,
        affected_document_or_segment_ids=ALL_CROP_IDS,
        author_or_source_component="scripts/run_satrn_repetition_diagnostic.py",
        creation_timestamp=at,
        observation_confidence=ObservationConfidence.HIGH,
        tags=(
            "satrn",
            "output-constancy",
            "technical-reliability-screening",
            "checkpoint-3-smoke-test",
            "integration-defect-excluded",
        ),
        unverified_hypothesis=(
            "Candidate explanation, NOT established by any measurement in this repository. The "
            f"checkpoint's own test_pipeline applies {DOCUMENTED_RESIZE}, so every crop is forced to "
            "a 6.25 width/height ratio regardless of its own; the crops in this sample span "
            f"{EXTREME_ASPECT_RATIO_SPAN[0]}-{EXTREME_ASPECT_RATIO_SPAN[1]}, meaning some are "
            "stretched ~9x horizontally and others compressed. Together with the model card's "
            "statement that training used binarized line images while these crops are unbinarized "
            "RGB, this is a plausible input-domain mismatch under which the decoder falls back on a "
            "high-frequency phrase prior from its training distribution -- which would also explain "
            "why all 7 distinct outputs occupy one narrow lexical neighbourhood "
            "('staden'/'stånd'/'tala(n)' + 'den' + '22'/'27' + 'dennes'/'att'). Confirming or "
            "refuting this needs a controlled preprocessing experiment that has not been run, and "
            "no preprocessing was changed by this diagnostic: the adapter's behaviour was verified "
            "to match the checkpoint's documented pipeline exactly, so there was no inconsistency to "
            "correct."
        ),
    )


def verify_against_diagnostic_report(report: dict) -> tuple[str, ...]:
    """Re-derives every claim in this module from a loaded `satrn_repetition_diagnostic.json` and
    returns a tuple of mismatch descriptions -- empty when the module and the artifact agree.

    Exists so the drift check is a real recomputation rather than a second transcription of the same
    numbers into a test file. `tests/htr/knowledge/test_satrn_repetition_knowledge.py` asserts the
    result is empty.
    """
    problems: list[str] = []
    manifest = report["crop_manifest"]

    def note(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    note(
        len(manifest) == CROPS_IN_SAMPLE,
        f"crop count {len(manifest)} != CROPS_IN_SAMPLE {CROPS_IN_SAMPLE}",
    )
    note(
        tuple(row["crop_id"] for row in manifest) == ALL_CROP_IDS,
        "ALL_CROP_IDS does not match the diagnostic's crop order",
    )

    repeated = tuple(
        row["crop_id"] for row in manifest if row["smoke_satrn_text"] == REPEATED_OUTPUT
    )
    note(repeated == REPEATED_CROP_IDS, f"REPEATED_CROP_IDS mismatch: artifact has {repeated}")
    second = tuple(
        row["crop_id"] for row in manifest if row["smoke_satrn_text"] == SECOND_REPEATED_OUTPUT
    )
    note(
        second == SECOND_REPEATED_CROP_IDS,
        f"SECOND_REPEATED_CROP_IDS mismatch: artifact has {second}",
    )
    controls = tuple(row["crop_id"] for row in manifest if not row["repeated_in_smoke"])
    note(
        controls == DISTINCT_OUTPUT_CROP_IDS,
        f"DISTINCT_OUTPUT_CROP_IDS mismatch: artifact has {controls}",
    )

    by_id = {row["crop_id"]: row for row in manifest}
    for crop_id, confidence in REPEATED_GROUP_CONFIDENCES.items():
        actual = by_id[crop_id]["smoke_satrn_confidence"]
        note(
            actual == confidence,
            f"confidence for {crop_id}: module {confidence} != artifact {actual}",
        )
    for crop_id, digest_prefix in REPEATED_CROP_SHA256.items():
        actual = by_id[crop_id]["sha256_png"]
        note(
            actual.startswith(digest_prefix[:16]),
            f"sha256 prefix for {crop_id}: module {digest_prefix[:16]} != artifact {actual[:16]}",
        )

    constancy = report["constancy"]["smoke_test_original"]
    note(
        constancy["distinct_outputs"] == SATRN_DISTINCT_OUTPUTS,
        f"distinct outputs {constancy['distinct_outputs']} != {SATRN_DISTINCT_OUTPUTS}",
    )
    note(
        constancy["distinct_ratio"] == SATRN_DISTINCT_RATIO,
        f"distinct ratio {constancy['distinct_ratio']} != {SATRN_DISTINCT_RATIO}",
    )
    note(
        constancy["most_repeated_output"] == REPEATED_OUTPUT,
        f"most repeated {constancy['most_repeated_output']!r} != {REPEATED_OUTPUT!r}",
    )
    note(
        constancy["most_repeated_count"] == len(REPEATED_CROP_IDS),
        f"most repeated count {constancy['most_repeated_count']} != {len(REPEATED_CROP_IDS)}",
    )

    ratios = [row["aspect_ratio"] for row in manifest]
    note(
        (min(ratios), max(ratios)) == EXTREME_ASPECT_RATIO_SPAN,
        f"aspect ratio span {(min(ratios), max(ratios))} != {EXTREME_ASPECT_RATIO_SPAN}",
    )

    determinism = report["determinism"]
    note(
        all(determinism.values()),
        f"not every rerun pass reproduced the smoke-test text: {determinism}",
    )
    checks = report["integration_checks"]
    note(
        checks["every_call_got_a_fresh_child_process"],
        "diagnostic did not confirm a fresh child process per call",
    )
    note(checks["no_temp_file_in_argv"], "diagnostic did not confirm the absence of a temp file")
    note(
        checks["distinct_model_revisions"] == [SATRN_MODEL_REVISION],
        f"revisions {checks['distinct_model_revisions']} != [{SATRN_MODEL_REVISION}]",
    )
    note(
        checks["distinct_confidences_within_repeated_group"] == len(REPEATED_CROP_IDS),
        "the repeated group did not carry one distinct confidence per crop",
    )

    preprocessing = report["documented_preprocessing"]
    note(
        preprocessing["test_pipeline_resize"] == DOCUMENTED_RESIZE,
        f"documented resize {preprocessing['test_pipeline_resize']!r} != {DOCUMENTED_RESIZE!r}",
    )
    note(
        tuple(preprocessing["data_preprocessor_mean"]) == DOCUMENTED_MEAN,
        "documented normalization mean mismatch",
    )
    note(
        tuple(preprocessing["data_preprocessor_std"]) == DOCUMENTED_STD,
        "documented normalization std mismatch",
    )
    return tuple(problems)
